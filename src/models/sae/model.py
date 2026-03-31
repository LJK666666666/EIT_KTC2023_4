"""
Spatial-Transformer Sparse AutoEncoder (ST-SAE) for EIT conductivity images.

Architecture:
  1. AngleCNN: predict rotation angle [cosθ, sinθ] from original image
  2. STN: rotate input by -θ to canonical orientation
  3. EncoderCNN: compress canonical image to z_shape (63-dim)
  4. Decoder: decompress z_shape to canonical logits (3, 256, 256)
  5. STN: rotate logits by +θ to restore original orientation

Key design:
  - Decoder only sees z_shape (63-dim), θ only participates in geometric transforms
  - Angle represented as 2D unit vector [cosθ, sinθ] (avoids 0°/360° discontinuity)
  - Training uses bilinear interpolation (gradient flow), inference uses nearest
  - L1 sparsity only on z_shape, never on angle
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleCNN(nn.Module):
    """Predict rotation angle as 2D unit vector [cosθ, sinθ] from input image."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=4, padding=2),  # 256→64
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 5, stride=4, padding=2),           # 64→16
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=4, padding=1),           # 16→4
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),                              # 4→1
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        """Returns L2-normalized [cosθ, sinθ]."""
        feat = self.features(x).flatten(1)  # (B, 64)
        angle_xy = self.head(feat)           # (B, 2)
        angle_xy = F.normalize(angle_xy, p=2, dim=-1)  # project to unit circle
        return angle_xy


class EncoderCNN(nn.Module):
    """Compress canonical image to z_shape vector."""

    def __init__(self, in_channels=3, channels=(32, 64, 128, 256),
                 shape_dim=63):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        # After 4x stride-2: 256→128→64→32→16, so 256*16*16
        self.fc = nn.Linear(channels[-1] * 16 * 16, shape_dim)

    def forward(self, x):
        feat = self.conv(x)        # (B, 256, 16, 16)
        return self.fc(feat.flatten(1))  # (B, 63)


class Decoder(nn.Module):
    """Decompress z_shape to canonical logits.

    Uses gradual expansion: FC → 4×4 → 6 upsample steps → 256×256
    Final 1×1 Conv from 8→3 for smooth channel transition.
    """

    def __init__(self, shape_dim=63, start_ch=256, start_size=4):
        super().__init__()
        self.start_ch = start_ch
        self.start_size = start_size
        self.fc = nn.Linear(shape_dim, start_ch * start_size * start_size)

        # 6 upsample steps: 4→8→16→32→64→128→256
        # channels:        256→256→128→64→32→16→8
        self.up = nn.Sequential(
            self._up_block(256, 256),   # 4→8
            self._up_block(256, 128),   # 8→16
            self._up_block(128, 64),    # 16→32
            self._up_block(64, 32),     # 32→64
            self._up_block(32, 16),     # 64→128
            self._up_block(16, 8),      # 128→256
        )
        # Smooth channel transition: 8→3 (no activation, raw logits)
        self.out_conv = nn.Conv2d(8, 3, 1)

    @staticmethod
    def _up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, z_shape):
        x = self.fc(z_shape)
        x = x.view(-1, self.start_ch, self.start_size, self.start_size)
        x = self.up(x)           # (B, 8, 256, 256)
        return self.out_conv(x)  # (B, 3, 256, 256) raw logits


def _build_rotation_matrix(theta):
    """Build 2×3 affine matrix for counter-clockwise rotation by theta (radians).

    Args:
        theta: (B,) tensor of angles in radians.

    Returns:
        (B, 2, 3) affine matrix for F.affine_grid.
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zeros = torch.zeros_like(theta)

    # Standard 2D rotation (no translation)
    row1 = torch.stack([cos_t, -sin_t, zeros], dim=1)
    row2 = torch.stack([sin_t, cos_t, zeros], dim=1)
    return torch.stack([row1, row2], dim=1)  # (B, 2, 3)


class SparseAutoEncoder(nn.Module):
    """ST-SAE: Spatial-Transformer Sparse AutoEncoder.

    Encodes images into 65-dim latent: z_shape (63) + angle_xy (2).
    Decoder only sees z_shape; angle only participates in STN rotation.
    """

    def __init__(self, in_channels=3, encoder_channels=(32, 64, 128, 256),
                 shape_dim=63, decoder_start_size=4):
        super().__init__()
        self.shape_dim = shape_dim

        self.angle_cnn = AngleCNN(in_channels)
        self.encoder = EncoderCNN(in_channels, encoder_channels, shape_dim)
        self.decoder = Decoder(shape_dim, start_ch=encoder_channels[-1],
                               start_size=decoder_start_size)

    def _stn_rotate(self, image, theta, inverse=False):
        """Differentiable rotation via Spatial Transformer Network.

        Training: bilinear (gradient flows to AngleCNN).
        Inference: nearest (preserves one-hot discreteness).
        """
        if inverse:
            theta = -theta

        affine = _build_rotation_matrix(theta)  # (B, 2, 3)
        grid = F.affine_grid(affine, image.shape, align_corners=False)

        mode = 'bilinear' if self.training else 'nearest'
        return F.grid_sample(image, grid, mode=mode,
                             padding_mode='zeros', align_corners=False)

    def encode(self, x):
        """Encode image to (z_shape, angle_xy).

        Args:
            x: (B, 3, 256, 256) one-hot image.

        Returns:
            z_shape: (B, 63) rotation-invariant shape features.
            angle_xy: (B, 2) normalized [cosθ, sinθ].
        """
        angle_xy = self.angle_cnn(x)                    # (B, 2)
        theta = torch.atan2(angle_xy[:, 1], angle_xy[:, 0])  # (B,)
        x_aligned = self._stn_rotate(x, theta, inverse=True)  # rotate by -θ
        z_shape = self.encoder(x_aligned)                # (B, 63)
        return z_shape, angle_xy

    def decode(self, z_shape, angle_xy):
        """Decode z_shape to output logits, rotated by +θ.

        Args:
            z_shape: (B, 63) shape features.
            angle_xy: (B, 2) normalized [cosθ, sinθ].

        Returns:
            logits: (B, 3, 256, 256) raw logits.
        """
        canonical_logits = self.decoder(z_shape)  # (B, 3, 256, 256)
        theta = torch.atan2(angle_xy[:, 1], angle_xy[:, 0])
        # Rotate canonical output by +θ (always bilinear for logits)
        affine = _build_rotation_matrix(theta)
        grid = F.affine_grid(affine, canonical_logits.shape,
                             align_corners=False)
        logits = F.grid_sample(canonical_logits, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False)
        return logits

    def forward(self, x):
        """Full forward pass: encode → decode.

        Returns:
            logits: (B, 3, 256, 256) raw logits.
            z_shape: (B, 63) for L1 sparsity.
            angle_xy: (B, 2) for storage.
        """
        z_shape, angle_xy = self.encode(x)
        logits = self.decode(z_shape, angle_xy)
        return logits, z_shape, angle_xy
