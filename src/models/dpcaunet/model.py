"""
Dual-Pooling Cross-Attention UNet (DPCA-UNet).

Architecture:
  1. Electrode Encoder: measurements (B,2356) → K,V (B,31,d_model)
  2. Spatial Query MLP: position grid (H*W,5) → Q (H*W,d_model)
  3. Multi-Head Cross-Attention with mask: Q,K,V → feature map (B,d_model,H,W)
  4. Dual-Pooling UNet: MaxPool+MinPool encoder → decoder → (B,3,H,W)

Electrode layout:
  - 32 electrodes on circular boundary
  - Electrode 1 at top (90°), counterclockwise, 11.25° spacing
  - 31 differential channels (adjacent pairs)
  - Level 1-7 progressively removes channels (vincl masking)

Angle convention:
  - Standard math: 0° = right (+x), counterclockwise positive
  - Electrode 1 center = 90°, Electrode 2 = 101.25°, ...
  - Pixel angles use atan2(y, x) — same convention, naturally aligned
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Electrode angle constants
# ---------------------------------------------------------------

def _compute_electrode_angles(n_electrodes=32):
    """Compute electrode center angles in radians (standard math convention).

    Electrode 1 at top (90°), counterclockwise, 11.25° spacing.
    Returns angles for all electrodes and midpoint angles for 31
    differential channels.
    """
    # Electrode centers: E1=90°, E2=101.25°, E3=112.5°, ...
    angles_deg = 90.0 + np.arange(n_electrodes) * (360.0 / n_electrodes)
    angles_rad = np.deg2rad(angles_deg)

    # 31 differential channels: midpoint of (E_i, E_{i+1})
    mid_angles = np.empty(n_electrodes - 1)
    for i in range(n_electrodes - 1):
        a1, a2 = angles_rad[i], angles_rad[i + 1]
        mid_angles[i] = a1 + (a2 - a1) / 2  # always 5.625° offset

    return angles_rad, mid_angles


# ---------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------

class ElectrodeEncoder(nn.Module):
    """Encode per-channel measurements + angle info into K and V.

    Input:  (B, 31, 78)  — 76 measurements + cos(θ) + sin(θ) per channel
    Output: K (B, 31, d_model), V (B, 31, d_model)
    """

    def __init__(self, input_dim=78, d_model=64):
        super().__init__()
        self.key_proj = nn.Linear(input_dim, d_model)
        self.val_proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.key_proj(x), self.val_proj(x)


class SpatialQueryMLP(nn.Module):
    """Generate spatial query vectors from position encoding.

    Pre-computes a (H*W, 5) position grid and learns a mapping to d_model.
    Position features: normalized x, y, distance to center, cos(θ), sin(θ).
    Angle uses atan2(y, x) — same convention as electrode angles.
    """

    def __init__(self, d_model=64, im_size=256, hidden_dim=64):
        super().__init__()
        self.im_size = im_size
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Pre-compute position encoding
        # Physical domain: [-0.115, 0.115], radius 0.098
        # Normalize so tank radius = 1.0
        pix_width = 0.23 / im_size
        coords = np.linspace(
            -0.115 + pix_width / 2, 0.115 - pix_width / 2 + pix_width, im_size)
        gx, gy = np.meshgrid(coords, coords, indexing='ij')
        scale = 0.098
        x_norm = gx / scale
        y_norm = gy / scale
        dist = np.sqrt(x_norm ** 2 + y_norm ** 2)
        # atan2(y, x): 0°=right, 90°=top — matches electrode convention
        angle = np.arctan2(y_norm, x_norm)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        pos = np.stack([x_norm.ravel(), y_norm.ravel(), dist.ravel(),
                        cos_a.ravel(), sin_a.ravel()], axis=-1).astype(np.float32)
        self.register_buffer('pos_encoding', torch.from_numpy(pos))

    def forward(self, batch_size):
        """Returns Q: (B, H*W, d_model)."""
        q = self.mlp(self.pos_encoding)
        return q.unsqueeze(0).expand(batch_size, -1, -1)


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention with optional key mask.

    Q: (B, N_q, d_model)  — spatial queries (N_q = H*W)
    K: (B, N_k, d_model)  — electrode keys (N_k = 31)
    V: (B, N_k, d_model)  — electrode values
    mask: (B, N_k) bool    — True = valid, False = masked out
    Output: (B, N_q, d_model)
    """

    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, Nq, D = Q.shape
        Nk = K.shape[1]

        Q = Q.view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B,h,Nq,Nk)

        # Apply key mask: masked channels get -inf before softmax
        if mask is not None:
            # mask: (B, Nk) → (B, 1, 1, Nk)
            attn_mask = mask[:, None, None, :]  # True=keep, False=mask
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from all-masked rows (shouldn't happen in practice)
        attn = attn.nan_to_num(0.0)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.out_proj(out)


# ---------------------------------------------------------------
# Dual-Pooling UNet building blocks
# ---------------------------------------------------------------

class DualPool(nn.Module):
    """MaxPool + MinPool → channel concatenation (doubles channels)."""

    def __init__(self, kernel_size=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        max_p = self.pool(x)
        min_p = -self.pool(-x)
        return torch.cat([max_p, min_p], dim=1)


class ConvBlock(nn.Module):
    """Two convolutions with BatchNorm and GELU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """DualPool → ConvBlock.  Input ch → 2*ch (from dual pool) → out_ch."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dual_pool = DualPool()
        self.conv = ConvBlock(in_ch * 2, out_ch)

    def forward(self, x):
        return self.conv(self.dual_pool(x))


class DecoderBlock(nn.Module):
    """Upsample → cat skip → ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------
# Main model
# ---------------------------------------------------------------

class DPCAUNet(nn.Module):
    """Dual-Pooling Cross-Attention UNet for EIT reconstruction.

    Args:
        n_channels: Number of measurement channels (31 for 32-electrode EIT).
        n_patterns: Number of excitation patterns (76).
        d_model: Hidden dimension for attention and initial feature map channels.
        n_heads: Number of attention heads.
        im_size: Output image size (256).
        encoder_channels: Tuple of channel sizes for each encoder level.
        out_channels: Number of output classes (3).
        max_period: Timestep embedding frequency (same as FCUNet).
    """

    def __init__(self, n_channels=31, n_patterns=76, d_model=64, n_heads=4,
                 im_size=256, encoder_channels=(64, 128, 256),
                 out_channels=3, max_period=0.25):
        super().__init__()
        self.n_channels = n_channels
        self.n_patterns = n_patterns
        self.d_model = d_model
        self.im_size = im_size
        self.max_period = max_period

        # Electrode angle encoding
        # E1 at top (90°), counterclockwise, 11.25° spacing
        _, mid_angles = _compute_electrode_angles(n_channels + 1)
        self.register_buffer('electrode_cos',
                             torch.from_numpy(np.cos(mid_angles).astype(np.float32)))
        self.register_buffer('electrode_sin',
                             torch.from_numpy(np.sin(mid_angles).astype(np.float32)))

        # 1. Electrode encoder: (B, 31, 78) → K, V
        self.electrode_encoder = ElectrodeEncoder(
            input_dim=n_patterns + 2, d_model=d_model)

        # 2. Spatial query MLP
        self.spatial_query = SpatialQueryMLP(
            d_model=d_model, im_size=im_size)

        # 3. Cross-attention (with mask support)
        self.cross_attn = MultiHeadCrossAttention(
            d_model=d_model, n_heads=n_heads)

        # 4. Level embedding
        self.level_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # 5. Initial conv
        self.initial_conv = ConvBlock(d_model, encoder_channels[0])

        # 6. Encoder (dual-pooling)
        self.encoders = nn.ModuleList()
        enc_in = [encoder_channels[0]] + list(encoder_channels[:-1])
        for c_in, c_out in zip(enc_in, encoder_channels):
            self.encoders.append(EncoderBlock(c_in, c_out))

        # 7. Bottleneck
        self.bottleneck = ConvBlock(encoder_channels[-1], encoder_channels[-1])

        # 8. Decoder
        self.decoders = nn.ModuleList()
        dec_channels = list(reversed(encoder_channels))
        for i in range(len(dec_channels) - 1):
            self.decoders.append(
                DecoderBlock(dec_channels[i], dec_channels[i + 1],
                             dec_channels[i + 1]))
        self.decoders.append(
            DecoderBlock(dec_channels[-1], encoder_channels[0],
                         encoder_channels[0]))

        # 9. Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0], out_channels, 1),
        )

    def _timestep_embedding(self, timesteps, dim):
        """Sinusoidal timestep embedding (same as FCUNet)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(
                half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def _build_channel_mask(self, measurements):
        """Detect active channels from vincl-masked measurements.

        A channel is active if any of its 76 pattern values is non-zero.

        Args:
            measurements: (B, 2356) with vincl masking already applied.

        Returns:
            mask: (B, 31) bool tensor, True = active channel.
        """
        x = measurements.view(-1, self.n_channels, self.n_patterns)  # (B,31,76)
        # A channel is active if it has at least one non-zero measurement
        return x.abs().sum(dim=-1) > 0  # (B, 31)

    def forward(self, measurements, level):
        """
        Args:
            measurements: (B, 2356) flattened voltage differences
                          (vincl masking already applied by trainer/pipeline).
            level: (B,) difficulty level (1-7).

        Returns:
            (B, 3, 256, 256) logits for 3-class segmentation.
        """
        B = measurements.shape[0]

        # --- 1. Build electrode input: (B, 31, 78) ---
        x = measurements.view(B, self.n_channels, self.n_patterns)
        cos_enc = self.electrode_cos.view(1, -1, 1).expand(B, -1, -1)
        sin_enc = self.electrode_sin.view(1, -1, 1).expand(B, -1, -1)
        x = torch.cat([x, cos_enc, sin_enc], dim=-1)  # (B, 31, 78)

        K, V = self.electrode_encoder(x)

        # --- 2. Channel mask (from vincl masking) ---
        channel_mask = self._build_channel_mask(measurements)  # (B, 31)

        # --- 3. Spatial queries ---
        Q = self.spatial_query(B)

        # --- 4. Masked cross-attention → feature map ---
        feat = self.cross_attn(Q, K, V, mask=channel_mask)
        feat = feat.view(B, self.im_size, self.im_size,
                         self.d_model).permute(0, 3, 1, 2)

        # --- 5. Level embedding (additive) ---
        level_emb = self._timestep_embedding(level, self.d_model)
        level_emb = self.level_embed(level_emb)
        feat = feat + level_emb[:, :, None, None]

        # --- 6. UNet ---
        x0 = self.initial_conv(feat)

        skips = [x0]
        h = x0
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        h = self.bottleneck(h)

        skips = skips[:-1]
        for dec in self.decoders:
            h = dec(h, skips.pop())

        return self.output_head(h)
