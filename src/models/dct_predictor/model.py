"""
Fixed-basis DCT low-frequency predictor for EIT segmentation.
"""

import math

import torch
import torch.nn as nn


def build_dct_basis(n: int, k: int) -> torch.Tensor:
    xs = torch.arange(n, dtype=torch.float32)
    rows = []
    for i in range(k):
        alpha = math.sqrt(1.0 / n) if i == 0 else math.sqrt(2.0 / n)
        rows.append(alpha * torch.cos(math.pi * (xs + 0.5) * i / n))
    return torch.stack(rows, dim=0)


class DCTDecoder(nn.Module):
    def __init__(self, image_size=256, coeff_size=16, out_channels=3):
        super().__init__()
        self.image_size = image_size
        self.coeff_size = coeff_size
        self.out_channels = out_channels
        self.register_buffer('basis', build_dct_basis(image_size, coeff_size))

    def coeffs_to_logits(self, coeffs: torch.Tensor) -> torch.Tensor:
        basis = self.basis.float()
        coeffs_f = coeffs.float()
        logits = torch.einsum('kn,bckl,lm->bcnm', basis, coeffs_f, basis)
        return logits.to(coeffs.dtype)

    def images_to_coeffs(self, images: torch.Tensor) -> torch.Tensor:
        basis = self.basis.float()
        images_f = images.float()
        coeffs = torch.einsum('kn,bcnm,lm->bckl', basis, images_f, basis)
        return coeffs.to(images.dtype)


class DCTPredictor(nn.Module):
    def __init__(self,
                 input_dim=2356,
                 hidden_dims=(1024, 512, 512),
                 level_embed_dim=32,
                 coeff_size=16,
                 out_channels=3,
                 dropout=0.1):
        super().__init__()
        self.coeff_size = coeff_size
        self.out_channels = out_channels
        self.level_embed = nn.Embedding(8, level_embed_dim)

        dims = [input_dim + level_embed_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(
            hidden_dims[-1], out_channels * coeff_size * coeff_size)
        self.decoder = DCTDecoder(
            image_size=256,
            coeff_size=coeff_size,
            out_channels=out_channels,
        )

    def _level_features(self, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.long().clamp(min=1, max=7)
        return self.level_embed(levels)

    def forward(self, measurements: torch.Tensor, levels: torch.Tensor):
        feat = torch.cat([measurements, self._level_features(levels)], dim=1)
        feat = self.backbone(feat)
        coeffs = self.head(feat).view(
            measurements.shape[0],
            self.out_channels,
            self.coeff_size,
            self.coeff_size,
        )
        logits = self.decoder.coeffs_to_logits(coeffs)
        return logits, coeffs

    def target_coeffs(self, gt_onehot: torch.Tensor) -> torch.Tensor:
        return self.decoder.images_to_coeffs(gt_onehot)
