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


class ChangeGatedDCTPredictor(nn.Module):
    """DCT predictor with a sample-level change gate.

    The gate is intended for time-difference pulmonary EIT where a non-trivial
    fraction of samples may contain nearly zero conductivity change. The model
    predicts low-frequency DCT coefficients as usual, and also predicts a
    scalar change probability used to suppress false positives on no-change
    samples.
    """

    def __init__(self,
                 input_dim=2356,
                 hidden_dims=(1024, 512, 512),
                 level_embed_dim=32,
                 coeff_size=16,
                 out_channels=1,
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
        hidden_last = hidden_dims[-1]
        self.coeff_head = nn.Linear(
            hidden_last, out_channels * coeff_size * coeff_size)
        self.gate_head = nn.Linear(hidden_last, 1)
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
        coeffs_raw = self.coeff_head(feat).view(
            measurements.shape[0],
            self.out_channels,
            self.coeff_size,
            self.coeff_size,
        )
        gate_logits = self.gate_head(feat)
        gate = torch.sigmoid(gate_logits).view(-1, 1, 1, 1)
        coeffs = coeffs_raw * gate
        logits = self.decoder.coeffs_to_logits(coeffs)
        return logits, coeffs, gate_logits

    def target_coeffs(self, gt_onehot: torch.Tensor) -> torch.Tensor:
        return self.decoder.images_to_coeffs(gt_onehot)


class SpatialChangeGatedDCTPredictor(nn.Module):
    """DCT predictor with a spatial change mask branch.

    The model predicts:
      1. low-frequency conductivity-change coefficients
      2. low-frequency change-mask coefficients

    The final conductivity prediction is obtained by multiplying the decoded
    residual field with the sigmoid of the decoded mask logits.
    """

    def __init__(self,
                 input_dim=2356,
                 hidden_dims=(1024, 512, 512),
                 level_embed_dim=32,
                 coeff_size=16,
                 out_channels=1,
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
        hidden_last = hidden_dims[-1]
        self.coeff_head = nn.Linear(
            hidden_last, out_channels * coeff_size * coeff_size)
        self.mask_head = nn.Linear(hidden_last, coeff_size * coeff_size)
        self.decoder = DCTDecoder(
            image_size=256,
            coeff_size=coeff_size,
            out_channels=out_channels,
        )
        self.mask_decoder = DCTDecoder(
            image_size=256,
            coeff_size=coeff_size,
            out_channels=1,
        )

    def _level_features(self, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.long().clamp(min=1, max=7)
        return self.level_embed(levels)

    def forward(self, measurements: torch.Tensor, levels: torch.Tensor):
        feat = torch.cat([measurements, self._level_features(levels)], dim=1)
        feat = self.backbone(feat)
        coeffs = self.coeff_head(feat).view(
            measurements.shape[0],
            self.out_channels,
            self.coeff_size,
            self.coeff_size,
        )
        mask_coeffs = self.mask_head(feat).view(
            measurements.shape[0], 1, self.coeff_size, self.coeff_size
        )
        residual = self.decoder.coeffs_to_logits(coeffs)
        mask_logits = self.mask_decoder.coeffs_to_logits(mask_coeffs)
        pred = residual * torch.sigmoid(mask_logits)
        return pred, coeffs, mask_logits

    def target_coeffs(self, gt_onehot: torch.Tensor) -> torch.Tensor:
        return self.decoder.images_to_coeffs(gt_onehot)


class ConditionalSpatialChangeDCTPredictor(nn.Module):
    """Spatial mask conditioned residual DCT predictor.

    Compared with :class:`SpatialChangeGatedDCTPredictor`, the residual
    coefficient branch is explicitly conditioned on the predicted spatial
    change-mask coefficients. During training, mask coefficient teacher forcing
    can be used by passing ``mask_coeffs_override``.
    """

    def __init__(self,
                 input_dim=2356,
                 hidden_dims=(1024, 512, 512),
                 level_embed_dim=32,
                 coeff_size=16,
                 out_channels=1,
                 dropout=0.1,
                 mask_condition_dim=128):
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
        hidden_last = hidden_dims[-1]
        self.mask_head = nn.Linear(hidden_last, coeff_size * coeff_size)
        self.mask_condition = nn.Sequential(
            nn.Linear(coeff_size * coeff_size, mask_condition_dim),
            nn.LayerNorm(mask_condition_dim),
            nn.GELU(),
        )
        self.coeff_head = nn.Linear(
            hidden_last + mask_condition_dim,
            out_channels * coeff_size * coeff_size,
        )
        self.decoder = DCTDecoder(
            image_size=256,
            coeff_size=coeff_size,
            out_channels=out_channels,
        )
        self.mask_decoder = DCTDecoder(
            image_size=256,
            coeff_size=coeff_size,
            out_channels=1,
        )

    def _level_features(self, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.long().clamp(min=1, max=7)
        return self.level_embed(levels)

    def forward(self, measurements: torch.Tensor, levels: torch.Tensor,
                mask_coeffs_override: torch.Tensor | None = None):
        feat = torch.cat([measurements, self._level_features(levels)], dim=1)
        feat = self.backbone(feat)
        mask_coeffs_pred = self.mask_head(feat).view(
            measurements.shape[0], 1, self.coeff_size, self.coeff_size
        )
        cond_coeffs = mask_coeffs_pred
        if mask_coeffs_override is not None:
            cond_coeffs = mask_coeffs_override
        mask_embed = self.mask_condition(cond_coeffs.reshape(measurements.shape[0], -1))
        coeff_feat = torch.cat([feat, mask_embed], dim=1)
        coeffs = self.coeff_head(coeff_feat).view(
            measurements.shape[0],
            self.out_channels,
            self.coeff_size,
            self.coeff_size,
        )
        residual = self.decoder.coeffs_to_logits(coeffs)
        mask_logits = self.mask_decoder.coeffs_to_logits(mask_coeffs_pred)
        pred = residual * torch.sigmoid(mask_logits)
        return pred, coeffs, mask_logits, mask_coeffs_pred

    def target_coeffs(self, gt_onehot: torch.Tensor) -> torch.Tensor:
        return self.decoder.images_to_coeffs(gt_onehot)

    def target_mask_coeffs(self, mask_target: torch.Tensor) -> torch.Tensor:
        return self.mask_decoder.images_to_coeffs(mask_target)


class MaskOnlyDCTPredictor(nn.Module):
    """Predict a low-frequency spatial change mask from TD16 measurements."""

    def __init__(self,
                 input_dim=208,
                 hidden_dims=(512, 256, 256),
                 level_embed_dim=16,
                 coeff_size=24,
                 dropout=0.1):
        super().__init__()
        self.coeff_size = coeff_size
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
        self.mask_head = nn.Linear(hidden_dims[-1], coeff_size * coeff_size)
        self.mask_decoder = DCTDecoder(
            image_size=256,
            coeff_size=coeff_size,
            out_channels=1,
        )

    def _level_features(self, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.long().clamp(min=1, max=7)
        return self.level_embed(levels)

    def forward(self, measurements: torch.Tensor, levels: torch.Tensor):
        feat = torch.cat([measurements, self._level_features(levels)], dim=1)
        feat = self.backbone(feat)
        mask_coeffs = self.mask_head(feat).view(
            measurements.shape[0], 1, self.coeff_size, self.coeff_size
        )
        mask_logits = self.mask_decoder.coeffs_to_logits(mask_coeffs)
        return mask_logits, mask_coeffs

    def target_mask_coeffs(self, mask_target: torch.Tensor) -> torch.Tensor:
        return self.mask_decoder.images_to_coeffs(mask_target)


class AtlasResidualDCTPredictor(nn.Module):
    """Predict residual DCT coefficients around a fixed atlas image.

    This is intended for pulmonary conductivity regression where the global
    thorax layout is highly stable and the task is dominated by local
    deviations around a canonical mean conductivity map.
    """

    def __init__(self,
                 input_dim=2356,
                 hidden_dims=(1024, 512, 512),
                 level_embed_dim=32,
                 coeff_size=32,
                 out_channels=1,
                 dropout=0.1,
                 image_size=256):
        super().__init__()
        self.predictor = DCTPredictor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            level_embed_dim=level_embed_dim,
            coeff_size=coeff_size,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.image_size = image_size
        self.out_channels = out_channels
        self.register_buffer(
            'atlas',
            torch.zeros(1, out_channels, image_size, image_size, dtype=torch.float32),
        )

    def set_atlas(self, atlas: torch.Tensor) -> None:
        atlas_t = torch.as_tensor(atlas, dtype=torch.float32)
        if atlas_t.ndim == 2:
            atlas_t = atlas_t.unsqueeze(0).unsqueeze(0)
        elif atlas_t.ndim == 3:
            atlas_t = atlas_t.unsqueeze(0)
        if atlas_t.shape != self.atlas.shape:
            raise ValueError(
                f'Atlas shape {tuple(atlas_t.shape)} does not match '
                f'{tuple(self.atlas.shape)}'
            )
        self.atlas.copy_(atlas_t)

    def forward(self, measurements: torch.Tensor, levels: torch.Tensor):
        residual, coeffs = self.predictor(measurements, levels)
        atlas = self.atlas.to(device=residual.device, dtype=residual.dtype)
        return atlas + residual, coeffs

    def target_coeffs(self, target_images: torch.Tensor) -> torch.Tensor:
        atlas = self.atlas.to(device=target_images.device, dtype=target_images.dtype)
        residual = target_images - atlas
        return self.predictor.decoder.images_to_coeffs(residual)


class _ResidualDecoder(nn.Module):
    def __init__(self, in_channels=32, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class AtlasRefineDCTPredictor(nn.Module):
    """Pulmonary-specific coarse-to-fine predictor.

    The model predicts:
      1. a coarse residual around a fixed atlas using low-frequency DCT
      2. a learned local refinement residual in image space
    """

    def __init__(self,
                 input_dim=2356,
                 hidden_dims=(1024, 512, 512),
                 level_embed_dim=32,
                 coeff_size=20,
                 out_channels=1,
                 dropout=0.1,
                 image_size=256,
                 refine_channels=32,
                 refine_seed_size=16):
        super().__init__()
        self.coeff_size = coeff_size
        self.out_channels = out_channels
        self.image_size = image_size
        self.refine_channels = refine_channels
        self.refine_seed_size = refine_seed_size

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
        hidden_last = hidden_dims[-1]
        self.coeff_head = nn.Linear(
            hidden_last, out_channels * coeff_size * coeff_size)
        self.refine_seed_head = nn.Linear(
            hidden_last,
            refine_channels * refine_seed_size * refine_seed_size,
        )
        self.decoder = DCTDecoder(
            image_size=image_size,
            coeff_size=coeff_size,
            out_channels=out_channels,
        )
        self.refine_decoder = _ResidualDecoder(
            in_channels=refine_channels,
            out_channels=out_channels,
        )
        self.refine_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer(
            'atlas',
            torch.zeros(1, out_channels, image_size, image_size, dtype=torch.float32),
        )

    def _level_features(self, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.long().clamp(min=1, max=7)
        return self.level_embed(levels)

    def set_atlas(self, atlas: torch.Tensor) -> None:
        atlas_t = torch.as_tensor(atlas, dtype=torch.float32)
        if atlas_t.ndim == 2:
            atlas_t = atlas_t.unsqueeze(0).unsqueeze(0)
        elif atlas_t.ndim == 3:
            atlas_t = atlas_t.unsqueeze(0)
        if atlas_t.shape != self.atlas.shape:
            raise ValueError(
                f'Atlas shape {tuple(atlas_t.shape)} does not match '
                f'{tuple(self.atlas.shape)}'
            )
        self.atlas.copy_(atlas_t)

    def forward(self, measurements: torch.Tensor, levels: torch.Tensor):
        feat = torch.cat([measurements, self._level_features(levels)], dim=1)
        feat = self.backbone(feat)
        coeffs = self.coeff_head(feat).view(
            measurements.shape[0],
            self.out_channels,
            self.coeff_size,
            self.coeff_size,
        )
        coarse = self.decoder.coeffs_to_logits(coeffs)
        seed = self.refine_seed_head(feat).view(
            measurements.shape[0],
            self.refine_channels,
            self.refine_seed_size,
            self.refine_seed_size,
        )
        refine = self.refine_decoder(seed)
        atlas = self.atlas.to(device=coarse.device, dtype=coarse.dtype)
        pred = atlas + coarse + self.refine_scale.to(coarse.dtype) * refine
        return pred, coeffs

    def target_coeffs(self, target_images: torch.Tensor) -> torch.Tensor:
        atlas = self.atlas.to(device=target_images.device, dtype=target_images.dtype)
        residual = target_images - atlas
        return self.decoder.images_to_coeffs(residual)


class AtlasResidualDecoderPredictor(nn.Module):
    """Atlas-aware direct residual image predictor without a fixed DCT basis."""

    def __init__(self,
                 input_dim=2356,
                 hidden_dims=(1024, 512, 512),
                 level_embed_dim=32,
                 out_channels=1,
                 dropout=0.1,
                 image_size=256,
                 seed_channels=32,
                 seed_size=16):
        super().__init__()
        self.out_channels = out_channels
        self.image_size = image_size
        self.seed_channels = seed_channels
        self.seed_size = seed_size

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
        self.seed_head = nn.Linear(
            hidden_dims[-1], seed_channels * seed_size * seed_size)
        self.decoder = _ResidualDecoder(
            in_channels=seed_channels,
            out_channels=out_channels,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer(
            'atlas',
            torch.zeros(1, out_channels, image_size, image_size, dtype=torch.float32),
        )

    def _level_features(self, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.long().clamp(min=1, max=7)
        return self.level_embed(levels)

    def set_atlas(self, atlas: torch.Tensor) -> None:
        atlas_t = torch.as_tensor(atlas, dtype=torch.float32)
        if atlas_t.ndim == 2:
            atlas_t = atlas_t.unsqueeze(0).unsqueeze(0)
        elif atlas_t.ndim == 3:
            atlas_t = atlas_t.unsqueeze(0)
        if atlas_t.shape != self.atlas.shape:
            raise ValueError(
                f'Atlas shape {tuple(atlas_t.shape)} does not match '
                f'{tuple(self.atlas.shape)}'
            )
        self.atlas.copy_(atlas_t)

    def forward(self, measurements: torch.Tensor, levels: torch.Tensor):
        feat = torch.cat([measurements, self._level_features(levels)], dim=1)
        feat = self.backbone(feat)
        seed = self.seed_head(feat).view(
            measurements.shape[0],
            self.seed_channels,
            self.seed_size,
            self.seed_size,
        )
        residual = self.decoder(seed)
        atlas = self.atlas.to(device=residual.device, dtype=residual.dtype)
        pred = atlas + self.residual_scale.to(residual.dtype) * residual
        return pred
