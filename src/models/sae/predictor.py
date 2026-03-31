"""
Pure MLP predictor: electrode measurements → latent code (z_shape + angle_xy).

Uses LayerNorm (batch-size insensitive) and dual output heads.
No HC-DPCA or harmonic encoding — avoids absolute coordinate bias
that would break the rotational decoupling of ST-SAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeasurementPredictor(nn.Module):
    """MLP predictor: measurements (2356) → z_shape (63) + angle_xy (2).

    Args:
        input_dim: Measurement vector dimension (default 2356).
        hidden_dims: Tuple of hidden layer sizes.
        shape_dim: z_shape output dimension (default 63).
        dropout: Dropout rate.
    """

    def __init__(self, input_dim=2356, hidden_dims=(512, 256, 128),
                 shape_dim=63, dropout=0.1):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        self.head_shape = nn.Linear(in_dim, shape_dim)
        self.head_angle = nn.Linear(in_dim, 2)

    def forward(self, x):
        """
        Args:
            x: (B, 2356) measurement vector.

        Returns:
            z_shape: (B, 63) predicted shape features.
            angle_xy: (B, 2) predicted [cosθ, sinθ], L2-normalized.
        """
        features = self.backbone(x)
        z_shape = self.head_shape(features)
        angle_xy = F.normalize(self.head_angle(features), p=2, dim=-1)
        return z_shape, angle_xy
