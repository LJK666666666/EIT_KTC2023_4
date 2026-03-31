"""MLP predictor: measurements -> discrete slot logits + angle."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQMeasurementPredictor(nn.Module):
    def __init__(self, input_dim=2356, hidden_dims=(512, 256, 128),
                 num_slots=16, codebook_size=512, dropout=0.1):
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
        self.head_slots = nn.Linear(in_dim, num_slots * codebook_size)
        self.head_angle = nn.Linear(in_dim, 2)
        self.num_slots = num_slots
        self.codebook_size = codebook_size

    def forward(self, x):
        feat = self.backbone(x)
        slot_logits = self.head_slots(feat).view(
            -1, self.num_slots, self.codebook_size)
        angle_xy = F.normalize(self.head_angle(feat), p=2, dim=-1)
        return slot_logits, angle_xy
