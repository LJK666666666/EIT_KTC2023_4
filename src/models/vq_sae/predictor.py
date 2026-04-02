"""MLP predictor: measurements -> discrete slot logits + angle."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQMeasurementPredictor(nn.Module):
    def __init__(self, input_dim=2356, hidden_dims=(512, 256, 128),
                 num_slots=16, codebook_size=512, dropout=0.1,
                 slot_num_classes=None):
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
        self.slot_num_classes = (list(slot_num_classes)
                                 if slot_num_classes is not None else None)
        if self.slot_num_classes is None:
            self.head_slots = nn.Linear(in_dim, num_slots * codebook_size)
            self.slot_heads = None
        else:
            self.head_slots = None
            self.slot_heads = nn.ModuleList([
                nn.Linear(in_dim, int(num_classes))
                for num_classes in self.slot_num_classes
            ])
        self.head_angle = nn.Linear(in_dim, 2)
        self.num_slots = num_slots
        self.codebook_size = codebook_size

    def forward(self, x):
        feat = self.backbone(x)
        if self.slot_heads is None:
            slot_logits = self.head_slots(feat).view(
                -1, self.num_slots, self.codebook_size)
        else:
            slot_logits = [head(feat) for head in self.slot_heads]
        angle_xy = F.normalize(self.head_angle(feat), p=2, dim=-1)
        return slot_logits, angle_xy
