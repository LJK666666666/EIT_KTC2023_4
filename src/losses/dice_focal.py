"""Dice + Focal combined loss for imbalanced segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_targets(targets, num_classes):
    """Accept class indices (B,H,W) or one-hot targets (B,C,H,W)."""
    if targets.dim() == 4:
        if targets.shape[1] != num_classes:
            raise ValueError(
                f'One-hot targets must have shape (B, {num_classes}, H, W), '
                f'got {tuple(targets.shape)}'
            )
        return torch.argmax(targets, dim=1)
    if targets.dim() == 3:
        return targets
    raise ValueError(
        f'Targets must have shape (B, H, W) or (B, C, H, W), '
        f'got {tuple(targets.shape)}'
    )


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Operates on softmax probabilities vs one-hot targets.
    Smooth term avoids division by zero.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw logits.
            targets: (B, H, W) integer class labels.
        """
        C = logits.shape[1]
        targets = _prepare_targets(targets, C)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        # One-hot encode targets
        targets_oh = F.one_hot(targets.long(), C)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Per-class Dice
        dims = (0, 2, 3)  # reduce over batch and spatial
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples, focuses on hard ones.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights, tensor of shape (C,)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw logits.
            targets: (B, H, W) integer class labels.
        """
        targets = _prepare_targets(targets, logits.shape[1])
        ce = F.cross_entropy(logits, targets.long(), reduction='none')
        p_t = torch.exp(-ce)  # probability of correct class
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets.long()]
            loss = alpha_t * loss

        return loss.mean()


class DiceFocalLoss(nn.Module):
    """Combined 0.5 * Dice + 0.5 * Focal loss.

    Args:
        dice_weight: Weight for Dice loss component.
        focal_weight: Weight for Focal loss component.
        focal_gamma: Focal loss gamma (focusing parameter).
        focal_alpha: Optional per-class weights for Focal loss.
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5,
                 focal_gamma=2.0, focal_alpha=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice(logits, targets) +
                self.focal_weight * self.focal(logits, targets))
