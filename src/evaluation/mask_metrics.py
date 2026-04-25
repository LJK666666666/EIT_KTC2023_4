"""Binary mask quality metrics for pulmonary TD16 change localization."""

from __future__ import annotations

import numpy as np


def binary_mask_metrics_batch(
    target: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute per-sample binary mask metrics.

    Args:
        target: Ground-truth binary mask, shape (N, H, W).
        pred: Predicted probabilities or logits after sigmoid, shape (N, H, W).
        valid_mask: Optional mask restricting the evaluation domain, same shape.
        threshold: Threshold applied to ``pred``.

    Returns:
        Dict of per-sample metrics: precision, recall, f1, iou, accuracy.
    """
    target_np = np.asarray(target) > 0.5
    pred_np = np.asarray(pred) > threshold
    if valid_mask is None:
        valid_np = np.ones_like(target_np, dtype=bool)
    else:
        valid_np = np.asarray(valid_mask) > 0.5

    tp = np.sum(pred_np & target_np & valid_np, axis=(1, 2)).astype(np.float64)
    fp = np.sum(pred_np & (~target_np) & valid_np, axis=(1, 2)).astype(np.float64)
    fn = np.sum((~pred_np) & target_np & valid_np, axis=(1, 2)).astype(np.float64)
    tn = np.sum((~pred_np) & (~target_np) & valid_np, axis=(1, 2)).astype(np.float64)

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    iou = tp / np.maximum(tp + fp + fn, 1.0)
    accuracy = (tp + tn) / np.maximum(tp + fp + fn + tn, 1.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }
