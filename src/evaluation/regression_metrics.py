"""Regression metrics for continuous conductivity reconstruction."""

import numpy as np


def masked_regression_metrics(target, pred, mask=None, eps=1e-8, active_threshold=None):
    """Compute masked MAE/RMSE/relative-L2 for conductivity images.

    Args:
        target: (..., H, W) ndarray
        pred: same shape as target
        mask: optional boolean ndarray broadcastable to target.
              If None, uses target > 0.
    """
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if mask is None:
        mask = target > 0
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != target.shape:
        mask = np.broadcast_to(mask, target.shape)

    diff = pred - target
    diff_masked = diff[mask]
    target_masked = target[mask]

    mae = float(np.mean(np.abs(diff_masked)))
    rmse = float(np.sqrt(np.mean(diff_masked ** 2)))
    rel_l2 = float(
        np.linalg.norm(diff_masked) / (np.linalg.norm(target_masked) + eps)
    )
    out = {
        'mae': mae,
        'rmse': rmse,
        'rel_l2': rel_l2,
    }
    if active_threshold is not None:
        active_mask = mask & (np.abs(target) > float(active_threshold))
        if np.any(active_mask):
            active_diff = diff[active_mask]
            active_target = target[active_mask]
            out['active_rel_l2'] = float(
                np.linalg.norm(active_diff) / (np.linalg.norm(active_target) + eps)
            )
        else:
            out['active_rel_l2'] = float('nan')
    return out


def masked_regression_metrics_batch(targets, preds, masks=None, active_threshold=None):
    """Compute per-sample metrics for a batch."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    if masks is None:
        masks = targets > 0
    maes, rmses, rels, active_rels = [], [], [], []
    for target, pred, mask in zip(targets, preds, masks):
        m = masked_regression_metrics(
            target, pred, mask=mask, active_threshold=active_threshold)
        maes.append(m['mae'])
        rmses.append(m['rmse'])
        rels.append(m['rel_l2'])
        if 'active_rel_l2' in m:
            active_rels.append(m['active_rel_l2'])
    out = {
        'mae': np.asarray(maes, dtype=np.float64),
        'rmse': np.asarray(rmses, dtype=np.float64),
        'rel_l2': np.asarray(rels, dtype=np.float64),
    }
    if active_rels:
        out['active_rel_l2'] = np.asarray(active_rels, dtype=np.float64)
    return out
