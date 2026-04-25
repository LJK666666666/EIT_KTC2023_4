"""Evaluate a pyEIT GREIT baseline on 16-electrode pulmonary TD16 datasets."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyeit.mesh as mesh
import pyeit.mesh.shape as shape
from pyeit.eit.greit import GREIT

from src.evaluation.regression_metrics import masked_regression_metrics_batch
from src.utils.pulmonary16 import build_draeger208_pyeit_protocol


def create_result_subdir(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (base / f"{tag}_{idx}").exists():
        idx += 1
    out_dir = base / f"{tag}_{idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def average_pool_8x8(images: np.ndarray) -> np.ndarray:
    """Downsample `(N, 256, 256)` to `(N, 32, 32)` by average pooling."""
    return images.reshape(images.shape[0], 32, 8, 32, 8).mean(axis=(2, 4))


def resize_32_to_256(images: np.ndarray) -> np.ndarray:
    """Upsample `(N, 32, 32)` to `(N, 256, 256)` using bilinear interpolation."""
    out = []
    for img in images:
        out.append(zoom(img, zoom=(8, 8), order=1))
    return np.stack(out, axis=0)


def build_split_indices(num_samples: int):
    train_end = int(round(num_samples * 0.8))
    val_end = int(round(num_samples * 0.9))
    indices = np.arange(num_samples, dtype=np.int64)
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def build_mesh(fd_name: str, h0: float):
    if fd_name == "thorax":
        return mesh.create(n_el=16, fd=shape.thorax, h0=h0)
    if fd_name == "circle":
        return mesh.create(n_el=16, fd=shape.circle, h0=h0)
    raise ValueError(f"Unsupported mesh shape: {fd_name}")


def calibrate_alpha(preds: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> float:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    num = np.sum(preds[masks] * targets[masks])
    den = np.sum(preds[masks] ** 2) + 1e-8
    return float(num / den)


def save_preview(targets256, preds256, out_path, max_rows=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(max_rows, len(targets256))
    fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.6 * n))
    axes = np.atleast_2d(axes)
    vmax = float(np.max(np.abs(targets256[:n]))) if n > 0 else 1.0
    vmax = max(vmax, 1e-3)
    for row in range(n):
        panels = [targets256[row].T, preds256[row].T]
        for col, img in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap="coolwarm", origin="lower", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GREIT on TD16 pulmonary delta-sigma data")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--calibration-split", choices=["train", "val"], default="val")
    parser.add_argument("--mesh-shape", choices=["thorax", "circle"], default="thorax")
    parser.add_argument("--mesh-h0", type=float, default=0.12)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--p", type=float, default=0.20)
    parser.add_argument("--lamb", type=float, default=1e-2)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--blur-s", type=float, default=20.0)
    parser.add_argument("--active-threshold", type=float, default=0.02)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--experiment-tag", default="greit_td16")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = create_result_subdir("results", args.experiment_tag)
    print(f"Output directory: {out_dir}")

    protocol = build_draeger208_pyeit_protocol()
    eit_mesh = build_mesh(args.mesh_shape, args.mesh_h0)
    solver = GREIT(eit_mesh, protocol)
    solver.setup(
        p=args.p,
        lamb=args.lamb,
        n=args.grid_size,
        s=args.blur_s,
        ratio=args.ratio,
    )
    grid_mask = ~solver.mask.reshape(args.grid_size, args.grid_size)

    with h5py.File(args.hdf5_path, "r") as h5f:
        measurements = h5f["measurements"][:]
        sigma_delta = h5f["sigma_delta"][:].astype(np.float32)
        domain_mask = h5f["domain_mask"][:].astype(np.float32) > 0.5

    splits = build_split_indices(len(measurements))
    split_indices = splits[args.split]
    calibration_indices = splits[args.calibration_split]

    sigma32 = average_pool_8x8(sigma_delta)
    mask32 = average_pool_8x8(domain_mask.astype(np.float32)) > 0.5

    def reconstruct(indices, desc):
        preds32 = []
        total_time = 0.0
        for idx in tqdm(indices, desc=desc, ncols=100):
            t0 = time.time()
            pred = solver.solve(measurements[idx], np.zeros_like(measurements[idx]))
            total_time += time.time() - t0
            pred = pred.reshape(args.grid_size, args.grid_size)
            pred = np.where(grid_mask, pred, 0.0).astype(np.float32)
            preds32.append(pred)
        return np.stack(preds32, axis=0), total_time

    cal_preds32, cal_time = reconstruct(calibration_indices, f"Calibrating {args.calibration_split}")
    alpha = calibrate_alpha(cal_preds32, sigma32[calibration_indices], mask32[calibration_indices])

    test_preds32_raw, test_time = reconstruct(split_indices, f"Evaluating {args.split}")
    test_preds32 = alpha * test_preds32_raw
    test_preds256 = resize_32_to_256(test_preds32)

    reg32 = masked_regression_metrics_batch(
        sigma32[split_indices],
        test_preds32,
        masks=mask32[split_indices],
        active_threshold=args.active_threshold,
    )
    reg256 = masked_regression_metrics_batch(
        sigma_delta[split_indices],
        test_preds256,
        masks=domain_mask[split_indices],
        active_threshold=args.active_threshold,
    )

    summary = {
        "hdf5_path": args.hdf5_path.replace("\\", "/"),
        "split": args.split,
        "calibration_split": args.calibration_split,
        "num_samples": int(len(split_indices)),
        "mesh_shape": args.mesh_shape,
        "mesh_h0": float(args.mesh_h0),
        "grid_size": int(args.grid_size),
        "p": float(args.p),
        "lamb": float(args.lamb),
        "ratio": float(args.ratio),
        "blur_s": float(args.blur_s),
        "active_threshold": float(args.active_threshold),
        "alpha": float(alpha),
        "reconstruction_time_sec": float(test_time),
        "calibration_time_sec": float(cal_time),
        "mae32_mean": float(np.mean(reg32["mae"])),
        "rmse32_mean": float(np.mean(reg32["rmse"])),
        "rel_l2_32_mean": float(np.mean(reg32["rel_l2"])),
        "active_rel_l2_32_mean": float(np.nanmean(reg32["active_rel_l2"])),
        "mae256_mean": float(np.mean(reg256["mae"])),
        "rmse256_mean": float(np.mean(reg256["rmse"])),
        "rel_l2_256_mean": float(np.mean(reg256["rel_l2"])),
        "active_rel_l2_256_mean": float(np.nanmean(reg256["active_rel_l2"])),
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_preview(
        sigma_delta[split_indices][:args.preview_count],
        test_preds256[:args.preview_count],
        os.path.join(out_dir, "comparison.png"),
        max_rows=args.preview_count,
    )

    print(f"alpha: {summary['alpha']:.6e}")
    print(f"RMSE32: {summary['rmse32_mean']:.6f}")
    print(f"ActiveRelL2_32: {summary['active_rel_l2_32_mean']:.6f}")
    print(f"RMSE256: {summary['rmse256_mean']:.6f}")
    print(f"ActiveRelL2_256: {summary['active_rel_l2_256_mean']:.6f}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
