"""Visualize continuous conductivity reconstruction samples."""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines import (
    FCSigmaUNetPipeline,
    DCTSigmaPredictorPipeline,
    DCTSigmaResidualPredictorPipeline,
    DCTSigmaHybridPredictorPipeline,
    AtlasSigmaPredictorPipeline,
)


def create_result_subdir(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    idx = 1
    while (base / f"{tag}_{idx}").exists():
        idx += 1
    out_dir = base / f"{tag}_{idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def load_runtime_info(result_dir: str):
    cfg = load_yaml(os.path.join(result_dir, "config.yaml")) or {}
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    return {
        "hdf5_path": data.get("hdf5_path") or cfg.get("hdf5_path"),
        "train_indices": data.get("train_indices") or training.get("train_indices") or cfg.get("train_indices"),
        "val_indices": data.get("val_indices") or training.get("val_indices") or cfg.get("val_indices"),
        "test_indices": data.get("test_indices") or training.get("test_indices") or cfg.get("test_indices"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize sigma regression samples")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--method",
        choices=["dct_sigma_predictor", "dct_sigma_residual_predictor",
                 "fc_sigmaunet",
                 "dct_sigma_hybrid_predictor", "atlas_sigma_predictor"],
        default="dct_sigma_predictor",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_info = load_runtime_info(args.weights_dir)
    hdf5_path = args.hdf5_path or runtime_info["hdf5_path"]
    split_key = f"{args.split}_indices"
    split_indices = np.asarray(runtime_info[split_key], dtype=np.int64)

    n_show = min(args.rows * args.cols, len(split_indices))
    ids = split_indices[:n_show]

    with h5py.File(hdf5_path, "r") as h5f:
        order = np.argsort(ids)
        ids_sorted = ids[order]
        inverse = np.argsort(order)
        sigma = h5f["sigma"][ids_sorted][inverse]
        measurements = h5f["measurements"][ids_sorted][inverse]

    if args.method == "dct_sigma_residual_predictor":
        pipeline = DCTSigmaResidualPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "fc_sigmaunet":
        pipeline = FCSigmaUNetPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "atlas_sigma_predictor":
        pipeline = AtlasSigmaPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "dct_sigma_hybrid_predictor":
        pipeline = DCTSigmaHybridPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    else:
        pipeline = DCTSigmaPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    pipeline.load_model(level=1)
    ref_data = loadmat("KTC2023/Codes_Python/TrainingData/ref.mat")
    preds = pipeline.reconstruct_batch(measurements, ref_data, level=1)

    out_dir = create_result_subdir(args.weights_dir, f"{args.method}_{args.split}_samples")
    fig, axes = plt.subplots(args.rows, args.cols * 2, figsize=(2.6 * args.cols * 2, 2.6 * args.rows))
    axes = np.atleast_2d(axes)
    vmin = float(np.min(sigma))
    vmax = float(np.max(sigma))
    for i in range(n_show):
        r = i // args.cols
        c = (i % args.cols) * 2
        ax_gt = axes[r, c]
        ax_pr = axes[r, c + 1]
        im0 = ax_gt.imshow(sigma[i].T, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        im1 = ax_pr.imshow(preds[i].T, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        for ax in (ax_gt, ax_pr):
            ax.set_xticks([])
            ax.set_yticks([])
    for j in range(n_show, args.rows * args.cols):
        r = j // args.cols
        c = (j % args.cols) * 2
        axes[r, c].set_visible(False)
        axes[r, c + 1].set_visible(False)
    fig.colorbar(im1, ax=axes, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{args.split}_comparison.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "weights_dir": args.weights_dir,
        "hdf5_path": hdf5_path,
        "split": args.split,
        "method": args.method,
        "num_samples": int(n_show),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
