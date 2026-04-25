"""Visualize GT / DCT / FCUNet on the same pulmonary test samples."""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines import DCTPredictorPipeline, FCUNetPipeline


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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize pulmonary model comparison")
    parser.add_argument("--fcunet-dir", required=True)
    parser.add_argument("--dct-dir", required=True)
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


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


def main():
    args = parse_args()
    info = load_runtime_info(args.fcunet_dir)
    hdf5_path = args.hdf5_path or info["hdf5_path"]
    if not hdf5_path:
        raise ValueError("No HDF5 path provided and none found in FCUNet config.yaml")

    split_indices = info.get(f"{args.split}_indices")
    if split_indices is None:
        raise ValueError(f"Missing {args.split}_indices in FCUNet config.yaml")
    split_indices = np.asarray(split_indices, dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    n = min(len(split_indices), args.rows * args.cols)
    chosen = rng.choice(split_indices, size=n, replace=False)
    chosen = np.asarray(chosen, dtype=np.int64)
    order = np.argsort(chosen)
    chosen_sorted = chosen[order]
    inverse = np.argsort(order)

    ref_data = loadmat("KTC2023/Codes_Python/TrainingData/ref.mat")
    fcunet = FCUNetPipeline(device=args.device, weights_base_dir=args.fcunet_dir)
    dct = DCTPredictorPipeline(device=args.device, weights_base_dir=args.dct_dir)
    fcunet.load_model(level=1)
    dct.load_model(level=1)

    with h5py.File(hdf5_path, "r") as h5f:
        gts = h5f["gt"][chosen_sorted][inverse]
        measurements = h5f["measurements"][chosen_sorted][inverse]

    preds_fc = fcunet.reconstruct_batch(measurements, ref_data, level=1)
    preds_dct = dct.reconstruct_batch(measurements, ref_data, level=1)

    out_dir = create_result_subdir(args.fcunet_dir, "pulmonary_compare")
    fig, axes = plt.subplots(
        args.rows, args.cols * 3, figsize=(2.2 * args.cols * 3, 2.2 * args.rows)
    )
    axes = np.atleast_2d(axes)
    for i in range(args.rows * args.cols):
        gt_ax = axes[i // args.cols, (i % args.cols) * 3]
        dct_ax = axes[i // args.cols, (i % args.cols) * 3 + 1]
        fc_ax = axes[i // args.cols, (i % args.cols) * 3 + 2]
        if i < n:
            gt_ax.imshow(gts[i].T, cmap="gray", origin="lower", vmin=0, vmax=2)
            dct_ax.imshow(preds_dct[i].T, cmap="gray", origin="lower", vmin=0, vmax=2)
            fc_ax.imshow(preds_fc[i].T, cmap="gray", origin="lower", vmin=0, vmax=2)
        for ax in (gt_ax, dct_ax, fc_ax):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(os.path.join(out_dir, f"{args.split}_comparison.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "fcunet_dir": args.fcunet_dir,
            "dct_dir": args.dct_dir,
            "hdf5_path": hdf5_path,
            "split": args.split,
            "indices": [int(x) for x in chosen.tolist()],
        }, f, indent=2, ensure_ascii=False)

    print(f"Output directory: {out_dir}")
    if args.show:
        img = plt.imread(os.path.join(out_dir, f"{args.split}_comparison.png"))
        plt.figure(figsize=(14, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
