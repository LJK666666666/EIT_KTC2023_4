"""Visualize reconstruction samples on a simulated HDF5 split."""

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

from src.pipelines import DCTPredictorEnsemblePipeline, DCTPredictorPipeline, FCUNetPipeline


def create_result_subdir(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    idx = 1
    while (base / f"{tag}_{idx}").exists():
        idx += 1
    out_dir = base / f"{tag}_{idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize reconstruction samples on simulated HDF5")
    parser.add_argument("--method", choices=["fcunet", "dct_predictor", "dct_predictor_ensemble"],
                        required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--ensemble-config", default="scripts/dct_predictor_lung_ensemble.yaml")
    parser.add_argument("--indices-source-dir", default=None)
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def resolve_indices_source_dir(args):
    if args.indices_source_dir:
        return args.indices_source_dir
    if args.method in {"fcunet", "dct_predictor"}:
        return args.weights_dir
    spec = load_yaml(args.ensemble_config) or {}
    members = spec.get("members", [])
    if not members:
        raise ValueError(f"No ensemble members configured in {args.ensemble_config}")
    return os.path.join(args.weights_dir, members[0]["result_dir"])


def resolve_output_base_dir(args):
    if args.method in {"fcunet", "dct_predictor"}:
        return args.weights_dir
    spec = load_yaml(args.ensemble_config) or {}
    members = spec.get("members", [])
    if not members:
        raise ValueError(f"No ensemble members configured in {args.ensemble_config}")
    return os.path.join(args.weights_dir, members[0]["result_dir"])


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


def build_pipeline(args):
    if args.method == "fcunet":
        pipeline = FCUNetPipeline(device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "dct_predictor":
        pipeline = DCTPredictorPipeline(device=args.device, weights_base_dir=args.weights_dir)
    else:
        pipeline = DCTPredictorEnsemblePipeline(
            device=args.device,
            weights_base_dir=args.weights_dir,
            config_path=args.ensemble_config,
        )
    pipeline.load_model(level=1)
    return pipeline


def main():
    args = parse_args()
    indices_source_dir = resolve_indices_source_dir(args)
    info = load_runtime_info(indices_source_dir)
    hdf5_path = args.hdf5_path or info["hdf5_path"]
    if not hdf5_path:
        raise ValueError("No HDF5 path provided and none found in config.yaml")

    split_indices = info.get(f"{args.split}_indices")
    if split_indices is None:
        raise ValueError(f"Missing {args.split}_indices in config.yaml")
    split_indices = np.asarray(split_indices, dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    n = min(len(split_indices), args.rows * args.cols)
    chosen = rng.choice(split_indices, size=n, replace=False)
    chosen = np.asarray(chosen, dtype=np.int64)
    order = np.argsort(chosen)
    chosen_sorted = chosen[order]
    inverse = np.argsort(order)

    pipeline = build_pipeline(args)
    ref_data = loadmat("KTC2023/Codes_Python/TrainingData/ref.mat")

    with h5py.File(hdf5_path, "r") as h5f:
        gts = h5f["gt"][chosen_sorted][inverse]
        measurements = h5f["measurements"][chosen_sorted][inverse]

    preds = pipeline.reconstruct_batch(measurements, ref_data, level=1)

    out_dir = create_result_subdir(resolve_output_base_dir(args), f"{args.method}_{args.split}_samples")
    fig, axes = plt.subplots(args.rows, args.cols * 2,
                             figsize=(2.4 * args.cols * 2, 2.4 * args.rows))
    axes = np.atleast_2d(axes)
    for i in range(args.rows * args.cols):
        gt_ax = axes[i // args.cols, (i % args.cols) * 2]
        pr_ax = axes[i // args.cols, (i % args.cols) * 2 + 1]
        if i < n:
            gt_ax.imshow(gts[i].T, cmap="gray", origin="lower", vmin=0, vmax=2)
            pr_ax.imshow(preds[i].T, cmap="gray", origin="lower", vmin=0, vmax=2)
        gt_ax.set_xticks([])
        gt_ax.set_yticks([])
        pr_ax.set_xticks([])
        pr_ax.set_yticks([])
    fig.savefig(os.path.join(out_dir, f"{args.split}_comparison.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "method": args.method,
            "weights_dir": args.weights_dir,
            "ensemble_config": args.ensemble_config if args.method == "dct_predictor_ensemble" else None,
            "indices_source_dir": indices_source_dir,
            "hdf5_path": hdf5_path,
            "split": args.split,
            "indices": [int(x) for x in chosen.tolist()],
        }, f, indent=2, ensure_ascii=False)

    print(f"Output directory: {out_dir}")
    if args.show:
        img = plt.imread(os.path.join(out_dir, f"{args.split}_comparison.png"))
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
