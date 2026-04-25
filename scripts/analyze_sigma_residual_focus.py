"""Analyze residual-region recovery for pulmonary conductivity predictors."""

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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.regression_metrics import (
    masked_regression_metrics,
    masked_regression_metrics_batch,
)
from src.pipelines import (
    AtlasSigmaPredictorPipeline,
    DCTSigmaHybridPredictorPipeline,
    DCTSigmaPredictorPipeline,
    DCTSigmaResidualPredictorPipeline,
    FCSigmaUNetPipeline,
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


def build_pipeline(method_type: str, weights_dir: str, device: str):
    if method_type == "dct_sigma_predictor":
        return DCTSigmaPredictorPipeline(device=device, weights_base_dir=weights_dir)
    if method_type == "dct_sigma_residual_predictor":
        return DCTSigmaResidualPredictorPipeline(device=device, weights_base_dir=weights_dir)
    if method_type == "dct_sigma_hybrid_predictor":
        return DCTSigmaHybridPredictorPipeline(device=device, weights_base_dir=weights_dir)
    if method_type == "atlas_sigma_predictor":
        return AtlasSigmaPredictorPipeline(device=device, weights_base_dir=weights_dir)
    if method_type == "fc_sigmaunet":
        return FCSigmaUNetPipeline(device=device, weights_base_dir=weights_dir)
    raise ValueError(f"Unsupported method type: {method_type}")


def predict_all(method_cfg, hdf5_path, split_indices, device, batch_size, ref_data):
    method_type = method_cfg["type"]
    weights_dir = method_cfg["weights_dir"]
    if method_type == "atlas_baseline":
        atlas = np.load(os.path.join(weights_dir, "atlas.npy")).astype(np.float32)
        return np.repeat(atlas[None, ...], len(split_indices), axis=0)

    pipeline = build_pipeline(method_type, weights_dir, device)
    pipeline.load_model(level=1)

    preds_all = []
    with h5py.File(hdf5_path, "r") as h5f:
        meas_ds = h5f["measurements"]
        num_batches = int(np.ceil(len(split_indices) / batch_size))
        pbar = tqdm(range(num_batches), desc=f"{Path(weights_dir).name}", ncols=100)
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(len(split_indices), start + batch_size)
            batch_ids = split_indices[start:end]

            order = np.argsort(batch_ids)
            batch_sorted = batch_ids[order]
            inverse = np.argsort(order)
            measurements = meas_ds[batch_sorted][inverse]

            preds = pipeline.reconstruct_batch(measurements, ref_data, level=1)
            preds_all.append(np.stack(preds, axis=0))

    return np.concatenate(preds_all, axis=0)


def gather_targets(hdf5_path, split_indices):
    with h5py.File(hdf5_path, "r") as h5f:
        order = np.argsort(split_indices)
        sorted_ids = split_indices[order]
        inverse = np.argsort(order)
        sigma = h5f["sigma"][sorted_ids][inverse]
    return sigma.astype(np.float32)


def compute_metrics(targets, preds, atlas, residual_threshold):
    global_mask = targets > 0
    residual_mask = np.logical_and(global_mask, np.abs(targets - atlas[None, ...]) > residual_threshold)

    global_metrics = masked_regression_metrics_batch(targets, preds, masks=global_mask)
    valid_counts = residual_mask.reshape(residual_mask.shape[0], -1).sum(axis=1)
    residual_fraction = float(residual_mask.mean())
    valid_fraction = float(np.mean(valid_counts > 0))
    residual_mae = []
    residual_rmse = []
    residual_rel = []
    for target, pred, mask, count in zip(targets, preds, residual_mask, valid_counts):
        if count <= 0:
            continue
        m = masked_regression_metrics(target, pred, mask=mask)
        residual_mae.append(m["mae"])
        residual_rmse.append(m["rmse"])
        residual_rel.append(m["rel_l2"])

    return {
        "global": {
            "mae_mean": float(np.mean(global_metrics["mae"])),
            "rmse_mean": float(np.mean(global_metrics["rmse"])),
            "rel_l2_mean": float(np.mean(global_metrics["rel_l2"])),
        },
        "residual_region": {
            "mae_mean": float(np.mean(residual_mae)) if residual_mae else float("nan"),
            "rmse_mean": float(np.mean(residual_rmse)) if residual_rmse else float("nan"),
            "rel_l2_mean": float(np.mean(residual_rel)) if residual_rel else float("nan"),
            "residual_fraction": residual_fraction,
            "samples_with_residual_fraction": valid_fraction,
        },
    }


def plot_summary(summary, out_dir):
    dataset_names = [d["name"] for d in summary["datasets"]]
    method_order = ["atlas", "direct", "residual", "hybrid", "atlas_decoder", "fc_sigmaunet"]
    method_labels = {
        "atlas": "Atlas",
        "direct": "Direct",
        "residual": "Residual",
        "hybrid": "Hybrid",
        "atlas_decoder": "Atlas-decoder",
        "fc_sigmaunet": "FC-SigmaUNet",
    }
    colors = {
        "atlas": "#6c757d",
        "direct": "#c0392b",
        "residual": "#2e86de",
        "hybrid": "#16a085",
        "atlas_decoder": "#8e44ad",
        "fc_sigmaunet": "#f39c12",
    }

    width = 0.13
    x = np.arange(len(dataset_names))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for idx, method_key in enumerate(method_order):
        if not all(method_key in d["methods"] for d in summary["datasets"]):
            continue
        rel_vals = [
            d["methods"][method_key]["residual_region"]["rel_l2_mean"]
            for d in summary["datasets"]
        ]
        mae_vals = [
            d["methods"][method_key]["residual_region"]["mae_mean"]
            for d in summary["datasets"]
        ]
        offset = (idx - (len(method_order) - 1) / 2) * width
        axes[0].bar(x + offset, rel_vals, width=width, color=colors[method_key], label=method_labels[method_key])
        axes[1].bar(x + offset, mae_vals, width=width, color=colors[method_key], label=method_labels[method_key])

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Residual-region RelL2")
    axes[1].set_ylabel("Residual-region MAE")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residual_focus_bar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze residual-region pulmonary reconstruction quality")
    parser.add_argument(
        "--config",
        default="scripts/pulmonary_residual_focus_analysis.yaml",
        help="YAML config for dataset/method selection",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    out_dir = create_result_subdir("results", "pulmonary_residual_focus")
    print(f"Output directory: {out_dir}")

    device = cfg.get("device", "cuda")
    batch_size = int(cfg.get("batch_size", 64))
    ref_data = loadmat("KTC2023/Codes_Python/TrainingData/ref.mat")

    summary = {
        "config": args.config,
        "device": device,
        "batch_size": batch_size,
        "datasets": [],
    }

    for dataset_cfg in cfg["datasets"]:
        dataset_name = dataset_cfg["name"]
        hdf5_path = dataset_cfg["hdf5_path"]
        split = dataset_cfg.get("split", "test")
        residual_threshold = float(dataset_cfg.get("residual_threshold", 0.08))
        atlas_dir = dataset_cfg["atlas_dir"]
        atlas = np.load(os.path.join(atlas_dir, "atlas.npy")).astype(np.float32)

        runtime_info = load_runtime_info(atlas_dir)
        split_indices = np.asarray(runtime_info[f"{split}_indices"], dtype=np.int64)
        targets = gather_targets(hdf5_path, split_indices)

        dataset_summary = {
            "name": dataset_name,
            "hdf5_path": hdf5_path,
            "split": split,
            "num_samples": int(len(split_indices)),
            "residual_threshold": residual_threshold,
            "methods": {},
        }

        for method_name, method_cfg in dataset_cfg["methods"].items():
            print(f"[{dataset_name}] {method_name}")
            preds = predict_all(
                method_cfg=method_cfg,
                hdf5_path=hdf5_path,
                split_indices=split_indices,
                device=device,
                batch_size=batch_size,
                ref_data=ref_data,
            )
            dataset_summary["methods"][method_name] = compute_metrics(
                targets=targets,
                preds=preds,
                atlas=atlas,
                residual_threshold=residual_threshold,
            )

        summary["datasets"].append(dataset_summary)

    plot_summary(summary, out_dir)
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
