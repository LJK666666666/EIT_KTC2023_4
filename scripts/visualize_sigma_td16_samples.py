"""Visualize 16-electrode pulmonary time-difference conductivity predictions."""

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines import (
    DCTSigmaTD16PredictorPipeline,
    DCTSigmaTD16ChangePredictorPipeline,
    DCTSigmaTD16SpatialChangePredictorPipeline,
    DCTSigmaTD16ConditionalPredictorPipeline,
    TD16VAEPredictorPipeline,
    TD16VAEConditionalPredictorPipeline,
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
        "config": cfg,
        "hdf5_path": data.get("hdf5_path") or cfg.get("hdf5_path"),
        "train_indices": data.get("train_indices") or training.get("train_indices") or cfg.get("train_indices"),
        "val_indices": data.get("val_indices") or training.get("val_indices") or cfg.get("val_indices"),
        "test_indices": data.get("test_indices") or training.get("test_indices") or cfg.get("test_indices"),
    }


def build_pipeline(result_dir: str, runtime_info, device: str):
    cfg = runtime_info.get("config") or {}
    training = cfg.get("training", {})
    method_name = cfg.get("method", '')
    if method_name == 'td16_vae_conditional_predictor':
        return TD16VAEConditionalPredictorPipeline(
            device=device, weights_base_dir=result_dir
        ), "td16_vae_conditional_predictor"
    if method_name == 'td16_vae_predictor':
        return TD16VAEPredictorPipeline(
            device=device, weights_base_dir=result_dir
        ), "td16_vae_predictor"
    if "mask_teacher_forcing_ratio" in training:
        return DCTSigmaTD16ConditionalPredictorPipeline(
            device=device, weights_base_dir=result_dir
        ), "dct_sigma_td16_conditional_predictor"
    if "lambda_mask" in training:
        return DCTSigmaTD16SpatialChangePredictorPipeline(
            device=device, weights_base_dir=result_dir
        ), "dct_sigma_td16_spatial_change_predictor"
    if "lambda_gate" in training:
        return DCTSigmaTD16ChangePredictorPipeline(
            device=device, weights_base_dir=result_dir
        ), "dct_sigma_td16_change_predictor"
    return DCTSigmaTD16PredictorPipeline(
        device=device, weights_base_dir=result_dir
    ), "dct_sigma_td16_predictor"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TD16 sigma regression samples")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--device", default="cuda")
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
        sigma_delta = h5f["sigma_delta"][ids_sorted][inverse]
        sigma_ref = h5f["sigma_ref"][ids_sorted][inverse]
        sigma_target = h5f["sigma_target"][ids_sorted][inverse]
        measurements = h5f["measurements"][ids_sorted][inverse]

    pipeline, method_name = build_pipeline(args.weights_dir, runtime_info, args.device)
    pipeline.load_model(level=1)
    preds = pipeline.reconstruct_batch(measurements, ref_data=None, level=1)
    preds = np.stack(preds, axis=0)
    pred_target = sigma_ref + preds

    out_dir = create_result_subdir(args.weights_dir, "td16_test_samples")
    fig, axes = plt.subplots(
        args.rows * 5, args.cols, figsize=(2.6 * args.cols, 2.0 * args.rows * 5)
    )
    axes = np.atleast_2d(axes)

    abs_vmin = float(np.min(sigma_target))
    abs_vmax = float(np.max(sigma_target))
    delta_abs = max(float(np.max(np.abs(sigma_delta))), float(np.max(np.abs(preds))), 1e-6)

    row_labels = ['Ref', 'GT Δσ', 'Pred Δσ', 'GT σ', 'Pred σ']
    for i in range(n_show):
        col = i % args.cols
        block = (i // args.cols) * 5
        panels = [
            (sigma_ref[i].T, "viridis", abs_vmin, abs_vmax),
            (sigma_delta[i].T, "coolwarm", -delta_abs, delta_abs),
            (preds[i].T, "coolwarm", -delta_abs, delta_abs),
            (sigma_target[i].T, "viridis", abs_vmin, abs_vmax),
            (pred_target[i].T, "viridis", abs_vmin, abs_vmax),
        ]
        for row_offset, (img, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[block + row_offset, col]
            ax.imshow(img, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row_offset])

    for j in range(n_show, args.rows * args.cols):
        col = j % args.cols
        block = (j // args.cols) * 5
        for row_offset in range(5):
            axes[block + row_offset, col].set_visible(False)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{args.split}_comparison.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "weights_dir": args.weights_dir,
        "hdf5_path": hdf5_path,
        "split": args.split,
        "method": method_name,
        "num_samples": int(n_show),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
