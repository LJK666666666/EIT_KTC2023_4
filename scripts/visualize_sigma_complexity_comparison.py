"""Visualize low/high pulmonary sigma predictions under matched complexity."""

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
    AtlasSigmaPredictorPipeline,
    DCTSigmaPredictorPipeline,
    DCTSigmaResidualPredictorPipeline,
    DCTSigmaHybridPredictorPipeline,
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
    with open(os.path.join(result_dir, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.unsafe_load(f) or {}
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    return {
        "train_indices": np.asarray(
            data.get("train_indices")
            or training.get("train_indices")
            or cfg.get("train_indices"),
            dtype=np.int64,
        ),
        "test_indices": np.asarray(
            data.get("test_indices")
            or training.get("test_indices")
            or cfg.get("test_indices"),
            dtype=np.int64,
        ),
    }


def load_sigma_subset(h5_path: str, sample_ids: np.ndarray):
    with h5py.File(h5_path, "r") as h5f:
        order = np.argsort(sample_ids)
        sorted_ids = sample_ids[order]
        inv = np.argsort(order)
        sigma = h5f["sigma"][sorted_ids][inv]
        measurements = h5f["measurements"][sorted_ids][inv]
    return sigma, measurements


def load_atlas(h5_path: str, train_indices: np.ndarray):
    with h5py.File(h5_path, "r") as h5f:
        order = np.argsort(train_indices)
        sorted_ids = train_indices[order]
        sigma = h5f["sigma"][sorted_ids]
    return sigma.mean(axis=0)


def build_pipeline(method: str, result_dir: str, device: str):
    if method == "direct":
        pipeline = DCTSigmaPredictorPipeline(device=device, weights_base_dir=result_dir)
    elif method == "residual":
        pipeline = DCTSigmaResidualPredictorPipeline(device=device, weights_base_dir=result_dir)
    elif method == "hybrid":
        pipeline = DCTSigmaHybridPredictorPipeline(device=device, weights_base_dir=result_dir)
    elif method == "atlas_decoder":
        pipeline = AtlasSigmaPredictorPipeline(device=device, weights_base_dir=result_dir)
    elif method == "fc_sigmaunet":
        pipeline = FCSigmaUNetPipeline(device=device, weights_base_dir=result_dir)
    else:
        raise ValueError(method)
    pipeline.load_model(level=1)
    return pipeline


def main():
    cfg = load_yaml("scripts/pulmonary_complexity_model_comparison.yaml")
    out_dir = create_result_subdir("results", "pulmonary_complexity_visual")
    ref_data = loadmat("KTC2023/Codes_Python/TrainingData/ref.mat")

    level_order = ["low", "high"]
    method_order = ["direct", "residual", "hybrid", "atlas_decoder", "fc_sigmaunet"]
    method_labels = {
        "direct": "Direct",
        "residual": "Residual",
        "hybrid": "Hybrid",
        "atlas_decoder": "Atlas-decoder",
        "fc_sigmaunet": "FC-SigmaUNet",
    }
    sample_positions = np.asarray(cfg.get("visualization", {}).get("sample_indices", [0]), dtype=int)
    split = cfg.get("visualization", {}).get("split", "test")

    fig, axes = plt.subplots(
        len(level_order) * len(sample_positions),
        7,
        figsize=(15.8, 2.4 * len(level_order) * len(sample_positions)),
    )
    axes = np.atleast_2d(axes)

    row_summary = []

    for level_idx, level in enumerate(level_order):
        ds_cfg = cfg["datasets"][level]
        runtime = load_runtime_info(ds_cfg["direct_dir"])
        split_indices = runtime[f"{split}_indices"]
        picked = split_indices[sample_positions]
        sigma, measurements = load_sigma_subset(ds_cfg["hdf5_path"], picked)
        atlas = load_atlas(ds_cfg["hdf5_path"], runtime["train_indices"])

        preds = {}
        for method in method_order:
            pipeline = build_pipeline(method, ds_cfg[f"{method}_dir"], device="cuda")
            preds[method] = pipeline.reconstruct_batch(measurements, ref_data, level=1)

        vmin = float(np.min(sigma))
        vmax = float(np.max(sigma))

        for local_row, sample_id in enumerate(picked):
            row = level_idx * len(sample_positions) + local_row
            panels = [
                ("GT", sigma[local_row]),
                ("Atlas", atlas),
                (method_labels["direct"], preds["direct"][local_row]),
                (method_labels["residual"], preds["residual"][local_row]),
                (method_labels["hybrid"], preds["hybrid"][local_row]),
                (method_labels["atlas_decoder"], preds["atlas_decoder"][local_row]),
                (method_labels["fc_sigmaunet"], preds["fc_sigmaunet"][local_row]),
            ]
            for col, (_, image) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(image.T, cmap="viridis", origin="lower",
                               vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"{level.capitalize()} #{int(sample_id)}")
            row_summary.append({"level": level, "sample_id": int(sample_id)})

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": row_summary}, f, indent=2, ensure_ascii=False)
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
