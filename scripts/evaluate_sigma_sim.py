"""Evaluate continuous conductivity predictors on simulated HDF5 splits."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.io import loadmat
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.regression_metrics import masked_regression_metrics_batch
from src.pipelines import (
    FCSigmaUNetPipeline,
    DCTSigmaPredictorPipeline,
    DCTSigmaResidualPredictorPipeline,
    DCTSigmaHybridPredictorPipeline,
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
    parser = argparse.ArgumentParser(description="Evaluate sigma predictor on simulated HDF5 data")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--method",
        choices=["dct_sigma_predictor", "dct_sigma_residual_predictor",
                 "fc_sigmaunet",
                 "atlas_sigma_predictor",
                 "dct_sigma_hybrid_predictor"],
        default="dct_sigma_predictor",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_info = load_runtime_info(args.weights_dir)
    hdf5_path = args.hdf5_path or runtime_info["hdf5_path"]
    if not hdf5_path:
        raise ValueError("No HDF5 path provided and none found in config.yaml")

    split_key = f"{args.split}_indices"
    split_indices = runtime_info.get(split_key)
    if split_indices is None:
        raise ValueError(f"Missing {split_key} in {args.weights_dir}/config.yaml")
    split_indices = np.asarray(split_indices, dtype=np.int64)

    out_dir = create_result_subdir(args.weights_dir, f"{args.method}_{args.split}_eval")
    print(f"Output directory: {out_dir}")

    if args.method == "dct_sigma_residual_predictor":
        pipeline = DCTSigmaResidualPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "fc_sigmaunet":
        pipeline = FCSigmaUNetPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "atlas_sigma_predictor":
        from src.pipelines import AtlasSigmaPredictorPipeline
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

    preds_all = []
    sigma_all = []
    total_recon_time = 0.0

    with h5py.File(hdf5_path, "r") as h5f:
        sigma_ds = h5f["sigma"]
        meas_ds = h5f["measurements"]

        num_batches = int(np.ceil(len(split_indices) / args.batch_size))
        pbar = tqdm(range(num_batches), desc=f"Evaluating {args.split}", ncols=100)
        for batch_idx in pbar:
            start = batch_idx * args.batch_size
            end = min(len(split_indices), start + args.batch_size)
            batch_ids = split_indices[start:end]

            order = np.argsort(batch_ids)
            batch_sorted = batch_ids[order]
            inverse = np.argsort(order)

            sigma = sigma_ds[batch_sorted][inverse]
            measurements = meas_ds[batch_sorted][inverse]

            t0 = time.time()
            preds = pipeline.reconstruct_batch(measurements, ref_data, level=1)
            total_recon_time += time.time() - t0

            preds_all.append(np.stack(preds, axis=0))
            sigma_all.append(sigma.astype(np.float32))

    pred_np = np.concatenate(preds_all, axis=0)
    sigma_np = np.concatenate(sigma_all, axis=0)
    reg = masked_regression_metrics_batch(sigma_np, pred_np)
    summary = {
        "weights_dir": args.weights_dir,
        "hdf5_path": hdf5_path,
        "split": args.split,
        "method": args.method,
        "num_samples": int(len(split_indices)),
        "mae_mean": float(np.mean(reg["mae"])),
        "rmse_mean": float(np.mean(reg["rmse"])),
        "rel_l2_mean": float(np.mean(reg["rel_l2"])),
        "reconstruction_time_sec": float(total_recon_time),
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"MAE: {summary['mae_mean']:.6f}")
    print(f"RMSE: {summary['rmse_mean']:.6f}")
    print(f"RelL2: {summary['rel_l2_mean']:.6f}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
