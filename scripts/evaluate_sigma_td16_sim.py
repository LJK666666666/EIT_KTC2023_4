"""Evaluate 16-electrode pulmonary time-difference conductivity predictors."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.regression_metrics import masked_regression_metrics_batch
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
        ), 'td16_vae_conditional_predictor'
    if method_name == 'td16_vae_predictor':
        return TD16VAEPredictorPipeline(
            device=device, weights_base_dir=result_dir
        ), 'td16_vae_predictor'
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
    parser = argparse.ArgumentParser(
        description="Evaluate TD16 pulmonary conductivity predictor")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
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

    out_dir = create_result_subdir(args.weights_dir, "td16_test_eval")
    print(f"Output directory: {out_dir}")

    pipeline, method_name = build_pipeline(args.weights_dir, runtime_info, args.device)
    pipeline.load_model(level=1)

    preds_all = []
    sigma_all = []
    mask_all = []
    total_recon_time = 0.0

    with h5py.File(hdf5_path, "r") as h5f:
        sigma_ds = h5f["sigma_delta"]
        mask_ds = h5f["domain_mask"]
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
            masks = mask_ds[batch_sorted][inverse]
            measurements = meas_ds[batch_sorted][inverse]

            t0 = time.time()
            preds = pipeline.reconstruct_batch(measurements, ref_data=None, level=1)
            total_recon_time += time.time() - t0

            preds_all.append(np.stack(preds, axis=0))
            sigma_all.append(sigma.astype(np.float32))
            mask_all.append(masks.astype(np.float32))

    pred_np = np.concatenate(preds_all, axis=0)
    sigma_np = np.concatenate(sigma_all, axis=0)
    mask_np = np.concatenate(mask_all, axis=0) > 0.5
    reg = masked_regression_metrics_batch(
        sigma_np, pred_np, masks=mask_np, active_threshold=0.02)
    active_rel = reg.get("active_rel_l2")
    target_norm = np.linalg.norm((sigma_np * mask_np), axis=(1, 2))
    summary = {
        "weights_dir": args.weights_dir,
        "hdf5_path": hdf5_path,
        "split": args.split,
        "method": method_name,
        "num_samples": int(len(split_indices)),
        "mae_mean": float(np.mean(reg["mae"])),
        "rmse_mean": float(np.mean(reg["rmse"])),
        "rel_l2_mean": float(np.mean(reg["rel_l2"])),
        "active_rel_l2_mean": float(np.nanmean(active_rel)),
        "zero_delta_fraction": float(np.mean(target_norm < 1e-6)),
        "reconstruction_time_sec": float(total_recon_time),
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"MAE: {summary['mae_mean']:.6f}")
    print(f"RMSE: {summary['rmse_mean']:.6f}")
    print(f"RelL2: {summary['rel_l2_mean']:.6f}")
    print(f"ActiveRelL2: {summary['active_rel_l2_mean']:.6f}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
