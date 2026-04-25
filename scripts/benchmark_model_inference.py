"""Benchmark parameter count and single/batch inference latency."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines import (
    AtlasSigmaPredictorPipeline,
    DCTPredictorPipeline,
    DCTSigmaHybridPredictorPipeline,
    DCTSigmaPredictorPipeline,
    DCTSigmaResidualPredictorPipeline,
    FCSigmaUNetPipeline,
    FCUNetPipeline,
)


def create_result_subdir(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    idx = 1
    while (base / f"{tag}_{idx}").exists():
        idx += 1
    out_dir = base / f"{tag}_{idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model inference latency")
    parser.add_argument(
        "--method",
        choices=[
            "fcunet",
            "dct_predictor",
            "dct_sigma_predictor",
            "dct_sigma_residual_predictor",
            "dct_sigma_hybrid_predictor",
            "atlas_sigma_predictor",
            "fc_sigmaunet",
        ],
        required=True,
    )
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def load_runtime_info(result_dir: str):
    cfg = load_yaml(os.path.join(result_dir, "config.yaml")) or {}
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    return {
        "train_indices": data.get("train_indices") or training.get("train_indices") or cfg.get("train_indices"),
        "val_indices": data.get("val_indices") or training.get("val_indices") or cfg.get("val_indices"),
        "test_indices": data.get("test_indices") or training.get("test_indices") or cfg.get("test_indices"),
    }


def build_pipeline(args):
    if args.method == "fcunet":
        pipeline = FCUNetPipeline(device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "dct_predictor":
        pipeline = DCTPredictorPipeline(device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "dct_sigma_predictor":
        pipeline = DCTSigmaPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "dct_sigma_residual_predictor":
        pipeline = DCTSigmaResidualPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "dct_sigma_hybrid_predictor":
        pipeline = DCTSigmaHybridPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "fc_sigmaunet":
        pipeline = FCSigmaUNetPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    else:
        pipeline = AtlasSigmaPredictorPipeline(
            device=args.device, weights_base_dir=args.weights_dir)
    pipeline.load_model(level=1)
    return pipeline


def sync_if_needed(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    info = load_runtime_info(args.weights_dir)
    split_indices = np.asarray(info[f"{args.split}_indices"], dtype=np.int64)
    if split_indices.size == 0:
        raise ValueError(f"No indices available for split {args.split}")

    with h5py.File(args.hdf5_path, "r") as h5f:
        sample_ids = split_indices[:max(args.batch_size, 1)]
        order = np.argsort(sample_ids)
        inv = np.argsort(order)
        sample_sorted = sample_ids[order]
        measurements = h5f["measurements"][sample_sorted][inv]

    ref_data = loadmat("KTC2023/Codes_Python/TrainingData/ref.mat")
    pipeline = build_pipeline(args)
    model = pipeline.model
    num_params = int(sum(p.numel() for p in model.parameters()))
    num_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    single = [measurements[0]]
    batch = list(measurements[:args.batch_size])

    for _ in range(args.warmup):
        _ = pipeline.reconstruct_batch(single, ref_data, level=1)
        _ = pipeline.reconstruct_batch(batch, ref_data, level=1)
    sync_if_needed(args.device)

    single_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = pipeline.reconstruct_batch(single, ref_data, level=1)
        sync_if_needed(args.device)
        single_times.append(time.perf_counter() - t0)

    batch_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = pipeline.reconstruct_batch(batch, ref_data, level=1)
        sync_if_needed(args.device)
        batch_times.append(time.perf_counter() - t0)

    out_dir = create_result_subdir(args.weights_dir, f"{args.method}_latency")
    summary = {
        "method": args.method,
        "weights_dir": args.weights_dir,
        "hdf5_path": args.hdf5_path,
        "split": args.split,
        "device": args.device,
        "num_params": num_params,
        "num_trainable_params": num_trainable,
        "single_latency_ms_mean": float(np.mean(single_times) * 1000.0),
        "single_latency_ms_std": float(np.std(single_times) * 1000.0),
        "batch_size": int(len(batch)),
        "batch_latency_ms_mean": float(np.mean(batch_times) * 1000.0),
        "batch_latency_ms_std": float(np.std(batch_times) * 1000.0),
        "batch_per_sample_ms_mean": float(np.mean(batch_times) * 1000.0 / max(len(batch), 1)),
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Output directory: {out_dir}")
    print(f"Params: {num_params}")
    print(f"Single latency: {summary['single_latency_ms_mean']:.3f} ms")
    print(f"Batch latency: {summary['batch_latency_ms_mean']:.3f} ms "
          f"(batch={len(batch)}, per-sample={summary['batch_per_sample_ms_mean']:.3f} ms)")


if __name__ == "__main__":
    main()
