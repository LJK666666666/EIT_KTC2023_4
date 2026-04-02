"""Evaluate reconstruction models on a simulated HDF5 split."""

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

from src.evaluation.scoring_torch import fast_score_batch_auto
from src.pipelines import (
    DCTPredictorEnsemblePipeline,
    DCTPredictorPipeline,
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
    parser = argparse.ArgumentParser(description="Evaluate reconstruction model on simulated HDF5 data")
    parser.add_argument("--method", choices=["fcunet", "dct_predictor", "dct_predictor_ensemble"],
                        default="dct_predictor")
    parser.add_argument("--weights-dir", required=True,
        help="Single-model result dir or results root for ensemble.")
    parser.add_argument("--ensemble-config", default="scripts/dct_predictor_lung_ensemble.yaml")
    parser.add_argument("--indices-source-dir", default=None,
                        help="Result dir whose config.yaml provides train/val/test split indices.")
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def resolve_indices_source_dir(args):
    if args.indices_source_dir:
        return args.indices_source_dir
    if args.method in {"dct_predictor", "fcunet"}:
        return args.weights_dir
    spec = load_yaml(args.ensemble_config) or {}
    members = spec.get("members", [])
    if not members:
        raise ValueError(f"No ensemble members configured in {args.ensemble_config}")
    return os.path.join(args.weights_dir, members[0]["result_dir"])


def resolve_output_base_dir(args):
    if args.method in {"dct_predictor", "fcunet"}:
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
    if args.method == "dct_predictor":
        pipeline = DCTPredictorPipeline(device=args.device, weights_base_dir=args.weights_dir)
    elif args.method == "fcunet":
        pipeline = FCUNetPipeline(device=args.device, weights_base_dir=args.weights_dir)
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
    runtime_info = load_runtime_info(indices_source_dir)
    hdf5_path = args.hdf5_path or runtime_info["hdf5_path"]
    if not hdf5_path:
        raise ValueError("No HDF5 path provided and none found in config.yaml")

    split_key = f"{args.split}_indices"
    split_indices = runtime_info.get(split_key)
    if split_indices is None:
        raise ValueError(f"Missing {split_key} in {indices_source_dir}/config.yaml")
    split_indices = np.asarray(split_indices, dtype=np.int64)

    out_dir = create_result_subdir(resolve_output_base_dir(args), f"{args.method}_{args.split}_eval")
    print(f"Output directory: {out_dir}")

    pipeline = build_pipeline(args)
    ref_data = loadmat("KTC2023/Codes_Python/TrainingData/ref.mat")

    all_scores = []
    total_recon_time = 0.0
    total_score_time = 0.0

    with h5py.File(hdf5_path, "r") as h5f:
        gt_ds = h5f["gt"]
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

            gts = gt_ds[batch_sorted][inverse]
            measurements = meas_ds[batch_sorted][inverse]

            t0 = time.time()
            preds = pipeline.reconstruct_batch(measurements, ref_data, level=1)
            total_recon_time += time.time() - t0

            t1 = time.time()
            scores = fast_score_batch_auto(gts, preds, device=args.score_device or args.device)
            total_score_time += time.time() - t1
            all_scores.extend(float(x) for x in scores)

            pbar.set_postfix({
                "mean": f"{np.mean(all_scores):.4f}",
                "recon_s": f"{total_recon_time:.1f}",
                "score_s": f"{total_score_time:.1f}",
            })

    summary = {
        "method": args.method,
        "weights_dir": args.weights_dir,
        "ensemble_config": args.ensemble_config if args.method == "dct_predictor_ensemble" else None,
        "indices_source_dir": indices_source_dir,
        "hdf5_path": hdf5_path,
        "split": args.split,
        "num_samples": int(len(split_indices)),
        "mean_score": float(np.mean(all_scores)),
        "total_score": float(np.sum(all_scores)),
        "reconstruction_time_sec": float(total_recon_time),
        "score_time_sec": float(total_score_time),
        "scores": [float(x) for x in all_scores],
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Mean score: {summary['mean_score']:.6f}")
    print(f"Total score: {summary['total_score']:.6f}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
