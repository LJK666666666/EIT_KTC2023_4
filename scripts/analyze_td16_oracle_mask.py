"""Oracle-mask diagnostics for TD16 pulmonary time-difference models."""

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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.regression_metrics import masked_regression_metrics_batch
from src.models.dct_predictor import (
    SpatialChangeGatedDCTPredictor,
    ConditionalSpatialChangeDCTPredictor,
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
    model_cfg = cfg.get("model", {})
    return {
        "config": cfg,
        "training": training,
        "model": model_cfg,
        "hdf5_path": data.get("hdf5_path") or cfg.get("hdf5_path"),
        "train_indices": data.get("train_indices") or training.get("train_indices") or cfg.get("train_indices"),
        "val_indices": data.get("val_indices") or training.get("val_indices") or cfg.get("val_indices"),
        "test_indices": data.get("test_indices") or training.get("test_indices") or cfg.get("test_indices"),
    }


def build_model(runtime_info, device: str):
    training = runtime_info["training"]
    model_cfg = runtime_info["model"]
    common_kwargs = dict(
        input_dim=model_cfg.get("input_dim", 208),
        hidden_dims=tuple(model_cfg.get("hidden_dims", (512, 256, 256))),
        level_embed_dim=model_cfg.get("level_embed_dim", 16),
        coeff_size=model_cfg.get("coeff_size", 24),
        out_channels=model_cfg.get("out_channels", 1),
        dropout=model_cfg.get("dropout", 0.1),
    )
    if "mask_teacher_forcing_ratio" in training:
        model = ConditionalSpatialChangeDCTPredictor(
            mask_condition_dim=model_cfg.get("mask_condition_dim", 128),
            **common_kwargs,
        )
        kind = "conditional"
    elif "lambda_mask" in training:
        model = SpatialChangeGatedDCTPredictor(**common_kwargs)
        kind = "spatial"
    else:
        raise ValueError("Oracle mask analysis is only defined for mask-based TD16 models.")
    weight_path = None
    for name in ("best.pt", "last.pt"):
        candidate = os.path.join(runtime_info["weights_dir"], name)
        if os.path.exists(candidate):
            weight_path = candidate
            break
    if weight_path is None:
        raise FileNotFoundError(f"No best.pt/last.pt found in {runtime_info['weights_dir']}")
    state = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, kind


def summarize_metrics(target, pred, masks, active_threshold):
    reg = masked_regression_metrics_batch(
        target, pred, masks=masks, active_threshold=active_threshold
    )
    out = {
        "mae_mean": float(np.mean(reg["mae"])),
        "rmse_mean": float(np.mean(reg["rmse"])),
        "rel_l2_mean": float(np.mean(reg["rel_l2"])),
    }
    if "active_rel_l2" in reg:
        out["active_rel_l2_mean"] = float(np.nanmean(reg["active_rel_l2"]))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Oracle mask diagnostics for TD16 models")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_info = load_runtime_info(args.weights_dir)
    runtime_info["weights_dir"] = args.weights_dir
    hdf5_path = args.hdf5_path or runtime_info["hdf5_path"]
    if not hdf5_path:
        raise ValueError("No HDF5 path provided and none found in config.yaml")

    split_indices = runtime_info.get(f"{args.split}_indices")
    if split_indices is None:
        raise ValueError(f"Missing {args.split}_indices in config.yaml")
    split_indices = np.asarray(split_indices, dtype=np.int64)

    out_dir = create_result_subdir(args.weights_dir, "td16_oracle_mask_eval")
    print(f"Output directory: {out_dir}")

    model, kind = build_model(runtime_info, args.device)
    active_threshold = float(runtime_info["training"].get("active_threshold", 0.02))
    mask_threshold = float(runtime_info["training"].get("mask_threshold", 0.02))

    targets_all = []
    domain_masks_all = []
    pred_standard_all = []
    pred_oracle_mask_all = []
    pred_oracle_cond_all = []
    total_time = 0.0

    with h5py.File(hdf5_path, "r") as h5f:
        sigma_ds = h5f["sigma_delta"]
        mask_ds = h5f["domain_mask"]
        meas_ds = h5f["measurements"]
        num_batches = int(np.ceil(len(split_indices) / args.batch_size))
        pbar = tqdm(range(num_batches), desc=f"Oracle {args.split}", ncols=100)
        for batch_idx in pbar:
            start = batch_idx * args.batch_size
            end = min(len(split_indices), start + args.batch_size)
            batch_ids = split_indices[start:end]
            order = np.argsort(batch_ids)
            batch_sorted = batch_ids[order]
            inverse = np.argsort(order)

            sigma = sigma_ds[batch_sorted][inverse].astype(np.float32)
            domain_mask = mask_ds[batch_sorted][inverse].astype(np.float32)
            measurements = meas_ds[batch_sorted][inverse].astype(np.float32)

            sigma_tensor = torch.from_numpy(sigma[:, None]).to(args.device)
            domain_mask_tensor = torch.from_numpy(domain_mask[:, None]).to(args.device)
            meas_tensor = torch.from_numpy(measurements).to(args.device)
            level_tensor = torch.ones(
                (meas_tensor.shape[0],), dtype=torch.float, device=args.device
            )
            mask_target = (
                (sigma_tensor.abs() > mask_threshold).float() * domain_mask_tensor.float()
            )

            t0 = time.time()
            with torch.no_grad():
                if kind == "conditional":
                    pred_std, coeffs_std, _, _ = model(meas_tensor, level_tensor)
                    residual_std = model.decoder.coeffs_to_logits(coeffs_std)
                    pred_oracle_mask = residual_std * mask_target

                    mask_coeffs_target = model.target_mask_coeffs(mask_target)
                    _, coeffs_oracle, _, _ = model(
                        meas_tensor,
                        level_tensor,
                        mask_coeffs_override=mask_coeffs_target,
                    )
                    residual_oracle = model.decoder.coeffs_to_logits(coeffs_oracle)
                    pred_oracle_cond = residual_oracle * mask_target
                else:
                    pred_std, coeffs_std, _ = model(meas_tensor, level_tensor)
                    residual_std = model.decoder.coeffs_to_logits(coeffs_std)
                    pred_oracle_mask = residual_std * mask_target
                    pred_oracle_cond = pred_oracle_mask
            total_time += time.time() - t0

            targets_all.append(sigma)
            domain_masks_all.append(domain_mask > 0.5)
            pred_standard_all.append(pred_std[:, 0].detach().float().cpu().numpy())
            pred_oracle_mask_all.append(
                pred_oracle_mask[:, 0].detach().float().cpu().numpy()
            )
            pred_oracle_cond_all.append(
                pred_oracle_cond[:, 0].detach().float().cpu().numpy()
            )

    target_np = np.concatenate(targets_all, axis=0)
    mask_np = np.concatenate(domain_masks_all, axis=0)
    pred_std_np = np.concatenate(pred_standard_all, axis=0)
    pred_oracle_mask_np = np.concatenate(pred_oracle_mask_all, axis=0)
    pred_oracle_cond_np = np.concatenate(pred_oracle_cond_all, axis=0)

    summary = {
        "weights_dir": args.weights_dir,
        "hdf5_path": hdf5_path,
        "split": args.split,
        "model_kind": kind,
        "num_samples": int(len(split_indices)),
        "active_threshold": active_threshold,
        "mask_threshold": mask_threshold,
        "standard": summarize_metrics(target_np, pred_std_np, mask_np, active_threshold),
        "oracle_output_mask": summarize_metrics(
            target_np, pred_oracle_mask_np, mask_np, active_threshold
        ),
        "oracle_condition_and_mask": summarize_metrics(
            target_np, pred_oracle_cond_np, mask_np, active_threshold
        ),
        "diagnostic_time_sec": float(total_time),
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
