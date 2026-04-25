"""Analyze TD16 external mask fusion and mask localization quality."""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.mask_metrics import binary_mask_metrics_batch
from src.evaluation.regression_metrics import masked_regression_metrics_batch
from src.models.dct_predictor import (
    DCTPredictor,
    SpatialChangeGatedDCTPredictor,
    ConditionalSpatialChangeDCTPredictor,
    ChangeGatedDCTPredictor,
    MaskOnlyDCTPredictor,
)
from src.pipelines.base_pipeline import BasePipeline


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
        "training": training,
        "hdf5_path": data.get("hdf5_path") or cfg.get("hdf5_path"),
        "train_indices": data.get("train_indices") or cfg.get("train_indices"),
        "val_indices": data.get("val_indices") or cfg.get("val_indices"),
        "test_indices": data.get("test_indices") or cfg.get("test_indices"),
    }


def _autocast_context(device: str):
    if device == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def build_mask_model(result_dir: str, device: str):
    info = load_runtime_info(result_dir)
    model_cfg = info["config"].get("model", {})
    model = MaskOnlyDCTPredictor(
        input_dim=model_cfg.get("input_dim", 208),
        hidden_dims=tuple(model_cfg.get("hidden_dims", (512, 256, 256))),
        level_embed_dim=model_cfg.get("level_embed_dim", 16),
        coeff_size=model_cfg.get("coeff_size", 24),
        dropout=model_cfg.get("dropout", 0.1),
    )
    state = BasePipeline._load_state_dict(
        BasePipeline._find_weight([
            os.path.join(result_dir, "best.pt"),
            os.path.join(result_dir, "last.pt"),
        ]),
        device,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, info


def build_residual_model(result_dir: str, device: str):
    info = load_runtime_info(result_dir)
    training = info["training"]
    model_cfg = info["config"].get("model", {})

    if "mask_teacher_forcing_ratio" in training:
        model = ConditionalSpatialChangeDCTPredictor(
            input_dim=model_cfg.get("input_dim", 208),
            hidden_dims=tuple(model_cfg.get("hidden_dims", (512, 256, 256))),
            level_embed_dim=model_cfg.get("level_embed_dim", 16),
            coeff_size=model_cfg.get("coeff_size", 24),
            out_channels=model_cfg.get("out_channels", 1),
            dropout=model_cfg.get("dropout", 0.1),
            mask_condition_dim=model_cfg.get("mask_condition_dim", 128),
        )
        model_type = "conditional"
    elif "lambda_mask" in training:
        model = SpatialChangeGatedDCTPredictor(
            input_dim=model_cfg.get("input_dim", 208),
            hidden_dims=tuple(model_cfg.get("hidden_dims", (512, 256, 256))),
            level_embed_dim=model_cfg.get("level_embed_dim", 16),
            coeff_size=model_cfg.get("coeff_size", 24),
            out_channels=model_cfg.get("out_channels", 1),
            dropout=model_cfg.get("dropout", 0.1),
        )
        model_type = "spatial"
    elif "lambda_gate" in training:
        model = ChangeGatedDCTPredictor(
            input_dim=model_cfg.get("input_dim", 208),
            hidden_dims=tuple(model_cfg.get("hidden_dims", (512, 256, 256))),
            level_embed_dim=model_cfg.get("level_embed_dim", 16),
            coeff_size=model_cfg.get("coeff_size", 24),
            out_channels=model_cfg.get("out_channels", 1),
            dropout=model_cfg.get("dropout", 0.1),
        )
        model_type = "change"
    else:
        model = DCTPredictor(
            input_dim=model_cfg.get("input_dim", 208),
            hidden_dims=tuple(model_cfg.get("hidden_dims", (512, 256, 256))),
            level_embed_dim=model_cfg.get("level_embed_dim", 16),
            coeff_size=model_cfg.get("coeff_size", 24),
            out_channels=model_cfg.get("out_channels", 1),
            dropout=model_cfg.get("dropout", 0.1),
        )
        model_type = "base"

    state = BasePipeline._load_state_dict(
        BasePipeline._find_weight([
            os.path.join(result_dir, "best.pt"),
            os.path.join(result_dir, "last.pt"),
        ]),
        device,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, info, model_type


def mask_metric_means(target, pred_probs, domain_mask, threshold=0.5):
    metrics = binary_mask_metrics_batch(
        target=target,
        pred=pred_probs,
        valid_mask=domain_mask,
        threshold=threshold,
    )
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def reg_metric_means(target, pred, domain_mask, active_threshold=0.02):
    reg = masked_regression_metrics_batch(
        target, pred, masks=domain_mask, active_threshold=active_threshold
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
    parser = argparse.ArgumentParser(
        description="Analyze TD16 external mask fusion.")
    parser.add_argument("--mask-weights-dir", required=True)
    parser.add_argument("--residual-weights-dir", required=True)
    parser.add_argument("--hdf5-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    mask_model, mask_info = build_mask_model(args.mask_weights_dir, device)
    residual_model, residual_info, residual_type = build_residual_model(
        args.residual_weights_dir, device
    )
    hdf5_path = args.hdf5_path or residual_info["hdf5_path"] or mask_info["hdf5_path"]
    if not hdf5_path:
        raise ValueError("No HDF5 path provided and none found in config.yaml")

    split_indices = residual_info.get(f"{args.split}_indices")
    if split_indices is None:
        split_indices = mask_info.get(f"{args.split}_indices")
    if split_indices is None:
        raise ValueError(f"Missing {args.split}_indices in config")
    split_indices = np.asarray(split_indices, dtype=np.int64)

    mask_threshold = args.mask_threshold
    if mask_threshold is None:
        mask_threshold = float(mask_info["training"].get("mask_threshold", 0.02))

    out_dir = create_result_subdir(args.residual_weights_dir, "td16_external_mask_eval")
    print(f"Output directory: {out_dir}")

    targets_all = []
    domain_masks_all = []
    target_masks_all = []
    mask_probs_all = []
    internal_mask_probs_all = []
    residual_only_all = []
    internal_pred_all = []
    external_soft_all = []
    external_hard_all = []

    with h5py.File(hdf5_path, "r") as h5f:
        meas_ds = h5f["measurements"]
        sigma_ds = h5f["sigma_delta"]
        domain_ds = h5f["domain_mask"]
        num_batches = int(np.ceil(len(split_indices) / args.batch_size))
        pbar = tqdm(range(num_batches), desc=f"Fusion {args.split}", ncols=100)
        for batch_idx in pbar:
            start = batch_idx * args.batch_size
            end = min(len(split_indices), start + args.batch_size)
            batch_ids = split_indices[start:end]
            order = np.argsort(batch_ids)
            batch_sorted = batch_ids[order]
            inverse = np.argsort(order)

            measurements = meas_ds[batch_sorted][inverse].astype(np.float32)
            sigma_delta = sigma_ds[batch_sorted][inverse].astype(np.float32)
            domain_mask = (domain_ds[batch_sorted][inverse].astype(np.float32) > 0.5)
            target_mask = ((np.abs(sigma_delta) > mask_threshold) & domain_mask)

            meas_tensor = torch.from_numpy(measurements).to(device)
            level_tensor = torch.ones(
                (meas_tensor.shape[0],), dtype=torch.float, device=device
            )
            with torch.no_grad():
                with _autocast_context(device):
                    mask_logits, _ = mask_model(meas_tensor, level_tensor)
                    mask_probs = torch.sigmoid(mask_logits)

                    internal_mask_probs = None
                    if residual_type == "conditional":
                        internal_pred, coeffs, mask_logits_internal, _ = residual_model(
                            meas_tensor, level_tensor
                        )
                        residual_only = residual_model.decoder.coeffs_to_logits(coeffs)
                        internal_mask_probs = torch.sigmoid(mask_logits_internal)
                    elif residual_type == "spatial":
                        internal_pred, coeffs, mask_logits_internal = residual_model(
                            meas_tensor, level_tensor
                        )
                        residual_only = residual_model.decoder.coeffs_to_logits(coeffs)
                        internal_mask_probs = torch.sigmoid(mask_logits_internal)
                    elif residual_type == "change":
                        internal_pred, coeffs, _ = residual_model(meas_tensor, level_tensor)
                        residual_only = residual_model.decoder.coeffs_to_logits(coeffs)
                    else:
                        internal_pred, _ = residual_model(meas_tensor, level_tensor)
                        residual_only = internal_pred

                    external_soft = residual_only * mask_probs
                    external_hard = residual_only * (mask_probs > args.prob_threshold).to(
                        residual_only.dtype
                    )

            targets_all.append(sigma_delta)
            domain_masks_all.append(domain_mask)
            target_masks_all.append(target_mask.astype(np.float32))
            mask_probs_all.append(mask_probs[:, 0].detach().float().cpu().numpy())
            residual_only_all.append(
                residual_only[:, 0].detach().float().cpu().numpy().astype(np.float32)
            )
            internal_pred_all.append(
                internal_pred[:, 0].detach().float().cpu().numpy().astype(np.float32)
            )
            external_soft_all.append(
                external_soft[:, 0].detach().float().cpu().numpy().astype(np.float32)
            )
            external_hard_all.append(
                external_hard[:, 0].detach().float().cpu().numpy().astype(np.float32)
            )
            if internal_mask_probs is not None:
                internal_mask_probs_all.append(
                    internal_mask_probs[:, 0].detach().float().cpu().numpy()
                )

    target_np = np.concatenate(targets_all, axis=0)
    domain_np = np.concatenate(domain_masks_all, axis=0)
    target_mask_np = np.concatenate(target_masks_all, axis=0)
    mask_probs_np = np.concatenate(mask_probs_all, axis=0)
    residual_only_np = np.concatenate(residual_only_all, axis=0)
    internal_pred_np = np.concatenate(internal_pred_all, axis=0)
    external_soft_np = np.concatenate(external_soft_all, axis=0)
    external_hard_np = np.concatenate(external_hard_all, axis=0)

    summary = {
        "mask_weights_dir": args.mask_weights_dir,
        "residual_weights_dir": args.residual_weights_dir,
        "split": args.split,
        "hdf5_path": hdf5_path,
        "residual_model_type": residual_type,
        "mask_threshold": float(mask_threshold),
        "prob_threshold": float(args.prob_threshold),
        "external_mask_metrics": mask_metric_means(
            target_mask_np, mask_probs_np, domain_np, threshold=args.prob_threshold
        ),
        "internal_prediction_metrics": reg_metric_means(
            target_np, internal_pred_np, domain_np, active_threshold=mask_threshold
        ),
        "residual_only_metrics": reg_metric_means(
            target_np, residual_only_np, domain_np, active_threshold=mask_threshold
        ),
        "external_soft_fusion_metrics": reg_metric_means(
            target_np, external_soft_np, domain_np, active_threshold=mask_threshold
        ),
        "external_hard_fusion_metrics": reg_metric_means(
            target_np, external_hard_np, domain_np, active_threshold=mask_threshold
        ),
    }
    if internal_mask_probs_all:
        internal_mask_probs_np = np.concatenate(internal_mask_probs_all, axis=0)
        summary["internal_mask_metrics"] = mask_metric_means(
            target_mask_np, internal_mask_probs_np, domain_np, threshold=args.prob_threshold
        )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
