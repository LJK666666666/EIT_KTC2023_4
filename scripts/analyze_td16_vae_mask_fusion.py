"""Fuse mixed-dataset change localization with active-only TD16 VAE predictor."""

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
from src.models.dct_predictor import MaskOnlyDCTPredictor
from src.models.pulmonary_vae import ConvVAE, LatentMLPPredictor
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
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.unsafe_load(f)


def load_runtime_info(result_dir: str):
    cfg = load_yaml(os.path.join(result_dir, 'config.yaml')) or {}
    training = cfg.get('training', {})
    data = cfg.get('data', {})
    return {
        'config': cfg,
        'training': training,
        'hdf5_path': data.get('hdf5_path') or cfg.get('hdf5_path'),
        'train_indices': data.get('train_indices') or cfg.get('train_indices'),
        'val_indices': data.get('val_indices') or cfg.get('val_indices'),
        'test_indices': data.get('test_indices') or cfg.get('test_indices'),
    }


def _autocast_context(device: str):
    if device == 'cuda' and torch.cuda.is_available():
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def build_mask_model(result_dir: str, device: str):
    info = load_runtime_info(result_dir)
    model_cfg = info['config'].get('model', {})
    model = MaskOnlyDCTPredictor(
        input_dim=model_cfg.get('input_dim', 208),
        hidden_dims=tuple(model_cfg.get('hidden_dims', (512, 256, 256))),
        level_embed_dim=model_cfg.get('level_embed_dim', 16),
        coeff_size=model_cfg.get('coeff_size', 24),
        dropout=model_cfg.get('dropout', 0.1),
    )
    state = BasePipeline._load_state_dict(
        BasePipeline._find_weight([
            os.path.join(result_dir, 'best.pt'),
            os.path.join(result_dir, 'last.pt'),
        ]),
        device,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, info


def build_vae_predictor(result_dir: str, device: str):
    info = load_runtime_info(result_dir)
    cfg = info['config']
    model_cfg = cfg.get('model', {})
    predictor = LatentMLPPredictor(
        input_dim=model_cfg.get('input_dim', 208),
        latent_dim=model_cfg.get('latent_dim', 32),
        hidden_dims=tuple(model_cfg.get('hidden_dims', (512, 256, 128))),
        dropout=model_cfg.get('dropout', 0.1),
    )
    pred_state = BasePipeline._load_state_dict(
        BasePipeline._find_weight([
            os.path.join(result_dir, 'best.pt'),
            os.path.join(result_dir, 'last.pt'),
        ]),
        device,
    )
    predictor.load_state_dict(pred_state)
    predictor.to(device)
    predictor.eval()

    vae_cfg = cfg.get('vae', {})
    vae = ConvVAE(
        in_channels=vae_cfg.get('in_channels', 1),
        latent_dim=vae_cfg.get('latent_dim', 32),
        base_channels=vae_cfg.get('base_channels', 32),
    )
    vae_state = torch.load(
        vae_cfg['checkpoint'], map_location=device, weights_only=False)
    if 'model_state_dict' in vae_state:
        vae_state = vae_state['model_state_dict']
    vae.load_state_dict(vae_state)
    vae.to(device)
    vae.eval()
    return predictor, vae, info


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
        'mae_mean': float(np.mean(reg['mae'])),
        'rmse_mean': float(np.mean(reg['rmse'])),
        'rel_l2_mean': float(np.mean(reg['rel_l2'])),
    }
    if 'active_rel_l2' in reg:
        out['active_rel_l2_mean'] = float(np.nanmean(reg['active_rel_l2']))
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse mixed mask-only model with active-only TD16 VAE predictor.')
    parser.add_argument('--mask-weights-dir', required=True)
    parser.add_argument('--vae-predictor-dir', required=True)
    parser.add_argument('--hdf5-path', default=None)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--mask-threshold', type=float, default=None)
    parser.add_argument('--prob-threshold', type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    mask_model, mask_info = build_mask_model(args.mask_weights_dir, device)
    predictor, vae, predictor_info = build_vae_predictor(args.vae_predictor_dir, device)

    hdf5_path = args.hdf5_path or mask_info['hdf5_path']
    if not hdf5_path:
        raise ValueError('No HDF5 path provided.')

    split_indices = mask_info.get(f'{args.split}_indices')
    if split_indices is None:
        raise ValueError(f'Missing {args.split}_indices in mask model config.')
    split_indices = np.asarray(split_indices, dtype=np.int64)

    mask_threshold = args.mask_threshold
    if mask_threshold is None:
        mask_threshold = float(mask_info['training'].get('mask_threshold', 0.02))

    out_dir = create_result_subdir(args.vae_predictor_dir, 'td16_vae_mask_fusion')
    print(f'Output directory: {out_dir}')

    target_all = []
    domain_all = []
    target_mask_all = []
    mask_probs_all = []
    base_pred_all = []
    soft_pred_all = []
    hard_pred_all = []

    with h5py.File(hdf5_path, 'r') as h5f:
        meas_ds = h5f['measurements']
        sigma_ds = h5f['sigma_delta']
        domain_ds = h5f['domain_mask']
        num_batches = int(np.ceil(len(split_indices) / args.batch_size))
        pbar = tqdm(range(num_batches), desc=f'Fusion {args.split}', ncols=100)
        for batch_idx in pbar:
            start = batch_idx * args.batch_size
            end = min(len(split_indices), start + args.batch_size)
            batch_ids = split_indices[start:end]
            order = np.argsort(batch_ids)
            batch_sorted = batch_ids[order]
            inverse = np.argsort(order)

            measurements = meas_ds[batch_sorted][inverse].astype(np.float32)
            sigma_delta = sigma_ds[batch_sorted][inverse].astype(np.float32)
            domain_mask = domain_ds[batch_sorted][inverse].astype(np.float32) > 0.5
            target_mask = ((np.abs(sigma_delta) > mask_threshold) & domain_mask)

            meas_tensor = torch.from_numpy(measurements).to(device)
            level_tensor = torch.ones(
                (meas_tensor.shape[0],), dtype=torch.float, device=device)
            with torch.no_grad():
                with _autocast_context(device):
                    mask_logits, _ = mask_model(meas_tensor, level_tensor)
                    mask_probs = torch.sigmoid(mask_logits)
                    latent = predictor(meas_tensor)
                    base_pred = vae.decode(latent)

            mask_np = mask_probs[:, 0].detach().float().cpu().numpy()
            base_np = base_pred[:, 0].detach().float().cpu().numpy()
            soft_np = base_np * mask_np
            hard_np = base_np * (mask_np > args.prob_threshold)

            target_all.append(sigma_delta)
            domain_all.append(domain_mask)
            target_mask_all.append(target_mask)
            mask_probs_all.append(mask_np)
            base_pred_all.append(base_np)
            soft_pred_all.append(soft_np)
            hard_pred_all.append(hard_np)

    target_np = np.concatenate(target_all, axis=0)
    domain_np = np.concatenate(domain_all, axis=0)
    target_mask_np = np.concatenate(target_mask_all, axis=0)
    mask_probs_np = np.concatenate(mask_probs_all, axis=0)
    base_np = np.concatenate(base_pred_all, axis=0)
    soft_np = np.concatenate(soft_pred_all, axis=0)
    hard_np = np.concatenate(hard_pred_all, axis=0)

    summary = {
        'mask_weights_dir': args.mask_weights_dir,
        'vae_predictor_dir': args.vae_predictor_dir,
        'hdf5_path': hdf5_path,
        'split': args.split,
        'num_samples': int(target_np.shape[0]),
        'mask_metrics': mask_metric_means(
            target_mask_np, mask_probs_np, domain_np, threshold=args.prob_threshold),
        'base_prediction': reg_metric_means(target_np, base_np, domain_np),
        'external_soft_fusion': reg_metric_means(target_np, soft_np, domain_np),
        'external_hard_fusion': reg_metric_means(target_np, hard_np, domain_np),
    }
    out_path = os.path.join(out_dir, 'summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f'Summary saved to: {out_path}')


if __name__ == '__main__':
    main()
