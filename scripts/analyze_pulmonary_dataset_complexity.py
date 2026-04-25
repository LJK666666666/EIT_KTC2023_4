"""Analyze pulmonary conductivity dataset complexity and compressibility."""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.regression_metrics import masked_regression_metrics_batch
from src.models.dct_predictor.model import DCTDecoder


def create_result_dir(base_tag: str) -> Path:
    base = Path('results')
    base.mkdir(exist_ok=True)
    idx = 1
    while (base / f'{base_tag}_{idx}').exists():
        idx += 1
    out_dir = base / f'{base_tag}_{idx}'
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def split_indices(n_total, seed=1, ratio=(8, 1, 1)):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    s = float(sum(ratio))
    n_train = int(round(n_total * ratio[0] / s))
    n_val = int(round(n_total * ratio[1] / s))
    n_test = n_total - n_train - n_val
    return {
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:n_train + n_val + n_test],
    }


def load_sigma_subset(h5_path, indices):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    with h5py.File(h5_path, 'r') as f:
        sigma_sorted = f['sigma'][sorted_idx]
    sigma_sorted = np.asarray(sigma_sorted, dtype=np.float32)
    inv_order = np.argsort(order)
    return sigma_sorted[inv_order]


def compute_train_atlas(h5_path, train_indices, chunk_size=128):
    atlas = None
    count = 0
    with h5py.File(h5_path, 'r') as f:
        ds = f['sigma']
        for start in tqdm(
                range(0, len(train_indices), chunk_size),
                desc='Atlas mean',
                leave=False):
            idx = np.sort(train_indices[start:start + chunk_size])
            batch = np.asarray(ds[idx], dtype=np.float64)
            batch_sum = batch.sum(axis=0)
            if atlas is None:
                atlas = batch_sum
            else:
                atlas += batch_sum
            count += batch.shape[0]
    atlas /= max(count, 1)
    return atlas.astype(np.float32)


def evaluate_atlas_baseline(targets, atlas):
    atlas_batch = np.broadcast_to(atlas[None, ...], targets.shape)
    reg = masked_regression_metrics_batch(targets, atlas_batch)
    target_norm = np.linalg.norm(targets.reshape(targets.shape[0], -1), axis=1)
    residual_norm = np.linalg.norm((targets - atlas_batch).reshape(targets.shape[0], -1), axis=1)
    residual_ratio = residual_norm / np.maximum(target_norm, 1e-8)
    return reg, residual_ratio


def _reconstruct_dct(images_np, coeff_size, batch_size=32):
    decoder = DCTDecoder(image_size=images_np.shape[-1], coeff_size=coeff_size, out_channels=1)
    outputs = []
    for start in tqdm(
            range(0, images_np.shape[0], batch_size),
            desc=f'DCT K={coeff_size}',
            leave=False):
        batch = torch.from_numpy(images_np[start:start + batch_size]).unsqueeze(1)
        with torch.no_grad():
            coeffs = decoder.images_to_coeffs(batch)
            recon = decoder.coeffs_to_logits(coeffs).squeeze(1).cpu().numpy()
        outputs.append(recon)
    return np.concatenate(outputs, axis=0)


def compute_dct_curves(targets, atlas, coeff_sizes):
    results = []
    residuals = targets - atlas[None, ...]
    for k in coeff_sizes:
        direct = _reconstruct_dct(targets, k)
        atlas_res = _reconstruct_dct(residuals, k) + atlas[None, ...]
        reg_direct = masked_regression_metrics_batch(targets, direct)
        reg_res = masked_regression_metrics_batch(targets, atlas_res)
        results.append({
            'coeff_size': int(k),
            'direct_rel_l2_mean': float(np.mean(reg_direct['rel_l2'])),
            'atlas_res_rel_l2_mean': float(np.mean(reg_res['rel_l2'])),
            'direct_rmse_mean': float(np.mean(reg_direct['rmse'])),
            'atlas_res_rmse_mean': float(np.mean(reg_res['rmse'])),
        })
    return results


def save_atlas_figure(atlas, std_map, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    im0 = axes[0].imshow(atlas.T, cmap='viridis', origin='lower')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(std_map.T, cmap='magma', origin='lower')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_curve_figure(curves, out_path):
    ks = [item['coeff_size'] for item in curves]
    direct = [item['direct_rel_l2_mean'] for item in curves]
    atlas_res = [item['atlas_res_rel_l2_mean'] for item in curves]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(ks, direct, marker='o', label='Direct DCT')
    ax.plot(ks, atlas_res, marker='s', label='Atlas-residual DCT')
    ax.set_xlabel('Coefficient size K')
    ax.set_ylabel('Mean relative L2')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_hist_figure(values, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.hist(values, bins=20, color='#4c78a8', alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Residual / target norm')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_focus_figure(focus_stats, out_path):
    labels = list(focus_stats.keys())
    values = [focus_stats[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(labels, values, color='#f58518')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Mean active fraction')
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    ax.grid(alpha=0.2, axis='y')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Analyze pulmonary conductivity dataset complexity')
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=256)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--coeff-sizes', type=int, nargs='+',
                        default=[4, 8, 12, 16, 20, 24, 32, 48, 64])
    parser.add_argument('--focus-thresholds', type=float, nargs='+',
                        default=[0.03, 0.05, 0.08, 0.12])
    parser.add_argument('--experiment-tag', default='pulmonary_complexity')
    args = parser.parse_args()

    with h5py.File(args.dataset_path, 'r') as f:
        if 'sigma' not in f:
            raise ValueError(f'{args.dataset_path} does not contain sigma dataset.')
        n_total = int(f['sigma'].shape[0])

    split_map = split_indices(n_total, seed=args.seed)
    train_indices = np.asarray(split_map['train'], dtype=np.int64)
    eval_indices = np.asarray(split_map[args.split], dtype=np.int64)
    if args.max_samples > 0 and len(eval_indices) > args.max_samples:
        eval_indices = eval_indices[:args.max_samples]

    atlas = compute_train_atlas(args.dataset_path, train_indices, chunk_size=args.chunk_size)
    targets = load_sigma_subset(args.dataset_path, eval_indices)
    std_map = targets.std(axis=0)

    reg, residual_ratio = evaluate_atlas_baseline(targets, atlas)
    curves = compute_dct_curves(targets, atlas, args.coeff_sizes)

    residual_abs = np.abs(targets - atlas[None, ...])
    inside_mask = targets > 0
    focus_stats = {}
    for thr in args.focus_thresholds:
        active = ((residual_abs > thr) & inside_mask).reshape(targets.shape[0], -1)
        focus_stats[f'{thr:.2f}'] = float(np.mean(active.mean(axis=1)))

    summary = {
        'dataset_path': args.dataset_path.replace('\\', '/'),
        'split': args.split,
        'num_total': n_total,
        'num_train': int(len(train_indices)),
        'num_eval': int(len(eval_indices)),
        'atlas_baseline': {
            'mae_mean': float(np.mean(reg['mae'])),
            'rmse_mean': float(np.mean(reg['rmse'])),
            'rel_l2_mean': float(np.mean(reg['rel_l2'])),
            'residual_ratio_mean': float(np.mean(residual_ratio)),
            'residual_ratio_std': float(np.std(residual_ratio)),
        },
        'focus_fraction': focus_stats,
        'dct_curves': curves,
    }

    out_dir = create_result_dir(args.experiment_tag)
    save_atlas_figure(atlas, std_map, out_dir / 'atlas_std.png')
    save_curve_figure(curves, out_dir / 'dct_compressibility.png')
    save_hist_figure(residual_ratio, out_dir / 'residual_ratio_hist.png')
    save_focus_figure(focus_stats, out_dir / 'focus_fraction.png')

    np.save(out_dir / 'atlas.npy', atlas)
    np.save(out_dir / 'std_map.npy', std_map)
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'Analysis saved to: {out_dir}')


if __name__ == '__main__':
    main()
