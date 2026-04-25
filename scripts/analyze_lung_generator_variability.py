"""Analyze controllable variability presets of the pulmonary phantom generator."""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.regression_metrics import masked_regression_metrics_batch
from src.models.dct_predictor.model import DCTDecoder
from src.data.lung_phantom import create_lung_conductivity, create_lung_phantom


PRESETS = {
    'low': {
        'anatomy_scale': 0.6,
        'pathology_scale': 0.6,
        'detail_scale': 0.5,
        'conductivity_scale': 0.8,
        'texture_scale': 0.00,
    },
    'medium': {
        'anatomy_scale': 1.0,
        'pathology_scale': 1.0,
        'detail_scale': 1.0,
        'conductivity_scale': 1.0,
        'texture_scale': 0.00,
    },
    'high': {
        'anatomy_scale': 1.6,
        'pathology_scale': 1.6,
        'detail_scale': 1.5,
        'conductivity_scale': 1.25,
        'texture_scale': 0.35,
    },
}


def create_result_dir(base_tag: str) -> Path:
    base = Path('results')
    base.mkdir(exist_ok=True)
    idx = 1
    while (base / f'{base_tag}_{idx}').exists():
        idx += 1
    out_dir = base / f'{base_tag}_{idx}'
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _reconstruct_dct(images_np, coeff_size, batch_size=32):
    decoder = DCTDecoder(image_size=images_np.shape[-1], coeff_size=coeff_size, out_channels=1)
    outputs = []
    for start in range(0, images_np.shape[0], batch_size):
        batch = torch.from_numpy(images_np[start:start + batch_size]).unsqueeze(1)
        with torch.no_grad():
            coeffs = decoder.images_to_coeffs(batch)
            recon = decoder.coeffs_to_logits(coeffs).squeeze(1).cpu().numpy()
        outputs.append(recon)
    return np.concatenate(outputs, axis=0)


def compute_metrics(sigmas, coeff_sizes):
    atlas = sigmas.mean(axis=0)
    atlas_batch = np.broadcast_to(atlas[None, ...], sigmas.shape)
    atlas_reg = masked_regression_metrics_batch(sigmas, atlas_batch)
    residual_ratio = np.linalg.norm(
        (sigmas - atlas_batch).reshape(sigmas.shape[0], -1), axis=1
    ) / np.maximum(np.linalg.norm(sigmas.reshape(sigmas.shape[0], -1), axis=1), 1e-8)

    curves = []
    residuals = sigmas - atlas[None, ...]
    for k in coeff_sizes:
        direct = _reconstruct_dct(sigmas, k)
        atlas_res = _reconstruct_dct(residuals, k) + atlas[None, ...]
        direct_reg = masked_regression_metrics_batch(sigmas, direct)
        atlas_reg_k = masked_regression_metrics_batch(sigmas, atlas_res)
        curves.append({
            'coeff_size': int(k),
            'direct_rel_l2_mean': float(np.mean(direct_reg['rel_l2'])),
            'atlas_res_rel_l2_mean': float(np.mean(atlas_reg_k['rel_l2'])),
        })
    return {
        'atlas': atlas,
        'std_map': sigmas.std(axis=0),
        'atlas_rel_l2_mean': float(np.mean(atlas_reg['rel_l2'])),
        'atlas_rmse_mean': float(np.mean(atlas_reg['rmse'])),
        'residual_ratio_mean': float(np.mean(residual_ratio)),
        'residual_ratio_std': float(np.std(residual_ratio)),
        'curves': curves,
    }


def save_sigma_grid(samples_by_name, out_path, n_show=4):
    rows = len(samples_by_name)
    cols = n_show
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.2 * rows))
    axes = np.atleast_2d(axes)
    for r, (_, samples) in enumerate(samples_by_name.items()):
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(samples[c].T, cmap='viridis', origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_atlas_std_grid(metrics_by_name, out_path):
    names = list(metrics_by_name.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(5.2, 2.4 * len(names)))
    axes = np.atleast_2d(axes)
    for r, name in enumerate(names):
        atlas = metrics_by_name[name]['atlas']
        std_map = metrics_by_name[name]['std_map']
        im0 = axes[r, 0].imshow(atlas.T, cmap='viridis', origin='lower')
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.02)
        im1 = axes[r, 1].imshow(std_map.T, cmap='magma', origin='lower')
        axes[r, 1].set_xticks([])
        axes[r, 1].set_yticks([])
        fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_curve(metrics_by_name, out_path):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for name, metrics in metrics_by_name.items():
        ks = [item['coeff_size'] for item in metrics['curves']]
        vals = [item['atlas_res_rel_l2_mean'] for item in metrics['curves']]
        ax.plot(ks, vals, marker='o', label=name)
    ax.set_xlabel('Coefficient size K')
    ax.set_ylabel('Mean relative L2')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_summary_bars(metrics_by_name, out_path):
    names = list(metrics_by_name.keys())
    atlas_vals = [metrics_by_name[name]['atlas_rel_l2_mean'] for name in names]
    residual_vals = [metrics_by_name[name]['residual_ratio_mean'] for name in names]
    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - width / 2, atlas_vals, width=width, label='Atlas baseline')
    ax.bar(x + width / 2, residual_vals, width=width, label='Residual ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Mean value')
    ax.grid(alpha=0.2, axis='y')
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Analyze lung phantom variability presets')
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--coeff-sizes', type=int, nargs='+',
                        default=[4, 8, 12, 16, 20, 24, 32, 48, 64])
    parser.add_argument('--experiment-tag', default='lung_variability_analysis')
    args = parser.parse_args()

    out_dir = create_result_dir(args.experiment_tag)
    rng = np.random.default_rng(args.seed)
    samples_by_name = {}
    metrics_by_name = {}
    summary = {'num_samples': args.num_samples, 'presets': {}}

    for name, cfg in PRESETS.items():
        sigmas = []
        previews = []
        for _ in tqdm(range(args.num_samples), desc=f'{name} preset'):
            mask = create_lung_phantom(
                rng=rng,
                anatomy_scale=cfg['anatomy_scale'],
                pathology_scale=cfg['pathology_scale'],
                detail_scale=cfg['detail_scale'],
            )
            sigma = create_lung_conductivity(
                mask,
                rng=rng,
                conductivity_scale=cfg['conductivity_scale'],
                texture_scale=cfg['texture_scale'],
            )
            sigmas.append(sigma.astype(np.float32))
            if len(previews) < 4:
                previews.append(sigma.astype(np.float32))
        sigmas = np.stack(sigmas, axis=0)
        samples_by_name[name] = previews
        metrics = compute_metrics(sigmas, args.coeff_sizes)
        metrics_by_name[name] = metrics
        summary['presets'][name] = {
            'config': cfg,
            'atlas_rel_l2_mean': metrics['atlas_rel_l2_mean'],
            'atlas_rmse_mean': metrics['atlas_rmse_mean'],
            'residual_ratio_mean': metrics['residual_ratio_mean'],
            'residual_ratio_std': metrics['residual_ratio_std'],
            'curves': metrics['curves'],
        }

    save_sigma_grid(samples_by_name, out_dir / 'sigma_samples.png')
    save_atlas_std_grid(metrics_by_name, out_dir / 'atlas_std.png')
    save_curve(metrics_by_name, out_dir / 'dct_curve.png')
    save_summary_bars(metrics_by_name, out_dir / 'summary_bars.png')

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'Analysis saved to: {out_dir}')


if __name__ == '__main__':
    main()
