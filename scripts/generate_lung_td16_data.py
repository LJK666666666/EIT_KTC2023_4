"""Generate 16-electrode pulmonary time-difference EIT simulation data."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import create_lung_pair_conductivity, create_lung_pair_phantom
from src.ktc_methods import EITFEM, image_to_mesh, load_mesh
from src.utils.pulmonary16 import (
    build_adjacent_cycle_mpat,
    build_adjacent_skip3_inj,
    make_16e_mesh,
    reorder_raw256_to_208,
)


def create_result_dir(base_tag: str) -> str:
    base = Path('results')
    base.mkdir(exist_ok=True)
    idx = 1
    while (base / f'{base_tag}_{idx}').exists():
        idx += 1
    out_dir = base / f'{base_tag}_{idx}'
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def save_preview(ref_masks, tgt_masks, deltas, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = min(len(ref_masks), len(tgt_masks), len(deltas))
    cols = 3
    fig, axes = plt.subplots(n, cols, figsize=(2.8 * cols, 2.8 * n))
    axes = np.atleast_2d(axes)
    for row in range(n):
        panels = [
            (ref_masks[row].T, 'gray', 0.0, 2.0),
            (tgt_masks[row].T, 'gray', 0.0, 2.0),
            (deltas[row].T, 'coolwarm', None, None),
        ]
        for col, (img, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def add_measurement_noise(meas, rng, noise_pct1=0.05, noise_pct2=0.01):
    meas = np.asarray(meas, dtype=np.float64).reshape(-1)
    std = np.sqrt(
        ((noise_pct1 / 100.0) * np.abs(meas)) ** 2 +
        ((noise_pct2 / 100.0) * np.max(np.abs(meas))) ** 2
    )
    return meas + std * rng.standard_normal(meas.shape[0])


def main():
    parser = argparse.ArgumentParser(
        description='Generate 16-electrode pulmonary time-difference dataset')
    parser.add_argument('--num-images', type=int, default=2000)
    parser.add_argument('--output-dir', default='dataset_lung_td16')
    parser.add_argument('--mesh-name', default='Mesh_dense.mat')
    parser.add_argument('--preview-count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--anatomy-scale', type=float, default=1.0)
    parser.add_argument('--pathology-scale', type=float, default=1.0)
    parser.add_argument('--detail-scale', type=float, default=1.0)
    parser.add_argument('--conductivity-scale', type=float, default=1.0)
    parser.add_argument('--texture-scale', type=float, default=0.0)
    parser.add_argument('--normal-prob', type=float, default=0.25)
    parser.add_argument('--noise-std1', type=float, default=0.05)
    parser.add_argument('--noise-std2', type=float, default=0.01)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    base_path = Path(args.output_dir) / 'level_1'
    base_path.mkdir(parents=True, exist_ok=True)
    h5_path = base_path / 'data.h5'

    mesh, mesh2 = load_mesh(args.mesh_name)
    mesh16, mesh216 = make_16e_mesh(mesh, mesh2)
    inj = build_adjacent_skip3_inj(16)
    mpat = build_adjacent_cycle_mpat(16)
    vincl = np.ones((16, 16), dtype=bool)
    solver = EITFEM(mesh216, inj, mpat, vincl, use_gpu=False)
    z = 1e-6 * np.ones(16, dtype=np.float64)

    import h5py
    with h5py.File(h5_path, 'w') as h5f:
        h5_gt = h5f.create_dataset(
            'gt', shape=(args.num_images, 256, 256), dtype='uint8',
            chunks=(1, 256, 256), compression='lzf')
        h5_meas = h5f.create_dataset(
            'measurements', shape=(args.num_images, 208), dtype='float32',
            chunks=(1, 208), compression='lzf')
        h5_sigma_delta = h5f.create_dataset(
            'sigma_delta', shape=(args.num_images, 256, 256), dtype='float32',
            chunks=(1, 256, 256), compression='lzf')
        h5_sigma_ref = h5f.create_dataset(
            'sigma_ref', shape=(args.num_images, 256, 256), dtype='float32',
            chunks=(1, 256, 256), compression='lzf')
        h5_sigma_tgt = h5f.create_dataset(
            'sigma_target', shape=(args.num_images, 256, 256), dtype='float32',
            chunks=(1, 256, 256), compression='lzf')
        h5_domain = h5f.create_dataset(
            'domain_mask', shape=(args.num_images, 256, 256), dtype='uint8',
            chunks=(1, 256, 256), compression='lzf')
        h5_idx = h5f.create_dataset(
            'indices', shape=(args.num_images,), dtype='int64',
            chunks=(min(args.num_images, 1024),), compression='lzf')

        previews_ref = []
        previews_tgt = []
        previews_delta = []
        forward_times = []
        total_times = []

        for i in tqdm(range(args.num_images), desc='Generating lung TD16 data'):
            t_total = time.time()
            ref_mask, tgt_mask = create_lung_pair_phantom(
                rng=rng,
                anatomy_scale=args.anatomy_scale,
                pathology_scale=args.pathology_scale,
                detail_scale=args.detail_scale,
                normal_prob=args.normal_prob,
            )
            sigma_ref, sigma_tgt = create_lung_pair_conductivity(
                ref_mask,
                tgt_mask,
                rng=rng,
                conductivity_scale=args.conductivity_scale,
                texture_scale=args.texture_scale,
            )

            sigma_ref_mesh = image_to_mesh(np.flipud(sigma_ref).T, mesh16)
            sigma_tgt_mesh = image_to_mesh(np.flipud(sigma_tgt).T, mesh16)

            t0 = time.time()
            raw_ref = np.asarray(solver.SolveForward(sigma_ref_mesh, z)).reshape(-1)
            raw_tgt = np.asarray(solver.SolveForward(sigma_tgt_mesh, z)).reshape(-1)
            meas_ref = reorder_raw256_to_208(raw_ref)
            meas_tgt = reorder_raw256_to_208(raw_tgt)
            meas_ref = add_measurement_noise(
                meas_ref, rng, args.noise_std1, args.noise_std2)
            meas_tgt = add_measurement_noise(
                meas_tgt, rng, args.noise_std1, args.noise_std2)
            meas_delta = (meas_tgt - meas_ref).astype(np.float32)
            forward_times.append(time.time() - t0)

            sigma_delta = (sigma_tgt - sigma_ref).astype(np.float32)
            domain_mask = (sigma_ref > 0).astype(np.uint8)

            h5_gt[i] = tgt_mask.astype(np.uint8)
            h5_meas[i] = meas_delta
            h5_sigma_delta[i] = sigma_delta
            h5_sigma_ref[i] = sigma_ref.astype(np.float32)
            h5_sigma_tgt[i] = sigma_tgt.astype(np.float32)
            h5_domain[i] = domain_mask
            h5_idx[i] = args.start_idx + i

            if len(previews_ref) < args.preview_count:
                previews_ref.append(ref_mask)
                previews_tgt.append(tgt_mask)
                previews_delta.append(sigma_delta)

            total_times.append(time.time() - t_total)

    info = {
        'num_images': args.num_images,
        'dataset_path': str(h5_path).replace('\\', '/'),
        'electrodes': 16,
        'raw_measurements_per_pattern': 16,
        'reordered_measurement_dim': 208,
        'target_type': 'time_difference_conductivity',
        'avg_forward_ms': float(np.mean(forward_times) * 1000.0),
        'avg_total_ms': float(np.mean(total_times) * 1000.0),
        'seed': args.seed,
        'anatomy_scale': args.anatomy_scale,
        'pathology_scale': args.pathology_scale,
        'detail_scale': args.detail_scale,
        'conductivity_scale': args.conductivity_scale,
        'texture_scale': args.texture_scale,
        'normal_prob': args.normal_prob,
    }
    with open(base_path / 'dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    preview_dir = create_result_dir('lung_td16_preview')
    save_preview(
        previews_ref,
        previews_tgt,
        previews_delta,
        os.path.join(preview_dir, 'paired_phantoms.png'),
    )
    with open(os.path.join(preview_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f'Saved dataset to: {h5_path}')
    print(f'Preview saved to: {preview_dir}')


if __name__ == '__main__':
    main()
