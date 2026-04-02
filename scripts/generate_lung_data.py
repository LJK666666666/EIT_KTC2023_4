"""Generate thorax-style pulmonary EIT training data."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.lung_phantom import create_lung_conductivity, create_lung_phantom
from src.ktc_methods import EITFEM, image_to_mesh, load_mesh


def create_result_dir(base_tag: str) -> str:
    base = Path('results')
    base.mkdir(exist_ok=True)
    idx = 1
    while (base / f'{base_tag}_{idx}').exists():
        idx += 1
    out_dir = base / f'{base_tag}_{idx}'
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def save_preview(samples, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(samples)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for idx, sample in enumerate(samples):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.imshow(sample.T, cmap='gray', origin='lower', vmin=0, vmax=2)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate thorax-style EIT data')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--num-images', type=int, default=2000)
    parser.add_argument('--output-dir', default='dataset_lung')
    parser.add_argument('--ref-path', default='KTC2023/Codes_Python/TrainingData/ref.mat')
    parser.add_argument('--mesh-name', default='Mesh_dense.mat')
    parser.add_argument('--preview-count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--save-sigma', action='store_true',
                        help='Save conductivity maps in HDF5')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    base_path = Path(args.output_dir) / f'level_{args.level}'
    base_path.mkdir(parents=True, exist_ok=True)
    h5_path = base_path / 'data.h5'

    y_ref = loadmat(args.ref_path)
    Injref = y_ref['Injref']
    Mpat = y_ref['Mpat']
    mesh, mesh2 = load_mesh(args.mesh_name)

    nel = 32
    z = 1e-6 * np.ones(nel)
    vincl = np.ones((nel - 1, 76), dtype=bool)
    solver = EITFEM(mesh2, Injref, Mpat, vincl, use_gpu=False)

    noise_std1 = 0.05
    noise_std2 = 0.01
    solver.SetInvGamma(noise_std1, noise_std2, y_ref['Uelref'])

    import h5py
    with h5py.File(h5_path, 'w') as h5f:
        n_meas = int(np.asarray(y_ref['Uelref']).reshape(-1).shape[0])
        h5_gt = h5f.create_dataset(
            'gt', shape=(args.num_images, 256, 256), dtype='uint8',
            chunks=(1, 256, 256), compression='lzf')
        h5_meas = h5f.create_dataset(
            'measurements', shape=(args.num_images, n_meas), dtype='float32',
            chunks=(1, n_meas), compression='lzf')
        h5_idx = h5f.create_dataset(
            'indices', shape=(args.num_images,), dtype='int64',
            chunks=(min(args.num_images, 1024),), compression='lzf')
        h5_sigma = None
        if args.save_sigma:
            h5_sigma = h5f.create_dataset(
                'sigma', shape=(args.num_images, 256, 256), dtype='float32',
                chunks=(1, 256, 256), compression='lzf')

        previews = []
        forward_times = []
        total_times = []

        for i in tqdm(range(args.num_images), desc=f'Lung level {args.level}'):
            t_total = time.time()
            phantom = create_lung_phantom(rng=rng)
            sigma = create_lung_conductivity(phantom, rng=rng)
            sigma_mesh = image_to_mesh(np.flipud(sigma).T, mesh)

            t0 = time.time()
            uel = np.asarray(solver.SolveForward(sigma_mesh, z)).reshape(-1, 1)
            noise = np.asarray(
                solver.InvLn @ rng.standard_normal((uel.shape[0], 1))).reshape(-1, 1)
            uel_noisy = (uel + noise).reshape(-1).astype(np.float32)
            forward_times.append(time.time() - t0)

            h5_gt[i] = phantom.astype(np.uint8)
            h5_meas[i] = uel_noisy
            h5_idx[i] = args.start_idx + i
            if h5_sigma is not None:
                h5_sigma[i] = sigma.astype(np.float32)

            if len(previews) < args.preview_count:
                previews.append(phantom)
            total_times.append(time.time() - t_total)

    info = {
        'level': args.level,
        'num_images': args.num_images,
        'dataset_path': str(h5_path).replace('\\', '/'),
        'avg_forward_ms': float(np.mean(forward_times) * 1000.0),
        'avg_total_ms': float(np.mean(total_times) * 1000.0),
        'seed': args.seed,
    }
    with open(base_path / 'dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    preview_dir = create_result_dir('lung_data_preview')
    save_preview(previews, os.path.join(preview_dir, 'lung_phantoms.png'))
    with open(os.path.join(preview_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f'HDF5 saved to: {h5_path}')
    print(f'Preview saved to: {preview_dir}')


if __name__ == '__main__':
    main()
