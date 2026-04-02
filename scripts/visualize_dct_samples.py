"""Visualize DCT predictor reconstructions on HDF5 train/val/test splits."""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.dct_predictor_pipeline import DCTPredictorPipeline


def create_result_subdir(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    idx = 1
    while (base / f'{tag}_{idx}').exists():
        idx += 1
    out_dir = base / f'{tag}_{idx}'
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize DCT predictor samples')
    parser.add_argument('--weights-dir', required=True)
    parser.add_argument('--hdf5-path', default=None)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def load_runtime_info(weights_dir: str):
    with open(os.path.join(weights_dir, 'config.yaml'), 'r', encoding='utf-8') as f:
        cfg = yaml.unsafe_load(f)
    training = cfg.get('training', {})
    data = cfg.get('data', {})
    return {
        'hdf5_path': data.get('hdf5_path') or cfg.get('hdf5_path'),
        'train_indices': data.get('train_indices') or training.get('train_indices') or cfg.get('train_indices'),
        'val_indices': data.get('val_indices') or training.get('val_indices') or cfg.get('val_indices'),
        'test_indices': data.get('test_indices') or training.get('test_indices') or cfg.get('test_indices'),
    }


def select_indices(info, split):
    key = f'{split}_indices'
    idx = info.get(key)
    if idx is None:
        raise ValueError(f'Missing {key} in config.yaml')
    return list(idx)


def main():
    args = parse_args()
    info = load_runtime_info(args.weights_dir)
    hdf5_path = args.hdf5_path or info['hdf5_path']
    if not hdf5_path:
        raise ValueError('No HDF5 path provided and none found in config.yaml')

    indices = select_indices(info, args.split)
    rng = np.random.default_rng(args.seed)
    n = min(len(indices), args.rows * args.cols)
    chosen = rng.choice(indices, size=n, replace=False)
    chosen = np.asarray(chosen, dtype=np.int64)
    order = np.argsort(chosen)
    chosen_sorted = chosen[order]
    inverse = np.argsort(order)

    pipeline = DCTPredictorPipeline(device=args.device, weights_base_dir=args.weights_dir)
    pipeline.load_model(level=1)

    with h5py.File(hdf5_path, 'r') as h5f:
        gts = h5f['gt'][chosen_sorted][inverse]
        measurements = h5f['measurements'][chosen_sorted][inverse]

    ref_data = {
        'Injref': pipeline.create_vincl(1, np.zeros((32, 76), dtype=bool)),  # placeholder
    }
    # Load proper reference from training ref.mat through pipeline helper path.
    from scipy.io import loadmat
    y_ref = loadmat('KTC2023/Codes_Python/TrainingData/ref.mat')
    ref_data = {
        'Injref': y_ref['Injref'],
        'Uelref': y_ref['Uelref'],
        'Mpat': y_ref['Mpat'],
    }

    preds = pipeline.reconstruct_batch(measurements, ref_data, level=1)

    out_dir = create_result_subdir(args.weights_dir, 'dct_samples')
    fig, axes = plt.subplots(args.rows, args.cols * 2, figsize=(2.4 * args.cols * 2, 2.4 * args.rows))
    axes = np.atleast_2d(axes)
    for i in range(args.rows * args.cols):
        gt_ax = axes[i // args.cols, (i % args.cols) * 2]
        pr_ax = axes[i // args.cols, (i % args.cols) * 2 + 1]
        if i < n:
            gt_ax.imshow(gts[i].T, cmap='gray', origin='lower', vmin=0, vmax=2)
            pr_ax.imshow(preds[i].T, cmap='gray', origin='lower', vmin=0, vmax=2)
        gt_ax.set_xticks([])
        gt_ax.set_yticks([])
        pr_ax.set_xticks([])
        pr_ax.set_yticks([])
    fig.savefig(os.path.join(out_dir, f'{args.split}_comparison.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'weights_dir': args.weights_dir,
            'hdf5_path': hdf5_path,
            'split': args.split,
            'indices': [int(x) for x in chosen.tolist()],
        }, f, indent=2, ensure_ascii=False)

    print(f'Output directory: {out_dir}')
    if args.show:
        img = plt.imread(os.path.join(out_dir, f'{args.split}_comparison.png'))
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
