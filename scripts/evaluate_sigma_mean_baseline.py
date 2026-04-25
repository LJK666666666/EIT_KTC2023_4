"""Evaluate a train-mean pulmonary conductivity atlas baseline."""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.regression_metrics import masked_regression_metrics_batch


def create_result_subdir(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    idx = 1
    while (base / f"{tag}_{idx}").exists():
        idx += 1
    out_dir = base / f"{tag}_{idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate train-mean sigma atlas baseline')
    parser.add_argument('--hdf5-path', required=True)
    parser.add_argument('--split-ratio', default='8:1:1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--result-tag', default='sigma_mean_baseline')
    return parser.parse_args()


def compute_split_indices(n_total: int, split_ratio: str, seed: int):
    parts = [float(x) for x in split_ratio.split(':')]
    total_ratio = sum(parts)
    train_r = parts[0] / total_ratio
    val_r = parts[1] / total_ratio if len(parts) > 1 else 0.0
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main():
    args = parse_args()
    out_dir = create_result_subdir('results', args.result_tag)
    with h5py.File(args.hdf5_path, 'r') as h5:
        sigma = h5['sigma'][:]

    train_idx, val_idx, test_idx = compute_split_indices(
        len(sigma), args.split_ratio, args.seed)
    split_map = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }
    target_idx = split_map[args.split]
    atlas = sigma[train_idx].mean(axis=0).astype(np.float32)
    preds = np.repeat(atlas[None, ...], len(target_idx), axis=0)
    gt = sigma[target_idx].astype(np.float32)
    reg = masked_regression_metrics_batch(gt, preds)

    np.save(os.path.join(out_dir, 'atlas.npy'), atlas)
    summary = {
        'hdf5_path': args.hdf5_path,
        'split_ratio': args.split_ratio,
        'seed': args.seed,
        'split': args.split,
        'num_samples': int(len(target_idx)),
        'mae_mean': float(np.mean(reg['mae'])),
        'rmse_mean': float(np.mean(reg['rmse'])),
        'rel_l2_mean': float(np.mean(reg['rel_l2'])),
    }
    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
