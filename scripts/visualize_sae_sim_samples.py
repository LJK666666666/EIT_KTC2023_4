"""
Visualize SAE/VQ-SAE predictor inference on random train/val/test simulated samples.

This script:
1. Loads predictor + decoder pipeline.
2. Reads train/val/test indices from the predictor result config.yaml.
3. Randomly samples N*M examples from each split.
4. Runs batch inference without score computation.
5. Saves GT vs prediction comparison grids under the predictor result directory.
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.io import loadmat
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.sae_pipeline import SAEPipeline
from src.pipelines.vq_sae_pipeline import VQSAEPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SAE/VQ-SAE predictor results on simulated train/val/test splits')
    parser.add_argument('--weights-dir', type=str, default='results',
                        help='Base results directory or a predictor result directory')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'sae', 'vq_sae'],
                        help='Pipeline family. auto = infer from result dir/config')
    parser.add_argument('--sae-config', type=str,
                        default='scripts/sae_pipeline.yaml',
                        help='YAML config with sae_dir / sae_predictor_dir')
    parser.add_argument('--vq-sae-config', type=str,
                        default='scripts/vq_sae_pipeline.yaml',
                        help='YAML config with vq_sae_dir / vq_sae_predictor_dir')
    parser.add_argument('--sae-dir', type=str, default='',
                        help='Override SAE result subdirectory under weights-dir')
    parser.add_argument('--sae-predictor-dir', type=str, default='',
                        help='Override SAE predictor result subdirectory under weights-dir')
    parser.add_argument('--vq-sae-dir', type=str, default='',
                        help='Override VQ-SAE result subdirectory under weights-dir')
    parser.add_argument('--vq-sae-predictor-dir', type=str, default='',
                        help='Override VQ-SAE predictor result subdirectory under weights-dir')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Inference device (cuda/cpu)')
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='Dataset splits to visualize')
    parser.add_argument('--rows', type=int, default=2,
                        help='Number of sample rows per split figure')
    parser.add_argument('--cols', type=int, default=4,
                        help='Number of sample columns per split figure')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Inference batch size')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for sampling')
    parser.add_argument('--show', action='store_true',
                        help='Also display figures interactively after saving')
    return parser.parse_args()


def create_result_subdir(base_dir, prefix='ae_sim_samples'):
    num = 1
    while True:
        out_dir = os.path.join(base_dir, f'{prefix}_{num}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        num += 1


def load_result_config(result_dir):
    config_path = os.path.join(result_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.unsafe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Invalid config format: {config_path}')
    return data


def infer_family(args, weights_name, predictor_cfg=None):
    if args.method != 'auto':
        return args.method
    if weights_name.startswith('vq_sae_predictor_') or weights_name.startswith('vq_sae_baseline'):
        return 'vq_sae'
    if weights_name.startswith('sae_predictor_') or weights_name.startswith('sae_baseline'):
        return 'sae'
    if predictor_cfg and isinstance(predictor_cfg, dict):
        if 'vq_sae' in predictor_cfg:
            return 'vq_sae'
        if 'sae' in predictor_cfg:
            return 'sae'
    return 'sae'


def sample_split_indices(indices, num_samples, seed):
    indices = list(indices) if indices is not None else []
    if not indices:
        return []
    n = min(len(indices), num_samples)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(indices, dtype=int), size=n, replace=False)
    return chosen.tolist()


def load_h5_samples(h5_path, indices):
    measurements = []
    ground_truths = []
    with h5py.File(h5_path, 'r') as f:
        for idx in indices:
            measurements.append(np.asarray(f['measurements'][idx]))
            ground_truths.append(np.asarray(f['gt'][idx]).astype(np.int32))
    return measurements, ground_truths


def batched_reconstruct(pipeline, measurements, ref_data, level, batch_size):
    predictions = []
    for start in tqdm(range(0, len(measurements), batch_size),
                      desc='Infer', leave=False):
        batch = measurements[start:start + batch_size]
        predictions.extend(pipeline.reconstruct_batch(batch, ref_data, level))
    return predictions


def render_split_grid(ground_truths, predictions, indices, rows, cols,
                      save_path, show=False):
    cmap = ListedColormap(['#111111', '#2ca02c', '#d62728'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(
        rows * 2, cols, figsize=(2.2 * cols, 2.2 * rows * 2), dpi=150)
    axes = np.atleast_2d(axes)

    total_slots = rows * cols
    for slot in range(total_slots):
        sample_row = slot // cols
        sample_col = slot % cols
        gt_ax = axes[2 * sample_row, sample_col]
        pred_ax = axes[2 * sample_row + 1, sample_col]

        if slot < len(indices):
            gt_ax.imshow(ground_truths[slot], cmap=cmap, norm=norm,
                         interpolation='nearest')
            pred_ax.imshow(predictions[slot], cmap=cmap, norm=norm,
                           interpolation='nearest')
            pred_ax.set_xlabel(f'idx={indices[slot]}', fontsize=8)
        else:
            gt_ax.axis('off')
            pred_ax.axis('off')
            continue

        gt_ax.set_xticks([])
        gt_ax.set_yticks([])
        pred_ax.set_xticks([])
        pred_ax.set_yticks([])

        if sample_col == 0:
            gt_ax.set_ylabel('GT', fontsize=9)
            pred_ax.set_ylabel('Pred', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    weights_base_dir = args.weights_dir
    sae_dir_override = args.sae_dir
    predictor_dir_override = args.sae_predictor_dir
    vq_sae_dir_override = args.vq_sae_dir
    vq_predictor_dir_override = args.vq_sae_predictor_dir

    norm_weights_dir = os.path.normpath(args.weights_dir)
    weights_name = os.path.basename(norm_weights_dir)
    weights_parent = os.path.dirname(norm_weights_dir) or '.'
    predictor_cfg = None

    if os.path.isdir(args.weights_dir) and os.path.exists(
            os.path.join(args.weights_dir, 'config.yaml')):
        predictor_cfg = load_result_config(args.weights_dir)
        cfg_family = infer_family(args, weights_name, predictor_cfg)
        if weights_name.startswith('sae_predictor_') and not predictor_dir_override:
            predictor_dir_override = weights_name
            weights_base_dir = weights_parent
            sae_ckpt = predictor_cfg.get('sae', {}).get('checkpoint', '')
            if sae_ckpt and not sae_dir_override:
                sae_dir_override = os.path.basename(
                    os.path.dirname(os.path.normpath(sae_ckpt)))
        elif weights_name.startswith('sae_baseline') and not sae_dir_override:
            sae_dir_override = weights_name
            weights_base_dir = weights_parent
        elif weights_name.startswith('vq_sae_predictor_') and not vq_predictor_dir_override:
            vq_predictor_dir_override = weights_name
            weights_base_dir = weights_parent
            vq_ckpt = predictor_cfg.get('vq_sae', {}).get('checkpoint', '')
            if vq_ckpt and not vq_sae_dir_override:
                vq_sae_dir_override = os.path.basename(
                    os.path.dirname(os.path.normpath(vq_ckpt)))
        elif weights_name.startswith('vq_sae_baseline') and not vq_sae_dir_override:
            vq_sae_dir_override = weights_name
            weights_base_dir = weights_parent
        elif cfg_family == 'sae' and not predictor_dir_override:
            predictor_dir_override = weights_name
            weights_base_dir = weights_parent
            sae_ckpt = predictor_cfg.get('sae', {}).get('checkpoint', '')
            if sae_ckpt and not sae_dir_override:
                sae_dir_override = os.path.basename(
                    os.path.dirname(os.path.normpath(sae_ckpt)))
        elif cfg_family == 'vq_sae' and not vq_predictor_dir_override:
            vq_predictor_dir_override = weights_name
            weights_base_dir = weights_parent
            vq_ckpt = predictor_cfg.get('vq_sae', {}).get('checkpoint', '')
            if vq_ckpt and not vq_sae_dir_override:
                vq_sae_dir_override = os.path.basename(
                    os.path.dirname(os.path.normpath(vq_ckpt)))

    family = infer_family(args, weights_name, predictor_cfg)

    if family == 'vq_sae':
        pipeline = VQSAEPipeline(
            device=args.device,
            weights_base_dir=weights_base_dir,
            config_path=args.vq_sae_config,
            vq_sae_dir_override=vq_sae_dir_override,
            predictor_dir_override=vq_predictor_dir_override)
    else:
        pipeline = SAEPipeline(
            device=args.device,
            weights_base_dir=weights_base_dir,
            config_path=args.sae_config,
            sae_dir_override=sae_dir_override,
            predictor_dir_override=predictor_dir_override)
    pipeline.load_model(level=1)

    if not pipeline.predictor_dir:
        raise RuntimeError('Failed to resolve predictor result directory.')

    predictor_config = load_result_config(pipeline.predictor_dir)
    data_cfg = predictor_config['data']
    training_cfg = predictor_config.get('training', {})

    h5_path = data_cfg['hdf5_path']
    ref_path = data_cfg['ref_path']
    fixed_level = training_cfg.get('fixed_level', None)
    level = 1 if fixed_level is None else int(fixed_level)

    ref_data = loadmat(ref_path)
    out_dir = create_result_subdir(pipeline.predictor_dir)

    print(f'Predictor dir: {pipeline.predictor_dir}')
    if family == 'vq_sae':
        print(f'VQ-SAE dir: {pipeline.vq_sae_dir}')
    else:
        print(f'SAE dir: {pipeline.sae_dir}')
    print(f'Output directory: {out_dir}')
    print(f'HDF5 path: {h5_path}')
    print(f'Level: {level}')

    num_samples = args.rows * args.cols
    split_to_indices = {
        'train': data_cfg.get('train_indices', None),
        'val': data_cfg.get('val_indices', None),
        'test': data_cfg.get('test_indices', None),
    }

    summary = {
        'family': family,
        'predictor_dir': pipeline.predictor_dir,
        'hdf5_path': h5_path,
        'level': level,
        'rows': args.rows,
        'cols': args.cols,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'splits': {},
    }
    if family == 'vq_sae':
        summary['vq_sae_dir'] = pipeline.vq_sae_dir
    else:
        summary['sae_dir'] = pipeline.sae_dir

    for split_idx, split in enumerate(args.splits):
        available = split_to_indices.get(split, None)
        chosen = sample_split_indices(
            available, num_samples, seed=args.seed + split_idx)
        if not chosen:
            print(f'Skip {split}: no indices found.')
            continue

        print(f'{split}: sampled {len(chosen)} examples')
        measurements, ground_truths = load_h5_samples(h5_path, chosen)
        predictions = batched_reconstruct(
            pipeline, measurements, ref_data, level, args.batch_size)

        save_path = os.path.join(out_dir, f'{split}_comparison.png')
        render_split_grid(
            ground_truths, predictions, chosen,
            rows=args.rows, cols=args.cols,
            save_path=save_path, show=args.show)

        summary['splits'][split] = {
            'indices': chosen,
            'plot': save_path,
            'num_samples': len(chosen),
        }
        print(f'  Saved: {save_path}')

    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
