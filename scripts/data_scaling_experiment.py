"""
Data scaling experiment for FCUNet on KTC2023 EIT level_1 data.

Studies how FCUNet reconstruction quality changes with training data size.
Uses the first 100 available samples for test, the next 100 for validation,
and all remaining samples as the training pool. Training subsets are nested.

Prerequisite:
    python scripts/generate_data.py --level 1 --num-images 1000 --measurements-only

Usage:
    # Quick test (verify pipeline)
    python scripts/data_scaling_experiment.py --mode train --max-iters 2

    # Full experiment
    python scripts/data_scaling_experiment.py --mode train --device cuda

    # Custom sizes / epochs
    python scripts/data_scaling_experiment.py --mode train --train-sizes 100 400 --epochs 200

    # Test only on existing checkpoints
    python scripts/data_scaling_experiment.py --mode test

    # Summarize and plot existing results
    python scripts/data_scaling_experiment.py --mode postprocess
"""

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from src.configs import get_fcunet_config
from src.trainers import FCUNetTrainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_summary_config(config_path):
    """Load summary/plot config from YAML if it exists."""
    if not os.path.exists(config_path):
        return {'result_subdirs': {}}

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f'Invalid YAML config format: {config_path}')

    result_subdirs = data.get('result_subdirs', {})
    if result_subdirs is None:
        result_subdirs = {}
    if not isinstance(result_subdirs, dict):
        raise ValueError(
            f'"result_subdirs" must be a mapping in {config_path}'
        )

    return {'result_subdirs': result_subdirs}


def find_latest_matching_dir(base_dir, experiment_name):
    """Find the latest result dir matching {experiment_name}_{num}."""
    pattern = os.path.join(base_dir, f'{experiment_name}_*')
    candidates = []

    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        suffix = os.path.basename(path).removeprefix(f'{experiment_name}_')
        if suffix.isdigit():
            candidates.append((int(suffix), path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def get_override_subdir(summary_config, n_train):
    """Get YAML override subdir for a given train size."""
    result_subdirs = summary_config.get('result_subdirs', {})
    value = result_subdirs.get(n_train, result_subdirs.get(str(n_train), ''))
    if value is None:
        return ''
    return str(value).strip()


def _extract_sample_index(filename):
    """Extract numeric sample index from gt_ztm_000042.npy -> 42."""
    stem = filename.rsplit('.', 1)[0]
    return int(stem.split('_')[-1])


def discover_available_indices(args):
    """Discover sorted sample indices from dataset files or HDF5."""
    config = get_fcunet_config()

    if args.hdf5_path is not None:
        import h5py
        with h5py.File(args.hdf5_path, 'r') as f:
            total_len = f['gt'].shape[0]
        return list(range(total_len))

    gt_dir = os.path.join(config.data.dataset_base_path, 'gt')
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f'Dataset gt directory not found: {gt_dir}')

    indices = []
    for filename in os.listdir(gt_dir):
        if filename.endswith('.npy'):
            indices.append(_extract_sample_index(filename))

    if not indices:
        raise FileNotFoundError(f'No .npy samples found under: {gt_dir}')

    return sorted(indices)


def build_data_split(args):
    """Build test/val/train splits from available sorted sample indices."""
    all_indices = discover_available_indices(args)

    if len(all_indices) < 201:
        raise ValueError(
            'Need at least 201 samples to create '
            '100 test + 100 val + remaining train split. '
            f'Found only {len(all_indices)} samples.'
        )

    test_indices = all_indices[:100]
    val_indices = all_indices[100:200]
    train_pool = all_indices[200:]

    return {
        'all_indices': all_indices,
        'test_indices': test_indices,
        'val_indices': val_indices,
        'train_pool': train_pool,
    }


def load_existing_result(n_train, args, summary_config):
    """Load an existing experiment result for summary/plot only mode."""
    result_dir = resolve_result_dir(n_train, args, summary_config)

    test_path = os.path.join(result_dir, 'test_results.json')
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f'Missing test_results.json for n_train={n_train}: {test_path}'
        )

    with open(test_path, 'r', encoding='utf-8') as f:
        test_results = json.load(f)

    return {
        'n_train': n_train,
        'result_dir': result_dir,
        **test_results,
    }


def resolve_result_dir(n_train, args, summary_config):
    """Resolve result directory using YAML override or latest auto match."""
    experiment_name = f'fcunet_scaling_n{n_train}'
    override_subdir = get_override_subdir(summary_config, n_train)

    if override_subdir:
        result_dir = os.path.join(args.result_dir, override_subdir)
    else:
        result_dir = find_latest_matching_dir(args.result_dir, experiment_name)

    if not result_dir or not os.path.exists(result_dir):
        raise FileNotFoundError(
            f'No result directory found for n_train={n_train}. '
            f'Looked under: {args.result_dir}'
        )

    return result_dir


def build_experiment_config(n_train, args, split):
    """Build FCUNet config for a given train size."""
    config = get_fcunet_config()

    config.training.fixed_level = 1
    if n_train > len(split['train_pool']):
        raise ValueError(
            f'n_train={n_train} exceeds available training pool size '
            f'{len(split["train_pool"])}.'
        )
    config.data.train_indices = split['train_pool'][:n_train]
    config.data.val_indices = split['val_indices']
    config.data.test_indices = split['test_indices']
    config.result_base_dir = args.result_dir

    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.init_epochs is not None:
        config.training.init_epochs = args.init_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.max_iters is not None:
        config.training.max_iters = args.max_iters
    if args.device is not None:
        config.device = args.device
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.hdf5_path is not None:
        config.data.use_hdf5 = True
        config.data.hdf5_path = args.hdf5_path

    return config


def run_single_experiment(n_train, args, split):
    """Run one training experiment with n_train samples.

    Returns dict with n_train, result_dir, test_loss, test_score.
    """
    config = build_experiment_config(n_train, args, split)
    experiment_name = f'fcunet_scaling_n{n_train}'
    trainer = FCUNetTrainer(config=config, experiment_name=experiment_name)

    set_seed(args.seed)
    trainer.train()

    # Evaluate on test set using best checkpoint
    test_results = trainer.evaluate_test()
    challenge_results = trainer.evaluate_challenge()

    result = {
        'n_train': n_train,
        'result_dir': trainer.result_dir,
        **test_results,
        **challenge_results,
    }

    test_path = os.path.join(trainer.result_dir, 'test_results.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in result.items() if k != 'result_dir'},
                  f, indent=2)

    # Free GPU memory
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_test_only(n_train, args, summary_config, split):
    """Evaluate existing checkpoint and write test_results.json."""
    config = build_experiment_config(n_train, args, split)
    experiment_name = f'fcunet_scaling_n{n_train}'
    result_dir = resolve_result_dir(n_train, args, summary_config)

    trainer = FCUNetTrainer(config=config, experiment_name=experiment_name)
    trainer.result_dir = result_dir

    set_seed(args.seed)
    trainer.build_model()
    trainer.build_datasets()

    test_results = trainer.evaluate_test()
    challenge_results = trainer.evaluate_challenge()

    result = {
        'n_train': n_train,
        'result_dir': result_dir,
        **test_results,
        **challenge_results,
    }

    test_path = os.path.join(result_dir, 'test_results.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in result.items() if k != 'result_dir'},
                  f, indent=2)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def plot_scaling_results(all_results, save_dir='results'):
    """Generate the data scaling comparison plot."""
    sizes = [r['n_train'] for r in all_results]
    scores = [r.get('test_score', 0) for r in all_results]
    losses = [r.get('test_loss', 0) for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Test Score vs Data Size
    ax1.plot(sizes, scores, 'o-', linewidth=2, markersize=8, color='#2196F3')
    for x, y in zip(sizes, scores):
        ax1.annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=9)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_xlabel('Training Data Size', fontsize=11)
    ax1.set_ylabel('Test Score (SSIM)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: Test Loss vs Data Size
    ax2.plot(sizes, losses, 'o-', linewidth=2, markersize=8, color='#F44336')
    for x, y in zip(sizes, losses):
        ax2.annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=9)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel('Training Data Size', fontsize=11)
    ax2.set_ylabel('Test CE Loss', fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, 'data_scaling_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Scaling comparison plot saved to: {path}')


def plot_training_curves(all_results, save_dir='results'):
    """Generate training/validation loss comparison plot from training_log.json."""
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, result in enumerate(all_results):
        log_path = os.path.join(result['result_dir'], 'training_log.json')
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            log = json.load(f)

        label = f'n={result["n_train"]}'
        color = colors[i % len(colors)]

        # Training loss (stage 2 only = entries without 'stage' key or stage != 'init')
        epochs_loss = []
        losses = []
        for entry in log:
            if entry.get('stage') == 'init':
                continue
            if 'avg_loss' in entry:
                epochs_loss.append(entry['epoch'])
                losses.append(entry['avg_loss'])

        if epochs_loss:
            ax1.plot(epochs_loss, losses, linewidth=1.5, color=color,
                     label=label, alpha=0.8)

        # Validation loss
        epochs_val_loss = []
        val_losses = []
        for entry in log:
            if 'val_loss' in entry:
                epochs_val_loss.append(entry['epoch'])
                val_losses.append(entry['val_loss'])

        if epochs_val_loss:
            ax2.plot(epochs_val_loss, val_losses, linewidth=1.5, color=color,
                     label=label, alpha=0.8)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, 'data_scaling_training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Training curves plot saved to: {path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='FCUNet data scaling experiment')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'postprocess'],
                        help='train: run full experiment; '
                             'test: evaluate existing checkpoints only; '
                             'postprocess: summarize and plot existing results')
    parser.add_argument('--train-sizes', nargs='+', type=int,
                        default=[100, 200, 400, 800],
                        help='Training set sizes (default: 100 200 400 800)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override total training epochs (config default: 500)')
    parser.add_argument('--init-epochs', type=int, default=None,
                        help='Override init pre-training epochs (config default: 15)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (config default: 6)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (config default: 3e-5)')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='Quick test: stop after N iterations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader num_workers (config default: 8)')
    parser.add_argument('--hdf5-path', type=str, default=None,
                        help='Path to HDF5 training data file '
                             '(enables HDF5 mode)')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Base directory for saving results '
                             '(default: results)')
    parser.add_argument('--summary-config', type=str,
                        default='scripts/data_scaling_experiment.yaml',
                        help='YAML config for overriding result subdir names '
                             'in test/postprocess mode')
    return parser.parse_args()


def main():
    args = parse_args()
    summary_config = load_summary_config(args.summary_config)
    split = build_data_split(args)

    print('='*60)
    print('FCUNet Data Scaling Experiment')
    print(f'Train sizes: {args.train_sizes}')
    print(f'Total available samples: {len(split["all_indices"])}')
    print(f'Test: {len(split["test_indices"])} samples '
          f'({split["test_indices"][0]}-{split["test_indices"][-1]})')
    print(f'Val:  {len(split["val_indices"])} samples '
          f'({split["val_indices"][0]}-{split["val_indices"][-1]})')
    print(f'Train pool: {len(split["train_pool"])} samples '
          f'({split["train_pool"][0]}-{split["train_pool"][-1]})')
    print(f'Seed: {args.seed}')
    print(f'Mode: {args.mode}')
    print(f'Summary config: {args.summary_config}')
    print('='*60)

    all_results = []

    for n_train in args.train_sizes:
        print(f'\n{"="*60}')
        print(f'Experiment: n_train = {n_train}')
        if args.mode == 'postprocess':
            override_subdir = get_override_subdir(summary_config, n_train)
            if override_subdir:
                print(f'  Using YAML override subdir: {override_subdir}')
            else:
                print('  Using latest matching result directory')
            result = load_existing_result(n_train, args, summary_config)
        elif args.mode == 'test':
            override_subdir = get_override_subdir(summary_config, n_train)
            if override_subdir:
                print(f'  Testing YAML override subdir: {override_subdir}')
            else:
                print('  Testing latest matching result directory')
            result = run_test_only(n_train, args, summary_config, split)
        else:
            print(f'  Train indices: '
                  f'{split["train_pool"][0]}-{split["train_pool"][n_train - 1]}')
            result = run_single_experiment(n_train, args, split)
        all_results.append(result)

        print(f'\n  n={n_train}: test_loss={result.get("test_loss", "N/A")}, '
              f'test_score={result.get("test_score", "N/A")}')
        print(f'  result_dir={result["result_dir"]}')

    # Save summary
    os.makedirs(args.result_dir, exist_ok=True)
    summary_path = os.path.join(args.result_dir, 'data_scaling_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSummary saved to: {summary_path}')

    # Print results table
    print(f'\n{"="*50}')
    print(f'{"n_train":>8} {"Test Score":>12} {"Test Loss":>12}')
    print(f'{"-"*50}')
    for r in all_results:
        score = r.get('test_score', 0)
        loss = r.get('test_loss', 0)
        print(f'{r["n_train"]:>8} {score:>12.4f} {loss:>12.4f}')
    print(f'{"="*50}')

    # Generate plots
    if len(all_results) > 1:
        plot_scaling_results(all_results, save_dir=args.result_dir)
        plot_training_curves(all_results, save_dir=args.result_dir)
    elif len(all_results) == 1:
        print('Only 1 experiment completed, skipping comparison plots.')

    print('\nAll experiments complete.')


if __name__ == '__main__':
    main()
