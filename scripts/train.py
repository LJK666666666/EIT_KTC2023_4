"""
Unified training entry point for KTC2023 EIT reconstruction models.

Usage:
    # Train FCUNet
    python scripts/train.py --method fcunet

    # Train PostP
    python scripts/train.py --method postp

    # Train CondD for level 3
    python scripts/train.py --method condd --level 3

    # Quick test (1 iteration)
    python scripts/train.py --method fcunet --max-iters 1

    # Resume training
    python scripts/train.py --method fcunet --resume results/fcunet_baseline_1/last.pt

    # Override hyperparameters
    python scripts/train.py --method postp --epochs 100 --lr 1e-4 --batch-size 4
"""

import argparse
import sys
import os
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train KTC2023 EIT reconstruction models')

    parser.add_argument('--method', type=str, required=True,
                        choices=['fcunet', 'postp', 'condd', 'dpcaunet',
                                 'hcdpcaunet'],
                        help='Training method')
    parser.add_argument('--level', type=int, default=1,
                        help='Difficulty level for CondD (1-7)')

    # Override hyperparameters
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max-iters', type=int, default=None,
                        help='Quick test: stop after N iterations')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader num_workers')
    parser.add_argument('--hdf5-path', type=str, default=None,
                        help='Path to HDF5 training data file '
                             '(enables HDF5 mode)')
    parser.add_argument('--precision', type=str, default=None,
                        choices=['fp32', 'bf16'],
                        help='Training precision')

    # Data split
    parser.add_argument('--split-ratio', type=str, default='8:1:1',
                        help='Train:Val:Test ratio (default: 8:1:1)')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to last.pt checkpoint for resume')

    # Misc
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/tpu)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name for result directory')
    parser.add_argument('--result-base-dir', type=str, default=None,
                        help='Base directory for auto-incremented result dirs')

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Build trainer ----
    if args.method == 'fcunet':
        from src.configs import get_fcunet_config
        from src.trainers import FCUNetTrainer

        config = get_fcunet_config()
        name = args.experiment_name or 'fcunet_baseline'
        _apply_overrides(config, args)
        trainer = FCUNetTrainer(config=config, experiment_name=name)

    elif args.method == 'postp':
        from src.configs import get_postp_config
        from src.trainers import PostPTrainer

        config = get_postp_config()
        name = args.experiment_name or 'postp_baseline'
        _apply_overrides(config, args)
        trainer = PostPTrainer(config=config, experiment_name=name)

    elif args.method == 'condd':
        from src.configs import get_condd_config
        from src.trainers import CondDTrainer

        config = get_condd_config()
        name = args.experiment_name or f'condd_level{args.level}'
        _apply_overrides(config, args)
        trainer = CondDTrainer(
            config=config, level=args.level, experiment_name=name)

    elif args.method == 'dpcaunet':
        from src.configs import get_dpcaunet_config
        from src.trainers import DPCAUNetTrainer

        config = get_dpcaunet_config()
        name = args.experiment_name or 'dpcaunet_baseline'
        _apply_overrides(config, args)
        trainer = DPCAUNetTrainer(config=config, experiment_name=name)

    elif args.method == 'hcdpcaunet':
        from src.configs import get_hcdpcaunet_config
        from src.trainers import HCDPCAUNetTrainer

        config = get_hcdpcaunet_config()
        name = args.experiment_name or 'hcdpcaunet_baseline'
        _apply_overrides(config, args)
        trainer = HCDPCAUNetTrainer(config=config, experiment_name=name)

    else:
        print(f'Unknown method: {args.method}')
        sys.exit(1)

    # Set seed
    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    # ---- Data split ----
    _apply_data_split(config, args, seed)

    print(f'Method: {args.method}')
    print(f'Device: {trainer.device}')
    print(f'Seed: {seed}')
    print(f'Precision: {config.training.precision}')
    if hasattr(config, 'result_base_dir'):
        print(f'Result base dir: {config.result_base_dir}')

    # ---- Train ----
    trainer.train()


def _apply_data_split(config, args, seed):
    """Apply train/val/test split based on --split-ratio and dataset size."""
    use_hdf5 = config.data.get('use_hdf5', False)
    if not use_hdf5:
        return  # npy mode uses directory listing, split not applicable

    # Already has explicit indices set (e.g. from resume)
    if config.data.get('train_indices', None) is not None:
        return

    # Parse ratio
    parts = [float(x) for x in args.split_ratio.split(':')]
    total_ratio = sum(parts)
    train_r = parts[0] / total_ratio
    val_r = parts[1] / total_ratio if len(parts) > 1 else 0.0

    # Read dataset size from HDF5
    import h5py
    if not os.path.exists(config.data.hdf5_path):
        print(f'Warning: HDF5 file not found: {config.data.hdf5_path}, '
              f'skipping data split')
        return
    with h5py.File(config.data.hdf5_path, 'r') as f:
        n_total = f['gt'].shape[0]

    # Shuffle indices with fixed seed (reproducible across resumes)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total).tolist()

    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    config.data.train_indices = train_idx
    config.data.val_indices = val_idx if val_idx else None
    config.data.test_indices = test_idx if test_idx else None

    print(f'Data split ({args.split_ratio}): '
          f'train={len(train_idx)}, val={len(val_idx)}, '
          f'test={len(test_idx)}, total={n_total}')


def _apply_overrides(config, args):
    """Apply CLI argument overrides to config."""
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.max_iters is not None:
        config.training.max_iters = args.max_iters
    if args.resume is not None:
        config.training.resume_from = args.resume
    if args.device is not None:
        config.device = args.device
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.hdf5_path is not None:
        config.data.use_hdf5 = True
        config.data.hdf5_path = args.hdf5_path
    if args.precision is not None:
        config.training.precision = args.precision
    if args.result_base_dir is not None:
        config.result_base_dir = args.result_base_dir


if __name__ == '__main__':
    main()
