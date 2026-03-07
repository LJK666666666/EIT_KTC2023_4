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
                        choices=['fcunet', 'postp', 'condd'],
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

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to last.pt checkpoint for resume')

    # Misc
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name for result directory')

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

    else:
        print(f'Unknown method: {args.method}')
        sys.exit(1)

    # Set seed
    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    print(f'Method: {args.method}')
    print(f'Device: {trainer.device}')
    print(f'Seed: {seed}')

    # ---- Train ----
    trainer.train()


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


if __name__ == '__main__':
    main()
