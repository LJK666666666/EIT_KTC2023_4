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
                                  'hcdpcaunet', 'sae', 'sae_predictor',
                                  'vq_sae', 'vq_sae_predictor',
                                   'td16_vae', 'td16_vae_predictor',
                                   'td16_vae_conditional_predictor',
                                   'dct_predictor', 'dct_sigma_predictor',
                                    'dct_sigma_td16_predictor',
                                    'dct_sigma_td16_change_predictor',
                                    'dct_sigma_td16_spatial_change_predictor',
                                    'dct_sigma_td16_conditional_predictor',
                                    'dct_sigma_td16_mask_predictor',
                                    'dct_sigma_residual_predictor',
                                  'dct_sigma_hybrid_predictor',
                                  'atlas_sigma_predictor',
                                  'fc_sigmaunet'],
                          help='Training method')
    parser.add_argument('--level', type=int, default=1,
                        help='Difficulty level for CondD (1-7)')

    # Override hyperparameters
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--init-epochs', type=int, default=None)
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

    # SAE predictor specific
    parser.add_argument('--sae-checkpoint', type=str, default=None,
                        help='Path to trained SAE best.pt (for sae_predictor)')
    parser.add_argument('--vae-checkpoint', type=str, default=None,
                        help='Path to trained TD16 VAE best.pt (for td16_vae_predictor / td16_vae_conditional_predictor)')
    parser.add_argument('--latent-h5-path', type=str, default=None,
                        help='Path to latent_codes.h5 (for sae_predictor)')
    parser.add_argument('--vq-sae-checkpoint', type=str, default=None,
                        help='Path to trained VQ SAE best.pt '
                             '(for vq_sae_predictor)')
    parser.add_argument('--vq-latent-h5-path', type=str, default=None,
                        help='Path to VQ latent_codes.h5 '
                             '(for vq_sae_predictor)')
    parser.add_argument('--num-slots', type=int, default=None,
                        help='Override VQ num_slots')
    parser.add_argument('--codebook-size', type=int, default=None,
                        help='Override VQ codebook size')
    parser.add_argument('--code-dim', type=int, default=None,
                        help='Override VQ code dimension')
    parser.add_argument('--vq-beta', type=float, default=None,
                        help='Override VQ commitment beta')
    parser.add_argument('--lambda-slot', type=float, default=None,
                        help='Override predictor slot loss weight')
    parser.add_argument('--lambda-image', type=float, default=None,
                        help='Override predictor image loss weight')
    parser.add_argument('--lambda-angle', type=float, default=None,
                        help='Override predictor angle loss weight')
    parser.add_argument('--coeff-size', type=int, default=None,
                        help='Override DCT coefficient size')
    parser.add_argument('--coeff-loss-weight', type=float, default=None,
                         help='Override DCT coefficient regression loss weight')
    parser.add_argument('--focus-loss-weight', type=float, default=None,
                        help='Override atlas-deviation focus loss weight')
    parser.add_argument('--focus-threshold', type=float, default=None,
                        help='Override atlas-deviation focus threshold')
    parser.add_argument('--inactive-weight', type=float, default=None,
                        help='Override inactive-region loss weight')
    parser.add_argument('--pred-l1-weight', type=float, default=None,
                        help='Override prediction sparsity L1 loss weight')
    parser.add_argument('--active-region-threshold', type=float, default=None,
                        help='Override threshold for defining active delta region')
    parser.add_argument('--change-threshold', type=float, default=None,
                        help='Override threshold for defining sample-level change')
    parser.add_argument('--lambda-gate', type=float, default=None,
                        help='Override sample-level gate BCE loss weight')
    parser.add_argument('--lambda-mask', type=float, default=None,
                        help='Override spatial change mask BCE loss weight')
    parser.add_argument('--lambda-mask-coeff', type=float, default=None,
                        help='Override low-frequency change-mask coeff loss weight')
    parser.add_argument('--mask-threshold', type=float, default=None,
                        help='Override threshold for spatial active-mask target')
    parser.add_argument('--mask-teacher-forcing-ratio', type=float, default=None,
                        help='Teacher-forcing ratio for conditional TD16 mask branch')
    parser.add_argument('--active-oversample-factor', type=float, default=None,
                        help='Oversample active-change samples in TD16 mixed training')
    parser.add_argument('--ce-weight', type=float, default=None,
                        help='Override cross-entropy loss weight')
    parser.add_argument('--dice-weight', type=float, default=None,
                        help='Override Dice loss weight')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override model dropout')
    parser.add_argument('--fixed-level', type=int, default=None,
                        help='Train/validate on a fixed level only')
    parser.add_argument('--score-probe-freq', type=int, default=None,
                        help='Run DCT fast probe every N epochs')
    parser.add_argument('--score-probe-max-samples', type=int, default=None,
                        help='Max validation samples used for fast probe score')

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

    elif args.method == 'sae':
        from src.configs import get_sae_config
        from src.trainers import SAETrainer

        config = get_sae_config()
        name = args.experiment_name or 'sae_baseline'
        _apply_overrides(config, args)
        trainer = SAETrainer(config=config, experiment_name=name)

    elif args.method == 'sae_predictor':
        from src.configs import get_sae_predictor_config
        from src.trainers import SAEPredictorTrainer

        config = get_sae_predictor_config()
        name = args.experiment_name or 'sae_predictor_baseline'
        _apply_overrides(config, args)
        # Accept extra args for SAE checkpoint and latent codes
        if args.sae_checkpoint:
            config.sae.checkpoint = args.sae_checkpoint
        if args.latent_h5_path:
            config.sae.latent_h5_path = args.latent_h5_path
        trainer = SAEPredictorTrainer(config=config, experiment_name=name)

    elif args.method == 'vq_sae':
        from src.configs import get_vq_sae_config
        from src.trainers import VQSAETrainer

        config = get_vq_sae_config()
        name = args.experiment_name or 'vq_sae_baseline'
        _apply_overrides(config, args)
        trainer = VQSAETrainer(config=config, experiment_name=name)

    elif args.method == 'vq_sae_predictor':
        from src.configs import get_vq_sae_predictor_config
        from src.trainers import VQSAEPredictorTrainer

        config = get_vq_sae_predictor_config()
        name = args.experiment_name or 'vq_sae_predictor_baseline'
        _apply_overrides(config, args)
        if args.vq_sae_checkpoint:
            config.vq_sae.checkpoint = args.vq_sae_checkpoint
        if args.vq_latent_h5_path:
            config.vq_sae.latent_h5_path = args.vq_latent_h5_path
        elif args.latent_h5_path:
            config.vq_sae.latent_h5_path = args.latent_h5_path
        trainer = VQSAEPredictorTrainer(config=config, experiment_name=name)

    elif args.method == 'td16_vae':
        from src.configs import get_td16_vae_config
        from src.trainers import TD16VAETrainer

        config = get_td16_vae_config()
        name = args.experiment_name or 'td16_vae_baseline'
        _apply_overrides(config, args)
        trainer = TD16VAETrainer(config=config, experiment_name=name)

    elif args.method == 'td16_vae_predictor':
        from src.configs import get_td16_vae_predictor_config
        from src.trainers import TD16VAEPredictorTrainer

        config = get_td16_vae_predictor_config()
        name = args.experiment_name or 'td16_vae_predictor_baseline'
        _apply_overrides(config, args)
        if args.vae_checkpoint:
            config.vae.checkpoint = args.vae_checkpoint
        trainer = TD16VAEPredictorTrainer(config=config, experiment_name=name)

    elif args.method == 'td16_vae_conditional_predictor':
        from src.configs import get_td16_vae_conditional_predictor_config
        from src.trainers import TD16VAEConditionalPredictorTrainer

        config = get_td16_vae_conditional_predictor_config()
        name = args.experiment_name or 'td16_vae_conditional_predictor_baseline'
        _apply_overrides(config, args)
        if args.vae_checkpoint:
            config.vae.checkpoint = args.vae_checkpoint
        trainer = TD16VAEConditionalPredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'dct_predictor':
        from src.configs import get_dct_predictor_config
        from src.trainers import DCTPredictorTrainer

        config = get_dct_predictor_config()
        name = args.experiment_name or 'dct_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTPredictorTrainer(config=config, experiment_name=name)

    elif args.method == 'dct_sigma_predictor':
        from src.configs import get_dct_sigma_predictor_config
        from src.trainers import DCTSigmaPredictorTrainer

        config = get_dct_sigma_predictor_config()
        name = args.experiment_name or 'dct_sigma_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTSigmaPredictorTrainer(config=config, experiment_name=name)

    elif args.method == 'dct_sigma_td16_predictor':
        from src.configs import get_dct_sigma_td16_predictor_config
        from src.trainers import DCTSigmaTD16PredictorTrainer

        config = get_dct_sigma_td16_predictor_config()
        name = args.experiment_name or 'dct_sigma_td16_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTSigmaTD16PredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'dct_sigma_td16_change_predictor':
        from src.configs import get_dct_sigma_td16_change_predictor_config
        from src.trainers import DCTSigmaTD16ChangePredictorTrainer

        config = get_dct_sigma_td16_change_predictor_config()
        name = args.experiment_name or 'dct_sigma_td16_change_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTSigmaTD16ChangePredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'dct_sigma_td16_spatial_change_predictor':
        from src.configs import get_dct_sigma_td16_spatial_change_predictor_config
        from src.trainers import DCTSigmaTD16SpatialChangePredictorTrainer

        config = get_dct_sigma_td16_spatial_change_predictor_config()
        name = args.experiment_name or 'dct_sigma_td16_spatial_change_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTSigmaTD16SpatialChangePredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'dct_sigma_td16_conditional_predictor':
        from src.configs import get_dct_sigma_td16_conditional_predictor_config
        from src.trainers import DCTSigmaTD16ConditionalPredictorTrainer

        config = get_dct_sigma_td16_conditional_predictor_config()
        name = args.experiment_name or 'dct_sigma_td16_conditional_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTSigmaTD16ConditionalPredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'dct_sigma_td16_mask_predictor':
        from src.configs import get_dct_sigma_td16_mask_predictor_config
        from src.trainers import DCTSigmaTD16MaskPredictorTrainer

        config = get_dct_sigma_td16_mask_predictor_config()
        name = args.experiment_name or 'dct_sigma_td16_mask_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTSigmaTD16MaskPredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'dct_sigma_residual_predictor':
        from src.configs import get_dct_sigma_residual_predictor_config
        from src.trainers import DCTSigmaResidualPredictorTrainer

        config = get_dct_sigma_residual_predictor_config()
        name = args.experiment_name or 'dct_sigma_residual_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTSigmaResidualPredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'dct_sigma_hybrid_predictor':
        from src.configs import get_dct_sigma_hybrid_predictor_config
        from src.trainers import DCTSigmaHybridPredictorTrainer

        config = get_dct_sigma_hybrid_predictor_config()
        name = args.experiment_name or 'dct_sigma_hybrid_predictor_baseline'
        _apply_overrides(config, args)
        trainer = DCTSigmaHybridPredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'atlas_sigma_predictor':
        from src.configs import get_atlas_sigma_predictor_config
        from src.trainers import AtlasSigmaPredictorTrainer

        config = get_atlas_sigma_predictor_config()
        name = args.experiment_name or 'atlas_sigma_predictor_baseline'
        _apply_overrides(config, args)
        trainer = AtlasSigmaPredictorTrainer(
            config=config, experiment_name=name)

    elif args.method == 'fc_sigmaunet':
        from src.configs import get_fc_sigmaunet_config
        from src.trainers import FCSigmaUNetTrainer

        config = get_fc_sigmaunet_config()
        name = args.experiment_name or 'fc_sigmaunet_baseline'
        _apply_overrides(config, args)
        trainer = FCSigmaUNetTrainer(config=config, experiment_name=name)

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
    if args.init_epochs is not None and 'init_epochs' in config.training:
        config.training.init_epochs = args.init_epochs
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
    if args.num_slots is not None and hasattr(config, 'model'):
        config.model.num_slots = args.num_slots
    if args.codebook_size is not None and hasattr(config, 'model'):
        config.model.codebook_size = args.codebook_size
    if args.code_dim is not None and hasattr(config, 'model'):
        config.model.code_dim = args.code_dim
    if args.vq_beta is not None:
        config.training.vq_beta = args.vq_beta
    if args.lambda_slot is not None:
        config.training.lambda_slot = args.lambda_slot
    if args.lambda_image is not None:
        config.training.lambda_image = args.lambda_image
    if args.lambda_angle is not None:
        config.training.lambda_angle = args.lambda_angle
    if args.coeff_size is not None and hasattr(config, 'model'):
        config.model.coeff_size = args.coeff_size
    if args.coeff_loss_weight is not None:
        config.training.coeff_loss_weight = args.coeff_loss_weight
    if args.focus_loss_weight is not None:
        config.training.focus_loss_weight = args.focus_loss_weight
    if args.focus_threshold is not None:
        config.training.focus_threshold = args.focus_threshold
    if args.inactive_weight is not None:
        config.training.inactive_weight = args.inactive_weight
    if args.pred_l1_weight is not None:
        config.training.pred_l1_weight = args.pred_l1_weight
    if args.active_region_threshold is not None:
        config.training.active_region_threshold = args.active_region_threshold
    if args.change_threshold is not None:
        config.training.change_threshold = args.change_threshold
    if args.lambda_gate is not None:
        config.training.lambda_gate = args.lambda_gate
    if args.lambda_mask is not None:
        config.training.lambda_mask = args.lambda_mask
    if args.lambda_mask_coeff is not None:
        config.training.lambda_mask_coeff = args.lambda_mask_coeff
    if args.mask_threshold is not None:
        config.training.mask_threshold = args.mask_threshold
    if args.mask_teacher_forcing_ratio is not None:
        config.training.mask_teacher_forcing_ratio = args.mask_teacher_forcing_ratio
    if args.active_oversample_factor is not None:
        config.training.active_oversample_factor = args.active_oversample_factor
    if args.ce_weight is not None:
        config.training.ce_weight = args.ce_weight
    if args.dice_weight is not None:
        config.training.dice_weight = args.dice_weight
    if args.dropout is not None and hasattr(config, 'model'):
        config.model.dropout = args.dropout
    if args.fixed_level is not None:
        config.training.fixed_level = args.fixed_level
    if args.score_probe_freq is not None:
        config.training.score_probe_freq = args.score_probe_freq
    if args.score_probe_max_samples is not None:
        config.training.score_probe_max_samples = args.score_probe_max_samples


if __name__ == '__main__':
    main()
