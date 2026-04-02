"""
SAE trainer: Phase 1 (autoencoder training) + Phase 2 (encode & save).

Loss = CrossEntropy + L1(z_shape) + Equivariance(z_shape under rotation).
Phase 2 runs automatically at training end: encode all GT → save latent_codes.h5.
"""

import math
import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..configs.sae_config import get_configs as get_sae_config
from ..models.sae import SparseAutoEncoder
from ..data import GTHDF5Dataset


class SAETrainer(BaseTrainer):
    """Phase 1+2 trainer for ST-SAE."""

    def __init__(self, config=None, experiment_name='sae_baseline'):
        if config is None:
            config = get_sae_config()
        super().__init__(config, experiment_name)

    def build_model(self):
        model = SparseAutoEncoder(
            in_channels=self.config.model.in_channels,
            encoder_channels=tuple(self.config.model.encoder_channels),
            shape_dim=self.config.model.shape_dim,
            decoder_start_size=self.config.model.decoder_start_size,
        )
        model.to(self.device)
        self.model = model

        wd = self.config.training.get('weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.training.lr,
            weight_decay=wd)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            min_lr=self.config.training.get('min_lr', 1e-5))

        total_params = sum(p.numel() for p in model.parameters())
        print(f'SparseAutoEncoder: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
        use_hdf5 = self.config.data.get('use_hdf5', False)
        if not use_hdf5:
            raise ValueError('SAE training requires HDF5 dataset. '
                             'Use --hdf5-path to specify.')

        h5_path = self.config.data.hdf5_path
        train_indices = self.config.data.get('train_indices', None)
        val_indices = self.config.data.get('val_indices', None)

        dataset = GTHDF5Dataset(h5_path, indices=train_indices)
        self._warn_if_dropping_last_batch(
            'train', len(dataset), self.config.training.batch_size)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=self._use_static_batches(),
            pin_memory=self._pin_memory_enabled(),
            num_workers=self.config.training.num_workers)

        # Validation loader
        self.val_sim_loader = None
        if val_indices is not None:
            val_ds = GTHDF5Dataset(h5_path, indices=val_indices)
            self._warn_if_dropping_last_batch(
                'val', len(val_ds), self.config.training.batch_size)
            self.val_sim_loader = DataLoader(
                val_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers)

        # Store all indices for Phase 2 encoding
        self._all_indices = train_indices
        if self._all_indices is not None and val_indices is not None:
            test_indices = self.config.data.get('test_indices', None)
            self._all_indices = list(train_indices)
            if val_indices:
                self._all_indices.extend(val_indices)
            if test_indices:
                self._all_indices.extend(test_indices)

        self.val_data = None  # Not used for SAE

    def _rotate_image(self, image, k):
        """Rotate one-hot image by k electrode steps (k × 2π/32).

        Uses bilinear for training (gradient flow).
        """
        angle = k * (2 * math.pi / 32)
        B = image.shape[0]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
        ], dtype=image.dtype, device=image.device).unsqueeze(0).expand(B, -1, -1)

        grid = F.affine_grid(theta, image.shape, align_corners=False)
        mode = 'bilinear' if self.training else 'nearest'
        return F.grid_sample(image, grid, mode=mode,
                             padding_mode='zeros', align_corners=False)

    def train_step(self, batch):
        gt_onehot, gt_indices = batch
        gt_onehot = gt_onehot.to(self.device)
        gt_indices = gt_indices.to(self.device)

        self.optimizer.zero_grad()

        with self._autocast_context():
            logits, z_shape, angle_xy = self.model(gt_onehot)

            # CrossEntropy reconstruction loss
            recon_loss = F.cross_entropy(logits, gt_indices)

            # L1 sparsity on z_shape only
            l1_lambda = self.config.training.l1_lambda
            sparsity_loss = l1_lambda * torch.mean(torch.abs(z_shape))

            # Equivariance loss: ensure z_shape is rotation-invariant
            equiv_lambda = self.config.training.equiv_lambda
            if equiv_lambda > 0:
                k = random.randint(1, 31)
                gt_rotated = self._rotate_image(gt_onehot, k)
                z_shape_rot, _ = self.model.encode(gt_rotated)
                equiv_loss = equiv_lambda * F.mse_loss(z_shape_rot, z_shape)
            else:
                equiv_loss = torch.tensor(0.0, device=self.device)

            total_loss = recon_loss + sparsity_loss + equiv_loss

        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)

        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'equiv_loss': equiv_loss.item(),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}

        total_loss = 0.0
        num_samples = 0

        for gt_onehot, gt_indices in self.val_sim_loader:
            gt_onehot = gt_onehot.to(self.device)
            gt_indices = gt_indices.to(self.device)

            with torch.no_grad():
                with self._autocast_context():
                    logits, z_shape, _ = self.model(gt_onehot)
                    loss = F.cross_entropy(logits, gt_indices)
            total_loss += loss.item() * gt_onehot.shape[0]
            num_samples += gt_onehot.shape[0]

        avg_loss = total_loss / max(num_samples, 1)
        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, epoch + 1)
        print(f'  Val loss: {avg_loss:.5f}')
        return {'val_loss': avg_loss}

    def train(self):
        """Phase 1 training + Phase 2 encoding."""
        # Run standard training (Phase 1)
        super().train()

        # Phase 2: encode all GT and save latent codes
        self._encode_and_save()

    def _encode_and_save(self):
        """Phase 2: encode all GT images, save latent_codes.h5."""
        print('\n--- Phase 2: Encoding all GT images ---')

        # Load best checkpoint
        best_path = os.path.join(self.result_dir, 'best.pt')
        if not os.path.exists(best_path):
            best_path = os.path.join(self.result_dir, 'last.pt')
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.device,
                               weights_only=False)
            self.model.load_state_dict(state['model_state_dict'])
            print(f'Loaded weights from {best_path}')

        self.model.eval()

        # Create dataset with ALL indices
        h5_path = self.config.data.hdf5_path
        all_ds = GTHDF5Dataset(h5_path, indices=self._all_indices)
        all_loader = DataLoader(
            all_ds,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers)

        all_codes = []
        all_indices_list = all_ds.indices

        with torch.no_grad():
            for gt_onehot, _ in tqdm(all_loader, desc='Encoding'):
                gt_onehot = gt_onehot.to(self.device)
                with self._autocast_context():
                    z_shape, angle_xy = self.model.encode(gt_onehot)
                # Concatenate: [z_shape(63), cosθ(1), sinθ(1)] = 65
                codes = torch.cat([z_shape, angle_xy], dim=1)
                all_codes.append(codes.cpu().numpy())

        all_codes = np.concatenate(all_codes, axis=0)  # (N, 65)
        all_indices_arr = np.array(all_indices_list, dtype=np.int64)

        # Compute sparsity threshold (5th percentile of |z_shape|)
        z_shapes = all_codes[:, :self.config.model.shape_dim]
        threshold = float(np.percentile(np.abs(z_shapes), 5))

        # Save
        out_path = os.path.join(self.result_dir, 'latent_codes.h5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('codes', data=all_codes, compression='lzf')
            f.create_dataset('indices', data=all_indices_arr)
            f.attrs['sparsity_threshold'] = threshold
            f.attrs['shape_dim'] = self.config.model.shape_dim

        print(f'Saved {all_codes.shape[0]} latent codes to {out_path}')
        print(f'  Code shape: {all_codes.shape}, threshold: {threshold:.6f}')
        print(f'  z_shape sparsity: {(np.abs(z_shapes) < threshold).mean():.1%} '
              f'of values below threshold')
