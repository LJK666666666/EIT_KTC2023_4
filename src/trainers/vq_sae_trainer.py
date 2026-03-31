"""Trainer for ST-1D-VQ-VAE."""

import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..configs.vq_sae_config import get_configs as get_vq_sae_config
from ..data import VQGTHDF5Dataset
from ..losses.dice_focal import DiceLoss
from ..models.vq_sae import ST1DVQVAE


class VQSAETrainer(BaseTrainer):
    def __init__(self, config=None, experiment_name='vq_sae_baseline'):
        if config is None:
            config = get_vq_sae_config()
        super().__init__(config, experiment_name)
        self.dice_loss = DiceLoss()

    def build_model(self):
        model = ST1DVQVAE(
            in_channels=self.config.model.in_channels,
            encoder_channels=tuple(self.config.model.encoder_channels),
            num_slots=self.config.model.num_slots,
            codebook_size=self.config.model.codebook_size,
            code_dim=self.config.model.code_dim,
            decoder_start_size=self.config.model.decoder_start_size,
            vq_beta=self.config.training.vq_beta,
        )
        model.to(self.device)
        self.model = model

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.get('weight_decay', 1e-4),
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            min_lr=self.config.training.get('min_lr', 1e-5),
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f'ST1DVQVAE: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
        if not self.config.data.get('use_hdf5', False):
            raise ValueError('VQ SAE training requires HDF5 dataset.')

        h5_path = self.config.data.hdf5_path
        train_indices = self.config.data.get('train_indices', None)
        val_indices = self.config.data.get('val_indices', None)

        train_ds = VQGTHDF5Dataset(h5_path, indices=train_indices)
        self._warn_if_dropping_last_batch(
            'train', len(train_ds), self.config.training.batch_size)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=self._use_static_batches(),
            pin_memory=self._pin_memory_enabled(),
            num_workers=self.config.training.num_workers,
        )

        self.val_sim_loader = None
        if val_indices is not None:
            val_ds = VQGTHDF5Dataset(h5_path, indices=val_indices)
            self._warn_if_dropping_last_batch(
                'val', len(val_ds), self.config.training.batch_size)
            self.val_sim_loader = DataLoader(
                val_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers,
            )

        self._all_indices = train_indices
        if self._all_indices is not None and val_indices is not None:
            test_indices = self.config.data.get('test_indices', None)
            self._all_indices = list(train_indices)
            self._all_indices.extend(val_indices)
            if test_indices:
                self._all_indices.extend(test_indices)

    def _recon_loss(self, logits, gt_indices):
        ce_weight = float(self.config.training.get('ce_weight', 1.0))
        dice_weight = float(self.config.training.get('dice_weight', 1.0))
        ce = F.cross_entropy(logits, gt_indices)
        dice = self.dice_loss(logits, gt_indices)
        return ce_weight * ce + dice_weight * dice, ce, dice

    def train_step(self, batch):
        gt_onehot, gt_indices = batch
        gt_onehot = gt_onehot.to(self.device)
        gt_indices = gt_indices.to(self.device)

        self.optimizer.zero_grad()
        with self._autocast_context():
            logits, _, _, vq_loss = self.model(gt_onehot)
            recon_loss, ce_loss, dice_loss = self._recon_loss(
                logits, gt_indices)
            total_loss = recon_loss + vq_loss

        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)

        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
            'vq_loss': vq_loss.item(),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}

        totals = {
            'loss': 0.0,
            'ce_loss': 0.0,
            'dice_loss': 0.0,
            'vq_loss': 0.0,
        }
        num_samples = 0
        for gt_onehot, gt_indices in self.val_sim_loader:
            gt_onehot = gt_onehot.to(self.device)
            gt_indices = gt_indices.to(self.device)
            with torch.no_grad():
                with self._autocast_context():
                    logits, _, _, vq_loss = self.model(gt_onehot)
                    recon_loss, ce_loss, dice_loss = self._recon_loss(
                        logits, gt_indices)
                    total = recon_loss + vq_loss
            batch_size = gt_onehot.shape[0]
            totals['loss'] += total.item() * batch_size
            totals['ce_loss'] += ce_loss.item() * batch_size
            totals['dice_loss'] += dice_loss.item() * batch_size
            totals['vq_loss'] += vq_loss.item() * batch_size
            num_samples += batch_size

        metrics = {
            'val_loss': totals['loss'] / max(num_samples, 1),
            'val_ce_loss': totals['ce_loss'] / max(num_samples, 1),
            'val_dice_loss': totals['dice_loss'] / max(num_samples, 1),
            'val_vq_loss': totals['vq_loss'] / max(num_samples, 1),
        }
        if self.writer:
            for key, val in metrics.items():
                self.writer.add_scalar(f'val/{key[4:]}', val, epoch + 1)
        print(f'  Val loss: {metrics["val_loss"]:.5f}')
        return metrics

    def train(self):
        super().train()
        self._encode_and_save()

    def _encode_and_save(self):
        print('\n--- Phase 2: Encoding all GT images to discrete slots ---')
        best_path = os.path.join(self.result_dir, 'best.pt')
        if not os.path.exists(best_path):
            best_path = os.path.join(self.result_dir, 'last.pt')
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.device,
                               weights_only=False)
            self.model.load_state_dict(state['model_state_dict'])

        self.model.eval()

        all_ds = VQGTHDF5Dataset(self.config.data.hdf5_path,
                                 indices=self._all_indices)
        all_loader = DataLoader(
            all_ds,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
        )

        all_slot_indices = []
        all_angles = []
        sample_indices = all_ds.indices

        with torch.no_grad():
            for gt_onehot, _ in tqdm(all_loader, desc='Encoding'):
                gt_onehot = gt_onehot.to(self.device)
                with self._autocast_context():
                    slot_indices, angle_xy = self.model.encode_indices(gt_onehot)
                all_slot_indices.append(slot_indices.cpu().numpy())
                all_angles.append(angle_xy.cpu().numpy())

        all_slot_indices = np.concatenate(all_slot_indices, axis=0).astype(np.int64)
        all_angles = np.concatenate(all_angles, axis=0).astype(np.float32)
        sample_indices = np.asarray(sample_indices, dtype=np.int64)

        out_path = os.path.join(self.result_dir, 'latent_codes.h5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('indices', data=all_slot_indices, compression='lzf')
            f.create_dataset('angle_xy', data=all_angles, compression='lzf')
            f.create_dataset('sample_indices', data=sample_indices)
            f.attrs['num_slots'] = self.config.model.num_slots
            f.attrs['codebook_size'] = self.config.model.codebook_size
            f.attrs['code_dim'] = self.config.model.code_dim

        print(f'Saved discrete latent codes to {out_path}')
        print(f'  indices shape: {all_slot_indices.shape}, angle shape: {all_angles.shape}')

