"""Trainer for VAE-style pulmonary TD16 delta-conductivity manifold learning."""

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..configs.td16_vae_config import get_configs as get_td16_vae_config
from ..data import DifferenceConductivityHDF5Dataset
from ..evaluation.regression_metrics import masked_regression_metrics_batch
from ..models.pulmonary_vae import ConvVAE


class TD16VAETrainer(BaseTrainer):
    def __init__(self, config=None, experiment_name='td16_vae_baseline'):
        if config is None:
            config = get_td16_vae_config()
        super().__init__(config, experiment_name)

    def build_model(self):
        model = ConvVAE(
            in_channels=self.config.model.in_channels,
            latent_dim=self.config.model.latent_dim,
            base_channels=self.config.model.base_channels,
        )
        model.to(self.device)
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.training.selection_metric_mode,
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            min_lr=self.config.training.get('min_lr', 1e-6),
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'TD16VAE: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
        h5_path = self.config.data.hdf5_path
        train_indices = self.config.data.get('train_indices', None)
        val_indices = self.config.data.get('val_indices', None)

        train_ds = DifferenceConductivityHDF5Dataset(h5_path, indices=train_indices)
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
            val_ds = DifferenceConductivityHDF5Dataset(h5_path, indices=val_indices)
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

    def _loss_terms(self, recon, target, domain_mask, mu, logvar):
        mask = domain_mask.float()
        diff = recon - target
        denom = mask.sum().clamp_min(1.0)
        mse = (diff.pow(2) * mask).sum() / denom
        mae = (diff.abs() * mask).sum() / denom
        kl = self.model.kl_loss(mu.float(), logvar.float())
        total = (
            float(self.config.training.get('mse_weight', 1.0)) * mse +
            float(self.config.training.get('mae_weight', 0.2)) * mae +
            float(self.config.training.get('kl_weight', 1e-4)) * kl
        )
        return total, mse, mae, kl

    def train_step(self, batch):
        _, sigma_delta, domain_mask, *rest = batch
        sigma_delta = sigma_delta.to(self.device)
        domain_mask = domain_mask.to(self.device)

        self.optimizer.zero_grad()
        with self._autocast_context():
            recon, mu, logvar = self.model(sigma_delta)
            total_loss, mse_loss, mae_loss, kl_loss = self._loss_terms(
                recon, sigma_delta, domain_mask, mu, logvar)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.grad_clip_norm)
        self.optimizer_step(self.optimizer)
        return {
            'loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'kl_loss': kl_loss.item(),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}
        totals = {'loss': 0.0, 'mse_loss': 0.0, 'mae_loss': 0.0, 'kl_loss': 0.0}
        preds_all, targets_all, masks_all = [], [], []
        num_samples = 0
        for _, sigma_delta, domain_mask, *rest in self.val_sim_loader:
            sigma_delta = sigma_delta.to(self.device)
            domain_mask = domain_mask.to(self.device)
            with torch.no_grad():
                with self._autocast_context():
                    recon, mu, logvar = self.model(sigma_delta)
                    total_loss, mse_loss, mae_loss, kl_loss = self._loss_terms(
                        recon, sigma_delta, domain_mask, mu, logvar)
            batch_size = sigma_delta.shape[0]
            totals['loss'] += total_loss.item() * batch_size
            totals['mse_loss'] += mse_loss.item() * batch_size
            totals['mae_loss'] += mae_loss.item() * batch_size
            totals['kl_loss'] += kl_loss.item() * batch_size
            num_samples += batch_size
            preds_all.append(recon.detach().float().cpu().numpy()[:, 0])
            targets_all.append(sigma_delta.detach().float().cpu().numpy()[:, 0])
            masks_all.append(domain_mask.detach().float().cpu().numpy()[:, 0] > 0.5)
        reg = masked_regression_metrics_batch(
            np.concatenate(targets_all, axis=0),
            np.concatenate(preds_all, axis=0),
            masks=np.concatenate(masks_all, axis=0),
            active_threshold=self.config.training.active_threshold,
        )
        metrics = {
            'val_loss': totals['loss'] / max(num_samples, 1),
            'val_mse_loss': totals['mse_loss'] / max(num_samples, 1),
            'val_mae_loss': totals['mae_loss'] / max(num_samples, 1),
            'val_kl_loss': totals['kl_loss'] / max(num_samples, 1),
            'val_rmse': float(np.mean(reg['rmse'])),
            'val_rel_l2': float(np.mean(reg['rel_l2'])),
        }
        if 'active_rel_l2' in reg:
            metrics['val_active_rel_l2'] = float(np.nanmean(reg['active_rel_l2']))
        if self.writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f'val/{key[4:]}', val, epoch + 1)
        print(
            f'  Val loss: {metrics["val_loss"]:.6f} '
            f'RMSE: {metrics["val_rmse"]:.6f} '
            f'RelL2: {metrics["val_rel_l2"]:.6f}'
        )
        return metrics
