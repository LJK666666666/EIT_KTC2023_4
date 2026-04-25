"""Trainer for change-aware 16-electrode pulmonary time-difference regression."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..configs.dct_sigma_td16_change_predictor_config import (
    get_configs as get_dct_sigma_td16_change_predictor_config,
)
from ..data import DifferenceConductivityHDF5Dataset
from ..evaluation.regression_metrics import masked_regression_metrics_batch
from ..models.dct_predictor import ChangeGatedDCTPredictor


class DCTSigmaTD16ChangePredictorTrainer(BaseTrainer):
    def __init__(self, config=None,
                 experiment_name='dct_sigma_td16_change_predictor_baseline'):
        if config is None:
            config = get_dct_sigma_td16_change_predictor_config()
        super().__init__(config, experiment_name)
        self.test_sim_loader = None

    def build_model(self):
        model = ChangeGatedDCTPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            coeff_size=self.config.model.coeff_size,
            out_channels=self.config.model.out_channels,
            dropout=self.config.model.dropout,
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
            mode=self.config.training.get('selection_metric_mode', 'min'),
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            min_lr=self.config.training.get('min_lr', 1e-6),
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'DCTSigmaTD16ChangePredictor: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
        if not self.config.data.get('use_hdf5', False):
            raise ValueError('DCT sigma TD16 change predictor requires HDF5 dataset.')

        h5_path = self.config.data.hdf5_path
        train_indices = self.config.data.get('train_indices', None)
        val_indices = self.config.data.get('val_indices', None)
        test_indices = self.config.data.get('test_indices', None)

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

        if test_indices is not None:
            test_ds = DifferenceConductivityHDF5Dataset(h5_path, indices=test_indices)
            self._warn_if_dropping_last_batch(
                'test', len(test_ds), self.config.training.batch_size)
            self.test_sim_loader = DataLoader(
                test_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers,
            )

    def _change_target(self, target_sigma, domain_mask):
        mask = domain_mask.float()
        threshold = float(self.config.training.get('change_threshold', 0.02))
        active = ((target_sigma.abs() > threshold).float() * mask)
        return (active.sum(dim=(1, 2, 3)) > 0).float().unsqueeze(1)

    def _masked_image_loss(self, pred_sigma, target_sigma, domain_mask):
        mask = domain_mask.float()
        diff = pred_sigma - target_sigma
        active_threshold = float(
            self.config.training.get('active_region_threshold', 0.02)
        )
        active_mask = ((target_sigma.abs() > active_threshold).float() * mask)
        inactive_mask = mask * (1.0 - active_mask)

        active_denom = active_mask.sum().clamp_min(1.0)
        inactive_denom = inactive_mask.sum().clamp_min(1.0)
        mse_active = (diff.pow(2) * active_mask).sum() / active_denom
        mae_active = (diff.abs() * active_mask).sum() / active_denom
        mse_inactive = (diff.pow(2) * inactive_mask).sum() / inactive_denom
        mae_inactive = (diff.abs() * inactive_mask).sum() / inactive_denom
        inactive_weight = float(self.config.training.get('inactive_weight', 1.0))
        mse = mse_active + inactive_weight * mse_inactive
        mae = mae_active + inactive_weight * mae_inactive
        pred_l1 = (pred_sigma.abs() * mask).sum() / mask.sum().clamp_min(1.0)
        total = (
            float(self.config.training.get('mse_weight', 1.0)) * mse +
            float(self.config.training.get('mae_weight', 0.2)) * mae +
            float(self.config.training.get('pred_l1_weight', 0.0)) * pred_l1
        )
        return total, mse, mae, pred_l1

    def train_step(self, batch):
        measurements, sigma_delta, domain_mask = batch[:3]
        measurements = measurements.to(self.device)
        sigma_delta = sigma_delta.to(self.device)
        domain_mask = domain_mask.to(self.device)
        levels_tensor = torch.ones(
            (measurements.shape[0],), dtype=torch.float, device=self.device)
        gate_target = self._change_target(sigma_delta, domain_mask)

        self.optimizer.zero_grad()
        with self._autocast_context():
            pred_sigma, coeffs, gate_logits = self.model(measurements, levels_tensor)
            target_coeffs = self.model.target_coeffs(sigma_delta)
            coeff_loss = F.mse_loss(coeffs.float(), target_coeffs.float())
            image_loss, mse_loss, mae_loss, pred_l1_loss = self._masked_image_loss(
                pred_sigma, sigma_delta, domain_mask)
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logits.float(), gate_target.float())
            total_loss = (
                image_loss +
                float(self.config.training.get('coeff_loss_weight', 0.25)) * coeff_loss +
                float(self.config.training.get('lambda_gate', 0.2)) * gate_loss
            )
        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)
        gate_pred = (torch.sigmoid(gate_logits) > 0.5).float()
        gate_acc = (gate_pred == gate_target).float().mean()
        return {
            'loss': total_loss.item(),
            'coeff_loss': coeff_loss.item(),
            'image_loss': image_loss.item(),
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'pred_l1_loss': pred_l1_loss.item(),
            'gate_loss': gate_loss.item(),
            'gate_acc': gate_acc.item(),
        }

    def _collect_metrics(self, loader):
        totals = {
            'loss': 0.0,
            'coeff_loss': 0.0,
            'image_loss': 0.0,
            'mse_loss': 0.0,
            'mae_loss': 0.0,
            'pred_l1_loss': 0.0,
            'gate_loss': 0.0,
            'gate_acc': 0.0,
        }
        num_samples = 0
        preds_all = []
        targets_all = []
        masks_all = []

        for measurements, sigma_delta, domain_mask, *rest in loader:
            measurements = measurements.to(self.device)
            sigma_delta = sigma_delta.to(self.device)
            domain_mask = domain_mask.to(self.device)
            levels_tensor = torch.ones(
                (measurements.shape[0],), dtype=torch.float, device=self.device)
            gate_target = self._change_target(sigma_delta, domain_mask)
            with torch.no_grad():
                with self._autocast_context():
                    pred_sigma, coeffs, gate_logits = self.model(
                        measurements, levels_tensor)
                    target_coeffs = self.model.target_coeffs(sigma_delta)
                    coeff_loss = F.mse_loss(coeffs.float(), target_coeffs.float())
                    image_loss, mse_loss, mae_loss, pred_l1_loss = self._masked_image_loss(
                        pred_sigma, sigma_delta, domain_mask)
                    gate_loss = F.binary_cross_entropy_with_logits(
                        gate_logits.float(), gate_target.float())
                    total_loss = (
                        image_loss +
                        float(self.config.training.get('coeff_loss_weight', 0.25)) * coeff_loss +
                        float(self.config.training.get('lambda_gate', 0.2)) * gate_loss
                    )
            gate_pred = (torch.sigmoid(gate_logits) > 0.5).float()
            gate_acc = (gate_pred == gate_target).float().mean()
            batch_size = measurements.shape[0]
            totals['loss'] += total_loss.item() * batch_size
            totals['coeff_loss'] += coeff_loss.item() * batch_size
            totals['image_loss'] += image_loss.item() * batch_size
            totals['mse_loss'] += mse_loss.item() * batch_size
            totals['mae_loss'] += mae_loss.item() * batch_size
            totals['pred_l1_loss'] += pred_l1_loss.item() * batch_size
            totals['gate_loss'] += gate_loss.item() * batch_size
            totals['gate_acc'] += gate_acc.item() * batch_size
            num_samples += batch_size
            preds_all.append(pred_sigma.detach().float().cpu().numpy())
            targets_all.append(sigma_delta.detach().float().cpu().numpy())
            masks_all.append(domain_mask.detach().float().cpu().numpy())

        pred_np = np.concatenate(preds_all, axis=0)[:, 0]
        target_np = np.concatenate(targets_all, axis=0)[:, 0]
        mask_np = np.concatenate(masks_all, axis=0)[:, 0] > 0.5
        reg = masked_regression_metrics_batch(
            target_np,
            pred_np,
            masks=mask_np,
            active_threshold=self.config.training.get('active_threshold', 0.02),
        )
        return {
            'loss': totals['loss'] / max(num_samples, 1),
            'coeff_loss': totals['coeff_loss'] / max(num_samples, 1),
            'image_loss': totals['image_loss'] / max(num_samples, 1),
            'mse_loss': totals['mse_loss'] / max(num_samples, 1),
            'mae_loss': totals['mae_loss'] / max(num_samples, 1),
            'pred_l1_loss': totals['pred_l1_loss'] / max(num_samples, 1),
            'gate_loss': totals['gate_loss'] / max(num_samples, 1),
            'gate_acc': totals['gate_acc'] / max(num_samples, 1),
            'preds': pred_np,
            'targets': target_np,
            'masks': mask_np,
            'reg': reg,
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}
        metrics_raw = self._collect_metrics(self.val_sim_loader)
        metrics = {
            'val_loss': metrics_raw['loss'],
            'val_coeff_loss': metrics_raw['coeff_loss'],
            'val_image_loss': metrics_raw['image_loss'],
            'val_mse_loss': metrics_raw['mse_loss'],
            'val_mae_loss': metrics_raw['mae_loss'],
            'val_pred_l1_loss': metrics_raw['pred_l1_loss'],
            'val_gate_loss': metrics_raw['gate_loss'],
            'val_gate_acc': metrics_raw['gate_acc'],
            'val_rmse': float(np.mean(metrics_raw['reg']['rmse'])),
            'val_rel_l2': float(np.mean(metrics_raw['reg']['rel_l2'])),
        }
        if 'active_rel_l2' in metrics_raw['reg']:
            metrics['val_active_rel_l2'] = float(
                np.nanmean(metrics_raw['reg']['active_rel_l2'])
            )
        if self.writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f'val/{key[4:]}', val, epoch + 1)
        print(
            f'  Val loss: {metrics["val_loss"]:.6f} '
            f'RMSE: {metrics["val_rmse"]:.6f} '
            f'RelL2: {metrics["val_rel_l2"]:.6f}'
            + (
                f' ActiveRelL2: {metrics["val_active_rel_l2"]:.6f}'
                if 'val_active_rel_l2' in metrics else ''
            )
            + f' GateAcc: {metrics["val_gate_acc"]:.6f}'
        )
        return metrics
