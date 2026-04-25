"""Trainer for dedicated TD16 spatial change-mask prediction."""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

from .base_trainer import BaseTrainer
from ..configs.dct_sigma_td16_mask_predictor_config import (
    get_configs as get_dct_sigma_td16_mask_predictor_config,
)
from ..data import DifferenceConductivityHDF5Dataset
from ..evaluation.mask_metrics import binary_mask_metrics_batch
from ..models.dct_predictor import MaskOnlyDCTPredictor


class DCTSigmaTD16MaskPredictorTrainer(BaseTrainer):
    def __init__(self, config=None,
                 experiment_name='dct_sigma_td16_mask_predictor_baseline'):
        if config is None:
            config = get_dct_sigma_td16_mask_predictor_config()
        super().__init__(config, experiment_name)
        self.test_sim_loader = None

    def build_model(self):
        model = MaskOnlyDCTPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            coeff_size=self.config.model.coeff_size,
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
            mode=self.config.training.get('selection_metric_mode', 'max'),
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            min_lr=self.config.training.get('min_lr', 1e-6),
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'DCTSigmaTD16MaskPredictor: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
        if not self.config.data.get('use_hdf5', False):
            raise ValueError('DCT sigma TD16 mask predictor requires HDF5 dataset.')

        h5_path = self.config.data.hdf5_path
        train_indices = self.config.data.get('train_indices', None)
        val_indices = self.config.data.get('val_indices', None)
        test_indices = self.config.data.get('test_indices', None)

        train_ds = DifferenceConductivityHDF5Dataset(h5_path, indices=train_indices)
        self._warn_if_dropping_last_batch(
            'train', len(train_ds), self.config.training.batch_size)
        sampler = None
        shuffle = True
        oversample_factor = float(
            self.config.training.get('active_oversample_factor', 1.0)
        )
        if oversample_factor > 1.0 and train_indices is not None:
            train_idx_np = np.asarray(train_indices, dtype=np.int64)
            order = np.argsort(train_idx_np)
            train_idx_sorted = train_idx_np[order]
            inverse = np.argsort(order)
            with h5py.File(h5_path, 'r') as h5f:
                sigma_delta = h5f['sigma_delta'][train_idx_sorted][inverse]
                domain_mask = h5f['domain_mask'][train_idx_sorted][inverse] > 0.5
            threshold = float(self.config.training.get('mask_threshold', 0.02))
            active = np.any((np.abs(sigma_delta) > threshold) & domain_mask, axis=(1, 2))
            weights = np.ones(len(train_indices), dtype=np.float64)
            weights[active] = oversample_factor
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(train_indices),
                replacement=True,
            )
            shuffle = False
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
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

    def _mask_target(self, target_sigma, domain_mask):
        threshold = float(self.config.training.get('mask_threshold', 0.02))
        return ((target_sigma.abs() > threshold).float() * domain_mask.float())

    @staticmethod
    def _dice_loss(mask_probs, mask_target):
        inter = (mask_probs * mask_target).sum(dim=(1, 2, 3))
        denom = mask_probs.sum(dim=(1, 2, 3)) + mask_target.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + 1.0) / (denom + 1.0)
        return 1.0 - dice.mean()

    def _loss_terms(self, mask_logits, mask_coeffs, mask_target):
        bce = F.binary_cross_entropy_with_logits(mask_logits.float(), mask_target.float())
        dice = self._dice_loss(torch.sigmoid(mask_logits).float(), mask_target.float())
        target_coeffs = self.model.target_mask_coeffs(mask_target)
        coeff = F.mse_loss(mask_coeffs.float(), target_coeffs.float())
        total = (
            float(self.config.training.get('lambda_bce', 1.0)) * bce +
            float(self.config.training.get('lambda_dice', 0.5)) * dice +
            float(self.config.training.get('coeff_loss_weight', 0.1)) * coeff
        )
        return total, bce, dice, coeff

    def _mask_metric_summary(self, mask_probs, mask_target, domain_mask):
        metrics = binary_mask_metrics_batch(
            target=mask_target[:, 0],
            pred=mask_probs[:, 0],
            valid_mask=domain_mask[:, 0],
            threshold=float(self.config.training.get('mask_prob_threshold', 0.5)),
        )
        return {
            'mask_precision': float(np.mean(metrics['precision'])),
            'mask_recall': float(np.mean(metrics['recall'])),
            'mask_f1': float(np.mean(metrics['f1'])),
            'mask_iou': float(np.mean(metrics['iou'])),
            'mask_acc': float(np.mean(metrics['accuracy'])),
        }

    def train_step(self, batch):
        measurements, sigma_delta, domain_mask = batch[:3]
        measurements = measurements.to(self.device)
        sigma_delta = sigma_delta.to(self.device)
        domain_mask = domain_mask.to(self.device)
        levels_tensor = torch.ones(
            (measurements.shape[0],), dtype=torch.float, device=self.device)
        mask_target = self._mask_target(sigma_delta, domain_mask)

        self.optimizer.zero_grad()
        with self._autocast_context():
            mask_logits, mask_coeffs = self.model(measurements, levels_tensor)
            total_loss, bce_loss, dice_loss, coeff_loss = self._loss_terms(
                mask_logits, mask_coeffs, mask_target
            )
        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)

        mask_probs = torch.sigmoid(mask_logits).detach().float().cpu().numpy()
        mask_target_np = mask_target.detach().float().cpu().numpy()
        domain_mask_np = domain_mask.detach().float().cpu().numpy()
        metrics = self._mask_metric_summary(mask_probs, mask_target_np, domain_mask_np)
        return {
            'loss': total_loss.item(),
            'mask_bce_loss': bce_loss.item(),
            'mask_dice_loss': dice_loss.item(),
            'mask_coeff_loss': coeff_loss.item(),
            **metrics,
        }

    def _collect_metrics(self, loader):
        totals = {
            'loss': 0.0,
            'mask_bce_loss': 0.0,
            'mask_dice_loss': 0.0,
            'mask_coeff_loss': 0.0,
        }
        mask_probs_all = []
        mask_target_all = []
        domain_mask_all = []
        num_samples = 0

        for measurements, sigma_delta, domain_mask, *rest in loader:
            measurements = measurements.to(self.device)
            sigma_delta = sigma_delta.to(self.device)
            domain_mask = domain_mask.to(self.device)
            levels_tensor = torch.ones(
                (measurements.shape[0],), dtype=torch.float, device=self.device)
            mask_target = self._mask_target(sigma_delta, domain_mask)
            with torch.no_grad():
                with self._autocast_context():
                    mask_logits, mask_coeffs = self.model(measurements, levels_tensor)
                    total_loss, bce_loss, dice_loss, coeff_loss = self._loss_terms(
                        mask_logits, mask_coeffs, mask_target
                    )
            batch_size = measurements.shape[0]
            totals['loss'] += total_loss.item() * batch_size
            totals['mask_bce_loss'] += bce_loss.item() * batch_size
            totals['mask_dice_loss'] += dice_loss.item() * batch_size
            totals['mask_coeff_loss'] += coeff_loss.item() * batch_size
            num_samples += batch_size

            mask_probs_all.append(torch.sigmoid(mask_logits).detach().float().cpu().numpy())
            mask_target_all.append(mask_target.detach().float().cpu().numpy())
            domain_mask_all.append(domain_mask.detach().float().cpu().numpy())

        mask_probs_np = np.concatenate(mask_probs_all, axis=0)
        mask_target_np = np.concatenate(mask_target_all, axis=0)
        domain_mask_np = np.concatenate(domain_mask_all, axis=0)
        metrics = self._mask_metric_summary(mask_probs_np, mask_target_np, domain_mask_np)
        return {
            'loss': totals['loss'] / max(num_samples, 1),
            'mask_bce_loss': totals['mask_bce_loss'] / max(num_samples, 1),
            'mask_dice_loss': totals['mask_dice_loss'] / max(num_samples, 1),
            'mask_coeff_loss': totals['mask_coeff_loss'] / max(num_samples, 1),
            **metrics,
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}
        metrics_raw = self._collect_metrics(self.val_sim_loader)
        metrics = {
            'val_loss': metrics_raw['loss'],
            'val_mask_bce_loss': metrics_raw['mask_bce_loss'],
            'val_mask_dice_loss': metrics_raw['mask_dice_loss'],
            'val_mask_coeff_loss': metrics_raw['mask_coeff_loss'],
            'val_mask_precision': metrics_raw['mask_precision'],
            'val_mask_recall': metrics_raw['mask_recall'],
            'val_mask_f1': metrics_raw['mask_f1'],
            'val_mask_iou': metrics_raw['mask_iou'],
            'val_mask_acc': metrics_raw['mask_acc'],
        }
        if self.writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f'val/{key[4:]}', val, epoch + 1)
        print(
            f'  Val loss: {metrics["val_loss"]:.6f} '
            f'IoU: {metrics["val_mask_iou"]:.6f} '
            f'F1: {metrics["val_mask_f1"]:.6f} '
            f'Recall: {metrics["val_mask_recall"]:.6f}'
        )
        return metrics
