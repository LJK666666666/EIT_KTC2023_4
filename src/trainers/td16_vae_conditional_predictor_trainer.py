"""Trainer for mask-conditioned TD16 VAE latent prediction on mixed data."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..configs.td16_vae_conditional_predictor_config import (
    get_configs as get_td16_vae_conditional_predictor_config,
)
from ..data import DifferenceConductivityHDF5Dataset
from ..evaluation.mask_metrics import binary_mask_metrics_batch
from ..evaluation.regression_metrics import masked_regression_metrics_batch
from ..models.pulmonary_vae import ConvVAE, ConditionalLatentMLPPredictor


class TD16VAEConditionalPredictorTrainer(BaseTrainer):
    def __init__(self, config=None, experiment_name='td16_vae_conditional_predictor_baseline'):
        if config is None:
            config = get_td16_vae_conditional_predictor_config()
        super().__init__(config, experiment_name)
        self.vae = None

    def build_model(self):
        if not self.config.vae.checkpoint:
            raise ValueError(
                'TD16 conditional VAE predictor requires --vae-checkpoint / config.vae.checkpoint'
            )
        predictor = ConditionalLatentMLPPredictor(
            input_dim=self.config.model.input_dim,
            latent_dim=self.config.model.latent_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            dropout=self.config.model.dropout,
            mask_coeff_size=self.config.model.mask_coeff_size,
            mask_condition_dim=self.config.model.mask_condition_dim,
        )
        predictor.to(self.device)
        self.model = predictor

        vae = ConvVAE(
            in_channels=self.config.vae.in_channels,
            latent_dim=self.config.vae.latent_dim,
            base_channels=self.config.vae.base_channels,
        )
        state = torch.load(self.config.vae.checkpoint, map_location=self.device, weights_only=False)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        vae.load_state_dict(state)
        vae.to(self.device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        self.vae = vae

        self.optimizer = torch.optim.AdamW(
            predictor.parameters(),
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
        total_params = sum(p.numel() for p in predictor.parameters())
        print(f'TD16VAEConditionalPredictor: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
        h5_path = self.config.data.hdf5_path
        train_indices = self.config.data.get('train_indices', None)
        val_indices = self.config.data.get('val_indices', None)

        train_ds = DifferenceConductivityHDF5Dataset(h5_path, indices=train_indices)
        self._warn_if_dropping_last_batch('train', len(train_ds), self.config.training.batch_size)
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
            self._warn_if_dropping_last_batch('val', len(val_ds), self.config.training.batch_size)
            self.val_sim_loader = DataLoader(
                val_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers,
            )

    def _encode_target(self, sigma_delta):
        mu, _ = self.vae.encode(sigma_delta)
        return mu

    def _mask_target(self, sigma_delta, domain_mask):
        threshold = float(self.config.training.get('mask_threshold', 0.02))
        return ((sigma_delta.abs() > threshold).float() * domain_mask.float())

    @staticmethod
    def _dice_loss(mask_probs, mask_target):
        inter = (mask_probs * mask_target).sum(dim=(1, 2, 3))
        denom = mask_probs.sum(dim=(1, 2, 3)) + mask_target.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + 1.0) / (denom + 1.0)
        return 1.0 - dice.mean()

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
        }

    def _loss_terms(self, latent_pred, latent_target, recon_pred, sigma_delta,
                    domain_mask, mask_logits, mask_coeffs_pred, mask_target, mask_coeffs_target):
        mask = domain_mask.float()
        latent_loss = F.mse_loss(latent_pred.float(), latent_target.float())
        diff = recon_pred - sigma_delta
        denom = mask.sum().clamp_min(1.0)
        image_mse = (diff.pow(2) * mask).sum() / denom
        image_mae = (diff.abs() * mask).sum() / denom
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits.float(), mask_target.float())
        mask_dice = self._dice_loss(torch.sigmoid(mask_logits).float(), mask_target.float())
        mask_coeff = F.mse_loss(mask_coeffs_pred.float(), mask_coeffs_target.float())
        total = (
            float(self.config.training.get('latent_weight', 1.0)) * latent_loss +
            float(self.config.training.get('image_mse_weight', 1.0)) * image_mse +
            float(self.config.training.get('image_mae_weight', 0.2)) * image_mae +
            float(self.config.training.get('lambda_mask_bce', 1.0)) * mask_bce +
            float(self.config.training.get('lambda_mask_dice', 0.5)) * mask_dice +
            float(self.config.training.get('lambda_mask_coeff', 0.1)) * mask_coeff
        )
        return total, latent_loss, image_mse, image_mae, mask_bce, mask_dice, mask_coeff

    def train_step(self, batch):
        measurements, sigma_delta, domain_mask, *rest = batch
        measurements = measurements.to(self.device)
        sigma_delta = sigma_delta.to(self.device)
        domain_mask = domain_mask.to(self.device)
        with torch.no_grad():
            latent_target = self._encode_target(sigma_delta)
        mask_target = self._mask_target(sigma_delta, domain_mask)
        mask_coeffs_target = self.model.target_mask_coeffs(mask_target)
        teacher_ratio = float(self.config.training.get('mask_teacher_forcing_ratio', 0.0))
        mask_override = mask_coeffs_target if np.random.rand() < teacher_ratio else None

        self.optimizer.zero_grad()
        with self._autocast_context():
            latent_pred, mask_logits, mask_coeffs_pred = self.model(
                measurements, mask_coeffs_override=mask_override)
            recon_raw = self.vae.decode(latent_pred)
            recon_pred = recon_raw * torch.sigmoid(mask_logits)
            losses = self._loss_terms(
                latent_pred, latent_target, recon_pred, sigma_delta, domain_mask,
                mask_logits, mask_coeffs_pred, mask_target, mask_coeffs_target)
            total_loss, latent_loss, image_mse, image_mae, mask_bce, mask_dice, mask_coeff = losses
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
        self.optimizer_step(self.optimizer)

        metrics = {
            'loss': total_loss.item(),
            'latent_loss': latent_loss.item(),
            'image_mse': image_mse.item(),
            'image_mae': image_mae.item(),
            'mask_bce_loss': mask_bce.item(),
            'mask_dice_loss': mask_dice.item(),
            'mask_coeff_loss': mask_coeff.item(),
        }
        mask_probs = torch.sigmoid(mask_logits).detach().float().cpu().numpy()
        mask_target_np = mask_target.detach().float().cpu().numpy()
        domain_mask_np = domain_mask.detach().float().cpu().numpy()
        metrics.update(self._mask_metric_summary(mask_probs, mask_target_np, domain_mask_np))
        return metrics

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}
        totals = {
            'loss': 0.0,
            'latent_loss': 0.0,
            'image_mse': 0.0,
            'image_mae': 0.0,
            'mask_bce_loss': 0.0,
            'mask_dice_loss': 0.0,
            'mask_coeff_loss': 0.0,
        }
        preds_all, targets_all, masks_all = [], [], []
        mask_probs_all, mask_target_all, domain_mask_all = [], [], []
        num_samples = 0
        for measurements, sigma_delta, domain_mask, *rest in self.val_sim_loader:
            measurements = measurements.to(self.device)
            sigma_delta = sigma_delta.to(self.device)
            domain_mask = domain_mask.to(self.device)
            with torch.no_grad():
                latent_target = self._encode_target(sigma_delta)
                mask_target = self._mask_target(sigma_delta, domain_mask)
                mask_coeffs_target = self.model.target_mask_coeffs(mask_target)
                with self._autocast_context():
                    latent_pred, mask_logits, mask_coeffs_pred = self.model(measurements)
                    recon_raw = self.vae.decode(latent_pred)
                    recon_pred = recon_raw * torch.sigmoid(mask_logits)
                    losses = self._loss_terms(
                        latent_pred, latent_target, recon_pred, sigma_delta, domain_mask,
                        mask_logits, mask_coeffs_pred, mask_target, mask_coeffs_target)
                    total_loss, latent_loss, image_mse, image_mae, mask_bce, mask_dice, mask_coeff = losses
            batch_size = measurements.shape[0]
            totals['loss'] += total_loss.item() * batch_size
            totals['latent_loss'] += latent_loss.item() * batch_size
            totals['image_mse'] += image_mse.item() * batch_size
            totals['image_mae'] += image_mae.item() * batch_size
            totals['mask_bce_loss'] += mask_bce.item() * batch_size
            totals['mask_dice_loss'] += mask_dice.item() * batch_size
            totals['mask_coeff_loss'] += mask_coeff.item() * batch_size
            num_samples += batch_size
            preds_all.append(recon_pred.detach().float().cpu().numpy()[:, 0])
            targets_all.append(sigma_delta.detach().float().cpu().numpy()[:, 0])
            masks_all.append(domain_mask.detach().float().cpu().numpy()[:, 0] > 0.5)
            mask_probs_all.append(torch.sigmoid(mask_logits).detach().float().cpu().numpy()[:, 0])
            mask_target_all.append(mask_target.detach().float().cpu().numpy()[:, 0])
            domain_mask_all.append(domain_mask.detach().float().cpu().numpy()[:, 0] > 0.5)

        reg = masked_regression_metrics_batch(
            np.concatenate(targets_all, axis=0),
            np.concatenate(preds_all, axis=0),
            masks=np.concatenate(masks_all, axis=0),
            active_threshold=self.config.training.active_threshold,
        )
        metrics = {
            'val_loss': totals['loss'] / max(num_samples, 1),
            'val_latent_loss': totals['latent_loss'] / max(num_samples, 1),
            'val_image_mse': totals['image_mse'] / max(num_samples, 1),
            'val_image_mae': totals['image_mae'] / max(num_samples, 1),
            'val_mask_bce_loss': totals['mask_bce_loss'] / max(num_samples, 1),
            'val_mask_dice_loss': totals['mask_dice_loss'] / max(num_samples, 1),
            'val_mask_coeff_loss': totals['mask_coeff_loss'] / max(num_samples, 1),
            'val_rmse': float(np.mean(reg['rmse'])),
            'val_rel_l2': float(np.mean(reg['rel_l2'])),
        }
        if 'active_rel_l2' in reg:
            metrics['val_active_rel_l2'] = float(np.nanmean(reg['active_rel_l2']))
        metrics.update({
            f'val_{k}': v for k, v in self._mask_metric_summary(
                np.concatenate(mask_probs_all, axis=0)[:, None, ...],
                np.concatenate(mask_target_all, axis=0)[:, None, ...],
                np.concatenate(domain_mask_all, axis=0)[:, None, ...],
            ).items()
        })
        if self.writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f'val/{key[4:]}', val, epoch + 1)
        print(
            f'  Val loss: {metrics["val_loss"]:.6f} '
            f'RMSE: {metrics["val_rmse"]:.6f} '
            f'ActiveRelL2: {metrics.get("val_active_rel_l2", float("nan")):.6f} '
            f'MaskIoU: {metrics.get("val_mask_iou", float("nan")):.6f}'
        )
        return metrics
