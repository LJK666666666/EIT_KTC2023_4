"""Trainer for atlas-residual DCT conductivity regression."""

import os

import h5py
import numpy as np

from .dct_sigma_predictor_trainer import DCTSigmaPredictorTrainer
from ..configs.dct_sigma_residual_predictor_config import (
    get_configs as get_dct_sigma_residual_predictor_config,
)
from ..models.dct_predictor import AtlasResidualDCTPredictor


class DCTSigmaResidualPredictorTrainer(DCTSigmaPredictorTrainer):
    def __init__(self, config=None,
                 experiment_name='dct_sigma_residual_predictor_baseline'):
        if config is None:
            config = get_dct_sigma_residual_predictor_config()
        super().__init__(config, experiment_name)
        self._last_focus_mse = 0.0
        self._last_focus_mae = 0.0

    def build_model(self):
        model = AtlasResidualDCTPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            coeff_size=self.config.model.coeff_size,
            out_channels=self.config.model.out_channels,
            dropout=self.config.model.dropout,
            image_size=self.config.model.get('image_size', 256),
        )
        model.to(self.device)
        self.model = model
        self.optimizer = self._build_optimizer(model)
        self.scheduler = self._build_scheduler()
        total_params = sum(p.numel() for p in model.parameters())
        print(f'DCTSigmaResidualPredictor: {total_params / 1e6:.2f}M parameters')

    def _build_optimizer(self, model):
        import torch
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.get('weight_decay', 1e-4),
        )

    def _build_scheduler(self):
        import torch.optim.lr_scheduler as lr_scheduler
        return lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.training.get('selection_metric_mode', 'min'),
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            min_lr=self.config.training.get('min_lr', 1e-6),
        )

    def _compute_train_atlas(self, h5_path, train_indices, chunk_size=256):
        train_indices = np.asarray(train_indices, dtype=np.int64)
        if train_indices.size == 0:
            raise ValueError('train_indices is empty, cannot build pulmonary atlas.')
        total = None
        count = 0
        with h5py.File(h5_path, 'r') as h5:
            sigma_ds = h5['sigma']
            for start in range(0, len(train_indices), chunk_size):
                batch_ids = train_indices[start:start + chunk_size]
                order = np.argsort(batch_ids)
                ids_sorted = batch_ids[order]
                inverse = np.argsort(order)
                sigma = sigma_ds[ids_sorted][inverse].astype(np.float64)
                batch_sum = sigma.sum(axis=0)
                if total is None:
                    total = batch_sum
                else:
                    total += batch_sum
                count += sigma.shape[0]
        atlas = (total / max(count, 1)).astype(np.float32)
        return atlas

    def _masked_image_loss(self, pred_sigma, target_sigma):
        total, mse, mae = super()._masked_image_loss(pred_sigma, target_sigma)
        focus_weight = float(self.config.training.get('focus_loss_weight', 0.0))
        focus_threshold = float(self.config.training.get('focus_threshold', 0.08))
        if focus_weight <= 0:
            return total, mse, mae

        atlas = self.model.atlas.to(device=target_sigma.device, dtype=target_sigma.dtype)
        base_mask = (target_sigma > 0).float()
        focus_mask = (((target_sigma - atlas).abs() > focus_threshold).float() * base_mask)
        denom = focus_mask.sum()
        if denom.item() < 1:
            return total, mse, mae

        diff = (pred_sigma - target_sigma) * focus_mask
        focus_mse = diff.pow(2).sum() / denom.clamp_min(1.0)
        focus_mae = diff.abs().sum() / denom.clamp_min(1.0)
        self._last_focus_mse = float(focus_mse.detach().item())
        self._last_focus_mae = float(focus_mae.detach().item())
        total = total + focus_weight * (
            float(self.config.training.get('mse_weight', 1.0)) * focus_mse +
            float(self.config.training.get('mae_weight', 0.1)) * focus_mae
        )
        return total, mse, mae

    def build_datasets(self):
        super().build_datasets()
        train_indices = self.config.data.get('train_indices', None)
        if train_indices is None:
            raise ValueError('Atlas-residual DCT predictor requires explicit train_indices.')
        atlas = self._compute_train_atlas(
            self.config.data.hdf5_path, train_indices)
        self.model.set_atlas(atlas)
        if self.config.training.get('save_atlas', True):
            np.save(os.path.join(self.result_dir, 'atlas.npy'), atlas)
