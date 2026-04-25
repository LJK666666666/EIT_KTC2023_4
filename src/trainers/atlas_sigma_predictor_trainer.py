"""Trainer for atlas-aware direct residual conductivity prediction."""

import numpy as np
import torch

from .dct_sigma_residual_predictor_trainer import (
    DCTSigmaResidualPredictorTrainer,
)
from ..configs.atlas_sigma_predictor_config import (
    get_configs as get_atlas_sigma_predictor_config,
)
from ..models.dct_predictor import AtlasResidualDecoderPredictor


class AtlasSigmaPredictorTrainer(DCTSigmaResidualPredictorTrainer):
    def __init__(self, config=None,
                 experiment_name='atlas_sigma_predictor_baseline'):
        if config is None:
            config = get_atlas_sigma_predictor_config()
        super().__init__(config, experiment_name)

    def build_model(self):
        model = AtlasResidualDecoderPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            out_channels=self.config.model.out_channels,
            dropout=self.config.model.dropout,
            image_size=self.config.model.get('image_size', 256),
            seed_channels=self.config.model.get('seed_channels', 32),
            seed_size=self.config.model.get('seed_size', 16),
        )
        model.to(self.device)
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.get('weight_decay', 1e-4),
        )
        self.scheduler = self._build_scheduler()
        total_params = sum(p.numel() for p in model.parameters())
        print(f'AtlasSigmaPredictor: {total_params / 1e6:.2f}M parameters')

    def train_step(self, batch):
        measurements, target_sigma, _ = batch
        measurements = measurements.to(self.device)
        target_sigma = target_sigma.to(self.device)
        levels = torch.ones(
            measurements.shape[0], device=self.device, dtype=torch.long)

        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            pred_sigma = self.model(measurements, levels)
            total_loss, mse, mae = self._masked_image_loss(pred_sigma, target_sigma)
        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)
        return {
            'loss': float(total_loss.detach().item()),
            'mse': float(mse.detach().item()),
            'mae': float(mae.detach().item()),
            'focus_mse': float(self._last_focus_mse),
            'focus_mae': float(self._last_focus_mae),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}
        self.model.eval()
        losses = []
        mses = []
        maes = []
        preds = []
        targets = []
        with torch.no_grad():
            for measurements, target_sigma, _ in self.val_sim_loader:
                measurements = measurements.to(self.device)
                target_sigma = target_sigma.to(self.device)
                levels = torch.ones(
                    measurements.shape[0], device=self.device, dtype=torch.long)
                with self._autocast_context():
                    pred_sigma = self.model(measurements, levels)
                    total_loss, mse, mae = self._masked_image_loss(
                        pred_sigma, target_sigma)
                losses.append(float(total_loss.detach().item()))
                mses.append(float(mse.detach().item()))
                maes.append(float(mae.detach().item()))
                preds.append(pred_sigma.detach().float().cpu().numpy())
                targets.append(target_sigma.detach().float().cpu().numpy())
        pred_np = np.concatenate(preds, axis=0)
        target_np = np.concatenate(targets, axis=0)
        from ..evaluation.regression_metrics import masked_regression_metrics_batch
        reg = masked_regression_metrics_batch(target_np, pred_np)
        return {
            'val_loss': float(sum(losses) / max(len(losses), 1)),
            'val_mse': float(sum(mses) / max(len(mses), 1)),
            'val_mae': float(sum(maes) / max(len(maes), 1)),
            'val_rmse': float(np.mean(reg['rmse'])),
            'val_rel_l2': float(np.mean(reg['rel_l2'])),
        }
