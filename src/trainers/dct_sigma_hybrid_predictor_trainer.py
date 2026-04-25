"""Trainer for pulmonary atlas-plus-refinement DCT regression."""

import torch
import torch.optim.lr_scheduler as lr_scheduler

from .dct_sigma_residual_predictor_trainer import (
    DCTSigmaResidualPredictorTrainer,
)
from ..configs.dct_sigma_hybrid_predictor_config import (
    get_configs as get_dct_sigma_hybrid_predictor_config,
)
from ..models.dct_predictor import AtlasRefineDCTPredictor


class DCTSigmaHybridPredictorTrainer(DCTSigmaResidualPredictorTrainer):
    def __init__(self, config=None,
                 experiment_name='dct_sigma_hybrid_predictor_baseline'):
        if config is None:
            config = get_dct_sigma_hybrid_predictor_config()
        super().__init__(config, experiment_name)

    def build_model(self):
        model = AtlasRefineDCTPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            coeff_size=self.config.model.coeff_size,
            out_channels=self.config.model.out_channels,
            dropout=self.config.model.dropout,
            image_size=self.config.model.get('image_size', 256),
            refine_channels=self.config.model.get('refine_channels', 32),
            refine_seed_size=self.config.model.get('refine_seed_size', 16),
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
        print(f'DCTSigmaHybridPredictor: {total_params / 1e6:.2f}M parameters')
