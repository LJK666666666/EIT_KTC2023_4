"""Inference pipeline for change-aware 16-electrode pulmonary TD16 regression."""

import os

import numpy as np
import torch
import yaml

from .base_pipeline import BasePipeline
from ..configs import get_dct_sigma_td16_change_predictor_config
from ..models.dct_predictor import ChangeGatedDCTPredictor
from ..utils.pulmonary16 import reorder_raw256_to_208


class DCTSigmaTD16ChangePredictorPipeline(BasePipeline):
    def __init__(self, device='cuda', weights_base_dir='results'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_dct_sigma_td16_change_predictor_config()
        self._model_loaded = False

    def _load_runtime_config(self):
        config_path = os.path.join(self.weights_base_dir, 'config.yaml')
        if not os.path.exists(config_path):
            return
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded = yaml.unsafe_load(f)
        if not isinstance(loaded, dict):
            return
        model_cfg = loaded.get('model', {})
        for key in ('input_dim', 'hidden_dims', 'level_embed_dim',
                    'coeff_size', 'out_channels', 'dropout'):
            if key in model_cfg:
                setattr(self.config.model, key, model_cfg[key])

    def load_model(self, level: int = 1) -> None:
        if self._model_loaded:
            return
        self._load_runtime_config()
        model = ChangeGatedDCTPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            coeff_size=self.config.model.coeff_size,
            out_channels=self.config.model.out_channels,
            dropout=self.config.model.dropout,
        )
        weight_path = self._find_weight([
            f'{self.weights_base_dir}/best.pt',
            f'{self.weights_base_dir}/last.pt',
        ])
        model.load_state_dict(self._load_state_dict(weight_path, self.device))
        model.eval()
        model.to(self.device)
        self.model = model
        self._model_loaded = True

    def _prepare_input(self, Uel):
        y = np.asarray(Uel, dtype=np.float32).reshape(-1)
        if y.size == 256:
            y = reorder_raw256_to_208(y)
        if y.size != self.config.model.input_dim:
            raise ValueError(
                f'Expected 208-dim delta measurement or raw 256-dim vector, got {y.size}.'
            )
        return y.astype(np.float32)

    def reconstruct(self, Uel, ref_data=None, level=1):
        return self.reconstruct_batch([Uel], ref_data=ref_data, level=level)[0]

    def reconstruct_batch(self, Uels, ref_data=None, level=1):
        y_batch = [self._prepare_input(Uel) for Uel in Uels]
        y_tensor = torch.from_numpy(np.stack(y_batch).astype(np.float32)).to(self.device)
        level_tensor = torch.ones(
            (y_tensor.shape[0],), dtype=torch.float, device=self.device)
        with torch.no_grad():
            with self._autocast_context():
                sigma, _, _ = self.model(y_tensor, level_tensor)
        sigma_np = sigma[:, 0].detach().float().cpu().numpy().astype(np.float32)
        return [arr for arr in sigma_np]
