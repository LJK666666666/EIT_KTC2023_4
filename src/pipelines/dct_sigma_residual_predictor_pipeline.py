"""Inference pipeline for atlas-residual DCT conductivity regression."""

import os
import numpy as np
import torch
import yaml

from .base_pipeline import BasePipeline
from ..configs import get_dct_sigma_residual_predictor_config
from ..models.dct_predictor import AtlasResidualDCTPredictor


class DCTSigmaResidualPredictorPipeline(BasePipeline):
    def __init__(self, device='cuda', weights_base_dir='results'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_dct_sigma_residual_predictor_config()
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
                    'coeff_size', 'out_channels', 'dropout', 'image_size'):
            if key in model_cfg:
                setattr(self.config.model, key, model_cfg[key])

    def load_model(self, level: int) -> None:
        if self._model_loaded:
            return
        self._load_runtime_config()
        model = AtlasResidualDCTPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            coeff_size=self.config.model.coeff_size,
            out_channels=self.config.model.out_channels,
            dropout=self.config.model.dropout,
            image_size=self.config.model.get('image_size', 256),
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

    def _prepare_input(self, Uel, ref_data, level):
        Injref = ref_data['Injref']
        Uelref = np.asarray(ref_data['Uelref']).reshape(-1)
        vincl = self.create_vincl(level, Injref).T.flatten()
        y = np.asarray(Uel).reshape(-1) - Uelref
        y[~vincl] = 0.0
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def reconstruct(self, Uel, ref_data, level):
        return self.reconstruct_batch([Uel], ref_data, level)[0]

    def reconstruct_batch(self, Uels, ref_data, level):
        y_batch = [self._prepare_input(Uel, ref_data, level) for Uel in Uels]
        y_tensor = torch.from_numpy(np.stack(y_batch).astype(np.float32)).to(self.device)
        level_tensor = torch.full(
            (y_tensor.shape[0],), level, dtype=torch.float, device=self.device)
        with torch.no_grad():
            with self._autocast_context():
                sigma, _ = self.model(y_tensor, level_tensor)
        sigma_np = sigma[:, 0].detach().float().cpu().numpy().astype(np.float32)
        return [arr for arr in sigma_np]
