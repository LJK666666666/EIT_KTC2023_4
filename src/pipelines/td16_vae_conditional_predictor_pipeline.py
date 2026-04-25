"""Inference pipeline for mask-conditioned TD16 VAE latent prediction."""

import os

import numpy as np
import torch
import yaml

from .base_pipeline import BasePipeline
from ..configs import get_td16_vae_conditional_predictor_config
from ..models.pulmonary_vae import ConvVAE, ConditionalLatentMLPPredictor
from ..utils.pulmonary16 import reorder_raw256_to_208


class TD16VAEConditionalPredictorPipeline(BasePipeline):
    def __init__(self, device='cuda', weights_base_dir='results'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_td16_vae_conditional_predictor_config()
        self.vae = None
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
        vae_cfg = loaded.get('vae', {})
        training_cfg = loaded.get('training', {})
        for key in ('input_dim', 'hidden_dims', 'latent_dim', 'dropout',
                    'mask_coeff_size', 'mask_condition_dim'):
            if key in model_cfg:
                setattr(self.config.model, key, model_cfg[key])
        for key in ('checkpoint', 'latent_dim', 'base_channels', 'in_channels'):
            if key in vae_cfg:
                setattr(self.config.vae, key, vae_cfg[key])
        for key in ('mask_prob_threshold',):
            if key in training_cfg:
                setattr(self.config.training, key, training_cfg[key])

    def load_model(self, level: int = 1) -> None:
        if self._model_loaded:
            return
        self._load_runtime_config()
        predictor = ConditionalLatentMLPPredictor(
            input_dim=self.config.model.input_dim,
            latent_dim=self.config.model.latent_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            dropout=self.config.model.dropout,
            mask_coeff_size=self.config.model.mask_coeff_size,
            mask_condition_dim=self.config.model.mask_condition_dim,
        )
        weight_path = self._find_weight([
            f'{self.weights_base_dir}/best.pt',
            f'{self.weights_base_dir}/last.pt',
        ])
        state = self._load_state_dict(weight_path, self.device)
        predictor.load_state_dict(state)
        predictor.eval()
        predictor.to(self.device)
        self.model = predictor

        if not self.config.vae.checkpoint:
            raise ValueError('TD16 conditional VAE predictor config missing vae.checkpoint')
        vae = ConvVAE(
            in_channels=self.config.vae.in_channels,
            latent_dim=self.config.vae.latent_dim,
            base_channels=self.config.vae.base_channels,
        )
        vae_state = torch.load(self.config.vae.checkpoint, map_location=self.device, weights_only=False)
        if 'model_state_dict' in vae_state:
            vae_state = vae_state['model_state_dict']
        vae.load_state_dict(vae_state)
        vae.eval()
        vae.to(self.device)
        self.vae = vae
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
        with torch.no_grad():
            with self._autocast_context():
                latent, mask_logits, _ = self.model(y_tensor)
                sigma = self.vae.decode(latent) * torch.sigmoid(mask_logits)
        sigma_np = sigma[:, 0].detach().float().cpu().numpy().astype(np.float32)
        return [arr for arr in sigma_np]
