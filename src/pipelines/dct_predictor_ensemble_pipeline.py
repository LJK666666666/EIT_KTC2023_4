"""Ensemble inference pipeline for DCT low-frequency predictors."""

import os

import numpy as np
import torch
import yaml

from .base_pipeline import BasePipeline
from ..models.dct_predictor import DCTPredictor


class DCTPredictorEnsemblePipeline(BasePipeline):
    def __init__(self,
                 device='cuda',
                 weights_base_dir='results',
                 config_path='scripts/dct_predictor_ensemble.yaml'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config_path = config_path
        self._models_loaded = False
        self.models = []
        self.model_weights = []

    def _load_ensemble_spec(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            spec = yaml.unsafe_load(f) or {}
        members = spec.get('members', [])
        if not members:
            raise ValueError(f'No ensemble members configured in {self.config_path}')
        norm = []
        for member in members:
            result_dir = member.get('result_dir', '').strip()
            if not result_dir:
                continue
            weight = float(member.get('weight', 1.0))
            norm.append({'result_dir': result_dir, 'weight': weight})
        if not norm:
            raise ValueError(f'No valid ensemble members configured in {self.config_path}')
        total = sum(max(m['weight'], 0.0) for m in norm)
        if total <= 0:
            raise ValueError('Ensemble weights must sum to a positive value.')
        for member in norm:
            member['weight'] = member['weight'] / total
        return norm

    def _build_model_from_dir(self, result_dir):
        config_path = os.path.join(result_dir, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Missing config.yaml in {result_dir}')
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.unsafe_load(f)
        model_cfg = cfg.get('model', {})
        model = DCTPredictor(
            input_dim=model_cfg['input_dim'],
            hidden_dims=tuple(model_cfg['hidden_dims']),
            level_embed_dim=model_cfg['level_embed_dim'],
            coeff_size=model_cfg['coeff_size'],
            out_channels=model_cfg['out_channels'],
            dropout=model_cfg['dropout'],
        )
        weight_path = self._find_weight([
            os.path.join(result_dir, 'best.pt'),
            os.path.join(result_dir, 'last.pt'),
        ])
        state_dict = self._load_state_dict(weight_path, self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    def load_model(self, level: int) -> None:
        if self._models_loaded:
            return
        members = self._load_ensemble_spec()
        self.models = []
        self.model_weights = []
        for member in members:
            result_dir = os.path.join(self.weights_base_dir, member['result_dir'])
            self.models.append(self._build_model_from_dir(result_dir))
            self.model_weights.append(member['weight'])
        self._models_loaded = True

    def _prepare_input(self, Uel, ref_data, level):
        Injref = ref_data['Injref']
        Uelref = np.asarray(ref_data['Uelref']).reshape(-1)
        vincl = self.create_vincl(level, Injref).T.flatten()
        y = np.asarray(Uel).reshape(-1) - Uelref
        y[~vincl] = 0.0
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def _predict_logits(self, y_tensor, level_tensor):
        logits_sum = None
        with torch.no_grad():
            with self._autocast_context():
                for model, weight in zip(self.models, self.model_weights):
                    logits, _ = model(y_tensor, level_tensor)
                    weighted = logits.float() * float(weight)
                    logits_sum = weighted if logits_sum is None else logits_sum + weighted
        return logits_sum

    def reconstruct(self, Uel, ref_data, level):
        return self.reconstruct_batch([Uel], ref_data, level)[0]

    def reconstruct_batch(self, Uels, ref_data, level):
        y_batch = [self._prepare_input(Uel, ref_data, level) for Uel in Uels]
        y_tensor = torch.from_numpy(np.stack(y_batch).astype(np.float32)).to(self.device)
        level_tensor = torch.full(
            (y_tensor.shape[0],), level, dtype=torch.float, device=self.device)
        logits = self._predict_logits(y_tensor, level_tensor)
        pred = torch.argmax(logits, dim=1)
        return [arr for arr in pred.cpu().numpy().astype(int)]

    def reconstruct_mixed_batch(self, samples):
        y_batch = [self._prepare_input(s['Uel'], s['ref_data'], s['level'])
                   for s in samples]
        level_tensor = torch.tensor(
            [s['level'] for s in samples], dtype=torch.float, device=self.device)
        y_tensor = torch.from_numpy(np.stack(y_batch).astype(np.float32)).to(self.device)
        logits = self._predict_logits(y_tensor, level_tensor)
        pred = torch.argmax(logits, dim=1)
        return [arr for arr in pred.cpu().numpy().astype(int)]
