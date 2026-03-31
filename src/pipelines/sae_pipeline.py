"""
SAE inference pipeline for EIT reconstruction.

Loads MLP predictor + SAE decoder. Reconstructs by:
1. Preprocess measurements (subtract ref, apply vincl)
2. MLP → (z_shape, angle_xy)
3. Soft threshold z_shape
4. Decoder(z_shape) → canonical logits
5. Rotate by +θ → argmax → segmentation
"""

import os
import pickle
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .base_pipeline import BasePipeline
from ..models.sae import SparseAutoEncoder, MeasurementPredictor


class SAEPipeline(BasePipeline):

    def __init__(self, device='cuda', weights_base_dir='results',
                 config_path='scripts/sae_pipeline.yaml',
                 sae_dir_override='',
                 predictor_dir_override=''):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.predictor = None
        self.decoder = None
        self.sparsity_threshold = 0.0
        self.shape_dim = 63
        self.config_path = config_path
        self.sae_dir_override = sae_dir_override
        self.predictor_dir_override = predictor_dir_override
        self.predictor_dir = None
        self.sae_dir = None

    def load_model(self, level: int) -> None:
        """Load predictor MLP + SAE decoder from results directory."""
        if self.predictor is not None:
            return  # Already loaded (level-independent)

        pipeline_cfg = self._load_pipeline_config()
        predictor_dir = self._resolve_result_dir(
            self.predictor_dir_override
            or pipeline_cfg.get('sae_predictor_dir', ''),
            prefix='sae_predictor_baseline')
        sae_dir = self._resolve_result_dir(
            self.sae_dir_override or pipeline_cfg.get('sae_dir', ''),
            prefix='sae_baseline')
        self.predictor_dir = predictor_dir
        self.sae_dir = sae_dir

        if predictor_dir is None or sae_dir is None:
            raise FileNotFoundError(
                'Cannot find SAE weight directories. '
                f'Searched in {self.weights_base_dir}')

        # Load predictor
        pred_path = self._find_weight([
            os.path.join(predictor_dir, 'best.pt'),
            os.path.join(predictor_dir, 'last.pt'),
        ])
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = np.core
        if 'numpy._core.multiarray' not in sys.modules:
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
        try:
            pred_ckpt = torch.load(
                pred_path, map_location=self.device, weights_only=False)
        except pickle.UnpicklingError:
            pred_ckpt = torch.load(
                pred_path, map_location=self.device, weights_only=True)
        pred_state = (pred_ckpt.get('model_state_dict', pred_ckpt)
                      if isinstance(pred_ckpt, dict) else pred_ckpt)
        predictor = MeasurementPredictor()
        predictor.load_state_dict(pred_state)
        predictor.to(self.device)
        predictor.eval()
        self.predictor = predictor

        # Load SAE (for decoder)
        sae_path = self._find_weight([
            os.path.join(sae_dir, 'best.pt'),
            os.path.join(sae_dir, 'last.pt'),
        ])
        sae_state = self._load_state_dict(sae_path, self.device)
        sae = SparseAutoEncoder()
        sae.load_state_dict(sae_state)

        if isinstance(pred_ckpt, dict) and 'sae_decoder_state_dict' in pred_ckpt:
            sae.decoder.load_state_dict(pred_ckpt['sae_decoder_state_dict'])
            print('SAEPipeline: using decoder weights stored in predictor checkpoint')

        sae.to(self.device)
        sae.eval()
        self.decoder = sae.decoder

        # Load sparsity threshold
        latent_path = os.path.join(sae_dir, 'latent_codes.h5')
        if os.path.exists(latent_path):
            with h5py.File(latent_path, 'r') as f:
                self.sparsity_threshold = float(
                    f.attrs.get('sparsity_threshold', 0.0))
                self.shape_dim = int(f.attrs.get('shape_dim', 63))

        print(f'SAEPipeline loaded: predictor={pred_path}, sae={sae_path}')

    def _load_pipeline_config(self):
        """Load SAE pipeline YAML config if it exists."""
        if not self.config_path or not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f'Invalid SAE pipeline config: {self.config_path}')
        return data

    def _find_latest_dir(self, prefix):
        """Find the latest auto-numbered results directory."""
        base = self.weights_base_dir
        num = 1
        latest = None
        while True:
            d = os.path.join(base, f'{prefix}_{num}')
            if os.path.exists(d):
                latest = d
                num += 1
            else:
                break
        return latest

    def _resolve_result_dir(self, configured_dir, prefix):
        """Resolve configured result dir, or fall back to latest auto-numbered."""
        if configured_dir:
            if os.path.isabs(configured_dir):
                return configured_dir
            return os.path.join(self.weights_base_dir, configured_dir)
        return self._find_latest_dir(prefix)

    def reconstruct(self, Uel: np.ndarray, ref_data: dict,
                    level: int) -> np.ndarray:
        """Reconstruct segmentation from voltage measurements."""
        return self.reconstruct_batch([Uel], ref_data, level)[0]

    def _prepare_input(self, Uel: np.ndarray, ref_data: dict, level: int):
        """Prepare one flattened masked input sample."""
        Injref = ref_data['Injref']
        Uelref = np.asarray(ref_data['Uelref']).reshape(-1)
        vincl = self.create_vincl(level, Injref).T.flatten()

        y = np.asarray(Uel).reshape(-1) - Uelref
        y[~vincl] = 0.0
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def reconstruct_batch(self, Uel_list, ref_data, level):
        """Batch reconstruction for efficiency."""
        y_batch = [self._prepare_input(Uel, ref_data, level)
                   for Uel in Uel_list]
        y_tensor = torch.from_numpy(np.stack(y_batch).astype(np.float32)).to(
            self.device)
        level_tensor = torch.full(
            (y_tensor.shape[0],), level, device=self.device)

        with torch.no_grad():
            with self._autocast_context():
                z_shape, angle_xy = self.predictor(y_tensor)

                # Soft threshold
                if self.sparsity_threshold > 0:
                    mask = (z_shape.abs() > self.sparsity_threshold).float()
                    z_shape = z_shape * mask

                # Decode
                canonical_logits = self.decoder(z_shape)  # (B, 3, 256, 256)

                # Rotate by +θ
                theta = torch.atan2(angle_xy[:, 1], angle_xy[:, 0])
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                zeros = torch.zeros_like(theta)
                row1 = torch.stack([cos_t, -sin_t, zeros], dim=1)
                row2 = torch.stack([sin_t, cos_t, zeros], dim=1)
                affine = torch.stack([row1, row2], dim=1)
                grid = F.affine_grid(affine, canonical_logits.shape,
                                     align_corners=False)
                logits = F.grid_sample(canonical_logits, grid,
                                       mode='bilinear',
                                       padding_mode='zeros',
                                       align_corners=False)

                pred = logits.argmax(dim=1)  # (B, 256, 256)

        pred_np = pred.cpu().numpy().astype(int)
        return [arr for arr in pred_np]
