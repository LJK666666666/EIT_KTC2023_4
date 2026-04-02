"""
SAE inference pipeline for EIT reconstruction.

Loads MLP predictor + SAE decoder. Reconstructs by:
1. Preprocess measurements (subtract ref, apply vincl)
2. MLP → (z_shape, angle_xy)
3. Soft threshold z_shape
4. Decoder(z_shape) → canonical logits
5. Rotate by +θ → argmax → segmentation
"""

import glob
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .base_pipeline import BasePipeline
from ..models.sae import SparseAutoEncoder, MeasurementPredictor


class SAEPipeline(BasePipeline):

    def __init__(self, device='cuda', weights_base_dir='results'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.predictor = None
        self.decoder = None
        self.sparsity_threshold = 0.0
        self.shape_dim = 63

    def load_model(self, level: int) -> None:
        """Load predictor MLP + SAE decoder from results directory."""
        if self.predictor is not None:
            return  # Already loaded (level-independent)

        # Find the latest sae_predictor result directory
        predictor_dir = self._find_latest_dir('sae_predictor_baseline')
        sae_dir = self._find_latest_dir('sae_baseline')

        if predictor_dir is None or sae_dir is None:
            raise FileNotFoundError(
                'Cannot find SAE weight directories. '
                f'Searched in {self.weights_base_dir}')

        # Load predictor
        pred_path = self._find_weight([
            os.path.join(predictor_dir, 'best.pt'),
            os.path.join(predictor_dir, 'last.pt'),
        ])
        pred_state = self._load_state_dict(pred_path, self.device)
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

    def reconstruct(self, Uel: np.ndarray, ref_data: dict,
                    level: int) -> np.ndarray:
        """Reconstruct segmentation from voltage measurements."""
        Injref = ref_data['Injref']
        Uelref = np.array(ref_data['Uelref']).flatten()

        # Difference measurements
        y_diff = (Uel - Uelref).flatten()

        # Apply vincl mask
        vincl = self.create_vincl(level, Injref).T.flatten()
        y_diff[~vincl] = 0.0

        # To tensor
        y_tensor = torch.from_numpy(
            y_diff.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with self._autocast_context():
                z_shape, angle_xy = self.predictor(y_tensor)

                # Soft threshold
                if self.sparsity_threshold > 0:
                    mask = (z_shape.abs() > self.sparsity_threshold).float()
                    z_shape = z_shape * mask

                # Decode
                canonical_logits = self.decoder(z_shape)  # (1, 3, 256, 256)

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

                pred = logits.argmax(dim=1)  # (1, 256, 256)

        return pred.squeeze(0).cpu().numpy().astype(int)

    def reconstruct_batch(self, Uel_list, ref_data, level):
        """Batch reconstruction for efficiency."""
        return [self.reconstruct(Uel, ref_data, level)
                for Uel in Uel_list]
