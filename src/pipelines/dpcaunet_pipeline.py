"""
DPCAUNet inference pipeline.

Reconstructs EIT images using cross-attention + dual-pooling UNet.
Same data flow as FCUNet: Uel → subtract reference → vincl mask → model → argmax.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .base_pipeline import BasePipeline
from ..configs.dpcaunet_config import get_configs as get_dpcaunet_config
from ..models.dpcaunet import DPCAUNet


class DPCAUNetPipeline(BasePipeline):
    """DPCA-UNet EIT reconstruction pipeline."""

    def __init__(self, device='cuda', weights_base_dir='results'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_dpcaunet_config()
        self._model_loaded = False

    def load_model(self, level: int) -> None:
        # Same model for all levels (level is encoded via embedding)
        if self._model_loaded:
            return

        model = DPCAUNet(
            n_channels=self.config.model.n_channels,
            n_patterns=self.config.model.n_patterns,
            d_model=self.config.model.d_model,
            n_heads=self.config.model.n_heads,
            im_size=self.config.data.im_size,
            encoder_channels=tuple(self.config.model.encoder_channels),
            out_channels=self.config.model.out_channels,
            max_period=self.config.model.max_period,
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

    def reconstruct(self, Uel: np.ndarray, ref_data: dict,
                    level: int) -> np.ndarray:
        return self.reconstruct_batch([Uel], ref_data, level)[0]

    def reconstruct_batch(self, Uels, ref_data: dict, level: int):
        """Batch reconstruction."""
        Injref = ref_data['Injref']
        Uelref = ref_data['Uelref']

        vincl = self.create_vincl(level, Injref).T.flatten()
        y_batch = []
        for Uel in Uels:
            y = np.array(Uel) - np.array(Uelref)
            y[~vincl] = 0
            y_batch.append(np.asarray(y).reshape(-1))

        y_tensor = torch.from_numpy(np.stack(y_batch)).float().to(self.device)
        level_tensor = torch.full(
            (y_tensor.shape[0],), level, device=self.device)

        with torch.no_grad():
            pred = self.model(y_tensor, level_tensor)
            pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1).cpu().numpy()

        return [arr.astype(int) for arr in pred_argmax]
