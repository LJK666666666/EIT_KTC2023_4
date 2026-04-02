"""
FCUNet inference pipeline.

Reconstructs EIT images by directly mapping voltage measurements to
segmentation maps through a fully-connected UNet.

Reference: programs/ktc2023_fcunet/main.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from .base_pipeline import BasePipeline
from ..configs import get_fcunet_config
from ..models.fcunet import FCUNet


class FCUNetPipeline(BasePipeline):
    """FCUNet-based EIT reconstruction pipeline.

    Flow: Uel -> subtract reference -> apply vincl mask -> FCUNet -> softmax -> argmax
    """

    def __init__(self, device='cuda', weights_base_dir='KTC2023_SubmissionFiles'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_fcunet_config()
        self._model_loaded = False

    def load_model(self, level: int) -> None:
        # FCUNet uses the same model for all levels
        if self._model_loaded:
            return

        model = FCUNet(
            image_size=self.config.data.im_size,
            in_channels=self.config.model.in_channels,
            model_channels=self.config.model.model_channels,
            out_channels=self.config.model.out_channels,
            num_res_blocks=self.config.model.num_res_blocks,
            attention_resolutions=self.config.model.attention_resolutions,
            channel_mult=self.config.model.channel_mult,
            conv_resample=self.config.model.conv_resample,
            dims=self.config.model.dims,
            num_heads=self.config.model.num_heads,
            num_head_channels=self.config.model.num_head_channels,
            num_heads_upsample=self.config.model.num_heads_upsample,
            use_scale_shift_norm=self.config.model.use_scale_shift_norm,
            resblock_updown=self.config.model.resblock_updown,
            use_new_attention_order=self.config.model.use_new_attention_order,
            max_period=self.config.model.max_period,
        )

        weight_path = self._find_weight([
            # Training checkpoint (best/last)
            f'{self.weights_base_dir}/best.pt',
            f'{self.weights_base_dir}/last.pt',
            # Original submission format
            f'{self.weights_base_dir}/fcunet_model/model.pt',
        ])
        model.load_state_dict(self._load_state_dict(weight_path, self.device))
        model.eval()
        model.to(self.device)

        self.model = model
        self._model_loaded = True

    def reconstruct(self, Uel: np.ndarray, ref_data: dict, level: int) -> np.ndarray:
        return self.reconstruct_batch([Uel], ref_data, level)[0]

    def _prepare_input(self, Uel: np.ndarray, ref_data: dict, level: int):
        """Prepare one flattened masked input sample."""
        Injref = ref_data['Injref']
        Uelref = np.asarray(ref_data['Uelref']).reshape(-1)
        vincl = self.create_vincl(level, Injref).T.flatten()

        y = np.asarray(Uel).reshape(-1) - Uelref
        y[~vincl] = 0
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def reconstruct_batch(self, Uels, ref_data: dict, level: int):
        """Batch reconstruction for identical results with lower overhead."""
        y_batch = [self._prepare_input(Uel, ref_data, level) for Uel in Uels]

        y_tensor = torch.from_numpy(np.stack(y_batch)).float().to(self.device)
        level_tensor = torch.full(
            (y_tensor.shape[0],), level, device=self.device)

        with torch.no_grad():
            pred = self.model(y_tensor, level_tensor)
            pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1).cpu().numpy()

        return [arr.astype(int) for arr in pred_argmax]

    def reconstruct_mixed_batch(self, samples):
        """Batch reconstruction across different levels."""
        y_batch = [self._prepare_input(s['Uel'], s['ref_data'], s['level'])
                   for s in samples]
        levels = [s['level'] for s in samples]

        y_tensor = torch.from_numpy(np.stack(y_batch)).float().to(self.device)
        level_tensor = torch.tensor(levels, dtype=torch.float,
                                    device=self.device)

        with torch.no_grad():
            pred = self.model(y_tensor, level_tensor)
            pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1).cpu().numpy()

        return [arr.astype(int) for arr in pred_argmax]
