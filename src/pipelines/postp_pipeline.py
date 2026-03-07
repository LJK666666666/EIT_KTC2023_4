"""
Post-processing UNet inference pipeline.

Performs linearised reconstruction with 5 different regularisation parameter
sets, then feeds the stacked reconstructions through an OpenAI UNet to produce
a segmentation map.

Reference: programs/ktc2023_postprocessing/main.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from .base_pipeline import BasePipeline
from ..configs import get_postp_config
from ..configs.alphas import POSTP_LEVEL_TO_ALPHAS
from ..models.openai_unet import OpenAiUNetModel
from ..reconstruction.linearised_reco import LinearisedRecoFenics


class PostPPipeline(BasePipeline):
    """Post-processing UNet EIT reconstruction pipeline.

    Flow: Uel -> 5 linearised reconstructions -> interpolate to 256x256
          -> stack as 5-channel input -> UNet -> softmax -> argmax
    """

    def __init__(self, device='cuda', weights_base_dir='KTC2023_SubmissionFiles'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_postp_config()
        self._model_loaded = False
        self.reconstructor = None
        self._current_level = None

    def load_model(self, level: int) -> None:
        # Load UNet model (same weights for all levels)
        if not self._model_loaded:
            model = OpenAiUNetModel(
                image_size=self.config.data.im_size,
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                marginal_prob_std=None,
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

            weight_path = f'{self.weights_base_dir}/postprocessing_model/version_01/model.pt'
            model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
            model.eval()
            model.to(self.device)

            self.model = model
            self._model_loaded = True

    def _setup_reconstructor(self, level: int, ref_data: dict) -> None:
        """Create the linearised reconstructor for the given level."""
        if self._current_level == level and self.reconstructor is not None:
            return

        Uelref = ref_data['Uelref']
        Mpat = ref_data['Mpat']
        Injref = ref_data['Injref']
        B = Mpat.T

        vincl_level = self.create_vincl(level, Injref)

        self.reconstructor = LinearisedRecoFenics(
            Uelref, B, vincl_level,
            mesh_name='sparse',
            base_path=f'{self.weights_base_dir}/data',
        )
        self._current_level = level

    def reconstruct(self, Uel: np.ndarray, ref_data: dict, level: int) -> np.ndarray:
        self._setup_reconstructor(level, ref_data)

        alphas = POSTP_LEVEL_TO_ALPHAS[level]

        # Get 5 initial reconstructions with different regularisation parameters
        delta_sigma_list = self.reconstructor.reconstruct_list(Uel, alphas)

        # Interpolate each to 256x256 and stack
        sigma_images = [self.reconstructor.interpolate_to_image(ds) for ds in delta_sigma_list]
        sigma_reco = np.stack(sigma_images)  # (5, 256, 256)

        reco = torch.from_numpy(sigma_reco).float().to(self.device).unsqueeze(0)  # (1, 5, 256, 256)
        level_tensor = torch.tensor([level]).to(self.device)

        with torch.no_grad():
            pred = self.model(reco, level_tensor)
            pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1).cpu().numpy()[0, :, :]

        return pred_argmax.astype(int)
