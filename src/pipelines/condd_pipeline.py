"""
Conditional Diffusion inference pipeline.

Performs linearised reconstruction, then uses a conditional DDPM score model
with DDIM sampling to generate multiple segmentation samples, and produces
a final segmentation via majority voting.

Reference: programs/ktc2023_conditional_diffusion/main.py
"""

import os
import numpy as np
import torch
from scipy.stats import mode

from .base_pipeline import BasePipeline
from ..configs import get_condd_config
from ..configs.condd_config import LEVEL_TO_ALPHAS, LEVEL_TO_HPARAMS
from ..diffusion.exp_utils import get_standard_score, get_standard_sde
from ..diffusion.ema import ExponentialMovingAverage
from ..samplers import BaseSampler, wrapper_ddim
from ..reconstruction.linearised_reco import LinearisedRecoFenics


class CondDPipeline(BasePipeline):
    """Conditional Diffusion EIT reconstruction pipeline.

    Flow: Uel -> 5 linearised reconstructions -> interpolate to 256x256
          -> DDIM conditional sampling (N samples) -> majority vote -> segmentation
    """

    def __init__(self, device='cuda', weights_base_dir='KTC2023_SubmissionFiles',
                 batch_mode=True, seed=42):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_condd_config()
        self.batch_mode = batch_mode
        self.seed = seed
        self.reconstructor = None
        self._current_level = None
        self.sampler = None

    def load_model(self, level: int) -> None:
        """Load diffusion model for the given level.

        Handles the EMA-only weight loading for levels 1/5/6 where only
        ema_model_training.pt is available.
        """
        torch.manual_seed(self.seed + level)

        sde = get_standard_sde(config=self.config)
        score = get_standard_score(config=self.config, sde=sde, use_ema=False, load_model=False)

        hparams = LEVEL_TO_HPARAMS[level]
        model_dir = f'{self.weights_base_dir}/diffusion_models/level_{level}/version_01'

        model_path = os.path.join(model_dir, 'model_training.pt')
        ema_path = os.path.join(model_dir, 'ema_model_training.pt')

        if os.path.exists(model_path):
            # Normal loading: load model weights, optionally apply EMA
            score.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            if hparams['use_ema'] and os.path.exists(ema_path):
                ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
                ema.load_state_dict(torch.load(ema_path, map_location='cpu', weights_only=True))
                ema.copy_to(score.parameters())
        elif os.path.exists(ema_path):
            # EMA-only loading (levels 1/5/6): initialise random model, load EMA, copy to model
            ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
            ema.load_state_dict(torch.load(ema_path, map_location='cpu', weights_only=True))
            ema.copy_to(score.parameters())
        else:
            raise FileNotFoundError(
                f'No model weights found for level {level} in {model_dir}'
            )

        score.to(self.device)
        score.eval()
        self.model = score

        # Set up sampler
        num_samples = hparams['num_samples']
        batch_size = num_samples if self.batch_mode else 1

        self.sampler = BaseSampler(
            score=score,
            sde=sde,
            predictor=wrapper_ddim,
            sample_kwargs={
                'num_steps': hparams['num_steps'],
                'batch_size': batch_size,
                'im_shape': [1, 256, 256],
                'travel_length': 1,
                'travel_repeat': 1,
                'predictor': {'eta': hparams['eta']},
            },
            device=self.device,
        )
        self._current_level = level

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

    def reconstruct(self, Uel: np.ndarray, ref_data: dict, level: int) -> np.ndarray:
        self._setup_reconstructor(level, ref_data)

        alphas = LEVEL_TO_ALPHAS[level]
        hparams = LEVEL_TO_HPARAMS[level]

        # Get 5 initial reconstructions
        delta_sigma_list = self.reconstructor.reconstruct_list(Uel, alphas)

        # Interpolate each to 256x256 and stack
        sigma_images = [self.reconstructor.interpolate_to_image(ds) for ds in delta_sigma_list]
        sigma_reco = np.stack(sigma_images)  # (5, 256, 256)

        reco = torch.from_numpy(sigma_reco).float().to(self.device).unsqueeze(0)  # (1, 5, 256, 256)

        num_samples = hparams['num_samples']

        # Conditional DDIM sampling
        if self.batch_mode:
            reco_batch = reco.repeat(num_samples, 1, 1, 1)  # (N, 5, 256, 256)
            x_mean = self.sampler.sample(reco_batch, logging=False)
        else:
            samples = []
            for _ in range(num_samples):
                x_ = self.sampler.sample(reco, logging=False)
                samples.append(x_)
            x_mean = torch.cat(samples)

        # Round and clip to valid range, then majority vote
        x_round = torch.round(x_mean).cpu().numpy()[:, 0, :, :]  # (N, 256, 256)
        x_round = np.clip(x_round, 0, 2)

        result = mode(x_round, axis=0, keepdims=False)[0]

        return result.astype(int)
