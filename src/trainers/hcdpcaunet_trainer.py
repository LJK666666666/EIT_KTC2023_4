"""
HC-DPCA-UNet trainer (same strategy as DPCAUNet).

Inherits all training logic including:
  - Dice + Focal loss
  - Grouped LR (attention 1/10)
  - Linear warmup + cosine annealing
  - Measurement normalization
  - Deep supervision with exponential weight decay
"""

import torch

from ..configs.hcdpcaunet_config import get_configs as get_hcdpcaunet_config
from ..models.hcdpcaunet import HCDPCAUNet
from .dpcaunet_trainer import DPCAUNetTrainer


class HCDPCAUNetTrainer(DPCAUNetTrainer):
    """Trainer for HC-DPCA-UNet.

    Inherits all training logic from DPCAUNetTrainer; only overrides
    build_model to instantiate the HC-DPCA-UNet architecture.
    """

    def __init__(self, config=None, experiment_name='hcdpcaunet_baseline'):
        if config is None:
            config = get_hcdpcaunet_config()
        super().__init__(config, experiment_name)

    def build_model(self):
        model = HCDPCAUNet(
            n_channels=self.config.model.n_channels,
            n_patterns=self.config.model.n_patterns,
            d_model=self.config.model.d_model,
            n_heads=self.config.model.n_heads,
            im_size=self.config.data.im_size,
            bottleneck_ch=self.config.model.bottleneck_ch,
            encoder_channels=tuple(self.config.model.encoder_channels),
            out_channels=self.config.model.out_channels,
            max_period=self.config.model.max_period,
            harmonic_L=self.config.model.harmonic_L,
            n_cascade_layers=self.config.model.n_cascade_layers,
        )
        model.to(self.device)
        self.model = model

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print(f'HCDPCAUNet parameters: {n_params:,}')

        # Placeholder (rebuilt per stage in train())
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.training.lr)
        self.scheduler = None
