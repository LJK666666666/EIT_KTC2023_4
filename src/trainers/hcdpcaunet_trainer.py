"""
HC-DPCA-UNet trainer: three-stage training (same strategy as DPCAUNet).

Stage 1: Train attention modules + pretrain_head only.
Stage 2: Freeze attention, train UNet + aux heads with deep supervision.
Stage 3: Unfreeze all, linearly decay aux loss weights to 0.
"""

from ..configs.hcdpcaunet_config import get_configs as get_hcdpcaunet_config
from ..models.hcdpcaunet import HCDPCAUNet
from .dpcaunet_trainer import DPCAUNetTrainer


class HCDPCAUNetTrainer(DPCAUNetTrainer):
    """Three-stage trainer for HC-DPCA-UNet.

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

        import torch
        import torch.optim.lr_scheduler as lr_scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.training.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor)

    def _attention_params(self):
        """Return parameters belonging to the attention modules."""
        names = ['electrode_encoder', 'spatial_query', 'cascaded_attn',
                 'level_embed']
        params = []
        for name, p in self.model.named_parameters():
            if any(name.startswith(n) for n in names):
                params.append(p)
        return params

    def _freeze_unet(self):
        """Freeze everything except attention + pretrain_head."""
        attn_names = ['electrode_encoder', 'spatial_query', 'cascaded_attn',
                      'level_embed', 'pretrain_head']
        for name, p in self.model.named_parameters():
            if not any(name.startswith(n) for n in attn_names):
                p.requires_grad = False
