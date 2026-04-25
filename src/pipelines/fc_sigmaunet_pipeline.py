"""FCUNet-style pipeline for continuous conductivity regression."""

import os

import numpy as np
import torch
import yaml

from .base_pipeline import BasePipeline
from ..configs import get_fc_sigmaunet_config
from ..models.fcunet import FCUNet


class FCSigmaUNetPipeline(BasePipeline):
    def __init__(self, device='cuda', weights_base_dir='results'):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_fc_sigmaunet_config()
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
        for key in (
            'in_channels', 'model_channels', 'out_channels', 'num_res_blocks',
            'attention_resolutions', 'channel_mult', 'conv_resample', 'dims',
            'num_heads', 'num_head_channels', 'num_heads_upsample',
            'use_scale_shift_norm', 'resblock_updown',
            'use_new_attention_order', 'max_period',
        ):
            if key in model_cfg:
                setattr(self.config.model, key, model_cfg[key])

    def load_model(self, level: int) -> None:
        if self._model_loaded:
            return
        self._load_runtime_config()
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
            f'{self.weights_base_dir}/best.pt',
            f'{self.weights_base_dir}/last.pt',
        ])
        model.load_state_dict(self._load_state_dict(weight_path, self.device))
        model.eval()
        model.to(self.device)
        self.model = model
        self._model_loaded = True

    def _prepare_input(self, Uel, ref_data, level):
        injref = ref_data['Injref']
        uelref = np.asarray(ref_data['Uelref']).reshape(-1)
        vincl = self.create_vincl(level, injref).T.flatten()
        y = np.asarray(Uel).reshape(-1) - uelref
        y[~vincl] = 0.0
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def reconstruct(self, Uel, ref_data, level):
        return self.reconstruct_batch([Uel], ref_data, level)[0]

    def reconstruct_batch(self, Uels, ref_data, level):
        y_batch = [self._prepare_input(Uel, ref_data, level) for Uel in Uels]
        y_tensor = torch.from_numpy(np.stack(y_batch)).float().to(self.device)
        level_tensor = torch.full(
            (y_tensor.shape[0],), level, dtype=torch.float, device=self.device)
        with torch.no_grad():
            with self._autocast_context():
                pred = self.model(y_tensor, level_tensor)
        pred_np = pred[:, 0].detach().float().cpu().numpy().astype(np.float32)
        return [arr for arr in pred_np]
