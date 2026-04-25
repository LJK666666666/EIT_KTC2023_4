"""Pipeline for atlas-aware direct residual conductivity prediction."""

import os

import numpy as np

from .base_pipeline import BasePipeline
from ..configs import get_atlas_sigma_predictor_config
from ..models.dct_predictor import AtlasResidualDecoderPredictor


class AtlasSigmaPredictorPipeline(BasePipeline):
    def __init__(self, device="cuda", weights_base_dir="results"):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.config = get_atlas_sigma_predictor_config()

    def _load_atlas(self, weights_dir: str):
        atlas_path = os.path.join(weights_dir, "atlas.npy")
        if not os.path.exists(atlas_path):
            raise FileNotFoundError(f"Missing atlas.npy under {weights_dir}")
        return np.load(atlas_path)

    def load_model(self, level):
        weight_path = self._find_weight([
            os.path.join(self.weights_base_dir, "best.pt"),
            os.path.join(self.weights_base_dir, "last.pt"),
        ])
        weights_dir = os.path.dirname(weight_path)
        model = AtlasResidualDecoderPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            out_channels=self.config.model.out_channels,
            dropout=self.config.model.dropout,
            image_size=self.config.model.get("image_size", 256),
            seed_channels=self.config.model.get("seed_channels", 32),
            seed_size=self.config.model.get("seed_size", 16),
        )
        model.set_atlas(self._load_atlas(weights_dir))
        state = self._load_state_dict(weight_path, self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        self.weights_base_dir = weights_dir

    def _prepare_measurements(self, measurements, ref_data, level):
        y = np.array(measurements, dtype=np.float32).reshape(-1)
        uelref = np.asarray(ref_data["Uelref"], dtype=np.float32).reshape(-1)
        injref = ref_data["Injref"]
        y_diff = y - uelref
        vincl = self.create_vincl(level, injref).T.flatten()
        y_diff[~vincl] = 0.0
        return y_diff

    def reconstruct(self, Uel, ref_data, level):
        import torch
        y = self._prepare_measurements(Uel, ref_data, level)
        y_t = torch.from_numpy(y).unsqueeze(0).to(self.device)
        level_t = torch.ones(1, device=self.device, dtype=torch.long)
        with torch.no_grad():
            with self._autocast_context():
                pred = self.model(y_t, level_t)
        return pred.squeeze(0).squeeze(0).float().cpu().numpy()

    def reconstruct_batch(self, measurements_batch, ref_data, level):
        import torch
        ys = [self._prepare_measurements(m, ref_data, level) for m in measurements_batch]
        y_t = torch.from_numpy(np.stack(ys, axis=0)).to(self.device)
        level_t = torch.ones(y_t.shape[0], device=self.device, dtype=torch.long)
        with torch.no_grad():
            with self._autocast_context():
                pred = self.model(y_t, level_t)
        return pred.squeeze(1).float().cpu().numpy()
