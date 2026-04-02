"""
SAE predictor trainer: Phase 3 — MLP maps measurements to latent codes.

Loss = MSE(z_shape) + λ_angle × MSE(angle_xy).
Frozen SAE decoder used for validation visualization.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from scipy.io import loadmat
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..configs.sae_predictor_config import get_configs as get_sae_predictor_config
from ..models.sae import SparseAutoEncoder, MeasurementPredictor
from ..data import SAEPredictorHDF5Dataset
from ..utils.measurement import create_vincl


class SAEPredictorTrainer(BaseTrainer):
    """Phase 3 trainer: MLP predictor for SAE latent codes."""

    def __init__(self, config=None, experiment_name='sae_predictor_baseline'):
        if config is None:
            config = get_sae_predictor_config()
        super().__init__(config, experiment_name)

        self.vincl_dict = None
        self.sae_decoder = None
        self.sparsity_threshold = 0.0

    def build_model(self):
        # Build predictor MLP
        model = MeasurementPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            shape_dim=self.config.model.shape_dim,
            dropout=self.config.model.dropout,
        )
        model.to(self.device)
        self.model = model

        wd = self.config.training.get('weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.training.lr,
            weight_decay=wd)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor)

        total_params = sum(p.numel() for p in model.parameters())
        print(f'MeasurementPredictor: {total_params / 1e6:.2f}M parameters')

        # Load frozen SAE decoder for validation visualization
        sae_ckpt = self.config.sae.checkpoint
        if sae_ckpt and os.path.exists(sae_ckpt):
            self._load_sae_decoder(sae_ckpt)

    def _load_sae_decoder(self, ckpt_path):
        """Load SAE decoder (frozen) for visualization during validation."""
        state = torch.load(ckpt_path, map_location=self.device,
                           weights_only=False)
        sd = state.get('model_state_dict', state)

        # Build SAE with default config to load decoder weights
        sae = SparseAutoEncoder()
        sae.load_state_dict(sd)
        sae.to(self.device)
        sae.eval()
        for p in sae.parameters():
            p.requires_grad = False

        self.sae_decoder = sae.decoder
        print(f'Loaded frozen SAE decoder from {ckpt_path}')

        # Load sparsity threshold from latent_codes.h5
        import h5py
        latent_path = self.config.sae.latent_h5_path
        if latent_path and os.path.exists(latent_path):
            with h5py.File(latent_path, 'r') as f:
                self.sparsity_threshold = float(
                    f.attrs.get('sparsity_threshold', 0.0))
            print(f'Sparsity threshold: {self.sparsity_threshold:.6f}')

    def build_datasets(self):
        use_hdf5 = self.config.data.get('use_hdf5', False)
        if not use_hdf5:
            raise ValueError('SAE predictor requires HDF5 dataset.')

        ref_path = self.config.data.ref_path
        mesh_name = self.config.data.mesh_name

        from ..ktc_methods import EITFEM, load_mesh
        y_ref = loadmat(ref_path)
        Injref = y_ref['Injref']
        Mpat = y_ref['Mpat']
        mesh, mesh2 = load_mesh(mesh_name)

        Nel = 32
        z = 1e-6 * np.ones((Nel, 1))
        vincl = np.ones((Nel - 1, 76), dtype=bool)

        solver = EITFEM(mesh2, Injref, Mpat, vincl)
        solver.SetInvGamma(
            self.config.data.noise_std1,
            self.config.data.noise_std2,
            y_ref['Uelref'])

        sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.745
        Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).flatten()

        h5_path = self.config.data.hdf5_path
        latent_h5_path = self.config.sae.latent_h5_path
        train_indices = self.config.data.get('train_indices', None)

        dataset = SAEPredictorHDF5Dataset(
            h5_path, latent_h5_path, Uelref, solver.InvLn,
            indices=train_indices, augment_noise=True,
            augment_rotation=True)

        self._warn_if_dropping_last_batch(
            'train', len(dataset), self.config.training.batch_size)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=self._use_static_batches(),
            pin_memory=self._pin_memory_enabled(),
            num_workers=self.config.training.num_workers)

        # Validation loader (no augmentation)
        self.val_sim_loader = None
        val_indices = self.config.data.get('val_indices', None)
        if val_indices is not None:
            val_ds = SAEPredictorHDF5Dataset(
                h5_path, latent_h5_path, Uelref, solver.InvLn,
                indices=val_indices, augment_noise=False,
                augment_rotation=False)
            self._warn_if_dropping_last_batch(
                'val', len(val_ds), self.config.training.batch_size)
            self.val_sim_loader = DataLoader(
                val_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers)

        # Vincl masks for level augmentation
        self.vincl_dict = {}
        for lvl in range(1, 8):
            self.vincl_dict[lvl] = create_vincl(lvl, Injref).T.flatten()

        self.val_data = None

    def train_step(self, batch):
        measurements, target_z = batch

        # Level augmentation: apply vincl mask
        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            levels = np.full(measurements.shape[0], fixed_level)
        else:
            levels = np.random.choice(np.arange(1, 8),
                                      size=measurements.shape[0])
        for k in range(measurements.shape[0]):
            measurements[k, ~self.vincl_dict[levels[k]]] = 0.0

        measurements = measurements.to(self.device)
        target_z = target_z.to(self.device)

        shape_dim = self.config.model.shape_dim
        target_shape = target_z[:, :shape_dim]
        target_angle = target_z[:, shape_dim:]

        self.optimizer.zero_grad()

        with self._autocast_context():
            pred_shape, pred_angle = self.model(measurements)

            mse_shape = F.mse_loss(pred_shape, target_shape)
            mse_angle = F.mse_loss(pred_angle, target_angle)
            lambda_angle = self.config.training.lambda_angle
            total_loss = mse_shape + lambda_angle * mse_angle

        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)

        return {
            'loss': total_loss.item(),
            'mse_shape': mse_shape.item(),
            'mse_angle': mse_angle.item(),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}

        shape_dim = self.config.model.shape_dim
        lambda_angle = self.config.training.lambda_angle
        total_loss = 0.0
        num_samples = 0

        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is None:
            fixed_level = 1

        for measurements, target_z in self.val_sim_loader:
            for k in range(measurements.shape[0]):
                measurements[k, ~self.vincl_dict[fixed_level]] = 0.0

            measurements = measurements.to(self.device)
            target_z = target_z.to(self.device)
            target_shape = target_z[:, :shape_dim]
            target_angle = target_z[:, shape_dim:]

            with torch.no_grad():
                with self._autocast_context():
                    pred_shape, pred_angle = self.model(measurements)
                    loss = (F.mse_loss(pred_shape, target_shape)
                            + lambda_angle * F.mse_loss(pred_angle,
                                                        target_angle))
            total_loss += loss.item() * measurements.shape[0]
            num_samples += measurements.shape[0]

        avg_loss = total_loss / max(num_samples, 1)
        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, epoch + 1)
        print(f'  Val loss: {avg_loss:.6f}')
        return {'val_loss': avg_loss}
