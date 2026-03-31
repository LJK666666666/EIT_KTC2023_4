"""
SAE predictor trainer: Phase 3 — MLP maps measurements to image space.

The predictor outputs latent code, then a trainable SAE decoder reconstructs
logits. Training loss is image-only: CE + Dice.
"""

import os
import math

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
from ..losses.dice_focal import DiceLoss
from ..utils.measurement import create_vincl


class SAEPredictorTrainer(BaseTrainer):
    """Phase 3 trainer: MLP + SAE decoder with image-space supervision."""

    def __init__(self, config=None, experiment_name='sae_predictor_baseline'):
        if config is None:
            config = get_sae_predictor_config()
        super().__init__(config, experiment_name)

        self.vincl_dict = None
        self.sae_decoder = None
        self.sae_model = None
        self.sparsity_threshold = 0.0
        self.dice_loss = DiceLoss()

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

        total_params = sum(p.numel() for p in model.parameters())
        print(f'MeasurementPredictor: {total_params / 1e6:.2f}M parameters')
        if not self.config.sae.checkpoint:
            raise ValueError(
                'SAE predictor requires sae.checkpoint / --sae-checkpoint')

        sae_ckpt = self.config.sae.checkpoint
        if sae_ckpt and os.path.exists(sae_ckpt):
            self._load_sae_model(sae_ckpt)
        else:
            raise FileNotFoundError(f'SAE checkpoint not found: {sae_ckpt}')

        wd = self.config.training.get('weight_decay', 1e-4)
        trainable_params = list(model.parameters()) + list(self.sae_decoder.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.config.training.lr,
            weight_decay=wd)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor)

    def _load_sae_model(self, ckpt_path):
        """Load SAE and unfreeze decoder only."""
        state = torch.load(ckpt_path, map_location=self.device,
                           weights_only=False)
        sd = state.get('model_state_dict', state)

        sae = SparseAutoEncoder()
        sae.load_state_dict(sd)
        sae.to(self.device)

        for p in sae.angle_cnn.parameters():
            p.requires_grad = False
        for p in sae.encoder.parameters():
            p.requires_grad = False
        for p in sae.decoder.parameters():
            p.requires_grad = True

        sae.angle_cnn.eval()
        sae.encoder.eval()
        sae.decoder.train()

        self.sae_model = sae
        self.sae_decoder = sae.decoder
        print(f'Loaded SAE model from {ckpt_path} (decoder trainable)')

        # Load sparsity threshold from latent_codes.h5
        import h5py
        latent_path = self.config.sae.latent_h5_path
        if latent_path and os.path.exists(latent_path):
            with h5py.File(latent_path, 'r') as f:
                self.sparsity_threshold = float(
                    f.attrs.get('sparsity_threshold', 0.0))
            print(f'Sparsity threshold: {self.sparsity_threshold:.6f}')

    def _image_loss(self, logits, gt_indices):
        ce_weight = float(self.config.training.get('ce_weight', 1.0))
        dice_weight = float(self.config.training.get('dice_weight', 1.0))
        ce = F.cross_entropy(logits, gt_indices)
        dice = self.dice_loss(logits, gt_indices)
        total = ce_weight * ce + dice_weight * dice
        return total, ce, dice

    @staticmethod
    def _rotate_gt_indices(gt_indices, rot_steps):
        """Rotate label maps by k * 2π/32 using nearest interpolation."""
        if torch.max(rot_steps).item() == 0:
            return gt_indices

        gt_onehot = F.one_hot(gt_indices.long(), num_classes=3).permute(
            0, 3, 1, 2).float()
        angle = rot_steps.float() * (2 * math.pi / 32)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        zeros = torch.zeros_like(angle)
        row1 = torch.stack([cos_a, -sin_a, zeros], dim=1)
        row2 = torch.stack([sin_a, cos_a, zeros], dim=1)
        theta = torch.stack([row1, row2], dim=1)
        grid = F.affine_grid(theta, gt_onehot.shape, align_corners=False)
        gt_rot = F.grid_sample(
            gt_onehot, grid, mode='nearest',
            padding_mode='zeros', align_corners=False)
        return torch.argmax(gt_rot, dim=1).long()

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
        z = 1e-6 * np.ones(Nel)
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

    def get_checkpoint_extra(self):
        return {
            'sae_decoder_state_dict': self.sae_decoder.state_dict(),
        }

    def load_checkpoint_extra(self, state):
        decoder_sd = state.get('sae_decoder_state_dict', None)
        if decoder_sd is not None and self.sae_decoder is not None:
            self.sae_decoder.load_state_dict(decoder_sd)

    def train_step(self, batch):
        measurements, target_z, gt_indices, rot_steps = batch

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
        gt_indices = gt_indices.to(self.device)
        rot_steps = rot_steps.to(self.device)
        gt_indices = self._rotate_gt_indices(gt_indices, rot_steps)

        shape_dim = self.config.model.shape_dim
        target_shape = target_z[:, :shape_dim]
        target_angle = target_z[:, shape_dim:]

        self.optimizer.zero_grad()

        with self._autocast_context():
            pred_shape, pred_angle = self.model(measurements)
            mse_shape = F.mse_loss(pred_shape, target_shape)
            mse_angle = F.mse_loss(pred_angle, target_angle)
            decoded_logits = self.sae_model.decode(pred_shape, pred_angle)
            total_loss, ce_loss, dice_loss = self._image_loss(
                decoded_logits, gt_indices)

        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.sae_decoder.parameters()),
            grad_clip)
        self.optimizer_step(self.optimizer)

        return {
            'loss': total_loss.item(),
            'mse_shape': mse_shape.item(),
            'mse_angle': mse_angle.item(),
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}

        shape_dim = self.config.model.shape_dim
        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        total_mse_shape = 0.0
        total_mse_angle = 0.0
        num_samples = 0

        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is None:
            fixed_level = 1

        self.sae_decoder.eval()

        for measurements, target_z, gt_indices, rot_steps in self.val_sim_loader:
            for k in range(measurements.shape[0]):
                measurements[k, ~self.vincl_dict[fixed_level]] = 0.0

            measurements = measurements.to(self.device)
            target_z = target_z.to(self.device)
            gt_indices = gt_indices.to(self.device)
            rot_steps = rot_steps.to(self.device)
            gt_indices = self._rotate_gt_indices(gt_indices, rot_steps)
            target_shape = target_z[:, :shape_dim]
            target_angle = target_z[:, shape_dim:]

            with torch.no_grad():
                with self._autocast_context():
                    pred_shape, pred_angle = self.model(measurements)
                    decoded_logits = self.sae_model.decode(pred_shape, pred_angle)
                    total, ce_loss, dice_loss = self._image_loss(
                        decoded_logits, gt_indices)
                    mse_shape = F.mse_loss(pred_shape, target_shape)
                    mse_angle = F.mse_loss(pred_angle, target_angle)

            total_loss += total.item() * measurements.shape[0]
            total_ce_loss += ce_loss.item() * measurements.shape[0]
            total_dice_loss += dice_loss.item() * measurements.shape[0]
            total_mse_shape += mse_shape.item() * measurements.shape[0]
            total_mse_angle += mse_angle.item() * measurements.shape[0]
            num_samples += measurements.shape[0]

        avg_total_loss = total_loss / max(num_samples, 1)
        avg_ce_loss = total_ce_loss / max(num_samples, 1)
        avg_dice_loss = total_dice_loss / max(num_samples, 1)
        avg_mse_shape = total_mse_shape / max(num_samples, 1)
        avg_mse_angle = total_mse_angle / max(num_samples, 1)
        if self.writer:
            self.writer.add_scalar('val/loss', avg_total_loss, epoch + 1)
            self.writer.add_scalar('val/ce_loss', avg_ce_loss, epoch + 1)
            self.writer.add_scalar('val/dice_loss', avg_dice_loss, epoch + 1)
            self.writer.add_scalar('val/mse_shape', avg_mse_shape, epoch + 1)
            self.writer.add_scalar('val/mse_angle', avg_mse_angle, epoch + 1)
        print(f'  Val loss: {avg_total_loss:.6f} | '
              f'CE: {avg_ce_loss:.6f} | Dice: {avg_dice_loss:.6f}')
        self.sae_decoder.train()
        return {
            'val_loss': avg_total_loss,
            'val_ce_loss': avg_ce_loss,
            'val_dice_loss': avg_dice_loss,
            'val_mse_shape': avg_mse_shape,
            'val_mse_angle': avg_mse_angle,
            'val_total_loss': avg_total_loss,
        }
