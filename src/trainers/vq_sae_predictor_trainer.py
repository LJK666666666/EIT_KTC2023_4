"""Trainer for measurement -> discrete slot predictor."""

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from scipy.io import loadmat
from torch.utils.data import DataLoader
import yaml

from .base_trainer import BaseTrainer
from ..configs.vq_sae_predictor_config import (
    get_configs as get_vq_sae_predictor_config,
)
from ..data import VQSAEPredictorHDF5Dataset
from ..ktc_methods import EITFEM, load_mesh
from ..models.vq_sae import ST1DVQVAE, VQMeasurementPredictor
from ..utils.measurement import create_vincl


class VQSAEPredictorTrainer(BaseTrainer):
    def __init__(self, config=None, experiment_name='vq_sae_predictor_baseline'):
        if config is None:
            config = get_vq_sae_predictor_config()
        super().__init__(config, experiment_name)
        self.vincl_dict = None
        self.vq_sae_model = None

    def build_model(self):
        model = VQMeasurementPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            num_slots=self.config.model.num_slots,
            codebook_size=self.config.model.codebook_size,
            dropout=self.config.model.dropout,
        )
        model.to(self.device)
        self.model = model

        if not self.config.vq_sae.checkpoint:
            raise ValueError(
                'VQ SAE predictor requires vq_sae.checkpoint / --vq-sae-checkpoint')
        self._load_vq_sae_model(self.config.vq_sae.checkpoint)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.get('weight_decay', 1e-4),
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'VQMeasurementPredictor: {total_params / 1e6:.2f}M parameters')

    def _load_vq_sae_model(self, ckpt_path):
        state = torch.load(ckpt_path, map_location=self.device,
                           weights_only=False)
        sd = state.get('model_state_dict', state)
        cfg_path = os.path.join(os.path.dirname(ckpt_path), 'config.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.unsafe_load(f)
        model_cfg = cfg['model']
        vq_sae = ST1DVQVAE(
            in_channels=model_cfg['in_channels'],
            encoder_channels=tuple(model_cfg['encoder_channels']),
            num_slots=model_cfg['num_slots'],
            codebook_size=model_cfg['codebook_size'],
            code_dim=model_cfg['code_dim'],
            decoder_start_size=model_cfg['decoder_start_size'],
            vq_beta=cfg['training']['vq_beta'],
        )
        vq_sae.load_state_dict(sd)
        vq_sae.to(self.device)
        vq_sae.eval()
        for p in vq_sae.parameters():
            p.requires_grad = False
        self.vq_sae_model = vq_sae
        print(f'Loaded frozen VQ SAE from {ckpt_path}')

    def build_datasets(self):
        if not self.config.data.get('use_hdf5', False):
            raise ValueError('VQ SAE predictor requires HDF5 dataset.')

        y_ref = loadmat(self.config.data.ref_path)
        Injref = y_ref['Injref']
        Mpat = y_ref['Mpat']
        mesh, mesh2 = load_mesh(self.config.data.mesh_name)
        nel = 32
        z = 1e-6 * np.ones((nel, 1))
        vincl = np.ones((nel - 1, 76), dtype=bool)
        solver = EITFEM(mesh2, Injref, Mpat, vincl)
        solver.SetInvGamma(
            self.config.data.noise_std1,
            self.config.data.noise_std2,
            y_ref['Uelref'],
        )
        sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.745
        Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).flatten()

        train_ds = VQSAEPredictorHDF5Dataset(
            self.config.data.hdf5_path,
            self.config.vq_sae.latent_h5_path,
            Uelref,
            solver.InvLn,
            indices=self.config.data.get('train_indices', None),
            augment_noise=True,
            augment_rotation=True,
        )
        self._warn_if_dropping_last_batch(
            'train', len(train_ds), self.config.training.batch_size)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=self._use_static_batches(),
            pin_memory=self._pin_memory_enabled(),
            num_workers=self.config.training.num_workers,
        )

        self.val_sim_loader = None
        val_indices = self.config.data.get('val_indices', None)
        if val_indices is not None:
            val_ds = VQSAEPredictorHDF5Dataset(
                self.config.data.hdf5_path,
                self.config.vq_sae.latent_h5_path,
                Uelref,
                solver.InvLn,
                indices=val_indices,
                augment_noise=False,
                augment_rotation=False,
            )
            self._warn_if_dropping_last_batch(
                'val', len(val_ds), self.config.training.batch_size)
            self.val_sim_loader = DataLoader(
                val_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers,
            )

        self.vincl_dict = {}
        for lvl in range(1, 8):
            self.vincl_dict[lvl] = create_vincl(lvl, Injref).T.flatten()

    def _slot_ce_loss(self, slot_logits, target_indices):
        bsz, num_slots, codebook_size = slot_logits.shape
        flat_logits = slot_logits.reshape(bsz * num_slots, codebook_size)
        flat_target = target_indices.reshape(bsz * num_slots)
        return F.cross_entropy(flat_logits, flat_target)

    def train_step(self, batch):
        measurements, target_indices, target_angle = batch

        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            levels = np.full(measurements.shape[0], fixed_level)
        else:
            levels = np.random.choice(np.arange(1, 8), size=measurements.shape[0])
        for k in range(measurements.shape[0]):
            measurements[k, ~self.vincl_dict[levels[k]]] = 0.0

        measurements = measurements.to(self.device)
        target_indices = target_indices.to(self.device)
        target_angle = target_angle.to(self.device)

        self.optimizer.zero_grad()
        with self._autocast_context():
            slot_logits, pred_angle = self.model(measurements)
            slot_loss = self._slot_ce_loss(slot_logits, target_indices)
            angle_loss = F.mse_loss(pred_angle, target_angle)
            total_loss = slot_loss + self.config.training.lambda_angle * angle_loss

        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)

        return {
            'loss': total_loss.item(),
            'slot_loss': slot_loss.item(),
            'angle_loss': angle_loss.item(),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}

        totals = {'loss': 0.0, 'slot_loss': 0.0, 'angle_loss': 0.0}
        num_samples = 0
        fixed_level = self.config.training.get('fixed_level', None) or 1
        for measurements, target_indices, target_angle in self.val_sim_loader:
            for k in range(measurements.shape[0]):
                measurements[k, ~self.vincl_dict[fixed_level]] = 0.0

            measurements = measurements.to(self.device)
            target_indices = target_indices.to(self.device)
            target_angle = target_angle.to(self.device)

            with torch.no_grad():
                with self._autocast_context():
                    slot_logits, pred_angle = self.model(measurements)
                    slot_loss = self._slot_ce_loss(slot_logits, target_indices)
                    angle_loss = F.mse_loss(pred_angle, target_angle)
                    total_loss = slot_loss + (
                        self.config.training.lambda_angle * angle_loss)
            batch_size = measurements.shape[0]
            totals['loss'] += total_loss.item() * batch_size
            totals['slot_loss'] += slot_loss.item() * batch_size
            totals['angle_loss'] += angle_loss.item() * batch_size
            num_samples += batch_size

        metrics = {
            'val_loss': totals['loss'] / max(num_samples, 1),
            'val_slot_loss': totals['slot_loss'] / max(num_samples, 1),
            'val_angle_loss': totals['angle_loss'] / max(num_samples, 1),
        }
        if self.writer:
            self.writer.add_scalar('val/loss', metrics['val_loss'], epoch + 1)
            self.writer.add_scalar('val/slot_loss', metrics['val_slot_loss'], epoch + 1)
            self.writer.add_scalar('val/angle_loss', metrics['val_angle_loss'], epoch + 1)
        print(f'  Val loss: {metrics["val_loss"]:.6f}')
        return metrics

