"""
DPCAUNet trainer: single-stage training for cross-attention EIT reconstruction.

Uses CrossEntropyLoss on 3-class segmentation, same data pipeline as FCUNet.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from scipy.io import loadmat
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..configs.dpcaunet_config import get_configs as get_dpcaunet_config
from ..models.dpcaunet import DPCAUNet
from ..data import FCUNetTrainingData
from ..evaluation.scoring import FastScoringFunction
from ..utils.measurement import create_vincl


class DPCAUNetTrainer(BaseTrainer):
    """Single-stage trainer for DPCA-UNet.

    Cross-attention maps measurements to spatial features, then
    dual-pooling UNet produces 3-class segmentation.
    """

    def __init__(self, config=None, experiment_name='dpcaunet_baseline'):
        if config is None:
            config = get_dpcaunet_config()
        super().__init__(config, experiment_name)

        self.vincl_dict = None
        self.loss_fn = nn.CrossEntropyLoss()

    def build_model(self):
        model = DPCAUNet(
            n_channels=self.config.model.n_channels,
            n_patterns=self.config.model.n_patterns,
            d_model=self.config.model.d_model,
            n_heads=self.config.model.n_heads,
            im_size=self.config.data.im_size,
            encoder_channels=tuple(self.config.model.encoder_channels),
            out_channels=self.config.model.out_channels,
            max_period=self.config.model.max_period,
        )
        model.to(self.device)
        self.model = model

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'DPCAUNet parameters: {n_params:,}')

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.training.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor)

    def build_datasets(self):
        ref_path = self.config.data.ref_path
        mesh_name = self.config.data.mesh_name
        base_path = self.config.data.dataset_base_path

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

        # Training dataset
        train_indices = self.config.data.get('train_indices', None)
        use_hdf5 = self.config.data.get('use_hdf5', False)

        if use_hdf5:
            from ..data import FCUNetHDF5Dataset
            h5_path = self.config.data.hdf5_path
            dataset = FCUNetHDF5Dataset(
                h5_path, Uelref, solver.InvLn,
                indices=train_indices, augment_noise=True)
        else:
            dataset = FCUNetTrainingData(
                Uelref, solver.InvLn, base_path,
                indices=train_indices, augment_noise=True)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            pin_memory=self.config.training.pin_memory,
            num_workers=self.config.training.num_workers)

        # Simulated val dataset
        self.val_sim_loader = None
        val_indices = self.config.data.get('val_indices', None)
        if val_indices is not None:
            if use_hdf5:
                from ..data import FCUNetHDF5Dataset
                val_ds = FCUNetHDF5Dataset(
                    self.config.data.hdf5_path, Uelref, solver.InvLn,
                    indices=val_indices, augment_noise=False)
            else:
                val_ds = FCUNetTrainingData(
                    Uelref, solver.InvLn, base_path,
                    indices=val_indices, augment_noise=False)
            self.val_sim_loader = DataLoader(
                val_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                pin_memory=self.config.training.pin_memory,
                num_workers=self.config.training.num_workers)

        # Pre-compute vincl masks
        self.vincl_dict = {}
        for lvl in range(1, 8):
            self.vincl_dict[lvl] = create_vincl(lvl, Injref).T.flatten()

        # Challenge validation data
        self._load_val_data()

    def _load_val_data(self):
        """Load challenge validation images."""
        gt_dir = self.config.validation.gt_dir
        data_dir = self.config.validation.data_dir
        num_val = self.config.validation.num_val_images

        ref_path = self.config.data.ref_path
        y_ref = np.array(loadmat(ref_path)['Uelref'])

        x_val, y_val_dict = [], {}
        for i in range(1, num_val + 1):
            x = loadmat(f'{gt_dir}/true{i}.mat')['truth']
            x_val.append(x)

            y_challenge = np.array(
                loadmat(f'{data_dir}/data{i}.mat')['Uel'])
            for level in range(1, 8):
                y_diff = y_challenge - y_ref
                y_diff[~self.vincl_dict[level]] = 0.0
                y_val_dict.setdefault(level, []).append(y_diff[:, 0])

        self.val_data = {
            'gt': np.stack(x_val),
            'measurements': {
                lvl: np.stack(arrs) for lvl, arrs in y_val_dict.items()
            },
        }

    def train_step(self, batch):
        y, gt = batch

        # Level augmentation
        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            levels = np.full(y.shape[0], fixed_level)
        else:
            levels = np.random.choice(np.arange(1, 8), size=y.shape[0])
        for k in range(y.shape[0]):
            y[k, ~self.vincl_dict[levels[k]]] = 0.0

        self.optimizer.zero_grad()

        levels_tensor = torch.from_numpy(levels).float()
        gt = gt.to(self.device)
        y = y.to(self.device)
        levels_tensor = levels_tensor.to(self.device)

        pred = self.model(y, levels_tensor)
        loss = self.loss_fn(pred, gt)

        loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return {'loss': loss.item()}

    def validate(self, epoch):
        """Validate: compute val_loss only."""
        metrics = {}
        if self.val_sim_loader is not None:
            metrics = self._validate_sim(epoch)
        return metrics

    def _validate_sim(self, epoch):
        """Validate on simulated val set: CE loss only."""
        fixed_level = self.config.training.get('fixed_level', 1)
        total_loss = 0.0
        num_samples = 0

        for y, gt in self.val_sim_loader:
            for k in range(y.shape[0]):
                y[k, ~self.vincl_dict[fixed_level]] = 0.0

            levels_tensor = torch.full(
                (y.shape[0],), fixed_level,
                dtype=torch.float, device=self.device)
            y = y.to(self.device)
            gt = gt.to(self.device)

            with torch.no_grad():
                pred = self.model(y, levels_tensor)
                loss = self.loss_fn(pred, gt)
            total_loss += loss.item() * y.shape[0]
            num_samples += y.shape[0]

        avg_loss = total_loss / max(num_samples, 1)

        if self.writer is not None:
            self.writer.add_scalar('val_sim/loss', avg_loss, epoch + 1)
        print(f'  Val(sim) loss: {avg_loss:.5f}')

        return {'val_loss': avg_loss}
