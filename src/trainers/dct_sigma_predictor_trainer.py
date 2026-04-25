"""Trainer for fixed-basis DCT conductivity regression."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from scipy.io import loadmat
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..configs.dct_sigma_predictor_config import (
    get_configs as get_dct_sigma_predictor_config,
)
from ..data import ConductivityHDF5Dataset
from ..evaluation.regression_metrics import masked_regression_metrics_batch
from ..models.dct_predictor import DCTPredictor
from ..utils.measurement import create_vincl


class DCTSigmaPredictorTrainer(BaseTrainer):
    def __init__(self, config=None, experiment_name='dct_sigma_predictor_baseline'):
        if config is None:
            config = get_dct_sigma_predictor_config()
        super().__init__(config, experiment_name)
        self.vincl_dict = None
        self.test_sim_loader = None

    def build_model(self):
        model = DCTPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            level_embed_dim=self.config.model.level_embed_dim,
            coeff_size=self.config.model.coeff_size,
            out_channels=self.config.model.out_channels,
            dropout=self.config.model.dropout,
        )
        model.to(self.device)
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.get('weight_decay', 1e-4),
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.training.get('selection_metric_mode', 'min'),
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            min_lr=self.config.training.get('min_lr', 1e-6),
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'DCTSigmaPredictor: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
        if not self.config.data.get('use_hdf5', False):
            raise ValueError('DCT sigma predictor currently requires HDF5 dataset.')

        from ..ktc_methods import EITFEM, load_mesh

        y_ref = loadmat(self.config.data.ref_path)
        Injref = y_ref['Injref']
        Mpat = y_ref['Mpat']
        mesh, mesh2 = load_mesh(self.config.data.mesh_name)
        nel = 32
        z = 1e-6 * np.ones(nel)
        vincl = np.ones((nel - 1, 76), dtype=bool)
        solver = EITFEM(mesh2, Injref, Mpat, vincl)
        solver.SetInvGamma(
            self.config.data.noise_std1,
            self.config.data.noise_std2,
            y_ref['Uelref'],
        )
        sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.745
        Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).reshape(-1)

        h5_path = self.config.data.hdf5_path
        train_indices = self.config.data.get('train_indices', None)
        val_indices = self.config.data.get('val_indices', None)
        test_indices = self.config.data.get('test_indices', None)

        train_ds = ConductivityHDF5Dataset(
            h5_path, Uelref, solver.InvLn,
            indices=train_indices, augment_noise=True)
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
        if val_indices is not None:
            val_ds = ConductivityHDF5Dataset(
                h5_path, Uelref, solver.InvLn,
                indices=val_indices, augment_noise=False)
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

        if test_indices is not None:
            test_ds = ConductivityHDF5Dataset(
                h5_path, Uelref, solver.InvLn,
                indices=test_indices, augment_noise=False)
            self._warn_if_dropping_last_batch(
                'test', len(test_ds), self.config.training.batch_size)
            self.test_sim_loader = DataLoader(
                test_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers,
            )

        self.vincl_dict = {}
        for lvl in range(1, 8):
            self.vincl_dict[lvl] = create_vincl(lvl, Injref).T.flatten()

    def _masked_image_loss(self, pred_sigma, target_sigma):
        mask = (target_sigma > 0).float()
        diff = (pred_sigma - target_sigma) * mask
        denom = mask.sum().clamp_min(1.0)
        mse = (diff.pow(2).sum() / denom)
        mae = (diff.abs().sum() / denom)
        total = (
            float(self.config.training.get('mse_weight', 1.0)) * mse +
            float(self.config.training.get('mae_weight', 0.1)) * mae
        )
        return total, mse, mae

    def _sample_levels(self, batch_size):
        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            return np.full(batch_size, fixed_level)
        return np.random.choice(np.arange(1, 8), size=batch_size)

    def train_step(self, batch):
        measurements, sigma, _ = batch
        levels = self._sample_levels(measurements.shape[0])
        for k in range(measurements.shape[0]):
            measurements[k, ~self.vincl_dict[levels[k]]] = 0.0

        measurements = measurements.to(self.device)
        sigma = sigma.to(self.device)
        levels_tensor = torch.from_numpy(levels).float().to(self.device)

        self.optimizer.zero_grad()
        with self._autocast_context():
            pred_sigma, coeffs = self.model(measurements, levels_tensor)
            target_coeffs = self.model.target_coeffs(sigma)
            coeff_loss = F.mse_loss(coeffs.float(), target_coeffs.float())
            image_loss, mse_loss, mae_loss = self._masked_image_loss(
                pred_sigma, sigma)
            total_loss = image_loss + float(
                self.config.training.get('coeff_loss_weight', 0.25)
            ) * coeff_loss
        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)
        return {
            'loss': total_loss.item(),
            'coeff_loss': coeff_loss.item(),
            'image_loss': image_loss.item(),
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
        }

    def _collect_metrics(self, loader, fixed_level):
        totals = {
            'loss': 0.0,
            'coeff_loss': 0.0,
            'image_loss': 0.0,
            'mse_loss': 0.0,
            'mae_loss': 0.0,
        }
        num_samples = 0
        preds_all = []
        sigma_all = []
        mask = torch.from_numpy(
            self.vincl_dict[fixed_level].astype(np.bool_)
        ).to(torch.bool)
        for measurements, sigma, _ in loader:
            measurements[:, ~mask] = 0.0
            measurements = measurements.to(self.device)
            sigma = sigma.to(self.device)
            levels_tensor = torch.full(
                (measurements.shape[0],), fixed_level,
                dtype=torch.float, device=self.device)
            with torch.no_grad():
                with self._autocast_context():
                    pred_sigma, coeffs = self.model(measurements, levels_tensor)
                    target_coeffs = self.model.target_coeffs(sigma)
                    coeff_loss = F.mse_loss(coeffs.float(), target_coeffs.float())
                    image_loss, mse_loss, mae_loss = self._masked_image_loss(
                        pred_sigma, sigma)
                    total_loss = image_loss + float(
                        self.config.training.get('coeff_loss_weight', 0.25)
                    ) * coeff_loss
            batch_size = measurements.shape[0]
            totals['loss'] += total_loss.item() * batch_size
            totals['coeff_loss'] += coeff_loss.item() * batch_size
            totals['image_loss'] += image_loss.item() * batch_size
            totals['mse_loss'] += mse_loss.item() * batch_size
            totals['mae_loss'] += mae_loss.item() * batch_size
            num_samples += batch_size
            preds_all.append(pred_sigma.detach().float().cpu().numpy())
            sigma_all.append(sigma.detach().float().cpu().numpy())

        pred_np = np.concatenate(preds_all, axis=0)[:, 0]
        sigma_np = np.concatenate(sigma_all, axis=0)[:, 0]
        reg = masked_regression_metrics_batch(sigma_np, pred_np)
        return {
            'loss': totals['loss'] / max(num_samples, 1),
            'coeff_loss': totals['coeff_loss'] / max(num_samples, 1),
            'image_loss': totals['image_loss'] / max(num_samples, 1),
            'mse_loss': totals['mse_loss'] / max(num_samples, 1),
            'mae_loss': totals['mae_loss'] / max(num_samples, 1),
            'preds': pred_np,
            'targets': sigma_np,
            'reg': reg,
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}
        fixed_level = self.config.training.get('fixed_level', 1)
        metrics_raw = self._collect_metrics(self.val_sim_loader, fixed_level)
        metrics = {
            'val_loss': metrics_raw['loss'],
            'val_coeff_loss': metrics_raw['coeff_loss'],
            'val_image_loss': metrics_raw['image_loss'],
            'val_mse_loss': metrics_raw['mse_loss'],
            'val_mae_loss': metrics_raw['mae_loss'],
            'val_rmse': float(np.mean(metrics_raw['reg']['rmse'])),
            'val_rel_l2': float(np.mean(metrics_raw['reg']['rel_l2'])),
        }
        if self.writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f'val/{key[4:]}', val, epoch + 1)
        print(
            f'  Val loss: {metrics["val_loss"]:.6f} '
            f'RMSE: {metrics["val_rmse"]:.6f} '
            f'RelL2: {metrics["val_rel_l2"]:.6f}'
        )
        return metrics

    def _get_eval_checkpoint_path(self):
        best_path = os.path.join(self.result_dir, 'best.pt')
        if not os.path.exists(best_path):
            best_path = os.path.join(self.result_dir, 'last.pt')
        return best_path

    def evaluate_test(self):
        if self.test_sim_loader is None:
            return {}
        self._load_checkpoint(self._get_eval_checkpoint_path())
        self.model.eval()
        fixed_level = self.config.training.get('fixed_level', 1)
        metrics_raw = self._collect_metrics(self.test_sim_loader, fixed_level)
        return {
            'test_loss': metrics_raw['loss'],
            'test_mae': float(np.mean(metrics_raw['reg']['mae'])),
            'test_rmse': float(np.mean(metrics_raw['reg']['rmse'])),
            'test_rel_l2': float(np.mean(metrics_raw['reg']['rel_l2'])),
        }
