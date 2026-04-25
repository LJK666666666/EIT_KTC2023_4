"""Trainer for fixed-basis DCT low-frequency predictor."""

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from scipy.io import loadmat
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..configs.dct_predictor_config import get_configs as get_dct_predictor_config
from ..data import DCTHDF5Dataset
from ..evaluation.scoring_torch import fast_score_batch_auto
from ..losses.dice_focal import DiceLoss
from ..models.dct_predictor import DCTPredictor
from ..utils.measurement import create_vincl


class DCTPredictorTrainer(BaseTrainer):
    def __init__(self, config=None, experiment_name='dct_predictor_baseline'):
        if config is None:
            config = get_dct_predictor_config()
        super().__init__(config, experiment_name)
        self.dice_loss = DiceLoss()
        self.vincl_dict = None
        self.test_sim_loader = None
        self._probe_cache = None

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
        print(f'DCTPredictor: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
        if not self.config.data.get('use_hdf5', False):
            raise ValueError('DCT predictor currently requires HDF5 dataset.')

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

        train_ds = DCTHDF5Dataset(
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
            val_ds = DCTHDF5Dataset(
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
            test_ds = DCTHDF5Dataset(
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

    def _image_loss(self, logits, gt_indices):
        ce = F.cross_entropy(logits, gt_indices)
        dice = self.dice_loss(logits, gt_indices)
        ce_weight = float(self.config.training.get('ce_weight', 1.0))
        dice_weight = float(self.config.training.get('dice_weight', 1.0))
        return ce_weight * ce + dice_weight * dice, ce, dice

    def _sample_levels(self, batch_size):
        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            return np.full(batch_size, fixed_level)
        return np.random.choice(np.arange(1, 8), size=batch_size)

    def _get_probe_cache(self, max_samples: int):
        if self.val_sim_loader is None:
            return None, None
        if self._probe_cache is not None:
            measurements_cpu, gt_cpu = self._probe_cache
        else:
            measurements_list = []
            gt_list = []
            seen = 0
            for measurements, gt_indices in self.val_sim_loader:
                measurements_list.append(measurements.clone())
                gt_list.append(gt_indices.clone())
                seen += measurements.shape[0]
                if max_samples > 0 and seen >= max_samples:
                    break
            measurements_cpu = torch.cat(measurements_list, dim=0)
            gt_cpu = torch.cat(gt_list, dim=0)
            if max_samples > 0:
                measurements_cpu = measurements_cpu[:max_samples]
                gt_cpu = gt_cpu[:max_samples]
            self._probe_cache = (measurements_cpu, gt_cpu)
        if max_samples > 0:
            return measurements_cpu[:max_samples], gt_cpu[:max_samples]
        return measurements_cpu, gt_cpu

    def _run_score_probe(self):
        if self.val_sim_loader is None:
            return {}
        max_samples = int(self.config.training.get(
            'score_probe_max_samples', 256) or 0)
        probe_batch_size = int(self.config.training.get(
            'score_probe_batch_size',
            max(self.config.training.batch_size, 1)) or 0)
        measurements_cpu, gt_cpu = self._get_probe_cache(max_samples)
        if measurements_cpu is None or gt_cpu is None or measurements_cpu.shape[0] == 0:
            return {}
        all_scores = []
        level_scores = {}
        for level in range(1, 8):
            level_preds = []
            mask = torch.from_numpy(
                self.vincl_dict[level].astype(np.bool_)
            ).to(torch.bool)
            total = measurements_cpu.shape[0]
            chunk = total if probe_batch_size <= 0 else probe_batch_size
            for start in range(0, total, chunk):
                end = min(start + chunk, total)
                measurements = measurements_cpu[start:end].clone()
                measurements[:, ~mask] = 0.0
                measurements = measurements.to(self.device)
                levels_tensor = torch.full(
                    (measurements.shape[0],), level,
                    dtype=torch.float, device=self.device)
                with torch.no_grad():
                    with self._autocast_context():
                        logits, _ = self.model(measurements, levels_tensor)
                        preds = torch.argmax(logits, dim=1)
                level_preds.append(preds.cpu().numpy().astype(np.int64))
            pred_np = np.concatenate(level_preds, axis=0)
            gt_np = gt_cpu.cpu().numpy().astype(np.int64)
            scores = fast_score_batch_auto(gt_np, pred_np, device=self.device)
            all_scores.extend(scores)
            level_scores[level] = float(np.mean(scores))
        return {
            'val_probe_score_mean': float(np.mean(all_scores)),
            'val_probe_score_total': float(np.sum(all_scores)),
            'val_probe_level_scores': level_scores,
        }

    def train_step(self, batch):
        measurements, gt_indices = batch
        levels = self._sample_levels(measurements.shape[0])
        for k in range(measurements.shape[0]):
            measurements[k, ~self.vincl_dict[levels[k]]] = 0.0

        measurements = measurements.to(self.device)
        gt_indices = gt_indices.to(self.device)
        gt_onehot = F.one_hot(
            gt_indices.long(), num_classes=3).permute(0, 3, 1, 2).float()
        levels_tensor = torch.from_numpy(levels).float().to(self.device)

        self.optimizer.zero_grad()
        with self._autocast_context():
            logits, coeffs = self.model(measurements, levels_tensor)
            target_coeffs = self.model.target_coeffs(gt_onehot)
            coeff_loss = F.mse_loss(coeffs.float(), target_coeffs.float())
            image_loss, ce_loss, dice_loss = self._image_loss(logits, gt_indices)
            total_loss = image_loss + float(
                self.config.training.get('coeff_loss_weight', 0.5)
            ) * coeff_loss
        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)
        return {
            'loss': total_loss.item(),
            'coeff_loss': coeff_loss.item(),
            'image_loss': image_loss.item(),
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}
        fixed_level = self.config.training.get('fixed_level', 1)
        if fixed_level is None:
            fixed_level = 1
        totals = {
            'loss': 0.0,
            'coeff_loss': 0.0,
            'image_loss': 0.0,
            'ce_loss': 0.0,
            'dice_loss': 0.0,
        }
        num_samples = 0
        mask = torch.from_numpy(
            self.vincl_dict[fixed_level].astype(np.bool_)
        ).to(torch.bool)
        for measurements, gt_indices in self.val_sim_loader:
            measurements[:, ~mask] = 0.0
            measurements = measurements.to(self.device)
            gt_indices = gt_indices.to(self.device)
            gt_onehot = F.one_hot(
                gt_indices.long(), num_classes=3).permute(0, 3, 1, 2).float()
            levels_tensor = torch.full(
                (measurements.shape[0],), fixed_level,
                dtype=torch.float, device=self.device)
            with torch.no_grad():
                with self._autocast_context():
                    logits, coeffs = self.model(measurements, levels_tensor)
                    target_coeffs = self.model.target_coeffs(gt_onehot)
                    coeff_loss = F.mse_loss(coeffs.float(), target_coeffs.float())
                    image_loss, ce_loss, dice_loss = self._image_loss(logits, gt_indices)
                    total_loss = image_loss + float(
                        self.config.training.get('coeff_loss_weight', 0.5)
                    ) * coeff_loss
            batch_size = measurements.shape[0]
            totals['loss'] += total_loss.item() * batch_size
            totals['coeff_loss'] += coeff_loss.item() * batch_size
            totals['image_loss'] += image_loss.item() * batch_size
            totals['ce_loss'] += ce_loss.item() * batch_size
            totals['dice_loss'] += dice_loss.item() * batch_size
            num_samples += batch_size
        metrics = {
            'val_loss': totals['loss'] / max(num_samples, 1),
            'val_coeff_loss': totals['coeff_loss'] / max(num_samples, 1),
            'val_image_loss': totals['image_loss'] / max(num_samples, 1),
            'val_ce_loss': totals['ce_loss'] / max(num_samples, 1),
            'val_dice_loss': totals['dice_loss'] / max(num_samples, 1),
        }
        probe_freq = int(self.config.training.get('score_probe_freq', 0) or 0)
        if probe_freq > 0 and ((epoch + 1) % probe_freq == 0):
            metrics.update(self._run_score_probe())
        if self.writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f'val/{key[4:]}', val, epoch + 1)
        print(f'  Val loss: {metrics["val_loss"]:.6f}')
        if 'val_probe_score_total' in metrics:
            print(
                f'  Probe score: mean={metrics["val_probe_score_mean"]:.6f}, '
                f'total={metrics["val_probe_score_total"]:.6f}'
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
        if fixed_level is None:
            fixed_level = 1
        total_loss = 0.0
        num_samples = 0
        all_preds = []
        all_gts = []
        mask = torch.from_numpy(
            self.vincl_dict[fixed_level].astype(np.bool_)
        ).to(torch.bool)
        for measurements, gt_indices in self.test_sim_loader:
            measurements[:, ~mask] = 0.0
            measurements = measurements.to(self.device)
            gt_indices = gt_indices.to(self.device)
            gt_onehot = F.one_hot(
                gt_indices.long(), num_classes=3).permute(0, 3, 1, 2).float()
            levels_tensor = torch.full(
                (measurements.shape[0],), fixed_level,
                dtype=torch.float, device=self.device)
            with torch.no_grad():
                with self._autocast_context():
                    logits, coeffs = self.model(measurements, levels_tensor)
                    target_coeffs = self.model.target_coeffs(gt_onehot)
                    coeff_loss = F.mse_loss(coeffs.float(), target_coeffs.float())
                    image_loss, _, _ = self._image_loss(logits, gt_indices)
                    total = image_loss + float(
                        self.config.training.get('coeff_loss_weight', 0.5)
                    ) * coeff_loss
                    preds = torch.argmax(logits, dim=1)
            batch_size = measurements.shape[0]
            total_loss += total.item() * batch_size
            num_samples += batch_size
            all_preds.append(preds.cpu().numpy())
            all_gts.append(gt_indices.cpu().numpy())
        pred_np = np.concatenate(all_preds, axis=0)
        gt_np = np.concatenate(all_gts, axis=0)
        scores = fast_score_batch_auto(gt_np, pred_np, device=self.device)
        return {
            'test_loss': total_loss / max(num_samples, 1),
            'test_score_mean': float(np.mean(scores)),
            'test_score_total': float(np.sum(scores)),
        }
