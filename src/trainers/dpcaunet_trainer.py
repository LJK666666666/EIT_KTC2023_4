"""
DPCAUNet trainer with convergence acceleration.

Stage 1 (stage1_epochs):
    Train attention module + pretrain_head only (1x1 conv).
    Goal: learn meaningful attention feature maps.

Stage 3 (remaining epochs):
    Unfreeze all, deep supervision with aux weight decay.
    Linear warmup + cosine annealing LR schedule.
    Grouped LR: attention layers use 1/10 of base LR.
    Loss: Dice + Focal (anti background-gradient domination).
    Measurement normalization: zero-mean, unit-variance.
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
from ..losses import DiceFocalLoss
from ..utils.measurement import create_vincl


class DPCAUNetTrainer(BaseTrainer):
    """Two-stage trainer for DPCA-UNet with convergence optimizations."""

    def __init__(self, config=None, experiment_name='dpcaunet_baseline'):
        if config is None:
            config = get_dpcaunet_config()
        super().__init__(config, experiment_name)

        self.vincl_dict = None
        self.loss_fn = DiceFocalLoss()
        self.training_stage = 1
        self._aux_weights = list(config.training.get(
            'aux_weights', (0.5, 0.25, 0.125)))
        # Measurement normalization stats (computed from training set)
        self._meas_mean = None  # tensor on device
        self._meas_std = None

    # ------------------------------------------------------------------
    # Model & optimizer
    # ------------------------------------------------------------------

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

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print(f'DPCAUNet parameters: {n_params:,}')

        # Placeholder optimizer (rebuilt per stage)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.training.lr)
        self.scheduler = None

    def _build_grouped_optimizer(self, lr, trainable_only=False):
        """Build AdamW with grouped LR: attention gets lr/10."""
        attn_names = ['electrode_encoder', 'spatial_query', 'cross_attn',
                      'level_embed', 'cascaded_attn']
        attn_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if any(name.startswith(n) for n in attn_names):
                attn_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        if attn_params:
            param_groups.append({
                'params': attn_params, 'lr': lr * 0.1})
        if other_params:
            param_groups.append({
                'params': other_params, 'lr': lr})

        wd = self.config.training.get('weight_decay', 1e-4)
        return torch.optim.AdamW(param_groups, weight_decay=wd)

    def _build_warmup_cosine_scheduler(self, optimizer, total_epochs,
                                       warmup_epochs):
        """Linear warmup + cosine annealing scheduler (per-epoch step)."""
        steps_per_epoch = len(self.train_loader)
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(
                total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _attention_params(self):
        names = ['electrode_encoder', 'spatial_query', 'cross_attn',
                 'level_embed', 'cascaded_attn']
        params = []
        for name, p in self.model.named_parameters():
            if any(name.startswith(n) for n in names):
                params.append(p)
        return params

    def _freeze_attention(self):
        for p in self._attention_params():
            p.requires_grad = False

    def _unfreeze_attention(self):
        for p in self._attention_params():
            p.requires_grad = True

    def _freeze_unet(self):
        attn_names = ['electrode_encoder', 'spatial_query', 'cross_attn',
                      'level_embed', 'cascaded_attn', 'pretrain_head']
        for name, p in self.model.named_parameters():
            if not any(name.startswith(n) for n in attn_names):
                p.requires_grad = False

    def _unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

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

        self.vincl_dict = {}
        for lvl in range(1, 8):
            self.vincl_dict[lvl] = create_vincl(lvl, Injref).T.flatten()

        self._load_val_data()
        self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute measurement mean/std from a pass over the training set."""
        print('Computing measurement normalization stats...')
        all_meas = []
        for y, _ in self.train_loader:
            all_meas.append(y)
            if len(all_meas) * y.shape[0] >= 200:
                break  # enough samples for stable statistics
        all_meas = torch.cat(all_meas, dim=0)
        # Only compute stats on non-zero elements (vincl-masked regions are 0)
        nonzero_mask = all_meas != 0
        if nonzero_mask.sum() > 0:
            mean_val = all_meas[nonzero_mask].mean().item()
            std_val = all_meas[nonzero_mask].std().item()
        else:
            mean_val, std_val = 0.0, 1.0
        std_val = max(std_val, 1e-8)
        self._meas_mean = torch.tensor(mean_val, device=self.device)
        self._meas_std = torch.tensor(std_val, device=self.device)
        print(f'  Measurement stats: mean={mean_val:.6f}, std={std_val:.6f}')

    def _normalize_measurements(self, y):
        """Normalize measurements: (y - mean) / std, keeping zeros at zero."""
        if self._meas_mean is None:
            return y
        # Use torch.where for XLA-compatible static-shape operation
        return torch.where(
            y != 0,
            (y - self._meas_mean) / self._meas_std,
            y)

    def _load_val_data(self):
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
                lvl: np.stack(arrs) for lvl, arrs in y_val_dict.items()},
        }

    # ------------------------------------------------------------------
    # Train / validate steps
    # ------------------------------------------------------------------

    def _apply_vincl(self, y, levels):
        for k in range(y.shape[0]):
            y[k, ~self.vincl_dict[levels[k]]] = 0.0

    def _sample_levels(self, batch_size):
        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            return np.full(batch_size, fixed_level)
        return np.random.choice(np.arange(1, 8), size=batch_size)

    def train_step(self, batch):
        y, gt = batch
        levels = self._sample_levels(y.shape[0])
        self._apply_vincl(y, levels)

        levels_tensor = torch.from_numpy(levels).float().to(self.device)
        gt = gt.to(self.device)
        y = y.to(self.device)
        y = self._normalize_measurements(y)

        self.optimizer.zero_grad()

        if self.training_stage == 1:
            with self._autocast_context():
                pred = self.model.forward_pretrain(y, levels_tensor)
                loss = self.loss_fn(pred, gt)
        elif self.training_stage in (2, 3):
            with self._autocast_context():
                main_out, aux_outs = self.model(
                    y, levels_tensor, deep_supervision=True)
                loss = self.loss_fn(main_out, gt)
                for i, aux in enumerate(aux_outs):
                    if i < len(self._aux_weights):
                        w = self._aux_weights[i]
                        if w > 0:
                            loss = loss + w * self.loss_fn(aux, gt)

        loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        # Step per-iteration scheduler (warmup + cosine)
        if self.scheduler is not None:
            self.scheduler.step()

        return {'loss': loss.item()}

    def validate(self, epoch):
        metrics = {}
        if self.val_sim_loader is not None:
            metrics = self._validate_sim(epoch)
        return metrics

    def _validate_sim(self, epoch):
        total_loss = 0.0
        num_samples = 0

        for y, gt in self.val_sim_loader:
            levels = self._sample_levels(y.shape[0])
            self._apply_vincl(y, levels)

            levels_tensor = torch.from_numpy(levels).float().to(self.device)
            y = y.to(self.device)
            gt = gt.to(self.device)
            y = self._normalize_measurements(y)

            with torch.no_grad():
                with self._autocast_context():
                    pred = self.model(y, levels_tensor)
                    loss = self.loss_fn(pred, gt)
            total_loss += loss.item() * y.shape[0]
            num_samples += y.shape[0]

        avg_loss = total_loss / max(num_samples, 1)

        if self.writer is not None:
            self.writer.add_scalar('val_sim/loss', avg_loss, epoch + 1)
        print(f'  Val(sim) loss: {avg_loss:.5f}')

        return {'val_loss': avg_loss}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        resume_path = self.config.training.get('resume_from', None)
        base_dir = getattr(self.config, 'result_base_dir', 'results')
        if resume_path:
            self.result_dir = os.path.dirname(resume_path)
        else:
            self.result_dir = self._create_result_dir(
                self.experiment_name, base_dir=base_dir)

        self.build_model()
        self.build_datasets()

        if resume_path:
            self._load_checkpoint(resume_path)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.result_dir)
        self._save_config()

        max_iters = self.config.training.get('max_iters', None)
        save_freq = self.config.training.get('save_freq', 5)

        s1_epochs = self.config.training.stage1_epochs
        total_epochs = self.config.training.epochs
        aux_decay_epochs = self.config.training.aux_decay_epochs
        warmup_epochs = self.config.training.get('warmup_epochs', 5)
        aux_weights_init = list(self.config.training.get(
            'aux_weights', (0.5, 0.25, 0.125)))

        # ---- Stage 1: Pretrain attention ----
        if self.training_stage == 1:
            print(f'Stage 1: Pretrain attention + lightweight head '
                  f'(epochs 1-{s1_epochs})')
            print(f'Results directory: {self.result_dir}')

            self._freeze_unet()
            self.optimizer = self._build_grouped_optimizer(
                self.config.training.stage1_lr, trainable_only=True)
            self.scheduler = self._build_warmup_cosine_scheduler(
                self.optimizer, s1_epochs,
                warmup_epochs=min(3, s1_epochs))

            for epoch in range(self.current_epoch, s1_epochs):
                self.current_epoch = epoch
                self.model.train()
                self.training_stage = 1

                epoch_metrics = self._train_epoch(epoch, max_iters=max_iters)

                val_metrics = {}
                val_freq = self.config.training.get('val_freq', 1)
                if val_freq > 0 and (epoch + 1) % val_freq == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_metrics = self.validate(epoch)
                    self.model.train()

                all_metrics = {**epoch_metrics, **val_metrics, 'stage': 1}
                all_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                self._log_epoch(epoch, all_metrics)

                if save_freq > 0 and (epoch + 1) % save_freq == 0:
                    self._save_checkpoint('last.pt')

                if max_iters and self.global_step >= max_iters:
                    self._save_checkpoint('last.pt')
                    print(f'Quick test: reached {max_iters} iterations.')
                    self._finish()
                    return

            self._save_checkpoint('last.pt')
            self.training_stage = 3
            self.current_epoch = 0
            self.global_step = 0

        # ---- Stage 2 (removed): directly enter stage 3 ----
        if self.training_stage == 2:
            self.training_stage = 3
            self.current_epoch = 0
            self.global_step = 0

        # ---- Stage 3: Unfreeze all, warmup + cosine, aux decay ----
        self._unfreeze_all()
        self._es_counter = 0
        self._es_best_val_loss = None
        s3_total = total_epochs - s1_epochs
        print(f'Stage 3: Full fine-tuning '
              f'(epochs 1-{s3_total}, warmup {warmup_epochs} epochs, '
              f'aux decay over {aux_decay_epochs})')

        self.optimizer = self._build_grouped_optimizer(
            self.config.training.lr)
        self.scheduler = self._build_warmup_cosine_scheduler(
            self.optimizer, s3_total, warmup_epochs)

        for epoch in range(self.current_epoch, s3_total):
            self.current_epoch = epoch
            self.training_stage = 3

            # Linearly decay auxiliary weights
            if epoch < aux_decay_epochs:
                decay = 1.0 - epoch / aux_decay_epochs
            else:
                decay = 0.0
            self._aux_weights = [w * decay for w in aux_weights_init]

            epoch_metrics = self._train_epoch(epoch, max_iters=max_iters)

            val_metrics = {}
            val_freq = self.config.training.get('val_freq', 1)
            if val_freq > 0 and (epoch + 1) % val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    val_metrics = self.validate(epoch)
                self.model.train()

            all_metrics = {**epoch_metrics, **val_metrics, 'stage': 3,
                           'aux_decay': decay}
            all_metrics['lr'] = self.optimizer.param_groups[-1]['lr']
            if self.writer:
                self.writer.add_scalar(
                    'train/lr', all_metrics['lr'], self.global_step)
                self.writer.add_scalar(
                    'train/lr_attn',
                    self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar(
                    'train/aux_decay', decay, self.global_step)

            self._log_epoch(epoch, all_metrics)

            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self._save_checkpoint('last.pt')

            if 'val_loss' in val_metrics:
                if (self.best_metric is None
                        or val_metrics['val_loss'] < self.best_metric):
                    self.best_metric = val_metrics['val_loss']
                    self._save_checkpoint('best.pt')
                    print(f'  New best val_loss: '
                          f'{val_metrics["val_loss"]:.5f}')

            # Early stopping
            es_loss = val_metrics.get(
                'val_loss', epoch_metrics.get('avg_loss'))
            if es_loss is not None:
                if (self._es_best_val_loss is None
                        or es_loss < self._es_best_val_loss):
                    self._es_best_val_loss = es_loss
                    self._es_counter = 0
                else:
                    self._es_counter += 1
            es_patience = self.config.training.get(
                'early_stopping_patience', None)
            if es_patience and self._es_counter >= es_patience:
                print(f'Early stopping: val_loss not improved for '
                      f'{es_patience} epochs')
                break

            if max_iters and self.global_step >= max_iters:
                print(f'Quick test: reached {max_iters} iterations.')
                break

        self._save_checkpoint('last.pt')
        self._finish()

    def _finish(self):
        if self.writer:
            self.writer.close()
        print(f'Training complete. Results: {self.result_dir}')

    # ------------------------------------------------------------------
    # Checkpoint extras
    # ------------------------------------------------------------------

    def get_checkpoint_extra(self):
        return {
            'training_stage': self.training_stage,
            'aux_weights': self._aux_weights,
            'meas_mean': self._meas_mean.item() if self._meas_mean is not None else None,
            'meas_std': self._meas_std.item() if self._meas_std is not None else None,
        }

    def load_checkpoint_extra(self, state):
        self.training_stage = state.get('training_stage', 3)
        self._aux_weights = state.get('aux_weights', self._aux_weights)
        if state.get('meas_mean') is not None:
            self._meas_mean = torch.tensor(
                state['meas_mean'], device=self.device)
            self._meas_std = torch.tensor(
                state['meas_std'], device=self.device)
