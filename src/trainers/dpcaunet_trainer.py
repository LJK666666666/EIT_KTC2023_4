"""
DPCAUNet trainer: three-stage training for cross-attention EIT reconstruction.

Stage 1 (stage1_epochs):
    Train attention module + pretrain_head only (1x1 conv).
    Loss: CrossEntropy on coarse prediction.
    Goal: learn meaningful attention feature maps.

Stage 2 (stage2_epochs):
    Freeze attention, train UNet + aux heads with deep supervision.
    Loss: main CE + weighted auxiliary CE from each decoder block.

Stage 3 (remaining epochs):
    Unfreeze all, linearly decay aux loss weights to 0.
    Network focuses on optimizing the main output.
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
    """Three-stage trainer for DPCA-UNet."""

    def __init__(self, config=None, experiment_name='dpcaunet_baseline'):
        if config is None:
            config = get_dpcaunet_config()
        super().__init__(config, experiment_name)

        self.vincl_dict = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.training_stage = 1
        # Auxiliary loss weights (current, may decay in stage 3)
        self._aux_weights = list(config.training.get(
            'aux_weights', (0.4, 0.2, 0.1)))

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

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.training.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor)

    def _attention_params(self):
        """Return parameters belonging to the attention module."""
        names = ['electrode_encoder', 'spatial_query', 'cross_attn',
                 'level_embed']
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
        """Freeze everything except attention + pretrain_head."""
        attn_names = ['electrode_encoder', 'spatial_query', 'cross_attn',
                      'level_embed', 'pretrain_head']
        for name, p in self.model.named_parameters():
            if not any(name.startswith(n) for n in attn_names):
                p.requires_grad = False

    def _unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True

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

    def _apply_vincl(self, y, levels):
        """Apply vincl mask in-place."""
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

        self.optimizer.zero_grad()

        if self.training_stage == 1:
            pred = self.model.forward_pretrain(y, levels_tensor)
            loss = self.loss_fn(pred, gt)
        elif self.training_stage in (2, 3):
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

    # ------------------------------------------------------------------
    # Three-stage training loop
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
        s2_epochs = self.config.training.stage2_epochs
        total_epochs = self.config.training.epochs
        aux_decay_epochs = self.config.training.aux_decay_epochs
        aux_weights_init = list(self.config.training.get(
            'aux_weights', (0.4, 0.2, 0.1)))

        # ---- Stage 1: Pretrain attention ----
        if self.training_stage == 1:
            print(f'Stage 1: Pretrain attention + lightweight head '
                  f'(epochs 1-{s1_epochs})')
            print(f'Results directory: {self.result_dir}')

            # Freeze UNet, only train attention + pretrain_head
            self._freeze_unet()
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.training.stage1_lr)

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

                self._log_epoch(epoch, {**epoch_metrics, **val_metrics,
                                        'stage': 1})

                if save_freq > 0 and (epoch + 1) % save_freq == 0:
                    self._save_checkpoint('last.pt')

                if max_iters and self.global_step >= max_iters:
                    self._save_checkpoint('last.pt')
                    print(f'Quick test: reached {max_iters} iterations.')
                    self._finish()
                    return

            self._save_checkpoint('last.pt')
            self.training_stage = 2
            self.current_epoch = 0
            self.global_step = 0

        # ---- Stage 2: Freeze attention, deep supervision ----
        if self.training_stage == 2:
            print(f'Stage 2: Freeze attention, deep supervision '
                  f'(epochs 1-{s2_epochs})')

            self._unfreeze_all()
            self._freeze_attention()
            self._aux_weights = list(aux_weights_init)
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.training.stage2_lr)
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                patience=self.config.training.scheduler_patience,
                factor=self.config.training.scheduler_factor)

            s2_start = self.current_epoch if self.current_epoch < s2_epochs \
                else 0
            for epoch in range(s2_start, s2_epochs):
                self.current_epoch = epoch
                self.model.train()
                self.training_stage = 2

                epoch_metrics = self._train_epoch(epoch, max_iters=max_iters)

                val_metrics = {}
                val_freq = self.config.training.get('val_freq', 1)
                if val_freq > 0 and (epoch + 1) % val_freq == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_metrics = self.validate(epoch)
                    self.model.train()

                all_metrics = {**epoch_metrics, **val_metrics, 'stage': 2}
                all_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                self._log_epoch(epoch, all_metrics)

                if self.scheduler is not None:
                    sched_metric = val_metrics.get(
                        'val_loss', epoch_metrics.get('avg_loss'))
                    if sched_metric is not None:
                        self.scheduler.step(sched_metric)

                if save_freq > 0 and (epoch + 1) % save_freq == 0:
                    self._save_checkpoint('last.pt')

                if 'val_loss' in val_metrics:
                    if (self.best_metric is None
                            or val_metrics['val_loss'] < self.best_metric):
                        self.best_metric = val_metrics['val_loss']
                        self._save_checkpoint('best.pt')

                if max_iters and self.global_step >= max_iters:
                    self._save_checkpoint('last.pt')
                    print(f'Quick test: reached {max_iters} iterations.')
                    self._finish()
                    return

            self._save_checkpoint('last.pt')
            self.training_stage = 3
            self.current_epoch = 0
            self.global_step = 0

        # ---- Stage 3: Unfreeze all, aux decay, fine-tune ----
        self._unfreeze_all()
        self._es_counter = 0
        self._es_best_val_loss = None
        s3_total = total_epochs - s1_epochs - s2_epochs
        print(f'Stage 3: Full fine-tuning with aux decay '
              f'(epochs 1-{s3_total}, aux decay over {aux_decay_epochs})')

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.training.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor)

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
            all_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            if self.writer:
                self.writer.add_scalar(
                    'train/lr', all_metrics['lr'], self.global_step)
                self.writer.add_scalar(
                    'train/aux_decay', decay, self.global_step)

            if self.scheduler is not None:
                sched_metric = val_metrics.get(
                    'val_loss', epoch_metrics.get('avg_loss'))
                if sched_metric is not None:
                    self.scheduler.step(sched_metric)

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
        }

    def load_checkpoint_extra(self, state):
        self.training_stage = state.get('training_stage', 3)
        self._aux_weights = state.get('aux_weights', self._aux_weights)
