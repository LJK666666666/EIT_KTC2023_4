"""FCUNet-style continuous conductivity regression baseline."""

import json
import os

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from scipy.io import loadmat
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..configs import get_fc_sigmaunet_config
from ..data import ConductivityHDF5Dataset
from ..evaluation.regression_metrics import masked_regression_metrics_batch
from ..models.fcunet import FCUNet
from ..utils.measurement import create_vincl


class FCSigmaUNetTrainer(BaseTrainer):
    """Two-stage FCUNet baseline for continuous conductivity regression."""

    def __init__(self, config=None, experiment_name='fc_sigmaunet_baseline'):
        if config is None:
            config = get_fc_sigmaunet_config()
        super().__init__(config, experiment_name)
        self.vincl_dict = None
        self.init_optimizer = None
        self.training_stage = 1
        self.val_sim_loader = None
        self.test_sim_loader = None

    def build_model(self):
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
        )
        self.init_optimizer = torch.optim.AdamW(
            model.initial_linear.parameters(),
            lr=self.config.training.init_lr,
            weight_decay=self.config.training.get('weight_decay', 1e-4),
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'FCSigmaUNet: {total_params / 1e6:.2f}M parameters')

    def build_datasets(self):
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
            y_ref['Uelref'])

        sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.745
        Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).reshape(-1)

        h5_path = self.config.data.hdf5_path
        train_indices = self.config.data.get('train_indices', None)
        val_indices = self.config.data.get('val_indices', None)
        test_indices = self.config.data.get('test_indices', None)

        train_ds = ConductivityHDF5Dataset(
            h5_path, Uelref, solver.InvLn, indices=train_indices, augment_noise=True)
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

        if val_indices is not None:
            val_ds = ConductivityHDF5Dataset(
                h5_path, Uelref, solver.InvLn, indices=val_indices, augment_noise=False)
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
                h5_path, Uelref, solver.InvLn, indices=test_indices, augment_noise=False)
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

    def _sample_levels(self, batch_size):
        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            return np.full(batch_size, fixed_level)
        return np.random.choice(np.arange(1, 8), size=batch_size)

    def _mask_measurements(self, measurements, levels):
        for k in range(measurements.shape[0]):
            measurements[k, ~self.vincl_dict[levels[k]]] = 0.0

    def _masked_image_loss(self, pred_sigma, target_sigma):
        mask = (target_sigma > 0).float()
        diff = (pred_sigma - target_sigma) * mask
        denom = mask.sum().clamp_min(1.0)
        mse = diff.pow(2).sum() / denom
        mae = diff.abs().sum() / denom
        total = (
            float(self.config.training.get('mse_weight', 1.0)) * mse
            + float(self.config.training.get('mae_weight', 0.1)) * mae
        )
        return total, mse, mae

    def train_step(self, batch):
        measurements, sigma, _ = batch
        levels = self._sample_levels(measurements.shape[0])
        self._mask_measurements(measurements, levels)

        if self.training_stage == 1:
            self.init_optimizer.zero_grad(set_to_none=True)
            measurements = measurements.to(self.device)
            sigma = sigma.to(self.device)
            with self._autocast_context():
                pred_sigma = self.model.linear_layer(measurements)
                loss, mse, mae = self._masked_image_loss(pred_sigma, sigma)
            loss.backward()
            self.optimizer_step(self.init_optimizer)
        else:
            self.optimizer.zero_grad(set_to_none=True)
            measurements = measurements.to(self.device)
            sigma = sigma.to(self.device)
            levels_tensor = torch.from_numpy(levels).float().to(self.device)
            with self._autocast_context():
                pred_sigma = self.model(measurements, levels_tensor)
                loss, mse, mae = self._masked_image_loss(pred_sigma, sigma)
            loss.backward()
            grad_clip = self.config.training.get('grad_clip_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer_step(self.optimizer)

        return {
            'loss': float(loss.detach().item()),
            'mse_loss': float(mse.detach().item()),
            'mae_loss': float(mae.detach().item()),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}
        fixed_level = self.config.training.get('fixed_level', 1)
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_samples = 0
        preds_all = []
        sigma_all = []

        with torch.no_grad():
            for measurements, sigma, _ in self.val_sim_loader:
                self._mask_measurements(
                    measurements, np.full(measurements.shape[0], fixed_level))
                measurements = measurements.to(self.device)
                sigma = sigma.to(self.device)
                levels_tensor = torch.full(
                    (measurements.shape[0],), fixed_level,
                    dtype=torch.float, device=self.device)
                with self._autocast_context():
                    pred_sigma = self.model(measurements, levels_tensor)
                    loss, mse, mae = self._masked_image_loss(pred_sigma, sigma)
                batch_size = measurements.shape[0]
                total_loss += float(loss.detach().item()) * batch_size
                total_mse += float(mse.detach().item()) * batch_size
                total_mae += float(mae.detach().item()) * batch_size
                num_samples += batch_size
                preds_all.append(pred_sigma.detach().float().cpu().numpy()[:, 0])
                sigma_all.append(sigma.detach().float().cpu().numpy()[:, 0])

        pred_np = np.concatenate(preds_all, axis=0)
        sigma_np = np.concatenate(sigma_all, axis=0)
        reg = masked_regression_metrics_batch(sigma_np, pred_np)
        metrics = {
            'val_loss': total_loss / max(num_samples, 1),
            'val_mse_loss': total_mse / max(num_samples, 1),
            'val_mae_loss': total_mae / max(num_samples, 1),
            'val_rmse': float(np.mean(reg['rmse'])),
            'val_rel_l2': float(np.mean(reg['rel_l2'])),
        }
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
        total_loss = 0.0
        num_samples = 0
        preds_all = []
        sigma_all = []
        with torch.no_grad():
            for measurements, sigma, _ in self.test_sim_loader:
                self._mask_measurements(
                    measurements, np.full(measurements.shape[0], fixed_level))
                measurements = measurements.to(self.device)
                sigma = sigma.to(self.device)
                levels_tensor = torch.full(
                    (measurements.shape[0],), fixed_level,
                    dtype=torch.float, device=self.device)
                with self._autocast_context():
                    pred_sigma = self.model(measurements, levels_tensor)
                    loss, _, _ = self._masked_image_loss(pred_sigma, sigma)
                batch_size = measurements.shape[0]
                total_loss += float(loss.detach().item()) * batch_size
                num_samples += batch_size
                preds_all.append(pred_sigma.detach().float().cpu().numpy()[:, 0])
                sigma_all.append(sigma.detach().float().cpu().numpy()[:, 0])

        pred_np = np.concatenate(preds_all, axis=0)
        sigma_np = np.concatenate(sigma_all, axis=0)
        reg = masked_regression_metrics_batch(sigma_np, pred_np)
        results = {
            'test_loss': total_loss / max(num_samples, 1),
            'test_mae': float(np.mean(reg['mae'])),
            'test_rmse': float(np.mean(reg['rmse'])),
            'test_rel_l2': float(np.mean(reg['rel_l2'])),
        }
        test_path = os.path.join(self.result_dir, 'test_results.json')
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'Test results saved to: {test_path}')
        return results

    def train(self):
        resume_path = self.config.training.get('resume_from', None)
        base_dir = getattr(self.config, 'result_base_dir', 'results')
        if resume_path:
            self.result_dir = self._find_checkpoint_dir(resume_path)
        else:
            self.result_dir = self._create_result_dir(
                self.experiment_name, base_dir=base_dir)

        self.build_model()
        self.build_datasets()

        if resume_path:
            self._load_checkpoint(resume_path)

        self._init_writer()
        self._save_config()

        max_iters = self.config.training.get('max_iters', None)
        init_epochs = self.config.training.init_epochs
        total_epochs = self.config.training.epochs
        save_freq = self.config.training.get('save_freq', 5)

        if self.training_stage == 1 and init_epochs > 0:
            init_start = self.current_epoch if self.current_epoch < init_epochs else init_epochs
            print(f'Stage 1: Pre-training initial_linear '
                  f'(epochs {init_start + 1}-{init_epochs})')
            print(f'Results directory: {self.result_dir}')
            for epoch in range(init_start, init_epochs):
                self.current_epoch = epoch
                self.model.train()
                self.training_stage = 1
                pbar = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc=f'Init {epoch + 1}/{init_epochs}',
                )
                total_loss = 0.0
                num_items = 0
                for _, batch in pbar:
                    loss_dict = self.train_step(batch)
                    self.mark_step()
                    batch_size = batch[0].shape[0]
                    total_loss += loss_dict['loss'] * batch_size
                    num_items += batch_size
                    self.global_step += 1
                    pbar.set_postfix(loss=f'{loss_dict["loss"]:.5f}')
                    if max_iters and self.global_step >= max_iters:
                        break

                avg_loss = total_loss / max(num_items, 1)
                print(f'  Init Epoch {epoch + 1} Avg Loss: {avg_loss:.5f}')
                self._log_epoch(epoch, {'avg_loss': avg_loss, 'stage': 'init'})
                if save_freq > 0 and (epoch + 1) % save_freq == 0:
                    self._save_checkpoint('last.pt')
                if max_iters and self.global_step >= max_iters:
                    self._save_checkpoint('last.pt')
                    print(f'Quick test: reached {max_iters} iterations.')
                    if self.writer:
                        self.writer.close()
                    print(f'Training complete. Results: {self.result_dir}')
                    return
            self._save_checkpoint('last.pt')
            self.training_stage = 2
            self.current_epoch = 0
            self.global_step = 0
        elif self.training_stage == 1:
            self.training_stage = 2

        self._es_counter = 0
        self._es_best_val_loss = None
        print(f'Stage 2: Full training (epochs {self.current_epoch + 1}-{total_epochs})')
        if max_iters:
            print(f'Quick test mode: stopping after {max_iters} iterations')

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch
            self.training_stage = 2
            epoch_metrics = self._train_epoch(epoch, max_iters=max_iters)

            val_metrics = {}
            val_freq = self.config.training.get('val_freq', 1)
            if val_freq > 0 and (epoch + 1) % val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    val_metrics = self.validate(epoch)
                self.model.train()

            all_metrics = {**epoch_metrics, **val_metrics}
            if self.scheduler is not None:
                sched_metric = val_metrics.get('val_loss', epoch_metrics.get('avg_loss'))
                if sched_metric is not None:
                    self.scheduler.step(sched_metric)
            all_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            if self.writer is not None:
                self.writer.add_scalar('train/lr', all_metrics['lr'], epoch + 1)

            self._log_epoch(epoch, all_metrics)
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self._save_checkpoint('last.pt')

            metric_name = self.config.training.get('selection_metric', 'val_loss')
            metric_value = val_metrics.get(metric_name, None)
            if metric_value is not None:
                is_better = (
                    self.best_metric is None
                    or metric_value < self.best_metric
                )
                if is_better:
                    self.best_metric = metric_value
                    self._save_checkpoint('best.pt')
                    print(f'  New best {metric_name}: {metric_value:.5f}')

            es_loss = val_metrics.get('val_loss', epoch_metrics.get('avg_loss'))
            if es_loss is not None:
                if self._es_best_val_loss is None or es_loss < self._es_best_val_loss:
                    self._es_best_val_loss = es_loss
                    self._es_counter = 0
                else:
                    self._es_counter += 1

            es_patience = self.config.training.get('early_stopping_patience', None)
            if es_patience and self._es_counter >= es_patience:
                print(f'Early stopping: val_loss not improved for {es_patience} epochs '
                      f'(best={self._es_best_val_loss:.4f})')
                break

            if max_iters and self.global_step >= max_iters:
                print(f'Quick test: reached {max_iters} iterations.')
                break

        self._save_checkpoint('last.pt')
        if self.writer:
            self.writer.close()
        print(f'Training complete. Results saved to: {self.result_dir}')

    def get_checkpoint_extra(self):
        return {
            'training_stage': self.training_stage,
            'init_optimizer_state_dict': (
                self.init_optimizer.state_dict()
                if self.init_optimizer else None),
        }

    def load_checkpoint_extra(self, state):
        self.training_stage = state.get('training_stage', 2)
        if self.init_optimizer is not None and state.get('init_optimizer_state_dict'):
            self.init_optimizer.load_state_dict(state['init_optimizer_state_dict'])

    @staticmethod
    def _find_checkpoint_dir(resume_path):
        return os.path.dirname(resume_path)
