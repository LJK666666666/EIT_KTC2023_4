"""
FCUNet trainer: two-stage training for direct measurement-to-segmentation.

Stage 1 (init_epochs):  Train only initial_linear layer with MSE loss.
Stage 2 (main epochs):  Train full model with CrossEntropyLoss.

Reference: KTC2023_SubmissionFiles/ktc_training/train_FCUNet.py
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
from ..configs import get_fcunet_config
from ..models.fcunet import FCUNet
from ..data import FCUNetTrainingData
from ..evaluation.scoring_torch import fast_score_batch_auto
from ..utils.measurement import create_vincl


class FCUNetTrainer(BaseTrainer):
    """Two-stage trainer for FCUNet.

    Stage 1: Pre-train initial_linear layer (MSE loss on difference map).
    Stage 2: Full model training (CrossEntropyLoss on 3-class segmentation).

    Both stages apply random level augmentation via vincl masks.
    """

    def __init__(self, config=None, experiment_name='fcunet_baseline'):
        if config is None:
            config = get_fcunet_config()
        super().__init__(config, experiment_name)

        self.vincl_dict = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_optimizer = None
        self.training_stage = 1  # 1 = init stage, 2 = main stage

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

        # Main optimizer (full model, stage 2)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.training.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor)

        # Init optimizer (initial_linear only, stage 1)
        self.init_optimizer = torch.optim.Adam(
            model.initial_linear.parameters(),
            lr=self.config.training.init_lr)

    def build_datasets(self):
        ref_path = self.config.data.ref_path
        mesh_name = self.config.data.mesh_name
        base_path = self.config.data.dataset_base_path

        # Load reference data and compute noise precision matrix
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

        # Simulate reference measurements
        # Flatten to 1D to avoid numpy.matrix shape issues from scipy sparse ops
        sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.745
        Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).flatten()

        # Training dataset (with noise augmentation)
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
        self._warn_if_dropping_last_batch(
            'train', len(dataset), self.config.training.batch_size)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=self._use_static_batches(),
            pin_memory=self._pin_memory_enabled(),
            num_workers=self.config.training.num_workers)

        # Simulated val/test datasets (no noise augmentation, deterministic)
        self.val_sim_loader = None
        self.test_sim_loader = None

        val_indices = self.config.data.get('val_indices', None)
        if val_indices is not None:
            if use_hdf5:
                val_ds = FCUNetHDF5Dataset(
                    h5_path, Uelref, solver.InvLn,
                    indices=val_indices, augment_noise=False)
            else:
                val_ds = FCUNetTrainingData(
                    Uelref, solver.InvLn, base_path,
                    indices=val_indices, augment_noise=False)
            self._warn_if_dropping_last_batch(
                'val', len(val_ds), self.config.training.batch_size)
            self.val_sim_loader = DataLoader(
                val_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers)

        test_indices = self.config.data.get('test_indices', None)
        if test_indices is not None:
            if use_hdf5:
                test_ds = FCUNetHDF5Dataset(
                    h5_path, Uelref, solver.InvLn,
                    indices=test_indices, augment_noise=False)
            else:
                test_ds = FCUNetTrainingData(
                    Uelref, solver.InvLn, base_path,
                    indices=test_indices, augment_noise=False)
            self._warn_if_dropping_last_batch(
                'test', len(test_ds), self.config.training.batch_size)
            self.test_sim_loader = DataLoader(
                test_ds,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                drop_last=self._use_static_batches(),
                pin_memory=self._pin_memory_enabled(),
                num_workers=self.config.training.num_workers)

        # Pre-compute vincl masks for all levels
        self.vincl_dict = {}
        for lvl in range(1, 8):
            self.vincl_dict[lvl] = create_vincl(lvl, Injref).T.flatten()

        # Challenge validation data (original behavior)
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
            'gt': np.stack(x_val),  # (4, 256, 256)
            'measurements': {
                lvl: np.stack(arrs) for lvl, arrs in y_val_dict.items()
            },  # {level: (4, 2356)}
        }

    def train_step(self, batch):
        y, gt = batch

        # Level augmentation: fixed or random
        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            levels = np.full(y.shape[0], fixed_level)
        else:
            levels = np.random.choice(np.arange(1, 8), size=y.shape[0])
        for k in range(y.shape[0]):
            y[k, ~self.vincl_dict[levels[k]]] = 0.0

        if self.training_stage == 1:
            # Stage 1: MSE on linear layer output vs GT difference map
            self.init_optimizer.zero_grad()

            gt_sum = torch.zeros(gt.shape[0], 256, 256)
            gt_sum = gt_sum - gt[:, 1, :, :] + gt[:, 2, :, :]
            gt_sum = gt_sum.unsqueeze(1)

            y = y.to(self.device)
            gt_sum = gt_sum.to(self.device)

            with self._autocast_context():
                x_pred = self.model.linear_layer(y)
                loss = torch.mean((x_pred - gt_sum) ** 2)

            loss.backward()
            self.optimizer_step(self.init_optimizer)
        else:
            # Stage 2: CrossEntropy on full model
            self.optimizer.zero_grad()

            levels_tensor = torch.from_numpy(levels).float()
            gt = gt.to(self.device)
            y = y.to(self.device)
            levels_tensor = levels_tensor.to(self.device)

            with self._autocast_context():
                pred = self.model(y, levels_tensor)
                loss = self.loss_fn(pred, gt)

            loss.backward()
            grad_clip = self.config.training.get('grad_clip_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer_step(self.optimizer)

        return {'loss': loss.item()}

    def validate(self, epoch):
        """Validate: compute val_loss only (no score computation)."""
        metrics = {}

        # Simulated val set: CE loss
        if self.val_sim_loader is not None:
            metrics = self._validate_sim(epoch)

        return metrics

    def _validate_challenge(self, epoch):
        """Evaluate on challenge test images across all 7 levels (score only).

        Not called during training. Use for testing/evaluation only.
        """
        gt_np = self.val_data['gt']  # (4, 256, 256)
        full_score = 0

        for level in range(1, 8):
            y_val = torch.from_numpy(
                self.val_data['measurements'][level]).float().to(self.device)
            level_inp = torch.full(
                (y_val.shape[0],), level, device=self.device)

            with torch.no_grad():
                with self._autocast_context():
                    pred = self.model(y_val, level_inp)
                    pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(
                pred_softmax, dim=1).cpu().numpy()

            level_scores = fast_score_batch_auto(
                gt_np[:pred_argmax.shape[0]], pred_argmax, device=self.device)
            mean_score = float(np.mean(level_scores)) if level_scores else 0.0

            if self.writer is not None:
                self.writer.add_scalar(
                    f'val/score_level{level}', mean_score, epoch + 1)
            full_score += mean_score

        avg_score = full_score / 7
        if self.writer is not None:
            self.writer.add_scalar('val/avg_score', avg_score, epoch + 1)
        print(f'  Val(challenge) score: {avg_score:.4f} (sum={full_score:.4f})')

        return {'challenge_score': avg_score}

    def _validate_sim(self, epoch):
        """Validate on simulated val set: CE loss only (no score computation)."""
        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is None:
            fixed_level = 1
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

    def _get_eval_checkpoint_path(self):
        """Return best checkpoint path, falling back to last.pt."""
        import os
        best_path = os.path.join(self.result_dir, 'best.pt')
        if not os.path.exists(best_path):
            best_path = os.path.join(self.result_dir, 'last.pt')
        return best_path

    def evaluate_test(self):
        """Evaluate best model on simulated test set. Returns loss + score."""
        if self.test_sim_loader is None:
            return {}

        self._load_checkpoint(self._get_eval_checkpoint_path())
        self.model.eval()

        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is None:
            fixed_level = 1
        total_loss = 0.0
        total_score = 0.0
        num_samples = 0

        with torch.no_grad():
            for y, gt in self.test_sim_loader:
                for k in range(y.shape[0]):
                    y[k, ~self.vincl_dict[fixed_level]] = 0.0

                levels_tensor = torch.full(
                    (y.shape[0],), fixed_level,
                    dtype=torch.float, device=self.device)
                y = y.to(self.device)
                gt = gt.to(self.device)

                with self._autocast_context():
                    pred = self.model(y, levels_tensor)
                    loss = self.loss_fn(pred, gt)
                total_loss += loss.item() * y.shape[0]

                pred_argmax = torch.argmax(
                    F.softmax(pred, dim=1), dim=1).cpu().numpy()
                gt_argmax = torch.argmax(gt, dim=1).cpu().numpy()
                batch_scores = fast_score_batch_auto(
                    gt_argmax, pred_argmax, device=self.device)
                total_score += float(np.sum(batch_scores))
                num_samples += y.shape[0]

        avg_loss = total_loss / max(num_samples, 1)
        avg_score = total_score / max(num_samples, 1)

        print(f'Test: loss={avg_loss:.5f}, score={avg_score:.4f} '
              f'({num_samples} samples)')

        import json
        results = {'test_loss': avg_loss, 'test_score': avg_score}
        test_path = os.path.join(self.result_dir, 'test_results.json')
        with open(test_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Test results saved to: {test_path}')

        return results

    def evaluate_challenge(self):
        """Evaluate best model on challenge validation images."""
        if self.val_data is None:
            return {}

        self._load_checkpoint(self._get_eval_checkpoint_path())
        self.model.eval()

        with torch.no_grad():
            return self._validate_challenge(epoch=self.current_epoch)

    # ------------------------------------------------------------------
    # Override main train loop for two-stage training
    # ------------------------------------------------------------------

    def train(self):
        """Two-stage training: init stage + main stage."""
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

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.result_dir)
        self._save_config()

        max_iters = self.config.training.get('max_iters', None)
        init_epochs = self.config.training.init_epochs
        total_epochs = self.config.training.epochs
        save_freq = self.config.training.get('save_freq', 5)

        # ---- Stage 1: Pre-train initial_linear ----
        if self.training_stage == 1:
            init_start = self.current_epoch if self.current_epoch < init_epochs else init_epochs
            print(f'Stage 1: Pre-training initial_linear '
                  f'(epochs {init_start + 1}-{init_epochs})')
            print(f'Results directory: {self.result_dir}')

            for epoch in range(init_start, init_epochs):
                self.current_epoch = epoch
                self.model.train()
                self.training_stage = 1

                pbar = tqdm(enumerate(self.train_loader),
                            total=len(self.train_loader),
                            desc=f'Init {epoch + 1}/{init_epochs}')
                total_loss, num_items = 0.0, 0
                for idx, batch in pbar:
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
                self._log_epoch(epoch, {'avg_loss': avg_loss,
                                        'stage': 'init'})
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

            # Transition to stage 2
            self.training_stage = 2
            self.current_epoch = 0
            self.global_step = 0

        # ---- Stage 2: Full model training ----
        # Reset early stopping for stage 2 (init stage scores are unreliable)
        self._es_counter = 0
        self._es_best_val_loss = None
        print(f'Stage 2: Full training (epochs {self.current_epoch + 1}'
              f'-{total_epochs})')
        if max_iters:
            print(f'Quick test mode: stopping after {max_iters} iterations')

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch
            self.training_stage = 2

            epoch_metrics = self._train_epoch(epoch, max_iters=max_iters)

            # Validate
            val_metrics = {}
            val_freq = self.config.training.get('val_freq', 1)
            if val_freq > 0 and (epoch + 1) % val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    val_metrics = self.validate(epoch)
                self.model.train()

            all_metrics = {**epoch_metrics, **val_metrics}

            # ReduceLROnPlateau: step with val_loss (preferred) or avg_loss
            if self.scheduler is not None:
                sched_metric = val_metrics.get(
                    'val_loss', epoch_metrics.get('avg_loss'))
                if sched_metric is not None:
                    self.scheduler.step(sched_metric)
            all_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', all_metrics['lr'],
                                   epoch + 1)

            self._log_epoch(epoch, all_metrics)
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self._save_checkpoint('last.pt')

            # Save best if val_loss improved (lower is better)
            if 'val_loss' in val_metrics:
                if (self.best_metric is None
                        or val_metrics['val_loss'] < self.best_metric):
                    self.best_metric = val_metrics['val_loss']
                    self._save_checkpoint('best.pt')
                    print(f'  New best val_loss: '
                          f'{val_metrics["val_loss"]:.5f}')

            # Early stopping based on val_loss (lower=better)
            es_loss = val_metrics.get(
                'val_loss', epoch_metrics.get('avg_loss'))
            if es_loss is not None:
                if (self._es_best_val_loss is None
                        or es_loss < self._es_best_val_loss):
                    self._es_best_val_loss = es_loss
                    self._es_counter = 0
                else:
                    self._es_counter += 1

            # Early stopping check
            es_patience = self.config.training.get(
                'early_stopping_patience', None)
            if es_patience and self._es_counter >= es_patience:
                print(f'Early stopping: val_loss not improved for '
                      f'{es_patience} epochs '
                      f'(best={self._es_best_val_loss:.4f})')
                break

            if max_iters and self.global_step >= max_iters:
                print(f'Quick test: reached {max_iters} iterations.')
                break

        self._save_checkpoint('last.pt')

        if self.writer:
            self.writer.close()
        print(f'Training complete. Results saved to: {self.result_dir}')

    # ------------------------------------------------------------------
    # Checkpoint extras
    # ------------------------------------------------------------------

    def get_checkpoint_extra(self):
        return {
            'training_stage': self.training_stage,
            'init_optimizer_state_dict': (
                self.init_optimizer.state_dict()
                if self.init_optimizer else None),
        }

    def load_checkpoint_extra(self, state):
        self.training_stage = state.get('training_stage', 2)
        if (self.init_optimizer is not None
                and state.get('init_optimizer_state_dict')):
            self.init_optimizer.load_state_dict(
                state['init_optimizer_state_dict'])

    @staticmethod
    def _find_checkpoint_dir(resume_path):
        """Get directory from checkpoint path."""
        import os
        return os.path.dirname(resume_path)
