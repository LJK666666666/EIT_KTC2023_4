"""
FCUNet trainer: two-stage training for direct measurement-to-segmentation.

Stage 1 (init_epochs):  Train only initial_linear layer with MSE loss.
Stage 2 (main epochs):  Train full model with CrossEntropyLoss.

Reference: KTC2023_SubmissionFiles/ktc_training/train_FCUNet.py
"""

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
from ..evaluation.scoring import FastScoringFunction
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
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.training.scheduler_step_size,
            gamma=self.config.training.scheduler_gamma)

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

        dataset = FCUNetTrainingData(Uelref, solver.InvLn, base_path)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            pin_memory=self.config.training.pin_memory,
            num_workers=self.config.training.num_workers)

        # Pre-compute vincl masks for all levels
        self.vincl_dict = {}
        for lvl in range(1, 8):
            self.vincl_dict[lvl] = create_vincl(lvl, Injref).T.flatten()

        # Validation data
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

        # Random level augmentation
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

            x_pred = self.model.linear_layer(y)
            loss = torch.mean((x_pred - gt_sum) ** 2)

            loss.backward()
            self.init_optimizer.step()
        else:
            # Stage 2: CrossEntropy on full model
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
        """Validate on challenge test images across all 7 levels."""
        if self.val_data is None:
            return {}

        gt_np = self.val_data['gt']  # (4, 256, 256)
        full_score = 0

        for level in range(1, 8):
            y_val = torch.from_numpy(
                self.val_data['measurements'][level]).float().to(self.device)
            level_inp = torch.full(
                (y_val.shape[0],), level, device=self.device)

            with torch.no_grad():
                pred = self.model(y_val, level_inp)
                pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(
                pred_softmax, dim=1).cpu().numpy()

            mean_score = 0
            for i in range(pred_argmax.shape[0]):
                score = FastScoringFunction(gt_np[i], pred_argmax[i])
                mean_score += score
            mean_score /= pred_argmax.shape[0]

            self.writer.add_scalar(
                f'val/score_level{level}', mean_score, epoch + 1)
            full_score += mean_score

        avg_score = full_score / 7
        self.writer.add_scalar('val/avg_score', avg_score, epoch + 1)
        print(f'  Val score: {avg_score:.4f} (sum={full_score:.4f})')

        return {'score': avg_score}

    # ------------------------------------------------------------------
    # Override main train loop for two-stage training
    # ------------------------------------------------------------------

    def train(self):
        """Two-stage training: init stage + main stage."""
        resume_path = self.config.training.get('resume_from', None)
        if resume_path:
            self.result_dir = self._find_checkpoint_dir(resume_path)
        else:
            self.result_dir = self._create_result_dir(self.experiment_name)

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
                self._save_checkpoint('last.pt')

                if max_iters and self.global_step >= max_iters:
                    print(f'Quick test: reached {max_iters} iterations.')
                    if self.writer:
                        self.writer.close()
                    print(f'Training complete. Results: {self.result_dir}')
                    return

            # Transition to stage 2
            self.training_stage = 2
            self.current_epoch = 0
            self.global_step = 0

        # ---- Stage 2: Full model training ----
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

            if self.scheduler is not None:
                self.scheduler.step()
                all_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/lr', all_metrics['lr'],
                                       epoch + 1)

            self._log_epoch(epoch, all_metrics)
            self._save_checkpoint('last.pt')

            if 'score' in val_metrics:
                if (self.best_metric is None
                        or val_metrics['score'] > self.best_metric):
                    self.best_metric = val_metrics['score']
                    self._save_checkpoint('best.pt')
                    print(f'  New best score: {val_metrics["score"]:.4f}')

            if max_iters and self.global_step >= max_iters:
                print(f'Quick test: reached {max_iters} iterations.')
                break

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
