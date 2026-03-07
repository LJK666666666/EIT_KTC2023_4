"""
Conditional Diffusion trainer: per-level DDPM/SDE training with EMA.

Trains a score model for a single difficulty level using epsilon-based
or score-based loss, depending on the SDE type. Includes EMA with
configurable warm-start.

Reference: KTC2023_SubmissionFiles/ktc_training/train_score.py
           src/diffusion/trainer.py
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..configs import get_condd_config
from ..diffusion.exp_utils import get_standard_score, get_standard_sde
from ..diffusion.losses import score_based_loss_fn, epsilon_based_loss_fn
from ..diffusion.ema import ExponentialMovingAverage
from ..diffusion.sde import _SCORE_PRED_CLASSES, _EPSILON_PRED_CLASSES
from ..data import MmapDataset, SimData
from ..evaluation.scoring import FastScoringFunction
from ..samplers import BaseSampler, wrapper_ddim


class CondDTrainer(BaseTrainer):
    """Trainer for conditional diffusion models (per-level).

    Each instance trains one difficulty level's model. The forward SDE
    (DDPM/VPSDE/VESDE) determines which loss function is used.

    EMA is initialised after ema_warm_start_steps and updated every step.
    Validation uses DDIM sampling + majority vote + FastScoringFunction.
    """

    def __init__(self, config=None, level=1,
                 experiment_name=None):
        if config is None:
            config = get_condd_config()
        if experiment_name is None:
            experiment_name = f'condd_level{level}'
        super().__init__(config, experiment_name)

        self.level = level
        self.sde = None
        self.ema = None
        self.ema_initialized = False
        self.loss_fn = None

    def build_model(self):
        self.sde = get_standard_sde(self.config)
        score = get_standard_score(
            self.config, self.sde, use_ema=False, load_model=False)
        score.to(self.device)
        self.model = score

        self.optimizer = torch.optim.Adam(
            score.parameters(), lr=float(self.config.training.lr))
        self.scheduler = None  # No scheduler for diffusion training

        # Select loss function based on SDE type
        if any(isinstance(self.sde, cls) for cls in _SCORE_PRED_CLASSES):
            self.loss_fn = score_based_loss_fn
        elif any(isinstance(self.sde, cls) for cls in _EPSILON_PRED_CLASSES):
            self.loss_fn = epsilon_based_loss_fn
        else:
            raise NotImplementedError(
                f'No loss function for SDE type: {type(self.sde)}')

    def build_datasets(self):
        use_mmap = self.config.data.get('use_mmap', False)
        base_path = self.config.data.get('dataset_base_path', 'dataset')
        mmap_base = self.config.data.get('mmap_base_path', base_path)
        level_to_num = self.config.data.level_to_num

        if use_mmap:
            dataset = MmapDataset(
                level=self.level,
                num_samples=level_to_num[self.level],
                base_path=mmap_base)
        else:
            dataset = SimData(
                level=self.level, base_path=base_path)

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            pin_memory=self.config.training.pin_memory,
            num_workers=self.config.training.num_workers)

        # Validation data
        self._load_val_data()

    def _load_val_data(self):
        """Load challenge validation data for this level."""
        gt_dir = self.config.validation.gt_dir
        reco_dir = self.config.validation.reco_dir
        num_val = self.config.validation.num_val_images

        x_val, c_val = [], []
        for i in range(1, num_val + 1):
            x = loadmat(f'{gt_dir}/true{i}.mat')['truth']
            reco_np = np.load(
                f'{reco_dir}/level_{self.level}/reco{i}.npy')
            x_val.append(x)
            c_val.append(reco_np)

        x_val = torch.from_numpy(np.stack(x_val)).float().unsqueeze(1)
        c_val = torch.from_numpy(np.stack(c_val)).float()

        self.val_data = {'gt': x_val, 'recos': c_val}

    def train_step(self, batch):
        cond, gt, _ = batch
        cond = cond.to(self.device)
        gt = gt.to(self.device)

        # gt shape: (B, 256, 256) → need (B, 1, 256, 256) for diffusion
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)

        self.optimizer.zero_grad()
        loss = self.loss_fn(gt, model=self.model, sde=self.sde, cond_inp=cond)
        loss.backward()

        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        # EMA management
        ema_warm = self.config.training.ema_warm_start_steps
        if not self.ema_initialized and self.global_step >= ema_warm:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.config.training.ema_decay)
            self.ema_initialized = True
        elif self.ema_initialized:
            self.ema.update(self.model.parameters())

        return {'loss': loss.item()}

    def validate(self, epoch):
        """Validate using DDIM sampling + FastScoringFunction."""
        if self.val_data is None:
            return {}

        c_test = self.val_data['recos'].to(self.device)
        x_test = self.val_data['gt']

        num_steps = self.config.validation.num_steps
        eps = self.config.validation.get('eps', 1e-3)

        sampler = BaseSampler(
            score=self.model,
            sde=self.sde,
            predictor=wrapper_ddim,
            sample_kwargs={
                'num_steps': num_steps,
                'start_time_step': 0,
                'batch_size': c_test.shape[0],
                'im_shape': [1, 256, 256],
                'eps': eps,
                'travel_length': 1,
                'travel_repeat': 1,
                'predictor': {'eta': 0.9},
            },
            device=self.device)

        x_mean = sampler.sample(c_test, logging=False)

        x_round = torch.round(x_mean).cpu().numpy()[:, 0, :, :]
        x_round = np.clip(x_round, 0, 2)

        mean_score = 0
        for i in range(x_round.shape[0]):
            score = FastScoringFunction(
                x_test[i, 0, :, :].numpy(), x_round[i])
            mean_score += score
        mean_score /= x_round.shape[0]

        self.writer.add_scalar('val/score', mean_score, epoch + 1)
        print(f'  Val score (level {self.level}): {mean_score:.4f}')

        return {'score': mean_score}

    # ------------------------------------------------------------------
    # Checkpoint extras (EMA state)
    # ------------------------------------------------------------------

    def get_checkpoint_extra(self):
        extra = {
            'ema_initialized': self.ema_initialized,
            'level': self.level,
        }
        if self.ema_initialized and self.ema is not None:
            extra['ema_state_dict'] = self.ema.state_dict()
        return extra

    def load_checkpoint_extra(self, state):
        self.ema_initialized = state.get('ema_initialized', False)
        if self.ema_initialized and 'ema_state_dict' in state:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.config.training.ema_decay)
            self.ema.load_state_dict(state['ema_state_dict'])

    # ------------------------------------------------------------------
    # Override save to also save EMA weights separately
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch, metrics):
        """Save EMA model weights separately for inference compatibility."""
        import os
        save_freq = self.config.training.get('save_model_every_n_epoch', 10)
        if (epoch + 1) % save_freq == 0 and self.ema_initialized:
            torch.save(self.ema.state_dict(),
                       os.path.join(self.result_dir,
                                    'ema_model_training.pt'))
            torch.save(self.model.state_dict(),
                       os.path.join(self.result_dir,
                                    'model_training.pt'))
