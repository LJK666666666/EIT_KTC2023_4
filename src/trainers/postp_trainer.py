"""
PostProcessing UNet trainer: single-stage CrossEntropy classification.

Trains a level-conditional UNet on 5-channel initial reconstructions
(ConcatDataset across all 7 difficulty levels).

Reference: KTC2023_SubmissionFiles/ktc_training/train_postprocessing.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from scipy.io import loadmat
from torch.utils.data import DataLoader, ConcatDataset

from .base_trainer import BaseTrainer
from ..configs import get_postp_config
from ..models.openai_unet import OpenAiUNetModel
from ..data import MmapDataset, SimData
from ..evaluation.scoring import FastScoringFunction


class PostPTrainer(BaseTrainer):
    """Trainer for the post-processing UNet method.

    Uses ConcatDataset of 7 difficulty levels. Each sample provides
    5-channel initial reconstructions and a GT segmentation map.
    The model predicts 3-class segmentation via CrossEntropyLoss.
    """

    def __init__(self, config=None, experiment_name='postp_baseline'):
        if config is None:
            config = get_postp_config()
        super().__init__(config, experiment_name)

        self.loss_fn = nn.CrossEntropyLoss()

    def build_model(self):
        model = OpenAiUNetModel(
            image_size=self.config.data.im_size,
            in_channels=self.config.model.in_channels,
            model_channels=self.config.model.model_channels,
            out_channels=self.config.model.out_channels,
            num_res_blocks=self.config.model.num_res_blocks,
            attention_resolutions=self.config.model.attention_resolutions,
            marginal_prob_std=None,
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

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.training.lr)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.training.scheduler_step_size,
            gamma=self.config.training.scheduler_gamma)

    def build_datasets(self):
        use_mmap = self.config.data.get('use_mmap', False)
        base_path = self.config.data.get('dataset_base_path', 'dataset')
        mmap_base = self.config.data.get('mmap_base_path', base_path)
        level_to_num = self.config.data.level_to_num

        dataset_list = []
        for level in range(1, 8):
            if use_mmap:
                ds = MmapDataset(
                    level=level,
                    num_samples=level_to_num[level],
                    base_path=mmap_base)
            else:
                ds = SimData(level=level, base_path=base_path)
            dataset_list.append(ds)

        dataset = ConcatDataset(dataset_list)
        print(f'PostP total training samples: {len(dataset)}')

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            pin_memory=self.config.training.pin_memory,
            num_workers=self.config.training.num_workers)

        # Validation data
        self._load_val_data()

    def _load_val_data(self):
        """Load pre-computed challenge reconstructions for validation."""
        gt_dir = self.config.validation.gt_dir
        reco_dir = self.config.validation.reco_dir
        num_val = self.config.validation.num_val_images

        x_val, c_val_dict = [], {}
        for i in range(1, num_val + 1):
            x = loadmat(f'{gt_dir}/true{i}.mat')['truth']
            x_val.append(x)

            for level in range(1, 8):
                reco_np = np.load(
                    f'{reco_dir}/level_{level}/reco{i}.npy')
                c_val_dict.setdefault(level, []).append(reco_np)

        self.val_data = {
            'gt': np.stack(x_val),
            'recos': {
                lvl: np.stack(arrs) for lvl, arrs in c_val_dict.items()
            },
        }

    def train_step(self, batch):
        reco, gt, level = batch

        # One-hot encode GT for CrossEntropyLoss
        gt_onehot = torch.zeros(gt.shape[0], 3, 256, 256)
        gt_onehot[:, 0, :, :] = (gt == 0).float()
        gt_onehot[:, 1, :, :] = (gt == 1).float()
        gt_onehot[:, 2, :, :] = (gt == 2).float()

        reco = reco.to(self.device)
        gt_onehot = gt_onehot.to(self.device)
        level = level.to(self.device)

        self.optimizer.zero_grad()
        pred = self.model(reco, level)
        loss = self.loss_fn(pred, gt_onehot)

        loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return {'loss': loss.item()}

    def validate(self, epoch):
        """Validate on challenge test images across all 7 levels."""
        if self.val_data is None:
            return {}

        gt_np = self.val_data['gt']
        full_score = 0

        for level in range(1, 8):
            c_val = torch.from_numpy(
                self.val_data['recos'][level]).float().to(self.device)
            level_inp = torch.full(
                (c_val.shape[0],), level,
                dtype=torch.float32, device=self.device)

            with torch.no_grad():
                pred = self.model(c_val, level_inp)
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
