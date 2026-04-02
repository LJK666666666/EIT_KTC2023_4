"""Trainer for measurement -> discrete slot predictor."""

import os
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from scipy.io import loadmat
from torch.utils.data import DataLoader
import yaml

from .base_trainer import BaseTrainer
from ..configs.vq_sae_predictor_config import (
    get_configs as get_vq_sae_predictor_config,
)
from ..data import VQSAEPredictorHDF5Dataset, EvaluationDataLoader
from ..evaluation.scoring_torch import fast_score_batch_auto
from ..ktc_methods import EITFEM, load_mesh
from ..losses.dice_focal import DiceLoss
from ..models.vq_sae import ST1DVQVAE, VQMeasurementPredictor
from ..utils.measurement import create_vincl


def _build_rotation_matrix(theta):
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    row1 = torch.stack([cos_t, -sin_t, zeros], dim=-1)
    row2 = torch.stack([sin_t, cos_t, zeros], dim=-1)
    return torch.stack([row1, row2], dim=1)


class VQSAEPredictorTrainer(BaseTrainer):
    def __init__(self, config=None, experiment_name='vq_sae_predictor_baseline'):
        if config is None:
            config = get_vq_sae_predictor_config()
        super().__init__(config, experiment_name)
        self.vincl_dict = None
        self.vq_sae_model = None
        self.slot_vocab_values = None
        self.slot_class_maps = None
        self.dice_loss = DiceLoss()
        self.eval_loader = None

    def _build_slot_vocab(self, latent_h5_path):
        import h5py

        with h5py.File(latent_h5_path, 'r') as f:
            all_indices = f['indices'][:]
        slot_vocab_values = []
        slot_class_maps = []
        for slot_idx in range(all_indices.shape[1]):
            vocab = np.unique(all_indices[:, slot_idx]).astype(np.int64)
            slot_vocab_values.append(vocab.tolist())
            slot_class_maps.append({
                int(code): local_idx for local_idx, code in enumerate(vocab)
            })
        self.slot_vocab_values = slot_vocab_values
        self.slot_class_maps = slot_class_maps

    def _save_slot_vocab(self):
        if self.slot_vocab_values is None:
            return
        os.makedirs(self.result_dir, exist_ok=True)
        path = os.path.join(self.result_dir, 'slot_vocab.json')
        payload = {
            'num_slots': len(self.slot_vocab_values),
            'slot_vocab_values': self.slot_vocab_values,
            'slot_num_classes': [len(v) for v in self.slot_vocab_values],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def build_model(self):
        if not self.config.vq_sae.latent_h5_path:
            raise ValueError(
                'VQ SAE predictor requires vq_sae.latent_h5_path / ' 
                '--vq-latent-h5-path')
        self._build_slot_vocab(self.config.vq_sae.latent_h5_path)

        model = VQMeasurementPredictor(
            input_dim=self.config.model.input_dim,
            hidden_dims=tuple(self.config.model.hidden_dims),
            num_slots=self.config.model.num_slots,
            codebook_size=self.config.model.codebook_size,
            dropout=self.config.model.dropout,
            slot_num_classes=[len(v) for v in self.slot_vocab_values],
        )
        model.to(self.device)
        self.model = model

        if not self.config.vq_sae.checkpoint:
            raise ValueError(
                'VQ SAE predictor requires vq_sae.checkpoint / --vq-sae-checkpoint')
        self._load_vq_sae_model(self.config.vq_sae.checkpoint)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.get('weight_decay', 1e-4),
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.training.get('selection_metric_mode', 'min'),
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'VQMeasurementPredictor: {total_params / 1e6:.2f}M parameters')
        print('Slot vocab sizes:', [len(v) for v in self.slot_vocab_values])
        self._save_slot_vocab()

    def _load_vq_sae_model(self, ckpt_path):
        state = torch.load(ckpt_path, map_location=self.device,
                           weights_only=False)
        sd = state.get('model_state_dict', state)
        cfg_path = os.path.join(os.path.dirname(ckpt_path), 'config.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.unsafe_load(f)
        model_cfg = cfg['model']
        vq_sae = ST1DVQVAE(
            in_channels=model_cfg['in_channels'],
            encoder_channels=tuple(model_cfg['encoder_channels']),
            num_slots=model_cfg['num_slots'],
            codebook_size=model_cfg['codebook_size'],
            code_dim=model_cfg['code_dim'],
            decoder_start_size=model_cfg['decoder_start_size'],
            vq_beta=cfg['training']['vq_beta'],
        )
        vq_sae.load_state_dict(sd)
        vq_sae.to(self.device)
        vq_sae.eval()
        for p in vq_sae.parameters():
            p.requires_grad = False
        self.vq_sae_model = vq_sae
        print(f'Loaded frozen VQ SAE from {ckpt_path}')

    def build_datasets(self):
        if not self.config.data.get('use_hdf5', False):
            raise ValueError('VQ SAE predictor requires HDF5 dataset.')

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
        Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).flatten()

        train_ds = VQSAEPredictorHDF5Dataset(
            self.config.data.hdf5_path,
            self.config.vq_sae.latent_h5_path,
            Uelref,
            solver.InvLn,
            indices=self.config.data.get('train_indices', None),
            augment_noise=True,
            augment_rotation=True,
            slot_class_maps=self.slot_class_maps,
        )
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
        val_indices = self.config.data.get('val_indices', None)
        if val_indices is not None:
            val_ds = VQSAEPredictorHDF5Dataset(
                self.config.data.hdf5_path,
                self.config.vq_sae.latent_h5_path,
                Uelref,
                solver.InvLn,
                indices=val_indices,
                augment_noise=False,
                augment_rotation=False,
                slot_class_maps=self.slot_class_maps,
            )
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

        self.vincl_dict = {}
        for lvl in range(1, 8):
            self.vincl_dict[lvl] = create_vincl(lvl, Injref).T.flatten()
        self.eval_loader = EvaluationDataLoader()

    def _slot_ce_loss(self, slot_logits, target_indices):
        if isinstance(slot_logits, list):
            losses = []
            for slot_idx, logits in enumerate(slot_logits):
                losses.append(F.cross_entropy(logits, target_indices[:, slot_idx]))
            return torch.stack(losses).mean()
        bsz, num_slots, codebook_size = slot_logits.shape
        flat_logits = slot_logits.reshape(bsz * num_slots, codebook_size)
        flat_target = target_indices.reshape(bsz * num_slots)
        return F.cross_entropy(flat_logits, flat_target)

    def _soft_decode_logits(self, slot_logits, pred_angle):
        emb_weight = self.vq_sae_model.quantizer.embedding.weight
        if isinstance(slot_logits, list):
            z_slots = []
            for slot_idx, logits in enumerate(slot_logits):
                probs = torch.softmax(logits.float(), dim=-1)
                global_ids = torch.as_tensor(
                    self.slot_vocab_values[slot_idx],
                    device=logits.device,
                    dtype=torch.long,
                )
                slot_emb = emb_weight[global_ids]  # (Ci, D)
                z_slots.append(probs @ slot_emb.float())
            z_q = torch.stack(z_slots, dim=1)
        else:
            probs = torch.softmax(slot_logits.float(), dim=-1)
            z_q = torch.einsum('bsk,kd->bsd', probs, emb_weight.float())
        return self.vq_sae_model.decode_quantized(z_q.to(pred_angle.dtype), pred_angle)

    def _hard_decode_logits(self, slot_logits, pred_angle):
        if isinstance(slot_logits, list):
            slot_indices = []
            for slot_idx, logits in enumerate(slot_logits):
                local_idx = torch.argmax(logits, dim=-1)
                global_ids = torch.as_tensor(
                    self.slot_vocab_values[slot_idx],
                    device=logits.device,
                    dtype=torch.long,
                )
                slot_indices.append(global_ids[local_idx.long()])
            slot_indices = torch.stack(slot_indices, dim=1)
        else:
            slot_indices = torch.argmax(slot_logits, dim=-1)
        return self.vq_sae_model.decode_from_indices(slot_indices, pred_angle)

    def _rotate_gt_indices(self, gt_indices, rot_steps):
        if rot_steps is None:
            return gt_indices
        theta = rot_steps.float() * (2.0 * np.pi / 32.0)
        affine = _build_rotation_matrix(theta).to(gt_indices.device, dtype=torch.float32)
        x = gt_indices.unsqueeze(1).float()
        grid = F.affine_grid(affine, size=x.shape, align_corners=False)
        x_rot = F.grid_sample(x, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        return x_rot[:, 0].long()

    def _image_loss(self, logits, gt_indices):
        ce = F.cross_entropy(logits, gt_indices)
        dice = self.dice_loss(logits, gt_indices)
        ce_weight = float(self.config.training.get('ce_weight', 1.0))
        dice_weight = float(self.config.training.get('dice_weight', 1.0))
        return ce_weight * ce + dice_weight * dice, ce, dice

    def _prepare_eval_input(self, Uel, ref_data, level):
        Uelref = np.asarray(ref_data['Uelref']).reshape(-1)
        vincl = self.vincl_dict[level]
        y = np.asarray(Uel).reshape(-1) - Uelref
        y[~vincl] = 0.0
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def _run_score_probe(self):
        all_scores = []
        level_scores = {}
        for level in range(1, 8):
            ref_data, uels, gts = self.eval_loader.load_all_for_level(level)
            y_batch = [
                self._prepare_eval_input(Uel, ref_data, level)
                for Uel in uels
            ]
            measurements = torch.from_numpy(
                np.stack(y_batch).astype(np.float32)
            ).to(self.device)
            with torch.no_grad():
                with self._autocast_context():
                    slot_logits, pred_angle = self.model(measurements)
                    decoded_logits = self._hard_decode_logits(slot_logits, pred_angle)
                    preds = torch.argmax(decoded_logits, dim=1).cpu().numpy().astype(int)
            scores = fast_score_batch_auto(gts, preds, device=self.device)
            all_scores.extend(scores)
            level_scores[level] = float(np.mean(scores))
        return {
            'val_probe_score_mean': float(np.mean(all_scores)),
            'val_probe_score_total': float(np.sum(all_scores)),
            'val_probe_level_scores': level_scores,
        }

    def train_step(self, batch):
        measurements, target_indices, target_angle, gt_indices, rot_steps = batch

        fixed_level = self.config.training.get('fixed_level', None)
        if fixed_level is not None:
            levels = np.full(measurements.shape[0], fixed_level)
        else:
            levels = np.random.choice(np.arange(1, 8), size=measurements.shape[0])
        for k in range(measurements.shape[0]):
            measurements[k, ~self.vincl_dict[levels[k]]] = 0.0

        measurements = measurements.to(self.device)
        target_indices = target_indices.to(self.device)
        target_angle = target_angle.to(self.device)
        gt_indices = gt_indices.to(self.device)
        rot_steps = rot_steps.to(self.device)
        gt_rot = self._rotate_gt_indices(gt_indices, rot_steps)

        self.optimizer.zero_grad()
        with self._autocast_context():
            slot_logits, pred_angle = self.model(measurements)
            slot_loss = self._slot_ce_loss(slot_logits, target_indices)
            angle_loss = F.mse_loss(pred_angle, target_angle)
            decoded_logits = self._soft_decode_logits(slot_logits, pred_angle)
            image_loss, ce_loss, dice_loss = self._image_loss(decoded_logits, gt_rot)
            total_loss = (
                float(self.config.training.get('lambda_slot', 1.0)) * slot_loss
                + float(self.config.training.get('lambda_angle', 0.5)) * angle_loss
                + float(self.config.training.get('lambda_image', 1.0)) * image_loss
            )

        total_loss.backward()
        grad_clip = self.config.training.get('grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer_step(self.optimizer)

        return {
            'loss': total_loss.item(),
            'slot_loss': slot_loss.item(),
            'angle_loss': angle_loss.item(),
            'image_loss': image_loss.item(),
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
        }

    def validate(self, epoch):
        if self.val_sim_loader is None:
            return {}

        totals = {
            'loss': 0.0,
            'slot_loss': 0.0,
            'angle_loss': 0.0,
            'image_loss': 0.0,
            'ce_loss': 0.0,
            'dice_loss': 0.0,
        }
        num_samples = 0
        fixed_level = self.config.training.get('fixed_level', None) or 1
        for measurements, target_indices, target_angle, gt_indices, rot_steps in self.val_sim_loader:
            for k in range(measurements.shape[0]):
                measurements[k, ~self.vincl_dict[fixed_level]] = 0.0

            measurements = measurements.to(self.device)
            target_indices = target_indices.to(self.device)
            target_angle = target_angle.to(self.device)
            gt_indices = gt_indices.to(self.device)
            rot_steps = rot_steps.to(self.device)
            gt_rot = self._rotate_gt_indices(gt_indices, rot_steps)

            with torch.no_grad():
                with self._autocast_context():
                    slot_logits, pred_angle = self.model(measurements)
                    slot_loss = self._slot_ce_loss(slot_logits, target_indices)
                    angle_loss = F.mse_loss(pred_angle, target_angle)
                    decoded_logits = self._soft_decode_logits(slot_logits, pred_angle)
                    image_loss, ce_loss, dice_loss = self._image_loss(decoded_logits, gt_rot)
                    total_loss = (
                        float(self.config.training.get('lambda_slot', 1.0)) * slot_loss
                        + float(self.config.training.get('lambda_angle', 0.5)) * angle_loss
                        + float(self.config.training.get('lambda_image', 1.0)) * image_loss
                    )
            batch_size = measurements.shape[0]
            totals['loss'] += total_loss.item() * batch_size
            totals['slot_loss'] += slot_loss.item() * batch_size
            totals['angle_loss'] += angle_loss.item() * batch_size
            totals['image_loss'] += image_loss.item() * batch_size
            totals['ce_loss'] += ce_loss.item() * batch_size
            totals['dice_loss'] += dice_loss.item() * batch_size
            num_samples += batch_size

        metrics = {
            'val_loss': totals['loss'] / max(num_samples, 1),
            'val_slot_loss': totals['slot_loss'] / max(num_samples, 1),
            'val_angle_loss': totals['angle_loss'] / max(num_samples, 1),
            'val_image_loss': totals['image_loss'] / max(num_samples, 1),
            'val_ce_loss': totals['ce_loss'] / max(num_samples, 1),
            'val_dice_loss': totals['dice_loss'] / max(num_samples, 1),
        }
        probe_freq = int(self.config.training.get('score_probe_freq', 0) or 0)
        if probe_freq > 0 and ((epoch + 1) % probe_freq == 0):
            metrics.update(self._run_score_probe())
        if self.writer:
            self.writer.add_scalar('val/loss', metrics['val_loss'], epoch + 1)
            self.writer.add_scalar('val/slot_loss', metrics['val_slot_loss'], epoch + 1)
            self.writer.add_scalar('val/angle_loss', metrics['val_angle_loss'], epoch + 1)
            self.writer.add_scalar('val/image_loss', metrics['val_image_loss'], epoch + 1)
            self.writer.add_scalar('val/ce_loss', metrics['val_ce_loss'], epoch + 1)
            self.writer.add_scalar('val/dice_loss', metrics['val_dice_loss'], epoch + 1)
            if 'val_probe_score_mean' in metrics:
                self.writer.add_scalar('val/probe_score_mean', metrics['val_probe_score_mean'], epoch + 1)
                self.writer.add_scalar('val/probe_score_total', metrics['val_probe_score_total'], epoch + 1)
        print(f'  Val loss: {metrics["val_loss"]:.6f}')
        if 'val_probe_score_total' in metrics:
            print(
                f'  Probe score: mean={metrics["val_probe_score_mean"]:.6f}, '
                f'total={metrics["val_probe_score_total"]:.6f}'
            )
        return metrics
