"""Inference pipeline for VQ SAE predictor + frozen VQ SAE decoder."""

import os
import json

import numpy as np
import torch
import yaml

from .base_pipeline import BasePipeline
from ..models.vq_sae import ST1DVQVAE, VQMeasurementPredictor


class VQSAEPipeline(BasePipeline):
    def __init__(self, device='cuda', weights_base_dir='results',
                 config_path='scripts/vq_sae_pipeline.yaml',
                 vq_sae_dir_override='',
                 predictor_dir_override=''):
        super().__init__(device=device, weights_base_dir=weights_base_dir)
        self.predictor = None
        self.vq_sae = None
        self.config_path = config_path
        self.vq_sae_dir_override = vq_sae_dir_override
        self.predictor_dir_override = predictor_dir_override
        self.predictor_dir = None
        self.vq_sae_dir = None
        self.slot_vocab_values = None

    def _load_pipeline_config(self):
        if not self.config_path or not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _find_latest_dir(self, prefix):
        num = 1
        latest = None
        while True:
            d = os.path.join(self.weights_base_dir, f'{prefix}_{num}')
            if os.path.exists(d):
                has_weight = (
                    os.path.exists(os.path.join(d, 'best.pt'))
                    or os.path.exists(os.path.join(d, 'last.pt'))
                )
                if has_weight:
                    latest = d
                num += 1
            else:
                break
        return latest

    def _resolve_result_dir(self, configured_dir, prefix):
        if configured_dir:
            if os.path.isabs(configured_dir):
                return configured_dir
            return os.path.join(self.weights_base_dir, configured_dir)
        return self._find_latest_dir(prefix)

    @staticmethod
    def _load_result_config(result_dir):
        path = os.path.join(result_dir, 'config.yaml')
        with open(path, 'r') as f:
            return yaml.unsafe_load(f)

    @staticmethod
    def _load_slot_vocab(result_dir):
        path = os.path.join(result_dir, 'slot_vocab.json')
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_model(self, level):
        if self.predictor is not None:
            return

        pipe_cfg = self._load_pipeline_config()
        predictor_dir = self._resolve_result_dir(
            self.predictor_dir_override
            or pipe_cfg.get('vq_sae_predictor_dir', ''),
            prefix='vq_sae_predictor_baseline')
        explicit_vq_sae_dir = self.vq_sae_dir_override or pipe_cfg.get('vq_sae_dir', '')
        vq_sae_dir = self._resolve_result_dir(
            explicit_vq_sae_dir,
            prefix='vq_sae_baseline')

        if predictor_dir is None:
            raise FileNotFoundError(
                f'Cannot find VQ SAE weight directories. Searched in {self.weights_base_dir}')

        pred_cfg = self._load_result_config(predictor_dir)
        if vq_sae_dir is None:
            ckpt = pred_cfg.get('vq_sae', {}).get('checkpoint', '')
            if ckpt:
                vq_sae_dir = os.path.dirname(os.path.normpath(ckpt))
            else:
                vq_sae_dir = self._find_latest_dir('vq_sae_baseline')
        if vq_sae_dir is None:
            raise FileNotFoundError(
                f'Cannot find paired VQ SAE directory. Searched in {self.weights_base_dir}')
        sae_cfg = self._load_result_config(vq_sae_dir)

        slot_vocab = self._load_slot_vocab(predictor_dir)
        predictor = VQMeasurementPredictor(
            input_dim=pred_cfg['model']['input_dim'],
            hidden_dims=tuple(pred_cfg['model']['hidden_dims']),
            num_slots=pred_cfg['model']['num_slots'],
            codebook_size=pred_cfg['model']['codebook_size'],
            dropout=pred_cfg['model']['dropout'],
            slot_num_classes=(slot_vocab or {}).get('slot_num_classes', None),
        )
        pred_path = self._find_weight([
            os.path.join(predictor_dir, 'best.pt'),
            os.path.join(predictor_dir, 'last.pt'),
        ])
        predictor.load_state_dict(self._load_state_dict(pred_path, self.device))
        predictor.to(self.device)
        predictor.eval()

        vq_sae = ST1DVQVAE(
            in_channels=sae_cfg['model']['in_channels'],
            encoder_channels=tuple(sae_cfg['model']['encoder_channels']),
            num_slots=sae_cfg['model']['num_slots'],
            codebook_size=sae_cfg['model']['codebook_size'],
            code_dim=sae_cfg['model']['code_dim'],
            decoder_start_size=sae_cfg['model']['decoder_start_size'],
            vq_beta=sae_cfg['training']['vq_beta'],
        )
        sae_path = self._find_weight([
            os.path.join(vq_sae_dir, 'best.pt'),
            os.path.join(vq_sae_dir, 'last.pt'),
        ])
        vq_sae.load_state_dict(self._load_state_dict(sae_path, self.device))
        vq_sae.to(self.device)
        vq_sae.eval()

        self.predictor = predictor
        self.vq_sae = vq_sae
        self.predictor_dir = predictor_dir
        self.vq_sae_dir = vq_sae_dir
        self.slot_vocab_values = (slot_vocab.get('slot_vocab_values')
                                  if slot_vocab else None)
        print(f'VQSAEPipeline loaded: predictor={pred_path}, vq_sae={sae_path}')

    def _prepare_input(self, Uel, ref_data, level):
        Injref = ref_data['Injref']
        Uelref = np.asarray(ref_data['Uelref']).reshape(-1)
        vincl = self.create_vincl(level, Injref).T.flatten()
        y = np.asarray(Uel).reshape(-1) - Uelref
        y[~vincl] = 0.0
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def reconstruct(self, Uel, ref_data, level):
        return self.reconstruct_batch([Uel], ref_data, level)[0]


    def _decode_slot_logits(self, slot_logits):
        if isinstance(slot_logits, list):
            local_indices = [torch.argmax(logits, dim=-1) for logits in slot_logits]
            slot_indices = []
            for slot_idx, local_idx in enumerate(local_indices):
                vocab = torch.as_tensor(
                    self.slot_vocab_values[slot_idx],
                    device=local_idx.device,
                    dtype=torch.long,
                )
                slot_indices.append(vocab[local_idx.long()])
            return torch.stack(slot_indices, dim=1)
        return torch.argmax(slot_logits, dim=-1)

    def reconstruct_batch(self, Uel_list, ref_data, level):
        y_batch = [self._prepare_input(Uel, ref_data, level) for Uel in Uel_list]
        y_tensor = torch.from_numpy(np.stack(y_batch).astype(np.float32)).to(
            self.device)
        with torch.no_grad():
            with self._autocast_context():
                slot_logits, angle_xy = self.predictor(y_tensor)
                slot_indices = self._decode_slot_logits(slot_logits)
                logits = self.vq_sae.decode_from_indices(slot_indices, angle_xy)
                pred = torch.argmax(logits, dim=1)
        pred_np = pred.cpu().numpy().astype(int)
        return [arr for arr in pred_np]

    def reconstruct_mixed_batch(self, mixed_samples):
        y_batch = []
        for sample in mixed_samples:
            y_batch.append(self._prepare_input(
                sample['Uel'], sample['ref_data'], sample['level']))
        y_tensor = torch.from_numpy(np.stack(y_batch).astype(np.float32)).to(
            self.device)
        with torch.no_grad():
            with self._autocast_context():
                slot_logits, angle_xy = self.predictor(y_tensor)
                slot_indices = self._decode_slot_logits(slot_logits)
                logits = self.vq_sae.decode_from_indices(slot_indices, angle_xy)
                pred = torch.argmax(logits, dim=1)
        return [arr for arr in pred.cpu().numpy().astype(int)]
