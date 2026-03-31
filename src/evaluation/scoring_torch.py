"""Torch-accelerated fast scorer for KTC2023 segmentation evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .scoring import FastScoringFunction


def _build_mask_pair(groundtruth, reconstruction):
    truth_c = np.zeros_like(groundtruth, dtype=np.float32)
    truth_c[np.abs(groundtruth - 2) < 0.1] = 1.0
    reco_c = np.zeros_like(reconstruction, dtype=np.float32)
    reco_c[np.abs(reconstruction - 2) < 0.1] = 1.0

    truth_d = np.zeros_like(groundtruth, dtype=np.float32)
    truth_d[np.abs(groundtruth - 1) < 0.1] = 1.0
    reco_d = np.zeros_like(reconstruction, dtype=np.float32)
    reco_d[np.abs(reconstruction - 1) < 0.1] = 1.0
    return truth_c, reco_c, truth_d, reco_d


class TorchFastScorer:
    """Fast scorer using separable Gaussian convolutions in Torch."""

    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.c1 = 1e-4
        self.c2 = 9e-4
        self.ker_h, self.ker_v, self.correction = self._build_kernel()
        self.ws = self.ker_h.shape[-1] // 2

    def _build_kernel(self):
        r = 80
        ws = int(np.ceil(2 * r))
        wr = np.arange(-ws, ws + 1, dtype=np.float32)
        ker1d = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * wr ** 2 / r ** 2)
        ker_h = torch.tensor(
            ker1d, device=self.device, dtype=self.dtype
        ).view(1, 1, 1, -1)
        ker_v = torch.tensor(
            ker1d, device=self.device, dtype=self.dtype
        ).view(1, 1, -1, 1)
        ones = torch.ones((1, 1, 256, 256), device=self.device, dtype=self.dtype)
        correction = F.conv2d(
            F.conv2d(ones, ker_h, padding=(0, ws)),
            ker_v, padding=(ws, 0)
        )
        return ker_h, ker_v, correction

    def _score_two_class_batch(self, truth, reco):
        gt = F.conv2d(F.conv2d(truth, self.ker_h, padding=(0, self.ws)),
                      self.ker_v, padding=(self.ws, 0)) / self.correction
        gr = F.conv2d(F.conv2d(reco, self.ker_h, padding=(0, self.ws)),
                      self.ker_v, padding=(self.ws, 0)) / self.correction

        mu_t2 = gt.square()
        mu_r2 = gr.square()
        mu_t_mu_r = gt * gr

        sigma_t2 = (
            F.conv2d(F.conv2d(truth.square(), self.ker_h, padding=(0, self.ws)),
                     self.ker_v, padding=(self.ws, 0)) / self.correction
        ) - mu_t2
        sigma_r2 = (
            F.conv2d(F.conv2d(reco.square(), self.ker_h, padding=(0, self.ws)),
                     self.ker_v, padding=(self.ws, 0)) / self.correction
        ) - mu_r2
        sigma_tr = (
            F.conv2d(F.conv2d(truth * reco, self.ker_h, padding=(0, self.ws)),
                     self.ker_v, padding=(self.ws, 0)) / self.correction
        ) - mu_t_mu_r

        ssim = ((2 * mu_t_mu_r + self.c1) * (2 * sigma_tr + self.c2)) / (
            (mu_t2 + mu_r2 + self.c1) * (sigma_t2 + sigma_r2 + self.c2)
        )
        return ssim.mean(dim=(1, 2, 3))

    def score_batch(self, groundtruths, reconstructions):
        """Score a batch of label images.

        Args:
            groundtruths: list/array of shape (N, 256, 256)
            reconstructions: list/array of shape (N, 256, 256)

        Returns:
            List[float] of length N.
        """
        gt_np = np.asarray(groundtruths)
        reco_np = np.asarray(reconstructions)
        if gt_np.ndim != 3 or reco_np.ndim != 3:
            raise ValueError("Expected (N, H, W) arrays for batch scoring.")
        if gt_np.shape != reco_np.shape:
            raise ValueError("Ground truth and reconstruction batch shapes mismatch.")
        if gt_np.shape[1:] != (256, 256):
            raise ValueError("Only 256x256 label maps are supported.")

        truth_c = (np.abs(gt_np - 2) < 0.1).astype(np.float32)
        reco_c = (np.abs(reco_np - 2) < 0.1).astype(np.float32)
        truth_d = (np.abs(gt_np - 1) < 0.1).astype(np.float32)
        reco_d = (np.abs(reco_np - 1) < 0.1).astype(np.float32)

        truth = torch.tensor(
            np.stack([truth_c, truth_d], axis=1),
            device=self.device, dtype=self.dtype
        )
        reco = torch.tensor(
            np.stack([reco_c, reco_d], axis=1),
            device=self.device, dtype=self.dtype
        )

        score_c = self._score_two_class_batch(truth[:, 0:1], reco[:, 0:1])
        score_d = self._score_two_class_batch(truth[:, 1:2], reco[:, 1:2])
        scores = 0.5 * (score_c + score_d)
        return [float(x) for x in scores.detach().cpu().numpy()]

    def score_single(self, groundtruth, reconstruction):
        return self.score_batch(
            np.asarray(groundtruth)[None], np.asarray(reconstruction)[None]
        )[0]


_SCORER_CACHE = {}


def _normalize_device(device):
    if isinstance(device, torch.device):
        device = str(device)
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    device = str(device)
    if device.startswith('cuda'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('xla') or device.startswith('tpu'):
        return 'cpu'
    return device


def get_auto_fast_scorer(device=None):
    resolved = _normalize_device(device)
    if resolved == 'cpu':
        return None
    if resolved not in _SCORER_CACHE:
        _SCORER_CACHE[resolved] = TorchFastScorer(device=resolved)
    return _SCORER_CACHE[resolved]


def fast_score_auto(groundtruth, reconstruction, device=None):
    scorer = get_auto_fast_scorer(device)
    if scorer is None:
        return float(FastScoringFunction(groundtruth, reconstruction))
    return float(scorer.score_single(groundtruth, reconstruction))


def fast_score_batch_auto(groundtruths, reconstructions, device=None):
    scorer = get_auto_fast_scorer(device)
    if scorer is None:
        return [
            float(FastScoringFunction(gt, reco))
            for gt, reco in zip(groundtruths, reconstructions)
        ]
    return [float(x) for x in scorer.score_batch(groundtruths, reconstructions)]
