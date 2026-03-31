"""Benchmark official / fast / torch-fast KTC2023 scoring on a single sample.

This script is intentionally standalone for experimentation. It does not
modify the default evaluation flow.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.scoring import FastScoringFunction, scoring_function


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-path",
        type=str,
        default="KTC2023/EvaluationData/GroundTruths/level_1/1_true.mat",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["identity", "shift", "flip", "class_swap"],
        default="shift",
    )
    parser.add_argument("--official-runs", type=int, default=1)
    parser.add_argument("--fast-runs", type=int, default=20)
    parser.add_argument("--torch-runs", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--result-tag", type=str, default="scoring_benchmark")
    return parser.parse_args()


def create_result_dir(tag):
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)
    idx = 1
    while True:
        path = os.path.join(base_dir, f"{tag}_{idx}")
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        idx += 1


def load_truth(path):
    data = sio.loadmat(path)
    if "truth" not in data:
        raise KeyError(f"'truth' not found in {path}")
    return np.asarray(data["truth"]).astype(np.int64)


def make_reconstruction(truth, variant):
    reco = truth.copy()
    if variant == "identity":
        return reco
    if variant == "shift":
        reco = np.roll(reco, shift=8, axis=1)
        reco = np.roll(reco, shift=-5, axis=0)
        return reco
    if variant == "flip":
        return np.flip(reco, axis=1).copy()
    if variant == "class_swap":
        out = reco.copy()
        out[reco == 1] = 2
        out[reco == 2] = 1
        return out
    raise ValueError(f"Unknown variant: {variant}")


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


def _torch_kernel(device, dtype):
    r = 80
    ws = int(np.ceil(2 * r))
    wr = np.arange(-ws, ws + 1, dtype=np.float32)
    ker1d = ((1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * wr ** 2 / r ** 2))
    ker_h = torch.tensor(ker1d, device=device, dtype=dtype).view(1, 1, 1, -1)
    ker_v = torch.tensor(ker1d, device=device, dtype=dtype).view(1, 1, -1, 1)
    ones = torch.ones((1, 1, 256, 256), device=device, dtype=dtype)
    correction = F.conv2d(F.conv2d(ones, ker_h, padding=(0, ws)),
                          ker_v, padding=(ws, 0))
    return ker_h, ker_v, correction


class TorchFastScorer:
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.ker_h, self.ker_v, self.correction = _torch_kernel(device, dtype)
        self.c1 = 1e-4
        self.c2 = 9e-4
        self.ws = self.ker_h.shape[-1] // 2

    def score_batched(self, groundtruth, reconstruction):
        truth_c, reco_c, truth_d, reco_d = _build_mask_pair(
            groundtruth, reconstruction
        )
        truth = torch.tensor(
            np.stack([truth_c, truth_d]), device=self.device, dtype=self.dtype
        ).unsqueeze(1)
        reco = torch.tensor(
            np.stack([reco_c, reco_d]), device=self.device, dtype=self.dtype
        ).unsqueeze(1)

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
        return float(ssim.mean(dim=(1, 2, 3)).mean().item())


def _torch_ssim_pair(truth, reco, ker_h, ker_v, correction):
    c1 = 1e-4
    c2 = 9e-4

    gt = F.conv2d(F.conv2d(truth, ker_h, padding=(0, ker_h.shape[-1] // 2)),
                  ker_v, padding=(ker_v.shape[-2] // 2, 0)) / correction
    gr = F.conv2d(F.conv2d(reco, ker_h, padding=(0, ker_h.shape[-1] // 2)),
                  ker_v, padding=(ker_v.shape[-2] // 2, 0)) / correction

    mu_t2 = gt.square()
    mu_r2 = gr.square()
    mu_t_mu_r = gt * gr

    sigma_t2 = (
        F.conv2d(F.conv2d(truth.square(), ker_h, padding=(0, ker_h.shape[-1] // 2)),
                 ker_v, padding=(ker_v.shape[-2] // 2, 0)) / correction
    ) - mu_t2
    sigma_r2 = (
        F.conv2d(F.conv2d(reco.square(), ker_h, padding=(0, ker_h.shape[-1] // 2)),
                 ker_v, padding=(ker_v.shape[-2] // 2, 0)) / correction
    ) - mu_r2
    sigma_tr = (
        F.conv2d(F.conv2d(truth * reco, ker_h, padding=(0, ker_h.shape[-1] // 2)),
                 ker_v, padding=(ker_v.shape[-2] // 2, 0)) / correction
    ) - mu_t_mu_r

    num = (2 * mu_t_mu_r + c1) * (2 * sigma_tr + c2)
    den = (mu_t2 + mu_r2 + c1) * (sigma_t2 + sigma_r2 + c2)
    return (num / den).mean()


def torch_fast_scoring_function(groundtruth, reconstruction, device="cpu"):
    truth_c, reco_c, truth_d, reco_d = _build_mask_pair(groundtruth, reconstruction)
    dtype = torch.float32
    ker_h, ker_v, correction = _torch_kernel(device, dtype)

    truth = torch.tensor(
        np.stack([truth_c, truth_d]), device=device, dtype=dtype
    ).unsqueeze(1)
    reco = torch.tensor(
        np.stack([reco_c, reco_d]), device=device, dtype=dtype
    ).unsqueeze(1)

    s1 = _torch_ssim_pair(truth[0:1], reco[0:1], ker_h, ker_v, correction)
    s2 = _torch_ssim_pair(truth[1:2], reco[1:2], ker_h, ker_v, correction)
    return float((0.5 * (s1 + s2)).item())


def torch_fast_scoring_function_batched(groundtruth, reconstruction, device="cpu"):
    truth_c, reco_c, truth_d, reco_d = _build_mask_pair(groundtruth, reconstruction)
    dtype = torch.float32
    ker_h, ker_v, correction = _torch_kernel(device, dtype)

    truth = torch.tensor(
        np.stack([truth_c, truth_d]), device=device, dtype=dtype
    ).unsqueeze(1)
    reco = torch.tensor(
        np.stack([reco_c, reco_d]), device=device, dtype=dtype
    ).unsqueeze(1)

    c1 = 1e-4
    c2 = 9e-4
    ws = ker_h.shape[-1] // 2

    gt = F.conv2d(F.conv2d(truth, ker_h, padding=(0, ws)),
                  ker_v, padding=(ws, 0)) / correction
    gr = F.conv2d(F.conv2d(reco, ker_h, padding=(0, ws)),
                  ker_v, padding=(ws, 0)) / correction
    mu_t2 = gt.square()
    mu_r2 = gr.square()
    mu_t_mu_r = gt * gr
    sigma_t2 = F.conv2d(F.conv2d(truth.square(), ker_h, padding=(0, ws)),
                        ker_v, padding=(ws, 0)) / correction - mu_t2
    sigma_r2 = F.conv2d(F.conv2d(reco.square(), ker_h, padding=(0, ws)),
                        ker_v, padding=(ws, 0)) / correction - mu_r2
    sigma_tr = F.conv2d(F.conv2d(truth * reco, ker_h, padding=(0, ws)),
                        ker_v, padding=(ws, 0)) / correction - mu_t_mu_r
    ssim = ((2 * mu_t_mu_r + c1) * (2 * sigma_tr + c2)) / (
        (mu_t2 + mu_r2 + c1) * (sigma_t2 + sigma_r2 + c2)
    )
    return float(ssim.mean(dim=(1, 2, 3)).mean().item())


def benchmark(name, fn, n_runs, warmup=1, sync=None):
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()
    times = []
    out = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "name": name,
        "score": float(out),
        "runs": n_runs,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def main():
    args = parse_args()
    gt = load_truth(args.gt_path)
    reco = make_reconstruction(gt, args.variant)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    sync = None
    if device.startswith("cuda") and torch.cuda.is_available():
        sync = torch.cuda.synchronize

    results = []
    results.append(
        benchmark(
            "official",
            lambda: scoring_function(gt, reco),
            n_runs=args.official_runs,
        )
    )
    results.append(
        benchmark(
            "fast_scipy",
            lambda: FastScoringFunction(gt, reco),
            n_runs=args.fast_runs,
        )
    )
    results.append(
        benchmark(
            f"fast_torch_{device}",
            lambda: torch_fast_scoring_function(gt, reco, device=device),
            n_runs=args.torch_runs,
            sync=sync,
        )
    )
    results.append(
        benchmark(
            f"fast_torch_batched_{device}",
            lambda: torch_fast_scoring_function_batched(gt, reco, device=device),
            n_runs=args.torch_runs,
            sync=sync,
        )
    )
    cached_scorer = TorchFastScorer(device=device, dtype=torch.float32)
    results.append(
        benchmark(
            f"fast_torch_cached_{device}",
            lambda: cached_scorer.score_batched(gt, reco),
            n_runs=args.torch_runs,
            sync=sync,
        )
    )

    official_score = results[0]["score"]
    for r in results[1:]:
        r["abs_diff_vs_official"] = abs(r["score"] - official_score)
        r["speedup_vs_official"] = (
            results[0]["mean_ms"] / r["mean_ms"] if r["mean_ms"] > 0 else None
        )

    out_dir = create_result_dir(args.result_tag)
    summary = {
        "gt_path": args.gt_path,
        "variant": args.variant,
        "device": device,
        "results": results,
    }
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Output directory: {out_dir}")
    for r in results:
        extra = ""
        if "speedup_vs_official" in r:
            extra = (
                f", diff={r['abs_diff_vs_official']:.6e}, "
                f"speedup={r['speedup_vs_official']:.2f}x"
            )
        print(
            f"{r['name']}: score={r['score']:.6f}, mean={r['mean_ms']:.3f} ms"
            f" ({r['runs']} runs){extra}"
        )
    print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
