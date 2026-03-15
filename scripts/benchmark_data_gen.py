"""
Benchmark CPU vs GPU data generation for KTC2023 EIT.

Generates N samples with both CPU and GPU modes, records per-step
timing, and saves comparison results.

Usage:
    python scripts/benchmark_data_gen.py --num-samples 10
    python scripts/benchmark_data_gen.py --num-samples 10 --measurements-only
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import loadmat

from src.ktc_methods import EITFEM, load_mesh, image_to_mesh
from src.data import create_phantoms
from src.utils.measurement import create_vincl
from src.configs.condd_config import LEVEL_TO_ALPHAS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_benchmark(num_samples, level, ref_path, mesh_name,
                  measurements_only, use_gpu, seed):
    """Run data generation benchmark, return per-step timing dict."""
    set_seed(seed)

    y_ref = loadmat(ref_path)
    Injref = y_ref['Injref']
    Mpat = y_ref['Mpat']
    mesh, mesh2 = load_mesh(mesh_name)

    Nel = 32
    z = 1e-6 * np.ones((Nel, 1))
    vincl = np.ones((Nel - 1, 76), dtype=bool)

    solver = EITFEM(mesh2, Injref, Mpat, vincl, use_gpu=use_gpu)
    solver.SetInvGamma(0.05, 0.01, y_ref['Uelref'])

    sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.745
    Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).reshape(-1, 1)
    noise = np.asarray(
        solver.InvLn * np.random.randn(Uelref.shape[0], 1)).reshape(-1, 1)
    Uelref = Uelref + noise

    reconstructor = None
    alphas = None
    if not measurements_only:
        from src.reconstruction.linearised_reco import LinearisedRecoFenics
        alphas = LEVEL_TO_ALPHAS[level]
        B = Mpat.T
        vincl_level = create_vincl(level, Injref)
        reconstructor = LinearisedRecoFenics(
            Uelref, B, vincl_level, mesh_name='sparse',
            base_path='KTC2023_SubmissionFiles/data',
            use_gpu=use_gpu)

    timing = {
        'phantom': [], 'forward': [], 'noise': [],
        'reco': [], 'interp': [], 'total': [],
    }

    # Warm-up: 1 sample (GPU needs warm-up for kernel compilation)
    set_seed(seed)
    _ = create_phantoms()
    sigma_bg_copy = np.ones((mesh.g.shape[0], 1)) * 0.745
    _ = solver.SolveForward(sigma_bg_copy, z.copy())

    set_seed(seed)
    for i in range(num_samples):
        t_total = time.time()

        t0 = time.time()
        sigma_pix = create_phantoms()
        timing['phantom'].append(time.time() - t0)

        background = 0.745
        resistive = np.random.rand() * 0.1 + 0.025
        conductive = np.random.rand() + 5.0
        sigma = np.zeros(sigma_pix.shape)
        sigma[sigma_pix == 0.0] = background
        sigma[sigma_pix == 1.0] = resistive
        sigma[sigma_pix == 2.0] = conductive
        sigma_gt = image_to_mesh(np.flipud(sigma).T, mesh)

        t0 = time.time()
        Uel_sim = np.asarray(
            solver.SolveForward(sigma_gt, z.copy())).reshape(-1, 1)
        timing['forward'].append(time.time() - t0)

        t0 = time.time()
        noise = np.asarray(
            solver.InvLn * np.random.randn(Uel_sim.shape[0], 1)
        ).reshape(-1, 1)
        Uel_noisy = Uel_sim + noise
        timing['noise'].append(time.time() - t0)

        t_reco, t_interp = 0, 0
        if reconstructor is not None:
            t0 = time.time()
            delta_sigma_list = reconstructor.reconstruct_list(
                Uel_noisy, alphas)
            t_reco = time.time() - t0

            t0 = time.time()
            for ds in delta_sigma_list:
                reconstructor.interpolate_to_image(ds)
            t_interp = time.time() - t0

        timing['reco'].append(t_reco)
        timing['interp'].append(t_interp)
        timing['total'].append(time.time() - t_total)

    return timing


def format_table(cpu_timing, gpu_timing, num_samples):
    """Format comparison table."""
    steps = ['phantom', 'forward', 'noise', 'reco', 'interp', 'total']
    header = f'{"Step":>10} {"CPU (ms)":>10} {"GPU (ms)":>10} {"Speedup":>8}'
    sep = '-' * len(header)

    lines = [sep, header, sep]
    for step in steps:
        cpu_avg = np.mean(cpu_timing[step]) * 1000
        gpu_avg = np.mean(gpu_timing[step]) * 1000
        if cpu_avg > 0.01:
            speedup = cpu_avg / gpu_avg if gpu_avg > 0.01 else float('inf')
            lines.append(
                f'{step:>10} {cpu_avg:>10.1f} {gpu_avg:>10.1f} {speedup:>7.1f}x')
    lines.append(sep)
    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark CPU vs GPU data generation')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to benchmark (default: 10)')
    parser.add_argument('--level', type=int, default=1,
                        help='Difficulty level (default: 1)')
    parser.add_argument('--measurements-only', action='store_true',
                        help='Benchmark measurements-only mode (FCUNet)')
    parser.add_argument('--ref-path', type=str,
                        default='KTC2023/Codes_Python/TrainingData/ref.mat')
    parser.add_argument('--mesh-name', type=str, default='Mesh_dense.mat')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/gpu_benchmark.json',
                        help='Output path for benchmark results')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import cupy as cp
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        print(f'GPU: {gpu_name}')
    except ImportError:
        print('ERROR: CuPy is required for GPU benchmark. '
              'Install with: pip install cupy-cuda12x')
        sys.exit(1)

    mode = 'measurements-only' if args.measurements_only else 'full'
    print(f'Benchmark: {args.num_samples} samples, level {args.level}, '
          f'mode={mode}')
    print()

    # CPU benchmark
    print('Running CPU benchmark...')
    t0 = time.time()
    cpu_timing = run_benchmark(
        args.num_samples, args.level, args.ref_path, args.mesh_name,
        args.measurements_only, use_gpu=False, seed=args.seed)
    cpu_total = time.time() - t0
    print(f'  CPU total: {cpu_total:.1f}s '
          f'({np.mean(cpu_timing["total"])*1000:.0f} ms/sample avg)')

    # GPU benchmark
    print('Running GPU benchmark...')
    t0 = time.time()
    gpu_timing = run_benchmark(
        args.num_samples, args.level, args.ref_path, args.mesh_name,
        args.measurements_only, use_gpu=True, seed=args.seed)
    gpu_total = time.time() - t0
    print(f'  GPU total: {gpu_total:.1f}s '
          f'({np.mean(gpu_timing["total"])*1000:.0f} ms/sample avg)')

    # Print comparison table
    print()
    print(format_table(cpu_timing, gpu_timing, args.num_samples))

    overall_speedup = (np.mean(cpu_timing['total'])
                       / np.mean(gpu_timing['total']))
    print(f'\nOverall speedup: {overall_speedup:.1f}x')

    # Save results
    results = {
        'num_samples': args.num_samples,
        'level': args.level,
        'mode': mode,
        'gpu_name': gpu_name,
        'seed': args.seed,
        'overall_speedup': float(overall_speedup),
        'cpu': {
            'total_s': float(cpu_total),
            'per_sample_ms': float(np.mean(cpu_timing['total']) * 1000),
            'per_step_avg_ms': {
                k: float(np.mean(v) * 1000)
                for k, v in cpu_timing.items()
            },
        },
        'gpu': {
            'total_s': float(gpu_total),
            'per_sample_ms': float(np.mean(gpu_timing['total']) * 1000),
            'per_step_avg_ms': {
                k: float(np.mean(v) * 1000)
                for k, v in gpu_timing.items()
            },
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
