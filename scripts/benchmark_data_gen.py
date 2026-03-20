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


def detect_optimizations(reconstructor, solver=None):
    """Detect active optimization items by inspecting runtime objects."""
    # A-G are always active (baked into source code)
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if (reconstructor is not None
            and getattr(reconstructor, '_R_precomputed', None) is not None):
        opts.append('H')
    if (solver is not None
            and getattr(solver, '_b_rank', None) is not None):
        opts.append('K3')
    return sorted(opts)


def get_next_output_path(base_path):
    """Auto-increment: results/gpu_benchmark_1.json, _2.json, ..."""
    directory = os.path.dirname(base_path) or '.'
    stem = os.path.splitext(os.path.basename(base_path))[0]
    ext = os.path.splitext(base_path)[1] or '.json'
    num = 1
    while os.path.exists(os.path.join(directory, f'{stem}_{num}{ext}')):
        num += 1
    return os.path.join(directory, f'{stem}_{num}{ext}')


def run_benchmark(num_samples, level, ref_path, mesh_name,
                  measurements_only, use_gpu, seed):
    """Run data generation benchmark, return (timing_dict, reconstructor, solver)."""
    # Use a FIXED seed for Uelref computation so all workers produce the
    # same Uelref → same R-matrix cache key → mmap cache hit (zero-copy).
    # The worker-specific seed is applied AFTER Uelref for sample generation.
    set_seed(42)

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

    # Now switch to worker-specific seed for sample generation
    set_seed(seed)

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
            use_gpu=use_gpu, alphas=alphas)

    timing = {
        'phantom': [], 'forward': [], 'noise': [],
        'reco': [], 'interp': [], 'total': [],
    }

    # Warm-up: 1 full sample (GPU needs warm-up for kernel compilation)
    set_seed(seed)
    _ = create_phantoms()
    sigma_bg_copy = np.ones((mesh.g.shape[0], 1)) * 0.745
    Uel_warmup = np.asarray(
        solver.SolveForward(sigma_bg_copy, z.copy())).reshape(-1, 1)
    if reconstructor is not None:
        noise_warmup = np.asarray(
            solver.InvLn * np.random.randn(Uel_warmup.shape[0], 1)
        ).reshape(-1, 1)
        _ = reconstructor.reconstruct_list(Uel_warmup + noise_warmup, alphas)

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

    return timing, reconstructor, solver


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
    parser.add_argument('--output', type=str,
                        default='results/gpu_benchmark.json',
                        help='Output base path (auto-increments _N suffix)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for multiprocess '
                             'throughput benchmark (default: 1 = skip)')
    return parser.parse_args()


def _mp_run_benchmark(kwargs):
    """Worker function for subprocess-isolated benchmark."""
    import os
    # Limit MKL/PARDISO to 1 thread per worker — forward solve is
    # memory-bandwidth-bound, not CPU-bound (16 threads = only 1.1x vs 1 thread)
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    timing, reco, solver = run_benchmark(**kwargs)
    opts = detect_optimizations(reco, solver)
    gpu_name = None
    if kwargs.get('use_gpu'):
        try:
            import cupy as cp
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
        except Exception:
            pass
    return timing, opts, gpu_name


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

    from concurrent.futures import ProcessPoolExecutor

    bench_kwargs = {
        'num_samples': args.num_samples,
        'level': args.level,
        'ref_path': args.ref_path,
        'mesh_name': args.mesh_name,
        'measurements_only': args.measurements_only,
        'seed': args.seed,
    }

    # Run CPU benchmark directly in main process
    print('Running CPU benchmark...')
    t0 = time.time()
    cpu_timing, cpu_reco, cpu_solver = run_benchmark(
        **bench_kwargs, use_gpu=False)
    cpu_total = time.time() - t0
    cpu_opts = detect_optimizations(cpu_reco, cpu_solver)
    print(f'  CPU total: {cpu_total:.1f}s '
          f'({np.mean(cpu_timing["total"])*1000:.0f} ms/sample avg)')

    # Free large arrays before GPU run
    del cpu_reco, cpu_solver
    import gc; gc.collect()

    # Run GPU benchmark directly (may fail with large precomputed matrices)
    gpu_timing = None
    gpu_total = 0
    gpu_opts = []
    print('Running GPU benchmark...')
    try:
        t0 = time.time()
        gpu_timing_raw, gpu_reco, gpu_solver = run_benchmark(
            **bench_kwargs, use_gpu=True)
        gpu_total = time.time() - t0
        gpu_opts = detect_optimizations(gpu_reco, gpu_solver)
        gpu_timing = gpu_timing_raw
        print(f'  GPU total: {gpu_total:.1f}s '
              f'({np.mean(gpu_timing["total"])*1000:.0f} ms/sample avg)')
        del gpu_reco, gpu_solver; gc.collect()
    except MemoryError:
        print('  GPU benchmark skipped (insufficient memory for both '
              'CPU + GPU precomputed matrices in same process)')

    del gc

    # Print comparison table
    print()
    if gpu_timing is not None:
        print(format_table(cpu_timing, gpu_timing, args.num_samples))
        overall_speedup = (np.mean(cpu_timing['total'])
                           / np.mean(gpu_timing['total']))
        print(f'\nOverall speedup: {overall_speedup:.1f}x')
    else:
        # CPU-only summary
        overall_speedup = 1.0
        steps = ['phantom', 'forward', 'noise', 'reco', 'interp', 'total']
        header = f'{"Step":>10} {"CPU (ms)":>10}'
        sep = '-' * len(header)
        print(sep)
        print(header)
        print(sep)
        for step in steps:
            avg = np.mean(cpu_timing[step]) * 1000
            if avg > 0.01:
                print(f'{step:>10} {avg:>10.1f}')
        print(sep)

    # Multiprocess throughput benchmark
    mp_result = None
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        n = args.num_samples
        w = args.workers
        chunk = n // w
        worker_args = []
        for i in range(w):
            count = chunk if i < w - 1 else n - i * chunk
            worker_args.append({
                'num_samples': count,
                'level': args.level,
                'ref_path': args.ref_path,
                'mesh_name': args.mesh_name,
                'measurements_only': args.measurements_only,
                'use_gpu': False,
                'seed': args.seed + i * 10000,
            })

        print(f'\nRunning multiprocess benchmark ({w} workers, '
              f'{n} samples total)...')
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=w) as pool:
            mp_results = list(pool.map(_mp_run_benchmark, worker_args))
        mp_wall = time.time() - t0

        # Merge per-step timings from all workers
        mp_timings_list = [r[0] for r in mp_results]
        mp_merged = {k: [] for k in mp_timings_list[0]}
        for t in mp_timings_list:
            for k, v in t.items():
                mp_merged[k].extend(v)

        mp_throughput = mp_wall / n * 1000
        serial_ms = np.mean(cpu_timing['total']) * 1000
        tp_speedup = serial_ms / mp_throughput

        print(f'  Wall time: {mp_wall:.1f}s '
              f'({mp_throughput:.0f} ms/sample throughput)')
        print(f'  vs serial CPU: {tp_speedup:.1f}x throughput gain')

        mp_result = {
            'workers': w,
            'wall_time_s': float(mp_wall),
            'throughput_ms_per_sample': float(mp_throughput),
            'throughput_speedup': float(tp_speedup),
            'per_step_avg_ms': {
                k: float(np.mean(v) * 1000)
                for k, v in mp_merged.items()
            },
        }

    # Detect active optimizations
    optimizations = sorted(set(gpu_opts) | set(cpu_opts))
    print(f'Active optimizations: {", ".join(optimizations)}')

    # Save results with auto-increment filename
    output_path = get_next_output_path(args.output)

    results = {
        'num_samples': args.num_samples,
        'level': args.level,
        'mode': mode,
        'gpu_name': gpu_name,
        'seed': args.seed,
        'optimizations': optimizations,
        'overall_speedup': float(overall_speedup),
        'cpu': {
            'total_s': float(cpu_total),
            'per_sample_ms': float(np.mean(cpu_timing['total']) * 1000),
            'per_step_avg_ms': {
                k: float(np.mean(v) * 1000)
                for k, v in cpu_timing.items()
            },
        },
    }

    if gpu_timing is not None:
        results['gpu'] = {
            'total_s': float(gpu_total),
            'per_sample_ms': float(np.mean(gpu_timing['total']) * 1000),
            'per_step_avg_ms': {
                k: float(np.mean(v) * 1000)
                for k, v in gpu_timing.items()
            },
        }

    if mp_result is not None:
        results['multiprocess'] = mp_result

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {output_path}')


if __name__ == '__main__':
    main()
