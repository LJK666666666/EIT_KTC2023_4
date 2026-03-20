"""
Generate training data for KTC2023 EIT reconstruction models.

Creates synthetic training samples by:
1. Generating random phantom images (3-class segmentation)
2. Simulating EIT measurements with EITFEM (forward solver + noise)
3. Computing 5 linearised reconstructions with different regularisation
4. Saving ground truth, measurements, and reconstructions

Usage:
    # Generate 2000 samples for level 3 (gt + gm_reco, for PostP/CondD)
    python scripts/generate_data.py --level 3 --num-images 2000

    # Generate for all levels
    python scripts/generate_data.py --all-levels --num-images 2000

    # FCUNet: only gt + measurements (fast, no linearised reconstruction)
    python scripts/generate_data.py --level 1 --num-images 2000 --measurements-only

    # Full: gt + measurements + gm_reco
    python scripts/generate_data.py --level 1 --num-images 2000 --save-measurements

    # GPU-accelerated generation (requires CuPy)
    python scripts/generate_data.py --level 1 --num-images 2000 --gpu

    # HDF5 output (single file instead of per-sample .npy)
    python scripts/generate_data.py --level 1 --num-images 2000 --hdf5

    # CPU multiprocess (4 workers, parallel forward solves)
    python scripts/generate_data.py --level 1 --num-images 2000 --workers 4
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ktc_methods import EITFEM, load_mesh, image_to_mesh
from src.data import create_phantoms
from src.utils.measurement import create_vincl
from src.configs.condd_config import LEVEL_TO_ALPHAS


def generate_data(level, num_images, output_dir='dataset',
                  ref_path='TrainingData/ref.mat',
                  mesh_name='Mesh_dense.mat',
                  save_measurements=False,
                  measurements_only=False,
                  start_idx=0,
                  use_gpu=False,
                  use_hdf5=False,
                  sys_bias=None):
    """Generate training data for a single level.

    Args:
        level: Difficulty level (1-7).
        num_images: Number of phantom images to generate.
        output_dir: Base output directory.
        ref_path: Path to reference data .mat file.
        mesh_name: Name of the mesh file.
        save_measurements: Whether to save raw measurements (for FCUNet).
        measurements_only: If True, only save gt + measurements, skip
            linearised reconstruction (much faster, for FCUNet only).
        start_idx: Starting index for file naming.
        use_gpu: If True, use CuPy GPU acceleration for FEM solve and
            linearised reconstruction.
        use_hdf5: If True, save to a single HDF5 file instead of per-sample
            .npy files. Reduces filesystem overhead for large datasets.
    """
    base_path = os.path.join(output_dir, f'level_{level}')
    os.makedirs(base_path, exist_ok=True)

    do_reco = not measurements_only
    do_meas = save_measurements or measurements_only

    # Set up output paths (HDF5 or per-sample .npy)
    h5_file = None
    gt_path = reco_path = meas_path = None
    if use_hdf5:
        import h5py
        h5_path = os.path.join(base_path, 'data.h5')
        h5_file = h5py.File(h5_path, 'w')
    else:
        gt_path = Path(os.path.join(base_path, 'gt'))
        gt_path.mkdir(parents=True, exist_ok=True)
        if do_reco:
            reco_path = Path(os.path.join(base_path, 'gm_reco'))
            reco_path.mkdir(parents=True, exist_ok=True)
        if do_meas:
            meas_path = Path(os.path.join(base_path, 'measurements'))
            meas_path.mkdir(parents=True, exist_ok=True)

    # Load reference data
    y_ref = loadmat(ref_path)
    Injref = y_ref['Injref']
    Mpat = y_ref['Mpat']

    mesh, mesh2 = load_mesh(mesh_name)

    Nel = 32
    z = 1e-6 * np.ones((Nel, 1))
    vincl = np.ones((Nel - 1, 76), dtype=bool)

    solver = EITFEM(mesh2, Injref, Mpat, vincl, use_gpu=use_gpu)

    noise_std1 = 0.05
    noise_std2 = 0.01
    solver.SetInvGamma(noise_std1, noise_std2, y_ref['Uelref'])

    # Simulate reference measurements
    # Note: SolveForward may return numpy.matrix (2D) due to scipy sparse ops.
    # Ensure consistent (N, 1) column vector shape to avoid broadcasting issues.
    sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.804
    Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).reshape(-1, 1)
    noise = np.asarray(
        solver.InvLn * np.random.randn(Uelref.shape[0], 1)).reshape(-1, 1)
    Uelref = Uelref + noise

    # Set up linearised reconstructor (only if needed)
    reconstructor = None
    if do_reco:
        from src.reconstruction.linearised_reco import LinearisedRecoFenics
        alphas = LEVEL_TO_ALPHAS[level]
        B = Mpat.T
        vincl_level = create_vincl(level, Injref)

        reconstructor = LinearisedRecoFenics(
            Uelref, B, vincl_level, mesh_name='sparse',
            base_path='KTC2023_SubmissionFiles/data',
            use_gpu=use_gpu, alphas=alphas)

    mode = 'measurements-only' if measurements_only else 'full'
    gpu_str = '+GPU' if use_gpu else 'CPU'
    hdf5_str = '+HDF5' if use_hdf5 else ''
    print(f'Generating {num_images} samples for level {level} '
          f'({mode}, {gpu_str}{hdf5_str})...')

    # Create HDF5 datasets (need measurement dimension from solver)
    if h5_file is not None:
        n_meas = Uelref.shape[0]
        h5_gt = h5_file.create_dataset(
            'gt', shape=(num_images, 256, 256), dtype='float32')
        h5_meas = None
        if do_meas:
            h5_meas = h5_file.create_dataset(
                'measurements', shape=(num_images, n_meas), dtype='float64')
        h5_reco = None
        if do_reco:
            h5_reco = h5_file.create_dataset(
                'reco', shape=(num_images, 5, 256, 256), dtype='float32')

    # Per-step timing records
    timing = {
        'phantom': [], 'forward': [], 'noise': [],
        'reco': [], 'interp': [], 'io': [], 'total': [],
    }

    for i in tqdm(range(num_images), desc=f'Level {level}'):
        t_total = time.time()

        # Generate random phantom
        t0 = time.time()
        sigma_pix = create_phantoms()
        timing['phantom'].append(time.time() - t0)

        idx = start_idx + i
        gt_name = f'gt_ztm_{idx:06d}.npy'

        # Random conductivity values
        background = 0.804
        resistive = np.random.rand() * 0.1 + 0.025
        conductive = np.random.rand() * 2.0 + 4.0

        sigma = np.zeros(sigma_pix.shape)
        sigma[sigma_pix == 0.0] = background
        sigma[sigma_pix == 1.0] = resistive
        sigma[sigma_pix == 2.0] = conductive

        sigma_gt = image_to_mesh(np.flipud(sigma).T, mesh)

        # Forward simulation with noise (reshape to column vector)
        t0 = time.time()
        Uel_sim = np.asarray(solver.SolveForward(sigma_gt, z)).reshape(-1, 1)
        timing['forward'].append(time.time() - t0)

        t0 = time.time()
        noise = np.asarray(
            solver.InvLn * np.random.randn(Uel_sim.shape[0], 1)).reshape(-1, 1)
        Uel_noisy = Uel_sim + noise
        if sys_bias is not None:
            Uel_noisy += sys_bias.reshape(-1, 1)
        timing['noise'].append(time.time() - t0)

        # Save gt and measurements
        t0 = time.time()
        if h5_file is not None:
            h5_gt[i] = sigma_pix.astype(np.float32)
            if do_meas and h5_meas is not None:
                h5_meas[i] = Uel_noisy.flatten()
        else:
            np.save(os.path.join(gt_path, gt_name), sigma_pix)
            if do_meas:
                u_name = f'u_ztm_{idx:06d}.npy'
                np.save(os.path.join(meas_path, u_name), Uel_noisy.flatten())

        # 5 linearised reconstructions (skip if measurements_only)
        t_reco, t_interp = 0, 0
        if do_reco:
            t1 = time.time()
            delta_sigma_list = reconstructor.reconstruct_list(
                Uel_noisy, alphas)
            t_reco = time.time() - t1

            t1 = time.time()
            sigma_images = [
                reconstructor.interpolate_to_image(ds)
                for ds in delta_sigma_list
            ]
            sigma_reco = np.stack(sigma_images)  # (5, 256, 256)
            t_interp = time.time() - t1

            reco_name = f'recos_ztm_{idx:06d}.npy'
            if h5_file is not None:
                h5_reco[i] = sigma_reco.astype(np.float32)
            else:
                np.save(os.path.join(reco_path, reco_name), sigma_reco)

        timing['reco'].append(t_reco)
        timing['interp'].append(t_interp)
        timing['io'].append(time.time() - t0 - t_reco - t_interp)
        timing['total'].append(time.time() - t_total)

    # Close HDF5 file if open
    if h5_file is not None:
        h5_file.close()
        print(f'  HDF5 saved to: {os.path.join(base_path, "data.h5")}')

    # Print timing summary
    avg_total = np.mean(timing['total'])
    print(f'Level {level}: {num_images} samples generated '
          f'({avg_total:.3f}s/sample avg)')

    if num_images > 0:
        print(f'  Avg timing breakdown (ms):')
        for key in ['phantom', 'forward', 'noise', 'reco', 'interp', 'io']:
            avg_ms = np.mean(timing[key]) * 1000
            if avg_ms > 0.01:
                print(f'    {key:>10}: {avg_ms:8.1f} ms')

    # Save timing to file
    timing_summary = {
        'level': level,
        'num_images': num_images,
        'mode': mode,
        'gpu': use_gpu,
        'avg_per_sample_s': float(avg_total) if num_images > 0 else 0,
        'per_step_avg_ms': {
            k: float(np.mean(v) * 1000) for k, v in timing.items()
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    timing_path = os.path.join(
        output_dir,
        f'timing_level{level}_{mode}_{"gpu" if use_gpu else "cpu"}.json')
    with open(timing_path, 'w') as f:
        json.dump(timing_summary, f, indent=2)
    print(f'  Timing saved to: {timing_path}')

    return timing_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate KTC2023 training data')
    parser.add_argument('--level', type=int, default=1,
                        help='Difficulty level (1-7)')
    parser.add_argument('--all-levels', action='store_true',
                        help='Generate for all levels 1-7')
    parser.add_argument('--num-images', type=int, default=2000,
                        help='Number of images per level')
    parser.add_argument('--output-dir', type=str, default='dataset',
                        help='Output directory')
    parser.add_argument('--ref-path', type=str,
                        default='KTC2023/Codes_Python/TrainingData/ref.mat',
                        help='Path to reference data')
    parser.add_argument('--mesh-name', type=str,
                        default='Mesh_dense.mat',
                        help='Mesh file name (looked up in src/ktc_methods/)')
    parser.add_argument('--save-measurements', action='store_true',
                        help='Also save raw measurements (for FCUNet)')
    parser.add_argument('--measurements-only', action='store_true',
                        help='Only save gt + measurements, skip linearised '
                             'reconstruction (fast mode for FCUNet)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index for file naming')
    parser.add_argument('--gpu', action='store_true',
                        help='Use CuPy GPU acceleration (requires CuPy)')
    parser.add_argument('--hdf5', action='store_true',
                        help='Save to single HDF5 file instead of per-sample '
                             '.npy files (reduces filesystem overhead)')
    parser.add_argument('--sys-bias', type=str, default=None,
                        help='Path to systematic bias .npy vector '
                             '(default: data/systematic_bias.npy if exists)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (CPU multiprocess, '
                             'default: 1 = serial)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Batch flush size for multiprocess workers '
                             '(default: 1000 samples per batch file)')
    return parser.parse_args()


def _generate_one_sample(solver, mesh, z, reconstructor, alphas,
                         do_meas, do_reco, sys_bias=None):
    """Generate one training sample. Returns (gt, measurements_or_None, reco_or_None)."""
    sigma_pix = create_phantoms()

    background = 0.804
    resistive = np.random.rand() * 0.1 + 0.025
    conductive = np.random.rand() * 2.0 + 4.0

    sigma = np.zeros(sigma_pix.shape)
    sigma[sigma_pix == 0.0] = background
    sigma[sigma_pix == 1.0] = resistive
    sigma[sigma_pix == 2.0] = conductive

    sigma_gt = image_to_mesh(np.flipud(sigma).T, mesh)

    Uel_sim = np.asarray(solver.SolveForward(sigma_gt, z)).reshape(-1, 1)
    noise = np.asarray(
        solver.InvLn * np.random.randn(Uel_sim.shape[0], 1)).reshape(-1, 1)
    Uel_noisy = Uel_sim + noise
    if sys_bias is not None:
        Uel_noisy += sys_bias.reshape(-1, 1)

    meas = Uel_noisy.flatten() if do_meas else None

    reco = None
    if do_reco:
        delta_sigma_list = reconstructor.reconstruct_list(Uel_noisy, alphas)
        sigma_images = [
            reconstructor.interpolate_to_image(ds)
            for ds in delta_sigma_list
        ]
        reco = np.stack(sigma_images)  # (5, 256, 256)

    return sigma_pix, meas, reco


def _init_solver_and_reco(level, ref_path, mesh_name, use_gpu,
                          measurements_only, save_measurements):
    """Shared initialisation for both single-process and worker paths."""
    y_ref = loadmat(ref_path)
    Injref = y_ref['Injref']
    Mpat = y_ref['Mpat']

    mesh, mesh2 = load_mesh(mesh_name)

    Nel = 32
    z = 1e-6 * np.ones((Nel, 1))
    vincl = np.ones((Nel - 1, 76), dtype=bool)

    solver = EITFEM(mesh2, Injref, Mpat, vincl, use_gpu=use_gpu)

    noise_std1 = 0.05
    noise_std2 = 0.01
    solver.SetInvGamma(noise_std1, noise_std2, y_ref['Uelref'])

    sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.804
    Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).reshape(-1, 1)
    noise = np.asarray(
        solver.InvLn * np.random.randn(Uelref.shape[0], 1)).reshape(-1, 1)
    Uelref = Uelref + noise

    do_reco = not measurements_only
    do_meas = save_measurements or measurements_only

    reconstructor = None
    alphas = None
    if do_reco:
        from src.reconstruction.linearised_reco import LinearisedRecoFenics
        from src.utils.measurement import create_vincl as _create_vincl
        alphas = LEVEL_TO_ALPHAS[level]
        B = Mpat.T
        vincl_level = _create_vincl(level, Injref)
        reconstructor = LinearisedRecoFenics(
            Uelref, B, vincl_level, mesh_name='sparse',
            base_path='KTC2023_SubmissionFiles/data',
            use_gpu=use_gpu, alphas=alphas)

    return solver, mesh, z, reconstructor, alphas, do_meas, do_reco


def _mp_worker_chunked(kwargs):
    """Worker that accumulates samples in memory and flushes to .npz batches."""
    import os
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    worker_id = kwargs['worker_id']
    chunk_size = kwargs.get('chunk_size', 1000)
    level = kwargs['level']
    num_images = kwargs['num_images']
    output_dir = kwargs['output_dir']

    base_path = os.path.join(output_dir, f'level_{level}')
    batch_dir = os.path.join(base_path, '_batches')
    os.makedirs(batch_dir, exist_ok=True)

    solver, mesh, z, reconstructor, alphas, do_meas, do_reco = \
        _init_solver_and_reco(
            level, kwargs['ref_path'], kwargs['mesh_name'],
            use_gpu=False,
            measurements_only=kwargs['measurements_only'],
            save_measurements=kwargs['save_measurements'])

    # Load systematic bias if provided
    sys_bias = None
    sys_bias_path = kwargs.get('sys_bias_path')
    if sys_bias_path and os.path.exists(sys_bias_path):
        sys_bias = np.load(sys_bias_path)

    gt_buf, meas_buf, reco_buf = [], [], []
    chunk_idx = 0

    def flush():
        nonlocal chunk_idx
        if not gt_buf:
            return
        data = {'gt': np.stack(gt_buf).astype(np.float32)}
        if meas_buf:
            data['measurements'] = np.stack(meas_buf).astype(np.float64)
        if reco_buf:
            data['reco'] = np.stack(reco_buf).astype(np.float32)
        path = os.path.join(
            batch_dir, f'batch_{worker_id:02d}_{chunk_idx:04d}.npz')
        np.savez(path, **data)
        gt_buf.clear()
        meas_buf.clear()
        reco_buf.clear()
        chunk_idx += 1

    for i in tqdm(range(num_images), desc=f'Worker {worker_id}',
                  position=worker_id, leave=False):
        gt, meas, reco = _generate_one_sample(
            solver, mesh, z, reconstructor, alphas, do_meas, do_reco,
            sys_bias=sys_bias)
        gt_buf.append(gt)
        if meas is not None:
            meas_buf.append(meas)
        if reco is not None:
            reco_buf.append(reco)

        if len(gt_buf) >= chunk_size:
            flush()

    flush()  # remaining
    return {'worker_id': worker_id, 'num_chunks': chunk_idx}


def _merge_batches_to_hdf5(base_path):
    """Merge .npz batch files from _batches/ into data.h5, then clean up."""
    import glob as _glob
    import h5py

    batch_dir = os.path.join(base_path, '_batches')
    h5_path = os.path.join(base_path, 'data.h5')
    batch_files = sorted(_glob.glob(os.path.join(batch_dir, 'batch_*.npz')))

    if not batch_files:
        print(f'  No batch files found in {batch_dir}')
        return

    with h5py.File(h5_path, 'w') as h5f:
        for bf in tqdm(batch_files, desc='Merging to HDF5'):
            data = np.load(bf)
            for key in data.files:
                arr = data[key]
                if key not in h5f:
                    h5f.create_dataset(
                        key, data=arr,
                        maxshape=(None, *arr.shape[1:]),
                        chunks=(1, *arr.shape[1:]))
                else:
                    ds = h5f[key]
                    old_len = ds.shape[0]
                    ds.resize(old_len + arr.shape[0], axis=0)
                    ds[old_len:] = arr

    # Clean up batch files
    import shutil
    shutil.rmtree(batch_dir)

    # Print summary
    with h5py.File(h5_path, 'r') as f:
        for k, v in f.items():
            print(f'  {k}: {v.shape} ({v.dtype})')
    print(f'  HDF5 saved to: {h5_path}')


def _mp_generate_chunk(kwargs):
    """Legacy wrapper for multiprocessing (per-sample .npy output)."""
    import os
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return generate_data(**kwargs)


def main():
    args = parse_args()

    if args.gpu:
        try:
            import cupy as cp
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            print(f'GPU acceleration enabled: {gpu_name}')
        except ImportError:
            print('ERROR: --gpu requires CuPy. Install with: '
                  'pip install cupy-cuda12x')
            sys.exit(1)

    levels = range(1, 8) if args.all_levels else [args.level]

    # Resolve systematic bias path
    sys_bias_path = args.sys_bias
    if sys_bias_path is None:
        default_path = 'data/systematic_bias.npy'
        if os.path.exists(default_path):
            sys_bias_path = default_path
    if sys_bias_path and os.path.exists(sys_bias_path):
        sys_bias = np.load(sys_bias_path)
        print(f'Systematic bias loaded: {sys_bias_path} '
              f'(mean={sys_bias.mean():.4e}, std={sys_bias.std():.4e})')
    else:
        sys_bias = None

    for level in levels:
        if args.workers > 1 and not args.gpu:
            # Multiprocess: chunked workers → .npz batches → merge to .h5
            from concurrent.futures import ProcessPoolExecutor
            n = args.num_images
            w = args.workers
            samples_per_worker = n // w
            worker_args = []
            for i in range(w):
                count = samples_per_worker if i < w - 1 \
                    else n - i * samples_per_worker
                worker_args.append({
                    'worker_id': i,
                    'chunk_size': args.chunk_size,
                    'level': level,
                    'num_images': count,
                    'output_dir': args.output_dir,
                    'ref_path': args.ref_path,
                    'mesh_name': args.mesh_name,
                    'save_measurements': args.save_measurements,
                    'measurements_only': args.measurements_only,
                    'sys_bias_path': sys_bias_path,
                })
            print(f'Using {w} workers for level {level} '
                  f'({samples_per_worker} samples/worker, '
                  f'chunk_size={args.chunk_size})...')
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=w) as pool:
                list(pool.map(_mp_worker_chunked, worker_args))
            elapsed = time.time() - t0
            print(f'Level {level}: {n} samples in {elapsed:.1f}s '
                  f'({elapsed/n*1000:.0f} ms/sample throughput)')

            # Merge .npz batches into final .h5
            base_path = os.path.join(args.output_dir, f'level_{level}')
            _merge_batches_to_hdf5(base_path)
        else:
            generate_data(
                level=level,
                num_images=args.num_images,
                output_dir=args.output_dir,
                ref_path=args.ref_path,
                mesh_name=args.mesh_name,
                save_measurements=args.save_measurements,
                measurements_only=args.measurements_only,
                start_idx=args.start_idx,
                use_gpu=args.gpu,
                use_hdf5=args.hdf5,
                sys_bias=sys_bias,
            )

    print('Data generation complete.')


if __name__ == '__main__':
    main()
