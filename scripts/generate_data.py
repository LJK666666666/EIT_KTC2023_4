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
"""

import argparse
import os
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
                  start_idx=0):
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
    """
    base_path = os.path.join(output_dir, f'level_{level}')
    gt_path = Path(os.path.join(base_path, 'gt'))
    gt_path.mkdir(parents=True, exist_ok=True)

    do_reco = not measurements_only
    do_meas = save_measurements or measurements_only

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

    solver = EITFEM(mesh2, Injref, Mpat, vincl)

    noise_std1 = 0.05
    noise_std2 = 0.01
    solver.SetInvGamma(noise_std1, noise_std2, y_ref['Uelref'])

    # Simulate reference measurements
    # Note: SolveForward may return numpy.matrix (2D) due to scipy sparse ops.
    # Flatten to ensure consistent 1D shape for downstream operations.
    sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.745
    Uelref = np.asarray(solver.SolveForward(sigma_bg, z)).flatten()
    noise = np.asarray(
        solver.InvLn * np.random.randn(Uelref.shape[0], 1)).flatten()
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
            base_path='KTC2023_SubmissionFiles/data')

    mode = 'measurements-only' if measurements_only else 'full'
    print(f'Generating {num_images} samples for level {level} ({mode})...')
    times = []

    for i in tqdm(range(num_images), desc=f'Level {level}'):
        t0 = time.time()

        # Generate random phantom
        sigma_pix = create_phantoms()

        idx = start_idx + i
        gt_name = f'gt_ztm_{idx:06d}.npy'

        np.save(os.path.join(gt_path, gt_name), sigma_pix)

        # Random conductivity values
        background = 0.745
        resistive = np.random.rand() * 0.1 + 0.025
        conductive = np.random.rand() + 5.0

        sigma = np.zeros(sigma_pix.shape)
        sigma[sigma_pix == 0.0] = background
        sigma[sigma_pix == 1.0] = resistive
        sigma[sigma_pix == 2.0] = conductive

        sigma_gt = image_to_mesh(np.flipud(sigma).T, mesh)

        # Forward simulation with noise (flatten to avoid numpy.matrix issues)
        Uel_sim = np.asarray(solver.SolveForward(sigma_gt, z)).flatten()
        noise = np.asarray(
            solver.InvLn * np.random.randn(Uel_sim.shape[0], 1)).flatten()
        Uel_noisy = Uel_sim + noise

        if do_meas:
            u_name = f'u_ztm_{idx:06d}.npy'
            np.save(os.path.join(meas_path, u_name), Uel_noisy)

        # 5 linearised reconstructions (skip if measurements_only)
        if do_reco:
            delta_sigma_list = reconstructor.reconstruct_list(
                Uel_noisy, alphas)
            sigma_images = [
                reconstructor.interpolate_to_image(ds)
                for ds in delta_sigma_list
            ]
            sigma_reco = np.stack(sigma_images)  # (5, 256, 256)
            reco_name = f'recos_ztm_{idx:06d}.npy'
            np.save(os.path.join(reco_path, reco_name), sigma_reco)

        times.append(time.time() - t0)

    avg_time = np.mean(times)
    print(f'Level {level}: {num_images} samples generated '
          f'({avg_time:.1f}s/sample avg)')


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
    return parser.parse_args()


def main():
    args = parse_args()

    levels = range(1, 8) if args.all_levels else [args.level]

    for level in levels:
        generate_data(
            level=level,
            num_images=args.num_images,
            output_dir=args.output_dir,
            ref_path=args.ref_path,
            mesh_name=args.mesh_name,
            save_measurements=args.save_measurements,
            measurements_only=args.measurements_only,
            start_idx=args.start_idx,
        )

    print('Data generation complete.')


if __name__ == '__main__':
    main()
