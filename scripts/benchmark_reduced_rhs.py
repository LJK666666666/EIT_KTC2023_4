"""
Isolated benchmark: reduced-RHS (K3) forward solve speedup.

Controls all variables — same A matrix, same solver, only difference is
solving 15 vs 76 RHS columns.

Usage:
    python scripts/benchmark_reduced_rhs.py
    python scripts/benchmark_reduced_rhs.py --num-samples 20
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ktc_methods import EITFEM, load_mesh, image_to_mesh
from src.data import create_phantoms


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark reduced-RHS (K3) forward solve')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--ref-path', type=str,
                        default='KTC2023/Codes_Python/TrainingData/ref.mat')
    parser.add_argument('--mesh-name', type=str, default='Mesh_dense.mat')
    args = parser.parse_args()

    # --- Setup (shared) ---
    y_ref = loadmat(args.ref_path)
    Injref = y_ref['Injref']
    Mpat = y_ref['Mpat']
    mesh, mesh2 = load_mesh(args.mesh_name)

    Nel = 32
    z = 1e-6 * np.ones((Nel, 1))
    vincl = np.ones((Nel - 1, 76), dtype=bool)

    solver = EITFEM(mesh2, Injref, Mpat, vincl)
    solver.SetInvGamma(0.05, 0.01, y_ref['Uelref'])

    print(f'RHS rank: {solver._b_rank}  '
          f'(basis {solver._b_basis.shape}, full {solver.b.shape})')

    # Detect solver backend
    from src.ktc_methods.KTCFwd import _HAS_PARDISO
    backend = 'pypardiso' if _HAS_PARDISO else 'scipy.spsolve'
    print(f'Solver backend: {backend}')

    # --- Pre-generate all phantoms so generation time is excluded ---
    np.random.seed(42)
    sigmas = []
    for _ in range(args.num_samples):
        sigma_pix = create_phantoms()
        sigma = np.zeros(sigma_pix.shape)
        sigma[sigma_pix == 0.0] = 0.745
        sigma[sigma_pix == 1.0] = np.random.rand() * 0.1 + 0.025
        sigma[sigma_pix == 2.0] = np.random.rand() + 5.0
        sigmas.append(image_to_mesh(np.flipud(sigma).T, mesh))

    # --- Benchmark: full 76-column solve ---
    from scipy.sparse.linalg import spsolve as scipy_spsolve
    if _HAS_PARDISO:
        from pypardiso import spsolve as pardiso_spsolve
        solve_fn = pardiso_spsolve
    else:
        solve_fn = scipy_spsolve

    rhs_full = np.asarray(solver.b, dtype=np.float64)

    # Warm-up
    solver.SolveForward(sigmas[0], z.copy())

    times_full = []
    for sigma_gt in sigmas:
        # Build A (assembly part — same for both paths)
        solver.SolveForward(sigma_gt, z.copy())  # sets solver.A
        A = solver.A

        t0 = time.perf_counter()
        UU_full = solve_fn(A, rhs_full)
        times_full.append(time.perf_counter() - t0)

    # --- Benchmark: reduced 15-column solve ---
    times_reduced = []
    for sigma_gt in sigmas:
        solver.SolveForward(sigma_gt, z.copy())
        A = solver.A

        t0 = time.perf_counter()
        UU_basis = solve_fn(A, solver._b_basis)
        UU_reduced = UU_basis @ solver._b_coeff
        times_reduced.append(time.perf_counter() - t0)

    # --- Verify correctness (last sample) ---
    Umeas_full = np.asarray(
        solver._MpatC * UU_full[solver.ng2:, :]).T[solver.mincl.T].T.flatten()
    Umeas_reduced = np.asarray(
        solver._MpatC * UU_reduced[solver.ng2:, :]).T[solver.mincl.T].T.flatten()
    rel_err = np.linalg.norm(Umeas_full - Umeas_reduced) / np.linalg.norm(Umeas_full)

    # --- Report ---
    full_ms = np.array(times_full) * 1000
    reduced_ms = np.array(times_reduced) * 1000

    print(f'\nSamples: {args.num_samples}')
    print(f'{"":20s} {"mean":>8s} {"std":>8s} {"min":>8s} {"max":>8s}')
    print('-' * 56)
    print(f'{"76-col solve (ms)":20s} {full_ms.mean():8.1f} {full_ms.std():8.1f} '
          f'{full_ms.min():8.1f} {full_ms.max():8.1f}')
    print(f'{"15-col solve (ms)":20s} {reduced_ms.mean():8.1f} {reduced_ms.std():8.1f} '
          f'{reduced_ms.min():8.1f} {reduced_ms.max():8.1f}')
    print('-' * 56)
    print(f'Speedup: {full_ms.mean() / reduced_ms.mean():.2f}x')
    print(f'Relative error: {rel_err:.2e}')


if __name__ == '__main__':
    main()
