"""
Benchmark batched PCG attempts for the forward step in scripts/generate_data.py.

This script does not modify the original data-generation pipeline. It compares:
1. Current exact direct solve (same linear system as EITFEM.SolveForward)
2. Exact reduced-RHS direct solve (76 RHS -> rank-15 basis)
3. GPU batched Jacobi-PCG on the reduced RHS basis with background warm start

The GPU path can benchmark both:
- PyTorch CUDA sparse CSR
- CuPy custom batched Jacobi-PCG

Outputs are saved to results/{experiment_name}_{num}/.

Usage:
    python scripts/benchmark_forward_batched_pcg.py --num-samples 5
    python scripts/benchmark_forward_batched_pcg.py --num-samples 5 --pcg-iters 200 600 1000
    python scripts/benchmark_forward_batched_pcg.py --num-samples 5 --gpu-backends cupy
"""

import argparse
import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
from scipy.io import loadmat
from scipy.linalg import qr
from scipy.sparse.linalg import spsolve as scipy_spsolve
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import create_phantoms
from src.ktc_methods import EITFEM, image_to_mesh, load_mesh

try:
    from pypardiso import spsolve as pardiso_spsolve
    HAS_PARDISO = True
except ImportError:
    HAS_PARDISO = False
    pardiso_spsolve = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_result_dir(experiment_name, base_dir='results'):
    num = 1
    while True:
        path = os.path.join(base_dir, f'{experiment_name}_{num}')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return path
        num += 1


def detect_gpu_name():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark forward-step batched PCG attempts')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random forward samples to benchmark')
    parser.add_argument('--level', type=int, default=1,
                        help='Difficulty level label used in result naming')
    parser.add_argument('--ref-path', type=str,
                        default='KTC2023/Codes_Python/TrainingData/ref.mat',
                        help='Path to reference data')
    parser.add_argument('--mesh-name', type=str, default='Mesh_dense.mat',
                        help='Mesh file name under src/ktc_methods/')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Base directory for outputs')
    parser.add_argument('--experiment-name', type=str,
                        default='forward_batched_pcg',
                        help='Experiment name prefix for results folder')
    parser.add_argument('--pcg-iters', type=int, nargs='+',
                        default=[200, 600, 1000],
                        help='GPU PCG iteration counts to compare')
    parser.add_argument('--rhs-basis-mode', type=str, default='columns',
                        choices=['columns', 'qr_ortho', 'svd'],
                        help='Basis used to reduce the 76 RHS columns')
    parser.add_argument('--gpu-backends', type=str, nargs='+',
                        default=['torch', 'cupy'],
                        choices=['torch', 'cupy'],
                        help='GPU backends to benchmark')
    parser.add_argument('--target-rel-error', type=float, default=1e-2,
                        help='Target mean relative error for recommendation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='CUDA device for GPU PCG benchmark')
    parser.add_argument('--warmup-iters', type=int, default=50,
                        help='Uncounted GPU warmup iterations before timing')
    return parser.parse_args()


def make_rhs_basis(rhs_full, mode='columns', tol=1e-12):
    """Reduce 76 RHS columns to a numerically independent basis."""
    q_mat, r_mat, piv = qr(rhs_full, mode='economic', pivoting=True)
    diag = np.abs(np.diag(r_mat))
    rank = int(np.sum(diag > tol))
    basis_cols = piv[:rank]

    if mode == 'columns':
        rhs_basis = rhs_full[:, basis_cols]
        coeff, *_ = np.linalg.lstsq(rhs_basis, rhs_full, rcond=None)
        basis_info = {'basis_cols': basis_cols.tolist()}
    elif mode == 'qr_ortho':
        rhs_basis = q_mat[:, :rank]
        coeff = rhs_basis.T @ rhs_full
        basis_info = {'basis_cols': None}
    elif mode == 'svd':
        u_mat, s_vec, vt_mat = np.linalg.svd(rhs_full, full_matrices=False)
        rhs_basis = u_mat[:, :rank]
        coeff = s_vec[:rank, None] * vt_mat[:rank, :]
        basis_info = {
            'basis_cols': None,
            'singular_values': s_vec[:rank].tolist(),
        }
    else:
        raise ValueError(f'Unsupported rhs basis mode: {mode}')

    return rhs_basis, coeff, rank, basis_info


def create_sigma_sample(mesh):
    sigma_pix = create_phantoms()
    sigma = np.zeros_like(sigma_pix, dtype=np.float64)
    sigma[sigma_pix == 0.0] = 0.745
    sigma[sigma_pix == 1.0] = np.random.rand() * 0.1 + 0.025
    sigma[sigma_pix == 2.0] = np.random.rand() + 5.0
    return image_to_mesh(np.flipud(sigma).T, mesh)


def assemble_system(solver, sigma, z):
    """Assemble A exactly like EITFEM.SolveForward, without solving."""
    sigma = np.array(sigma, copy=True)
    z = np.array(z, copy=True)
    sigma[sigma < solver.sigmamin] = solver.sigmamin
    sigma[sigma > solver.sigmamax] = solver.sigmamax
    z[z < solver.zmin] = solver.zmin

    t0 = time.time()
    all_ss = sigma[solver._H_vertex_idx]
    if all_ss.ndim == 3:
        all_ss = all_ss[:, :, 0]

    all_Ke = np.zeros((solver.nH2, 6, 6))
    for qq in range(3):
        sigma_w = all_ss @ solver._quad_S[qq]
        all_Ke += (
            solver._quad_w[qq] * sigma_w * solver._quad_abs_det[qq]
        )[:, None, None] * solver._quad_GtG[qq]

    A0 = sp.sparse.coo_matrix(
        (all_Ke.ravel(), (solver._row_A0, solver._col_A0)),
        shape=(solver._N, solver._N),
    ).tocsr()

    z_flat = z.flatten()
    if solver._S0_cached is None or not np.array_equal(z_flat, solver._z_cached):
        gN = solver._gN
        g = solver.Mesh2.g
        M_data, M_row, M_col = [], [], []
        K_data, K_row, K_col = [], [], []
        s = np.zeros((solver.Nel, 1))
        for ii in range(solver.nH2):
            ind = solver.Mesh2.Element[ii].Topology
            if solver.Mesh2.Element[ii].Electrode:
                Ind = solver.Mesh2.Element[ii].Electrode[1]
                a_nd = g[Ind[0], :]
                b_nd = g[Ind[1], :]
                c_nd = g[Ind[2], :]
                InE = solver.Mesh2.Element[ii].Electrode[0]
                z_inv = 1.0 / float(z.flat[InE])
                s[InE] += z_inv * solver.electrlen(np.array([a_nd, c_nd]))
                bb1 = solver.bound_quad1(np.array([a_nd, b_nd, c_nd]))
                bb2 = solver.bound_quad2(np.array([a_nd, b_nd, c_nd]))
                for il in range(6):
                    eind = np.where(ind[il] == Ind)[0]
                    if eind.size != 0:
                        M_row.append(ind[il])
                        M_col.append(InE)
                        M_data.append(-z_inv * bb1[eind[0]])
                    for im in range(6):
                        eind1 = np.where(ind[il] == Ind)[0]
                        eind2 = np.where(ind[im] == Ind)[0]
                        if eind1.size != 0 and eind2.size != 0:
                            K_row.append(ind[il])
                            K_col.append(ind[im])
                            K_data.append(z_inv * bb2[eind1[0], eind2[0]])

        if M_data:
            M = sp.sparse.coo_matrix(
                (M_data, (M_row, M_col)), shape=(gN, solver.Nel),
            ).tocsr()
        else:
            M = sp.sparse.csr_matrix((gN, solver.Nel))

        if K_data:
            K = sp.sparse.coo_matrix(
                (K_data, (K_row, K_col)), shape=(gN, gN),
            ).tocsr()
        else:
            K = sp.sparse.csr_matrix((gN, gN))

        tS = sp.sparse.diags(s.flatten())
        S = sp.sparse.csr_matrix(solver.C.T * tS * solver.C)
        M = M * solver.C
        solver._S0_cached = sp.sparse.bmat([[K, M], [M.T, S]]).tocsr()
        solver._S0_cached.sum_duplicates()
        solver._S0_cached.sort_indices()
        solver._z_cached = z_flat.copy()

    A = (A0 + solver._S0_cached).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    solver.A = A
    assembly_ms = (time.time() - t0) * 1000.0
    return A, assembly_ms


def direct_solve(A, rhs):
    t0 = time.time()
    if HAS_PARDISO:
        theta = pardiso_spsolve(A, rhs)
    else:
        theta = scipy_spsolve(A, rhs)
    solve_ms = (time.time() - t0) * 1000.0
    return np.asarray(theta), solve_ms


def measurements_from_theta(solver, theta):
    theta = np.asarray(theta)
    Umeas = np.asarray(solver._MpatC * theta[solver.ng2:, :])
    Umeas = Umeas.T[solver.mincl.T].T
    return Umeas.reshape(-1, 1)


def torch_batched_pcg(A_t, B_t, M_inv_t, x0_t, maxiter):
    with torch.inference_mode():
        X = x0_t.clone()
        R = B_t - torch.sparse.mm(A_t, X)
        Z = M_inv_t[:, None] * R
        P = Z.clone()
        rz_old = torch.sum(R * Z, dim=0)
        for _ in range(maxiter):
            AP = torch.sparse.mm(A_t, P)
            pAp = torch.sum(P * AP, dim=0).clamp_min(1e-30)
            alpha = rz_old / pAp
            X = X + P * alpha.unsqueeze(0)
            R = R - AP * alpha.unsqueeze(0)
            Z = M_inv_t[:, None] * R
            rz_new = torch.sum(R * Z, dim=0)
            beta = rz_new / rz_old.clamp_min(1e-30)
            P = Z + P * beta.unsqueeze(0)
            rz_old = rz_new
    return X


def prepare_cupy_runtime(temp_dir):
    libdir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.isdir(libdir):
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(libdir)
            except OSError:
                pass
        path_value = os.environ.get('PATH', '')
        if libdir not in path_value.split(os.pathsep):
            os.environ['PATH'] = libdir + os.pathsep + path_value

    os.makedirs(temp_dir, exist_ok=True)
    os.environ['TMP'] = temp_dir
    os.environ['TEMP'] = temp_dir

    import cupy._environment as cupy_env

    # Some Windows wheel setups fail to locate CUDA headers through the
    # default pathfinder. The sparse runtime itself works without them here.
    cupy_env._get_include_dir_from_conda_or_wheel = lambda major, minor: []

    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse

    return cp, cpx_sparse


def cupy_batched_pcg(cp, A_gpu, B_gpu, M_inv_gpu, x0_gpu, maxiter):
    X = x0_gpu.copy()
    R = B_gpu - A_gpu @ X
    Z = M_inv_gpu[:, None] * R
    P = Z.copy()
    rz_old = cp.sum(R * Z, axis=0)

    for _ in range(maxiter):
        AP = A_gpu @ P
        pAp = cp.sum(P * AP, axis=0)
        alpha = rz_old / cp.maximum(pAp, 1e-30)
        X = X + P * alpha[None, :]
        R = R - AP * alpha[None, :]
        Z = M_inv_gpu[:, None] * R
        rz_new = cp.sum(R * Z, axis=0)
        beta = rz_new / cp.maximum(rz_old, 1e-30)
        P = Z + P * beta[None, :]
        rz_old = rz_new
    return X


class TorchPCGRunner:
    def __init__(self, A_pattern, rhs_basis, coeff, theta_bg_basis,
                 ng2, mpat_c, mincl_t, device='cuda'):
        self.device = device
        self.shape = A_pattern.shape
        self.indptr_cpu = A_pattern.indptr.copy()
        self.indices_cpu = A_pattern.indices.copy()
        self.indptr_t = torch.tensor(
            self.indptr_cpu.astype(np.int64), device=device)
        self.indices_t = torch.tensor(
            self.indices_cpu.astype(np.int64), device=device)
        self.values_t = torch.tensor(
            np.asarray(A_pattern.data, dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        self.a_t = torch.sparse_csr_tensor(
            self.indptr_t,
            self.indices_t,
            self.values_t,
            size=self.shape,
            device=device,
        )
        self.m_inv_t = torch.tensor(
            np.asarray(1.0 / A_pattern.diagonal(), dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        self.rhs_basis_t = torch.tensor(
            rhs_basis, dtype=torch.float64, device=device)
        self.coeff_t = torch.tensor(
            coeff, dtype=torch.float64, device=device)
        self.theta_bg_basis_t = torch.tensor(
            theta_bg_basis, dtype=torch.float64, device=device)
        self.ng2 = ng2
        self.mpat_c = np.asarray(mpat_c)
        self.mincl_t = mincl_t

    def solve_measurements(self, A, maxiter):
        if (not np.array_equal(A.indptr, self.indptr_cpu)
                or not np.array_equal(A.indices, self.indices_cpu)):
            self.indptr_cpu = A.indptr.copy()
            self.indices_cpu = A.indices.copy()
            self.indptr_t = torch.tensor(
                self.indptr_cpu.astype(np.int64), device=self.device)
            self.indices_t = torch.tensor(
                self.indices_cpu.astype(np.int64), device=self.device)
            self.values_t = torch.tensor(
                np.asarray(A.data, dtype=np.float64),
                dtype=torch.float64,
                device=self.device,
            )
            self.a_t = torch.sparse_csr_tensor(
                self.indptr_t,
                self.indices_t,
                self.values_t,
                size=self.shape,
                device=self.device,
            )
            self.m_inv_t = torch.tensor(
                np.asarray(1.0 / A.diagonal(), dtype=np.float64),
                dtype=torch.float64,
                device=self.device,
            )
        else:
            self.values_t.copy_(torch.from_numpy(np.asarray(A.data, dtype=np.float64)))
            self.m_inv_t.copy_(
                torch.from_numpy(np.asarray(1.0 / A.diagonal(), dtype=np.float64)))
        torch.cuda.synchronize()
        t0 = time.time()
        theta_basis_t = torch_batched_pcg(
            self.a_t, self.rhs_basis_t, self.m_inv_t,
            self.theta_bg_basis_t, maxiter=maxiter)
        theta_elec_t = theta_basis_t[self.ng2:, :] @ self.coeff_t
        torch.cuda.synchronize()
        solve_ms = (time.time() - t0) * 1000.0

        t0 = time.time()
        theta_elec = theta_elec_t.cpu().numpy()
        Umeas = np.asarray(self.mpat_c @ theta_elec)
        Umeas = Umeas.T[self.mincl_t].T.reshape(-1, 1)
        post_ms = (time.time() - t0) * 1000.0
        return Umeas, solve_ms, post_ms


class CupyPCGRunner:
    def __init__(self, cp, cpx_sparse, A_pattern, rhs_basis, coeff,
                 theta_bg_basis, ng2, mpat_c, mincl_t):
        self.cp = cp
        self.cpx_sparse = cpx_sparse
        self.shape = A_pattern.shape
        self.indptr_cpu = A_pattern.indptr.copy()
        self.indices_cpu = A_pattern.indices.copy()
        self.indptr_gpu = cp.asarray(self.indptr_cpu.astype(np.int32))
        self.indices_gpu = cp.asarray(self.indices_cpu.astype(np.int32))
        self.values_gpu = cp.asarray(np.asarray(A_pattern.data, dtype=np.float64))
        self.a_gpu = cpx_sparse.csr_matrix(
            (self.values_gpu, self.indices_gpu, self.indptr_gpu),
            shape=self.shape,
        )
        self.m_inv_gpu = cp.asarray(
            np.asarray(1.0 / A_pattern.diagonal(), dtype=np.float64))
        self.rhs_basis_gpu = cp.asarray(rhs_basis, dtype=cp.float64)
        self.coeff_gpu = cp.asarray(coeff, dtype=cp.float64)
        self.theta_bg_basis_gpu = cp.asarray(theta_bg_basis, dtype=cp.float64)
        self.ng2 = ng2
        self.mpat_c = np.asarray(mpat_c)
        self.mincl_t = np.asarray(mincl_t, dtype=bool)

    def solve_measurements(self, A, maxiter):
        cp = self.cp
        if (not np.array_equal(A.indptr, self.indptr_cpu)
                or not np.array_equal(A.indices, self.indices_cpu)):
            self.indptr_cpu = A.indptr.copy()
            self.indices_cpu = A.indices.copy()
            self.indptr_gpu = cp.asarray(self.indptr_cpu.astype(np.int32))
            self.indices_gpu = cp.asarray(self.indices_cpu.astype(np.int32))
            self.values_gpu = cp.asarray(np.asarray(A.data, dtype=np.float64))
            self.a_gpu = self.cpx_sparse.csr_matrix(
                (self.values_gpu, self.indices_gpu, self.indptr_gpu),
                shape=self.shape,
            )
            self.m_inv_gpu = cp.asarray(
                np.asarray(1.0 / A.diagonal(), dtype=np.float64))
        else:
            self.values_gpu.set(np.asarray(A.data, dtype=np.float64))
            self.m_inv_gpu.set(np.asarray(1.0 / A.diagonal(), dtype=np.float64))

        cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        theta_basis_gpu = cupy_batched_pcg(
            cp,
            self.a_gpu,
            self.rhs_basis_gpu,
            self.m_inv_gpu,
            self.theta_bg_basis_gpu,
            maxiter=maxiter,
        )
        theta_elec_gpu = theta_basis_gpu[self.ng2:, :] @ self.coeff_gpu
        cp.cuda.Stream.null.synchronize()
        solve_ms = (time.time() - t0) * 1000.0

        t0 = time.time()
        theta_elec = cp.asnumpy(theta_elec_gpu)
        Umeas = np.asarray(self.mpat_c @ theta_elec)
        Umeas = Umeas.T[self.mincl_t].T.reshape(-1, 1)
        post_ms = (time.time() - t0) * 1000.0
        return Umeas, solve_ms, post_ms


def plot_time_bars(summary, out_path):
    labels = list(summary.keys())
    values = [summary[k]['mean_total_ms'] for k in labels]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(labels)), values, color='#4C72B0')
    plt.xticks(range(len(labels)), labels, rotation=20, ha='right')
    plt.ylabel('Mean Forward Time (ms)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_pareto(summary, out_path):
    plt.figure(figsize=(6, 4))
    for label, stats in summary.items():
        plt.scatter(
            stats['mean_rel_error'], stats['mean_total_ms'],
            s=60, label=label)
        plt.annotate(label, (stats['mean_rel_error'], stats['mean_total_ms']),
                     xytext=(4, 4), textcoords='offset points', fontsize=9)
    plt.xlabel('Mean Measurement Relative Error')
    plt.ylabel('Mean Forward Time (ms)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def summarise_mode(records, baseline_ms):
    mean_total_ms = float(np.mean([r['total_ms'] for r in records]))
    mean_rel_error = float(np.mean([r['rel_error'] for r in records]))
    max_rel_error = float(np.max([r['rel_error'] for r in records]))
    mean_max_abs_error = float(np.mean([r['max_abs_error'] for r in records]))
    return {
        'num_samples': len(records),
        'mean_total_ms': mean_total_ms,
        'mean_rel_error': mean_rel_error,
        'max_rel_error': max_rel_error,
        'mean_max_abs_error': mean_max_abs_error,
        'speedup_vs_current': float(baseline_ms / mean_total_ms),
    }


def choose_recommendation(summary, target_rel_error):
    candidates = [
        (name, stats) for name, stats in summary.items()
        if stats['mean_rel_error'] <= target_rel_error
    ]
    if not candidates:
        return {
            'target_rel_error': target_rel_error,
            'recommended_mode': None,
            'reason': 'No tested mode met the target error threshold.',
        }

    best_name, best_stats = min(
        candidates, key=lambda item: item[1]['mean_total_ms'])
    return {
        'target_rel_error': target_rel_error,
        'recommended_mode': best_name,
        'mean_total_ms': best_stats['mean_total_ms'],
        'mean_rel_error': best_stats['mean_rel_error'],
        'speedup_vs_current': best_stats['speedup_vs_current'],
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but torch.cuda.is_available() is False.')

    experiment_name = f'{args.experiment_name}_l{args.level}_n{args.num_samples}'
    result_dir = create_result_dir(experiment_name, args.result_dir)

    print('=' * 60)
    print('Forward Batched PCG Benchmark')
    print(f'Result directory: {result_dir}')
    print(f'Num samples: {args.num_samples}')
    print(f'PCG iters: {args.pcg_iters}')
    print(f'RHS basis mode: {args.rhs_basis_mode}')
    print(f'GPU backends: {args.gpu_backends}')
    print(f'Device: {args.device}')
    print(f'GPU: {detect_gpu_name()}')
    print(f'PARDISO: {HAS_PARDISO}')
    print('=' * 60)

    y_ref = loadmat(args.ref_path)
    Injref = y_ref['Injref']
    Mpat = y_ref['Mpat']
    mesh, mesh2 = load_mesh(args.mesh_name)

    Nel = 32
    z = 1e-6 * np.ones((Nel, 1))
    vincl = np.ones((Nel - 1, 76), dtype=bool)

    solver = EITFEM(mesh2, Injref, Mpat, vincl, use_gpu=False)

    sigma_bg = np.ones((mesh.g.shape[0], 1)) * 0.745
    A_bg, _ = assemble_system(solver, sigma_bg, z)
    rhs_full = np.asarray(solver.b, dtype=np.float64)
    rhs_basis, coeff, rhs_rank, basis_info = make_rhs_basis(
        rhs_full, mode=args.rhs_basis_mode)
    theta_bg_basis, _ = direct_solve(A_bg, rhs_basis)

    torch_runner = None
    cupy_runner = None
    cupy_backend_status = {'available': False}
    if args.device.startswith('cuda'):
        if 'torch' in args.gpu_backends:
            torch_runner = TorchPCGRunner(
                A_pattern=A_bg,
                rhs_basis=rhs_basis,
                coeff=coeff,
                theta_bg_basis=theta_bg_basis,
                ng2=solver.ng2,
                mpat_c=solver._MpatC,
                mincl_t=solver.mincl.T,
                device=args.device,
            )

        if 'cupy' in args.gpu_backends:
            try:
                cp, cpx_sparse = prepare_cupy_runtime(
                    os.path.join(result_dir, 'cupy_build_tmp'))
                cupy_runner = CupyPCGRunner(
                    cp=cp,
                    cpx_sparse=cpx_sparse,
                    A_pattern=A_bg,
                    rhs_basis=rhs_basis,
                    coeff=coeff,
                    theta_bg_basis=theta_bg_basis,
                    ng2=solver.ng2,
                    mpat_c=solver._MpatC,
                    mincl_t=solver.mincl.T,
                )
                cupy_backend_status = {
                    'available': True,
                    'cupy_version': cp.__version__,
                }
            except Exception as exc:
                cupy_backend_status = {
                    'available': False,
                    'error': repr(exc),
                }
                print(f'CuPy backend unavailable: {exc!r}')

    mode_records = {
        'current_cpu_direct': [],
        'cpu_direct_reduced_exact': [],
    }
    if torch_runner is not None:
        for maxiter in args.pcg_iters:
            mode_records[f'gpu_torch_batched_pcg_iter{maxiter}'] = []
    if cupy_runner is not None:
        for maxiter in args.pcg_iters:
            mode_records[f'gpu_cupy_batched_pcg_iter{maxiter}'] = []

    if args.warmup_iters > 0:
        warmup_iters = min(args.warmup_iters, max(args.pcg_iters))
        print(f'Running GPU warmup ({warmup_iters} iters, not timed)...')
        if torch_runner is not None:
            torch_runner.solve_measurements(A_bg, maxiter=warmup_iters)
        if cupy_runner is not None:
            cupy_runner.solve_measurements(A_bg, maxiter=warmup_iters)

    sample_iter = tqdm(range(args.num_samples), desc='Benchmark samples')
    for _ in sample_iter:
        sigma_gt = create_sigma_sample(mesh)
        A, assembly_ms = assemble_system(solver, sigma_gt, z)

        theta_full, direct_solve_ms = direct_solve(A, rhs_full)
        U_current = measurements_from_theta(solver, theta_full)
        mode_records['current_cpu_direct'].append({
            'assembly_ms': assembly_ms,
            'solve_ms': direct_solve_ms,
            'post_ms': 0.0,
            'total_ms': assembly_ms + direct_solve_ms,
            'rel_error': 0.0,
            'max_abs_error': 0.0,
        })

        theta_basis_exact, reduced_solve_ms = direct_solve(A, rhs_basis)
        theta_reduced = theta_basis_exact @ coeff
        U_reduced = measurements_from_theta(solver, theta_reduced)
        mode_records['cpu_direct_reduced_exact'].append({
            'assembly_ms': assembly_ms,
            'solve_ms': reduced_solve_ms,
            'post_ms': 0.0,
            'total_ms': assembly_ms + reduced_solve_ms,
            'rel_error': float(
                np.linalg.norm(U_reduced - U_current) / np.linalg.norm(U_current)),
            'max_abs_error': float(np.max(np.abs(U_reduced - U_current))),
        })

        if torch_runner is not None:
            for maxiter in args.pcg_iters:
                mode_name = f'gpu_torch_batched_pcg_iter{maxiter}'
                U_gpu, gpu_solve_ms, post_ms = torch_runner.solve_measurements(
                    A, maxiter=maxiter)
                rel_error = float(
                    np.linalg.norm(U_gpu - U_current) / np.linalg.norm(U_current))
                max_abs_error = float(np.max(np.abs(U_gpu - U_current)))
                mode_records[mode_name].append({
                    'assembly_ms': assembly_ms,
                    'solve_ms': gpu_solve_ms,
                    'post_ms': post_ms,
                    'total_ms': assembly_ms + gpu_solve_ms + post_ms,
                    'rel_error': rel_error,
                    'max_abs_error': max_abs_error,
                })

        if cupy_runner is not None:
            for maxiter in args.pcg_iters:
                mode_name = f'gpu_cupy_batched_pcg_iter{maxiter}'
                U_gpu, gpu_solve_ms, post_ms = cupy_runner.solve_measurements(
                    A, maxiter=maxiter)
                rel_error = float(
                    np.linalg.norm(U_gpu - U_current) / np.linalg.norm(U_current))
                max_abs_error = float(np.max(np.abs(U_gpu - U_current)))
                mode_records[mode_name].append({
                    'assembly_ms': assembly_ms,
                    'solve_ms': gpu_solve_ms,
                    'post_ms': post_ms,
                    'total_ms': assembly_ms + gpu_solve_ms + post_ms,
                    'rel_error': rel_error,
                    'max_abs_error': max_abs_error,
                })

    baseline_ms = np.mean(
        [r['total_ms'] for r in mode_records['current_cpu_direct']])
    summary = {
        name: summarise_mode(records, baseline_ms)
        for name, records in mode_records.items()
    }
    recommendation = choose_recommendation(summary, args.target_rel_error)

    summary_path = os.path.join(result_dir, 'summary.json')
    payload = {
        'config': {
            'num_samples': args.num_samples,
            'level': args.level,
            'ref_path': args.ref_path,
            'mesh_name': args.mesh_name,
            'seed': args.seed,
            'pcg_iters': args.pcg_iters,
            'rhs_basis_mode': args.rhs_basis_mode,
            'gpu_backends': args.gpu_backends,
            'target_rel_error': args.target_rel_error,
            'device': args.device,
            'warmup_iters': args.warmup_iters,
        },
        'environment': {
            'gpu_name': detect_gpu_name(),
            'torch_version': torch.__version__,
            'pardiso': HAS_PARDISO,
            'cupy_backend': cupy_backend_status,
        },
        'static_info': {
            'rhs_full_shape': list(rhs_full.shape),
            'rhs_reduced_rank': rhs_rank,
            'basis_info': basis_info,
            'matrix_shape': list(A_bg.shape),
            'matrix_nnz': int(A_bg.nnz),
        },
        'summary': summary,
        'per_sample_records': mode_records,
        'recommendation': recommendation,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    plot_time_bars(summary, os.path.join(result_dir, 'forward_time_comparison.png'))
    plot_pareto(summary, os.path.join(result_dir, 'forward_pareto.png'))

    print('\nSummary:')
    for name, stats in summary.items():
        print(
            f'  {name:>26}: '
            f'{stats["mean_total_ms"]:8.1f} ms, '
            f'rel_err={stats["mean_rel_error"]:.4e}, '
            f'speedup={stats["speedup_vs_current"]:.2f}x')
    print(f'\nSaved to: {summary_path}')


if __name__ == '__main__':
    main()
