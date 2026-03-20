"""
Digital Twin Residual Analysis: calibrate real conductivity values
and extract Sim-to-Real noise distribution from KTC2023 challenge data.

Uses the 4 labelled real measurements + ground truth masks to:
1. Fit physical conductivity scalars (sigma_bg, sigma_1, sigma_2) per sample
2. Compute residual E_gap = U_real - U_sim_best
3. Profile the noise distribution (mean, std, covariance structure)

Usage:
    python scripts/calibrate_real_data.py
    python scripts/calibrate_real_data.py --result-dir results/calibration_1
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ktc_methods import EITFEM, load_mesh, image_to_mesh


def get_next_result_dir(base_dir):
    """Return auto-incremented result dir: results/calibration_1, _2, ..."""
    parent = os.path.dirname(base_dir) or '.'
    stem = os.path.basename(base_dir.rstrip('/\\'))
    num = 1
    while os.path.exists(os.path.join(parent, f'{stem}_{num}')):
        num += 1
    return os.path.join(parent, f'{stem}_{num}')


def setup_solver(ref_path, mesh_name):
    """Initialize EITFEM solver with reference data."""
    y_ref = loadmat(ref_path)
    Injref = y_ref['Injref']
    Mpat = y_ref['Mpat']
    mesh, mesh2 = load_mesh(mesh_name)

    Nel = 32
    z = 1e-6 * np.ones((Nel, 1))
    vincl = np.ones((Nel - 1, 76), dtype=bool)

    solver = EITFEM(mesh2, Injref, Mpat, vincl)
    return solver, mesh, z, y_ref


def build_sigma_from_mask(truth, sigma_bg, sigma_1, sigma_2, mesh):
    """Build mesh conductivity from pixel-space truth mask + scalar values."""
    sigma_pix = np.zeros_like(truth, dtype=np.float64)
    sigma_pix[truth == 0] = sigma_bg
    sigma_pix[truth == 1] = sigma_1
    sigma_pix[truth == 2] = sigma_2
    return image_to_mesh(np.flipud(sigma_pix).T, mesh)


def fit_sample(solver, mesh, z, truth, Uel_real, fix_bg=None):
    """Fit conductivity scalars for one sample.

    Args:
        fix_bg: If not None, fix background conductivity to this value
                and only optimize sigma_1/sigma_2.

    Returns: (fit_info, U_sim_best)
    """
    classes_present = np.unique(truth)
    has_class1 = 1 in classes_present
    has_class2 = 2 in classes_present

    call_count = [0]

    def objective(params):
        call_count[0] += 1
        if fix_bg is not None:
            sigma_bg = fix_bg
            idx = 0
        else:
            sigma_bg = params[0]
            idx = 1

        if has_class1:
            sigma_1 = params[idx]; idx += 1
        else:
            sigma_1 = sigma_bg
        if has_class2:
            sigma_2 = params[idx]
        else:
            sigma_2 = sigma_bg

        sigma_mesh = build_sigma_from_mask(truth, sigma_bg, sigma_1, sigma_2, mesh)
        U_sim = np.asarray(solver.SolveForward(sigma_mesh, z.copy())).flatten()

        residual = U_sim - Uel_real
        return np.sum(residual ** 2)

    x0 = []
    bounds = []
    if fix_bg is None:
        x0.append(0.8)
        bounds.append((0.01, 10.0))
    if has_class1:
        x0.append(0.05)
        bounds.append((0.0001, 5.0))
    if has_class2:
        x0.append(5.0)
        bounds.append((0.1, 50.0))

    t0 = time.time()
    if not x0:
        # Only background class, nothing to optimize
        elapsed = 0.0
        sigma_bg = fix_bg if fix_bg is not None else 0.8
        sigma_mesh = build_sigma_from_mask(truth, sigma_bg, sigma_bg, sigma_bg, mesh)
        U_sim_best = np.asarray(solver.SolveForward(sigma_mesh, z.copy())).flatten()
        fit_info = {
            'sigma_bg': float(sigma_bg),
            'sigma_1': None, 'sigma_2': None,
            'has_class1': False, 'has_class2': False,
            'loss': float(np.sum((U_sim_best - Uel_real) ** 2)),
            'nfev': 1, 'elapsed_s': 0.0, 'success': True,
            'fix_bg': fix_bg is not None,
        }
        return fit_info, U_sim_best

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-10})
    elapsed = time.time() - t0

    # Extract final params
    if fix_bg is not None:
        sigma_bg = fix_bg
        idx = 0
    else:
        sigma_bg = result.x[0]
        idx = 1

    if has_class1:
        sigma_1 = result.x[idx]; idx += 1
    else:
        sigma_1 = sigma_bg
    if has_class2:
        sigma_2 = result.x[idx]
    else:
        sigma_2 = sigma_bg

    # Compute final U_sim
    sigma_mesh = build_sigma_from_mask(truth, sigma_bg, sigma_1, sigma_2, mesh)
    U_sim_best = np.asarray(solver.SolveForward(sigma_mesh, z.copy())).flatten()

    fit_info = {
        'sigma_bg': float(sigma_bg),
        'sigma_1': float(sigma_1) if has_class1 else None,
        'sigma_2': float(sigma_2) if has_class2 else None,
        'has_class1': bool(has_class1),
        'has_class2': bool(has_class2),
        'fix_bg': fix_bg is not None,
        'loss': float(result.fun),
        'nfev': call_count[0],
        'elapsed_s': elapsed,
        'success': bool(result.success),
    }

    return fit_info, U_sim_best


def fit_and_report(solver, mesh, z, truth, Uel_real, label, fix_bg=None):
    """Fit one sample, print results, return (fit_info, E_gap)."""
    classes = np.unique(truth)
    print(f'  Classes present: {classes}')
    if fix_bg is not None:
        print(f'  sigma_bg fixed to {fix_bg:.6f} S/m')

    fit_info, U_sim_best = fit_sample(solver, mesh, z, truth, Uel_real,
                                      fix_bg=fix_bg)
    print(f'  sigma_bg = {fit_info["sigma_bg"]:.6f} S/m')
    if fit_info['has_class1']:
        print(f'  sigma_1 (resistive) = {fit_info["sigma_1"]:.6f} S/m')
    if fit_info['has_class2']:
        print(f'  sigma_2 (conductive) = {fit_info["sigma_2"]:.6f} S/m')
    print(f'  Loss = {fit_info["loss"]:.6e}, '
          f'nfev = {fit_info["nfev"]}, '
          f'time = {fit_info["elapsed_s"]:.1f}s')

    E_gap = Uel_real - U_sim_best
    rel_err = np.linalg.norm(E_gap) / np.linalg.norm(Uel_real)
    print(f'  E_gap: mean={E_gap.mean():.6e}, std={E_gap.std():.6e}, '
          f'max_abs={np.max(np.abs(E_gap)):.6e}')
    print(f'  Relative fit error: {rel_err:.4e}')

    fit_info['E_gap_mean'] = float(E_gap.mean())
    fit_info['E_gap_std'] = float(E_gap.std())
    fit_info['relative_error'] = float(rel_err)
    return fit_info, E_gap


def noise_profiling(E_list, Uelref, group_name):
    """Print noise statistics for a group of E_gap vectors."""
    if not E_list:
        print(f'\n{"=" * 60}')
        print(f'Noise Distribution Analysis — {group_name} (0 samples)')
        print(f'{"=" * 60}')
        print('  No samples available, skipping.')
        return None

    E_all = np.stack(E_list)  # (N, 2356)
    n = E_all.shape[0]

    print(f'\n{"=" * 60}')
    print(f'Noise Distribution Analysis — {group_name} ({n} samples)')
    print(f'{"=" * 60}')

    E_mean = E_all.mean(axis=0)
    print(f'\n  Mean bias (systematic): mean={E_mean.mean():.6e}, '
          f'std={E_mean.std():.6e}')
    print(f'  Per-sample noise std: '
          f'{[f"{E_all[i].std():.4e}" for i in range(n)]}')
    print(f'  Overall noise std: {E_all.std():.4e}')

    # SNR (use Uelref as signal reference)
    signal_power = np.mean(Uelref ** 2)
    for i in range(n):
        noise_power = np.mean(E_list[i] ** 2)
        snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-30))
        print(f'  Sample {i+1} SNR: {snr_db:.1f} dB')

    if n >= 2:
        E_var = np.var(E_all, axis=0)
        print(f'\n  Per-channel variance: mean={E_var.mean():.4e}, '
              f'max={E_var.max():.4e}, min={E_var.min():.4e}')
        corr = np.corrcoef(E_all)
        off_diag = corr[np.triu_indices(n, k=1)]
        print(f'  Cross-sample correlation: '
              f'mean={off_diag.mean():.3f}, '
              f'min={off_diag.min():.3f}, max={off_diag.max():.3f}')

    return E_all


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate real conductivity and extract noise')
    parser.add_argument('--data-dir', type=str,
                        default='KTC2023/Codes_Python/TrainingData')
    parser.add_argument('--gt-dir', type=str,
                        default='KTC2023/Codes_Python/GroundTruths')
    parser.add_argument('--eval-data-dir', type=str,
                        default='KTC2023/EvaluationData_full/evaluation_datasets')
    parser.add_argument('--eval-gt-dir', type=str,
                        default='KTC2023/EvaluationData_full/GroundTruths')
    parser.add_argument('--ref-path', type=str,
                        default='KTC2023/Codes_Python/TrainingData/ref.mat')
    parser.add_argument('--mesh-name', type=str, default='Mesh_dense.mat')
    parser.add_argument('--result-dir', type=str,
                        default='results/calibration')
    parser.add_argument('--fix-bg', action='store_true',
                        help='Fix background conductivity to value fitted '
                             'from Uelref, only optimize sigma_1/sigma_2')
    args = parser.parse_args()

    args.result_dir = get_next_result_dir(args.result_dir)
    os.makedirs(args.result_dir, exist_ok=True)

    # Setup solver
    solver, mesh, z, y_ref = setup_solver(args.ref_path, args.mesh_name)
    Uelref = y_ref['Uelref'].flatten()

    print('=' * 60)
    print('Digital Twin Residual Analysis')
    print('=' * 60)

    # === Step 0: Fit background conductivity from Uelref ===
    print('\n--- Step 0: Fit background conductivity from Uelref ---')

    def bg_objective(params):
        sigma_uniform = np.ones((mesh.g.shape[0], 1)) * params[0]
        U_sim = np.asarray(solver.SolveForward(sigma_uniform, z.copy())).flatten()
        return np.sum((U_sim - Uelref) ** 2)

    result_bg = minimize(bg_objective, [0.8], method='L-BFGS-B',
                         bounds=[(0.01, 10.0)],
                         options={'maxiter': 200, 'ftol': 1e-10})
    sigma_bg_ref = float(result_bg.x[0])
    print(f'  Background conductivity (from Uelref): {sigma_bg_ref:.6f} S/m')
    print(f'  Loss: {result_bg.fun:.6e}, nfev: {result_bg.nfev}')

    sigma_uniform = np.ones((mesh.g.shape[0], 1)) * sigma_bg_ref
    U_ref_sim = np.asarray(solver.SolveForward(sigma_uniform, z.copy())).flatten()
    E_ref = Uelref - U_ref_sim
    print(f'  Uelref residual: mean={E_ref.mean():.6e}, std={E_ref.std():.6e}, '
          f'max_abs={np.max(np.abs(E_ref)):.6e}')

    all_results = {'background_sigma': sigma_bg_ref,
                    'fix_bg': args.fix_bg}
    fix_bg_val = sigma_bg_ref if args.fix_bg else None
    if args.fix_bg:
        print(f'\n  --fix-bg enabled: all samples use sigma_bg={sigma_bg_ref:.6f}')

    # ==========================================================
    # Part A: Training data (4 samples)
    # ==========================================================
    print(f'\n{"=" * 60}')
    print('Part A: Training Data (4 samples)')
    print(f'{"=" * 60}')

    train_E_gap = []
    for i in range(1, 5):
        print(f'\n--- Train Sample {i} ---')
        data = loadmat(os.path.join(args.data_dir, f'data{i}.mat'))
        truth = loadmat(os.path.join(args.gt_dir, f'true{i}.mat'))['truth']
        Uel_real = data['Uel'].flatten()
        fit_info, E_gap = fit_and_report(
            solver, mesh, z, truth, Uel_real, f'train_{i}',
            fix_bg=fix_bg_val)
        train_E_gap.append(E_gap)
        all_results[f'train_sample_{i}'] = fit_info

    train_E_all = noise_profiling(train_E_gap, Uelref, 'Training Data')

    # ==========================================================
    # Part B: Evaluation data (7 levels × 3 samples)
    # ==========================================================
    print(f'\n\n{"=" * 60}')
    print('Part B: Evaluation Data (7 levels x 3 samples)')
    print(f'{"=" * 60}')

    eval_E_gap = []
    for level in range(1, 8):
        data_dir = os.path.join(args.eval_data_dir, f'level{level}')
        gt_dir = os.path.join(args.eval_gt_dir, f'level_{level}')
        if not os.path.isdir(data_dir):
            print(f'\n  Skipping level {level}: {data_dir} not found')
            continue

        for sid in range(1, 4):
            data_path = os.path.join(data_dir, f'data{sid}.mat')
            gt_path = os.path.join(gt_dir, f'{sid}_true.mat')
            if not os.path.exists(data_path) or not os.path.exists(gt_path):
                continue

            label = f'eval_L{level}_S{sid}'
            print(f'\n--- {label} ---')
            data = loadmat(data_path)
            truth = loadmat(gt_path)['truth']
            Uel_real = data['Uel'].flatten()
            fit_info, E_gap = fit_and_report(
                solver, mesh, z, truth, Uel_real, label,
                fix_bg=fix_bg_val)
            eval_E_gap.append(E_gap)
            all_results[label] = fit_info

    eval_E_all = noise_profiling(eval_E_gap, Uelref, 'Evaluation Data')

    # ==========================================================
    # Combined noise profiling
    # ==========================================================
    all_E_gap = train_E_gap + eval_E_gap
    combined_E_all = noise_profiling(all_E_gap, Uelref,
                                     f'Combined ({len(all_E_gap)} samples)')

    # Save results
    results_path = os.path.join(args.result_dir, 'calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to: {results_path}')

    # Save CSV
    csv_path = os.path.join(args.result_dir, 'calibration_results.csv')
    csv_fields = ['sample', 'sigma_bg', 'fix_bg', 'sigma_1', 'sigma_2',
                  'has_class1', 'has_class2', 'loss', 'nfev',
                  'elapsed_s', 'E_gap_mean', 'E_gap_std', 'relative_error']
    csv_rows = [{'sample': k, **v} for k, v in all_results.items()
                if isinstance(v, dict)]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(csv_rows)
    print(f'CSV saved to: {csv_path}')

    if train_E_all is not None:
        np.save(os.path.join(args.result_dir, 'E_gap_train.npy'), train_E_all)
    if eval_E_all is not None:
        np.save(os.path.join(args.result_dir, 'E_gap_eval.npy'), eval_E_all)
    if combined_E_all is not None:
        np.save(os.path.join(args.result_dir, 'E_gap_all.npy'), combined_E_all)
    print(f'E_gap arrays saved to: {args.result_dir}/')


if __name__ == '__main__':
    main()
