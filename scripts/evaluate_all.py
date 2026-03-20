"""
Unified evaluation script for KTC2023 EIT reconstruction methods.

Evaluates FCUNet, PostProcessing UNet, and Conditional Diffusion pipelines
on the official evaluation datasets (7 difficulty levels, 3 samples each).

Usage:
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --methods fcunet postp
    python scripts/evaluate_all.py --methods condd --levels 1 2 3
    python scripts/evaluate_all.py --device cpu --no-plot
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.eval_dataset import EvaluationDataLoader
from src.evaluation.scoring import scoring_function
from src.evaluation.visualization import (
    plot_reconstruction_comparison,
    plot_method_overview,
    plot_cross_method_comparison,
    plot_scores_summary,
)


# Paper-reported reference scores for comparison
PAPER_SCORES = {
    'FCUNet': {1: 2.72, 2: 2.64, 3: 2.31, 4: 1.80, 5: 2.06, 6: 2.07, 7: 1.53},
    'PostP':  {1: 2.76, 2: 2.56, 3: 2.54, 4: 1.71, 5: 2.06, 6: 1.92, 7: 1.69},
    'CondD':  {1: 2.67, 2: 2.49, 3: 2.47, 4: 1.61, 5: 1.94, 6: 1.76, 7: 1.65},
}


def get_output_dir(method_name, base_dir='results'):
    """Create a non-conflicting output directory for results."""
    num = 1
    while True:
        dir_name = os.path.join(base_dir, f'eval_{method_name}_{num}')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            return dir_name
        num += 1


def build_pipeline(method_name, device, weights_base_dir, batch_mode):
    """Build a pipeline instance by method name."""
    if method_name == 'fcunet':
        from src.pipelines import FCUNetPipeline
        return FCUNetPipeline(device=device, weights_base_dir=weights_base_dir)
    elif method_name == 'postp':
        from src.pipelines import PostPPipeline
        return PostPPipeline(device=device, weights_base_dir=weights_base_dir)
    elif method_name == 'condd':
        from src.pipelines import CondDPipeline
        return CondDPipeline(device=device, weights_base_dir=weights_base_dir,
                             batch_mode=batch_mode)
    else:
        raise ValueError(f'Unknown method: {method_name}')


def evaluate_method(method_name, pipeline, levels, eval_loader, output_dir,
                    save_mat=True, save_plots=True):
    """Evaluate a single method across the specified levels.

    Returns:
        results: {level: {'scores': [...], 'sum_score': float, 'mean_score': float}}
        level_data: {level: {'ground_truths': [...], 'reconstructions': [...],
                     'scores': [...]}}  (for plot generation)
    """
    results = {}
    level_data = {}

    for level in levels:
        print(f'\n  Level {level}:')

        pipeline.load_model(level)
        ref_data = eval_loader.load_reference(level)
        measurements = eval_loader.load_measurements(level)
        ground_truths = eval_loader.load_ground_truths(level)

        reconstructions = []
        scores = []

        if hasattr(pipeline, 'reconstruct_batch'):
            t0 = time.time()
            reconstructions = pipeline.reconstruct_batch(
                measurements, ref_data, level)
            elapsed = time.time() - t0

            pbar = tqdm(range(len(reconstructions)), total=len(reconstructions),
                        desc='    Reconstructing', leave=False)
            for i in pbar:
                if i < len(ground_truths):
                    score = scoring_function(ground_truths[i],
                                             reconstructions[i])
                    scores.append(score)
                    pbar.set_postfix(score=f'{score:.4f}',
                                     time=f'{elapsed / len(reconstructions):.1f}s')
        else:
            pbar = tqdm(enumerate(measurements), total=len(measurements),
                        desc='    Reconstructing', leave=False)
            for i, Uel in pbar:
                t0 = time.time()
                reco = pipeline.reconstruct(Uel, ref_data, level)
                elapsed = time.time() - t0
                reconstructions.append(reco)

                if i < len(ground_truths):
                    score = scoring_function(ground_truths[i], reco)
                    scores.append(score)
                    pbar.set_postfix(score=f'{score:.4f}',
                                     time=f'{elapsed:.1f}s')

        sum_score = float(np.sum(scores))
        mean_score = float(np.mean(scores)) if scores else 0.0

        results[level] = {
            'scores': [float(s) for s in scores],
            'sum_score': sum_score,
            'mean_score': mean_score,
        }

        level_data[level] = {
            'ground_truths': ground_truths,
            'reconstructions': reconstructions,
            'scores': [float(s) for s in scores],
        }

        print(f'    Scores: {[f"{s:.4f}" for s in scores]}')
        print(f'    Sum: {sum_score:.4f}, Mean: {mean_score:.4f}')

        # Save reconstruction .mat files
        if save_mat:
            level_dir = os.path.join(output_dir, f'level_{level}')
            os.makedirs(level_dir, exist_ok=True)
            for i, reco in enumerate(reconstructions):
                mat_path = os.path.join(level_dir, f'{i + 1}.mat')
                savemat(mat_path, {'reconstruction': reco.astype(int)})

        # Save per-level comparison plot (GT vs Reco)
        if save_plots:
            plot_path = os.path.join(output_dir, f'comparison_level_{level}.png')
            plot_reconstruction_comparison(
                ground_truths, reconstructions, level,
                scores=scores, save_path=plot_path
            )

    # Save per-method multi-level overview plot
    if save_plots and len(level_data) > 0:
        overview_path = os.path.join(output_dir, f'overview_{method_name}.png')
        plot_method_overview(level_data, method_name, save_path=overview_path)
        print(f'  Overview plot saved to: {overview_path}')

    return results, level_data


def print_summary_table(all_results):
    """Print a formatted summary table comparing with paper scores."""
    methods = list(all_results.keys())
    levels = sorted(set(lvl for m in methods for lvl in all_results[m].keys()))

    # Header
    header = f'{"Method":<10} ' + ' '.join(f'{"L" + str(l):>8}' for l in levels) + f' {"Total":>8}'
    print('\n' + '=' * len(header))
    print(header)
    print('-' * len(header))

    for method in methods:
        level_sums = []
        for level in levels:
            if level in all_results[method]:
                s = all_results[method][level]['sum_score']
                level_sums.append(s)
            else:
                level_sums.append(0.0)
        total = sum(level_sums)
        row = f'{method:<10} ' + ' '.join(f'{s:>8.4f}' for s in level_sums) + f' {total:>8.4f}'
        print(row)

    # Print paper reference scores
    print('-' * len(header))
    paper_method_map = {'fcunet': 'FCUNet', 'postp': 'PostP', 'condd': 'CondD'}
    for method in methods:
        paper_key = paper_method_map.get(method)
        if paper_key and paper_key in PAPER_SCORES:
            paper = PAPER_SCORES[paper_key]
            level_sums = [paper.get(l, 0.0) for l in levels]
            total = sum(level_sums)
            row = f'{method + "(ref)":<10} ' + ' '.join(f'{s:>8.4f}' for s in level_sums) + f' {total:>8.4f}'
            print(row)

    print('=' * len(header))


def main():
    parser = argparse.ArgumentParser(description='Evaluate KTC2023 EIT reconstruction methods')
    parser.add_argument('--methods', nargs='+', default=['fcunet', 'postp', 'condd'],
                        choices=['fcunet', 'postp', 'condd'],
                        help='Methods to evaluate (default: all three)')
    parser.add_argument('--levels', nargs='+', type=int, default=list(range(1, 8)),
                        help='Difficulty levels to evaluate (default: 1-7)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Compute device (default: cuda)')
    parser.add_argument('--weights-dir', default='KTC2023_SubmissionFiles',
                        help='Base directory for pre-trained weights')
    parser.add_argument('--eval-data-dir', default='KTC2023/EvaluationData/evaluation_datasets',
                        help='Evaluation datasets directory')
    parser.add_argument('--gt-dir', default='KTC2023/EvaluationData/GroundTruths',
                        help='Ground truths directory')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating comparison plots')
    parser.add_argument('--no-mat', action='store_true',
                        help='Skip saving .mat reconstruction files')
    parser.add_argument('--condd-sequential', action='store_true',
                        help='Run conditional diffusion in sequential mode (less VRAM)')
    args = parser.parse_args()

    eval_loader = EvaluationDataLoader(
        eval_data_dir=args.eval_data_dir,
        gt_dir=args.gt_dir,
    )

    all_results = {}
    all_level_data = {}  # For cross-method comparison plots
    output_dirs = {}

    for method in args.methods:
        print(f'\n{"=" * 60}')
        print(f'Evaluating method: {method.upper()}')
        print(f'{"=" * 60}')

        output_dir = get_output_dir(method)
        output_dirs[method] = output_dir
        print(f'Results will be saved to: {output_dir}')

        batch_mode = not args.condd_sequential
        pipeline = build_pipeline(method, args.device, args.weights_dir, batch_mode)

        results, level_data = evaluate_method(
            method_name=method,
            pipeline=pipeline,
            levels=args.levels,
            eval_loader=eval_loader,
            output_dir=output_dir,
            save_mat=not args.no_mat,
            save_plots=not args.no_plot,
        )

        # Save per-method scores
        scores_path = os.path.join(output_dir, 'scores.json')
        with open(scores_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nScores saved to: {scores_path}')

        all_results[method] = results
        all_level_data[method] = level_data

        # Clean up pipeline to free memory
        del pipeline

    # Print summary table
    print_summary_table(all_results)

    # Generate cross-method plots
    if not args.no_plot and len(all_results) > 0:
        # Use the first method's output dir as the base for cross-method plots
        first_dir = output_dirs[args.methods[0]]
        base_dir = os.path.dirname(first_dir)

        # Cross-method comparison plot (shared levels)
        if len(all_level_data) > 1:
            cross_path = os.path.join(base_dir, 'cross_method_comparison.png')
            plot_cross_method_comparison(all_level_data, save_path=cross_path)
            print(f'\nCross-method comparison saved to: {cross_path}')

        # Scores bar chart
        scores_for_plot = {}
        for method, method_results in all_results.items():
            scores_for_plot[method] = {
                level: data['scores'] for level, data in method_results.items()
            }
        bar_path = os.path.join(base_dir, 'scores_bar_chart.png')
        plot_scores_summary(scores_for_plot, save_path=bar_path)
        print(f'Scores bar chart saved to: {bar_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
