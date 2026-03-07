"""
Visualization module for KTC2023 EIT reconstruction evaluation.

Provides plotting utilities for:
  - Side-by-side comparison of ground truth vs reconstruction (per level)
  - Per-method multi-level overview (all levels in one figure)
  - Cross-method comparison on shared levels
  - Score summaries across methods and difficulty levels

All plots use English labels, no titles (per project conventions),
and discrete colormaps for the 3-class segmentation {0, 1, 2}.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def _get_discrete_cmap():
    """Create a discrete colormap for the 3-class EIT segmentation."""
    colors = ['#D9D9D9', '#4393C3', '#D6604D']  # gray, blue, red
    cmap = mcolors.ListedColormap(colors, name='eit_3class')
    norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)
    return cmap, norm


def _add_colorbar(fig, cmap, norm, rect=None, shrink=None, ax=None):
    """Add a colorbar to the figure without overlapping plot content."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if rect is not None:
        cbar_ax = fig.add_axes(rect)
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1, 2])
    else:
        cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2],
                            shrink=shrink or 0.6,
                            orientation='vertical', pad=0.03)
    cbar.set_label('Class', fontsize=10)
    cbar.ax.set_yticklabels(['0 (BG)', '1 (Res)', '2 (Con)'], fontsize=8)
    return cbar


def _save_or_show(fig, save_path, dpi):
    """Save figure to file or display interactively."""
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_reconstruction_comparison(ground_truths, reconstructions, level,
                                   scores=None, save_path=None, dpi=150):
    """Plot ground truth vs reconstruction side by side for a single level.

    Args:
        ground_truths: List of 256x256 arrays with values in {0, 1, 2}.
        reconstructions: List of 256x256 arrays with values in {0, 1, 2}.
        level: Difficulty level (int).
        scores: Optional list of SSIM scores per sample.
        save_path: Path to save figure, or None for interactive display.
        dpi: Resolution of saved figure.
    """
    n_samples = len(ground_truths)
    cmap, norm = _get_discrete_cmap()

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6),
                             squeeze=False)

    for j in range(n_samples):
        ax_gt = axes[0, j]
        ax_gt.imshow(ground_truths[j], cmap=cmap, norm=norm,
                     interpolation='nearest')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if j == 0:
            ax_gt.set_ylabel('Ground Truth', fontsize=11)
        ax_gt.set_xlabel(f'Sample {j + 1}', fontsize=9)

        ax_re = axes[1, j]
        ax_re.imshow(reconstructions[j], cmap=cmap, norm=norm,
                     interpolation='nearest')
        ax_re.set_xticks([])
        ax_re.set_yticks([])
        if j == 0:
            ax_re.set_ylabel(f'Reconstruction (L{level})', fontsize=11)
        if scores and j < len(scores) and scores[j] is not None:
            ax_re.set_xlabel(f'SSIM={scores[j]:.4f}', fontsize=8)

    fig.subplots_adjust(right=0.88)
    _add_colorbar(fig, cmap, norm, rect=[0.91, 0.25, 0.015, 0.5])
    _save_or_show(fig, save_path, dpi)


def plot_method_overview(level_data, method_name, save_path=None, dpi=150):
    """Plot multi-level overview for a single method.

    Creates a figure with 2 rows per level (GT + Reco), 3 sample columns.

    Args:
        level_data: dict {level_int: {'ground_truths': [...],
                    'reconstructions': [...], 'scores': [...]}}
        method_name: Name string for labeling.
        save_path: Path to save figure, or None for interactive display.
        dpi: Resolution of saved figure.
    """
    levels = sorted(level_data.keys())
    n_levels = len(levels)
    if n_levels == 0:
        return
    n_samples = 3
    cmap, norm = _get_discrete_cmap()

    fig, axes = plt.subplots(
        n_levels * 2, n_samples,
        figsize=(3 * n_samples, 2.5 * n_levels * 2),
        squeeze=False,
    )

    for li, level in enumerate(levels):
        data = level_data[level]
        gts = data['ground_truths']
        recos = data['reconstructions']
        scores = data.get('scores', [None] * n_samples)

        gt_row = li * 2
        reco_row = li * 2 + 1

        for j in range(n_samples):
            ax_gt = axes[gt_row, j]
            if j < len(gts):
                ax_gt.imshow(gts[j], cmap=cmap, norm=norm,
                             interpolation='nearest')
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            if j == 0:
                ax_gt.set_ylabel(f'L{level} GT', fontsize=10,
                                 fontweight='bold')

            ax_re = axes[reco_row, j]
            if j < len(recos):
                ax_re.imshow(recos[j], cmap=cmap, norm=norm,
                             interpolation='nearest')
            ax_re.set_xticks([])
            ax_re.set_yticks([])
            if j == 0:
                ax_re.set_ylabel(f'L{level} {method_name}', fontsize=10)
            if j < len(scores) and scores[j] is not None:
                ax_re.set_xlabel(f'SSIM={scores[j]:.4f}', fontsize=8)

    for j in range(n_samples):
        axes[0, j].set_xlabel(f'Sample {j + 1}', fontsize=10)
        axes[0, j].xaxis.set_label_position('top')

    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.15)
    _add_colorbar(fig, cmap, norm, rect=[0.91, 0.35, 0.015, 0.3])
    _save_or_show(fig, save_path, dpi)


def plot_cross_method_comparison(methods_data, save_path=None, dpi=150):
    """Plot cross-method comparison on shared levels.

    For each shared level, shows 1 GT row + 1 row per method, 3 sample columns.

    Args:
        methods_data: dict {method_name: {level_int: {
                      'ground_truths': [...], 'reconstructions': [...],
                      'scores': [...]}}}
        save_path: Path to save figure, or None for interactive display.
        dpi: Resolution of saved figure.
    """
    methods = list(methods_data.keys())
    if not methods:
        return

    shared_levels = sorted(set.intersection(
        *[set(methods_data[m].keys()) for m in methods]
    ))
    if not shared_levels:
        print('No shared levels found across methods, skipping cross-method plot.')
        return

    n_methods = len(methods)
    n_samples = 3
    rows_per_level = 1 + n_methods
    total_rows = len(shared_levels) * rows_per_level
    cmap, norm = _get_discrete_cmap()

    fig, axes = plt.subplots(
        total_rows, n_samples,
        figsize=(3 * n_samples, 2.2 * total_rows),
        squeeze=False,
    )

    for li, level in enumerate(shared_levels):
        base_row = li * rows_per_level
        # Use ground truths from the first method that has them
        gts = None
        for m in methods:
            gts = methods_data[m][level].get('ground_truths')
            if gts:
                break

        # GT row
        for j in range(n_samples):
            ax = axes[base_row, j]
            if gts and j < len(gts):
                ax.imshow(gts[j], cmap=cmap, norm=norm,
                          interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f'L{level} GT', fontsize=10, fontweight='bold')

        # Method rows
        for mi, method_name in enumerate(methods):
            row = base_row + 1 + mi
            data = methods_data[method_name].get(level, {})
            recos = data.get('reconstructions', [])
            scores = data.get('scores', [None] * n_samples)

            for j in range(n_samples):
                ax = axes[row, j]
                if j < len(recos) and recos[j] is not None:
                    ax.imshow(recos[j], cmap=cmap, norm=norm,
                              interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(f'L{level} {method_name}', fontsize=9)
                if j < len(scores) and scores[j] is not None:
                    ax.set_xlabel(f'{scores[j]:.4f}', fontsize=7)

    for j in range(n_samples):
        axes[0, j].set_xlabel(f'Sample {j + 1}', fontsize=10)
        axes[0, j].xaxis.set_label_position('top')

    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.15)
    _add_colorbar(fig, cmap, norm, rect=[0.91, 0.35, 0.015, 0.3])
    _save_or_show(fig, save_path, dpi)


def plot_scores_summary(scores_dict, save_path=None, dpi=150):
    """Plot scores as a grouped bar chart (sum of 3 samples per level).

    Args:
        scores_dict: {method_name: {level_int: [score1, score2, ...], ...}}
        save_path: Path to save figure, or None for interactive display.
        dpi: Resolution of saved figure.
    """
    methods = list(scores_dict.keys())
    all_levels = sorted(set(
        lvl for m in methods for lvl in scores_dict[m].keys()
    ))
    n_methods = len(methods)
    n_levels = len(all_levels)

    sum_scores = np.zeros((n_methods, n_levels))
    for i, method in enumerate(methods):
        for j, level in enumerate(all_levels):
            if level in scores_dict[method]:
                sum_scores[i, j] = np.sum(scores_dict[method][level])

    x = np.arange(n_levels)
    bar_width = 0.8 / n_methods
    colors = ['#4393C3', '#D6604D', '#8DA0CB', '#66C2A5']

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_levels), 5))
    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, sum_scores[i], bar_width,
                      label=method, color=colors[i % len(colors)],
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, sum_scores[i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Difficulty Level', fontsize=11)
    ax.set_ylabel('Score (sum of 3 samples)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Level {lvl}' for lvl in all_levels], fontsize=9)
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()
    _save_or_show(fig, save_path, dpi)
