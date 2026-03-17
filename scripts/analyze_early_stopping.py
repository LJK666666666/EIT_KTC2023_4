"""
Analyze training logs from data scaling experiments and report where
early stopping (patience=15) would have triggered.

Supports two metrics:
  - val_loss (lower is better, default) — matches BaseTrainer behaviour
  - score   (higher is better)

Handles logs that contain a pre-training "init" stage (entries with
stage="init" are excluded from the early-stopping simulation).

Usage:
    python scripts/analyze_early_stopping.py
    python scripts/analyze_early_stopping.py --patience 10
    python scripts/analyze_early_stopping.py --pattern "results/fcunet_scaling_n*"
    python scripts/analyze_early_stopping.py --metric score
"""

import argparse
import glob
import json
import os
import shutil
import sys


def analyze_log(log_path, patience=15, metric='val_loss'):
    """Analyze a training log and find the early stopping point.

    Parameters
    ----------
    log_path : str
        Path to training_log.json.
    patience : int
        Number of epochs without improvement before early stopping.
    metric : str
        'val_loss' (lower=better) or 'score' (higher=better).

    Returns
    -------
    dict with: total_epochs, best_epoch, best_value, stop_epoch,
    wasted_epochs, entries (full log list for truncation).
    """
    with open(log_path) as f:
        log = json.load(f)

    if not log:
        return None

    # Separate init-stage entries from main training
    init_entries = [e for e in log if e.get('stage') == 'init']
    main_entries = [e for e in log if e.get('stage') != 'init']

    if not main_entries:
        return None

    lower_is_better = (metric == 'val_loss')

    # Fallback key: if val_loss is requested but missing, use avg_loss
    def _get_metric(entry):
        val = entry.get(metric)
        if val is None and metric == 'val_loss':
            val = entry.get('avg_loss')
        return val

    best_value = None
    best_epoch = 0
    es_counter = 0
    stop_epoch = None

    for entry in main_entries:
        epoch = entry.get('epoch', 0)
        value = _get_metric(entry)

        if value is None:
            continue

        improved = False
        if best_value is None:
            improved = True
        elif lower_is_better and value < best_value:
            improved = True
        elif not lower_is_better and value > best_value:
            improved = True

        if improved:
            best_value = value
            best_epoch = epoch
            es_counter = 0
        else:
            es_counter += 1

        if stop_epoch is None and es_counter >= patience:
            stop_epoch = epoch

    total_epochs = main_entries[-1].get('epoch', len(main_entries))

    if stop_epoch is None:
        stop_epoch = total_epochs  # Never triggered

    return {
        'total_epochs': total_epochs,
        'best_epoch': best_epoch,
        'best_value': best_value,
        'stop_epoch': stop_epoch,
        'wasted_epochs': total_epochs - stop_epoch,
        'init_entries': init_entries,
        'main_entries': main_entries,
    }


def truncate_log(result, patience):
    """Return a truncated log list that ends at the early-stopping epoch."""
    stop_ep = result['stop_epoch']
    init = result['init_entries']
    main = [e for e in result['main_entries'] if e['epoch'] <= stop_ep]
    return init + main


def main():
    parser = argparse.ArgumentParser(
        description='Analyze early stopping on existing training logs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--pattern', type=str,
                        default='results/fcunet_scaling_n*',
                        help='Glob pattern for result directories')
    parser.add_argument('--metric', type=str, default='val_loss',
                        choices=['val_loss', 'score'],
                        help='Metric for early stopping (default: val_loss)')
    parser.add_argument('--truncate', action='store_true',
                        help='Backup .bak then write truncated log in-place')
    parser.add_argument('dirs', nargs='*',
                        help='Explicit result directories (overrides --pattern)')
    args = parser.parse_args()

    if args.dirs:
        dirs = args.dirs
    else:
        dirs = sorted(glob.glob(args.pattern))
    if not dirs:
        print(f'No directories matching: {args.pattern}')
        sys.exit(1)

    better = 'lower' if args.metric == 'val_loss' else 'higher'
    print(f'Early stopping analysis  metric={args.metric} ({better}=better)  '
          f'patience={args.patience}')
    print(f'{"Directory":<40} {"Total":>6} {"Best@":>6} '
          f'{"BestVal":>10} {"Stop@":>6} {"Wasted":>7}')
    print('-' * 80)

    for d in dirs:
        log_path = os.path.join(d, 'training_log.json')
        if not os.path.exists(log_path):
            continue

        result = analyze_log(log_path, patience=args.patience,
                             metric=args.metric)
        if result is None:
            continue

        name = os.path.basename(d)
        val_str = (f'{result["best_value"]:.6f}'
                   if result['best_value'] is not None else 'N/A')
        print(f'{name:<40} {result["total_epochs"]:>6} '
              f'{result["best_epoch"]:>6} {val_str:>10} '
              f'{result["stop_epoch"]:>6} {result["wasted_epochs"]:>7}')

        if args.truncate and result['wasted_epochs'] > 0:
            truncated = truncate_log(result, args.patience)
            bak_path = log_path + '.bak'
            shutil.copy2(log_path, bak_path)
            with open(log_path, 'w') as f:
                json.dump(truncated, f, indent=2)
            orig_len = len(result['init_entries']) + len(result['main_entries'])
            print(f'  -> Backup: {bak_path}')
            print(f'  -> Truncated: {orig_len} -> {len(truncated)} entries')

    print('-' * 80)


if __name__ == '__main__':
    main()
