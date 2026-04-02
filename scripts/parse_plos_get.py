"""Parse Draeger .get files from the PLOS One subject dataset."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def create_result_dir(base_tag: str) -> str:
    base = Path('results')
    base.mkdir(exist_ok=True)
    idx = 1
    while (base / f'{base_tag}_{idx}').exists():
        idx += 1
    out_dir = base / f'{base_tag}_{idx}'
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def find_get_files(root: str):
    return sorted(str(p) for p in Path(root).rglob('*.get'))


def load_get_raw(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 256 != 0:
        raise ValueError(
            f'{path} contains {data.size} float32 values, not divisible by 256.')
    frames = data.size // 256
    return data.reshape(frames, 256).T.copy()


def reorder_to_208(voltages: np.ndarray) -> np.ndarray:
    chunks = []
    for i in range(16):
        temp = voltages[i * 16:(i + 1) * 16, :]
        if i == 15:
            temp = np.delete(temp, [0, 13, 14], axis=0)
        else:
            temp = np.concatenate([temp, temp[:i + 2, :]], axis=0)
            temp = np.delete(temp, np.arange(0, i + 2), axis=0)
            temp = np.delete(temp, [13, 14, 15], axis=0)
        chunks.append(temp)
    return np.concatenate(chunks, axis=0)


def dominant_signal(volt_reorder: np.ndarray) -> np.ndarray:
    x = volt_reorder.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    sig = vt[0]
    sig = sig - sig.mean()
    denom = np.std(sig)
    return sig / denom if denom > 0 else sig


def plot_overview(raw256: np.ndarray,
                  reordered208: np.ndarray,
                  signal: np.ndarray,
                  save_path: str,
                  max_frames: int = 600):
    frames = min(max_frames, raw256.shape[1])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    im0 = axes[0].imshow(raw256[:, :frames], aspect='auto', cmap='coolwarm')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Raw channel')
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02)

    im1 = axes[1].imshow(reordered208[:, :frames], aspect='auto', cmap='coolwarm')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Reordered channel')
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

    axes[2].plot(signal[:min(signal.size, 3000)], linewidth=1.0)
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('PC1 signal')

    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def relative_subject_path(path: str, root: str) -> str:
    return str(Path(path).relative_to(Path(root))).replace('\\', '/')


def main():
    parser = argparse.ArgumentParser(description='Parse PLOS One .get EIT data')
    parser.add_argument('--data-root', default='Subjects Data/Plos One Data')
    parser.add_argument('--export-npz', action='store_true',
                        help='Export parsed 256xT and 208xT arrays to NPZ')
    parser.add_argument('--max-plots', type=int, default=3,
                        help='Maximum number of overview plots to save')
    parser.add_argument('--focus-pattern', default='Subject A/1.Day',
                        help='Prefer plotting files whose relative path contains this pattern')
    args = parser.parse_args()

    files = find_get_files(args.data_root)
    if not files:
        raise FileNotFoundError(f'No .get files found under {args.data_root}')

    out_dir = create_result_dir('plos_get_analysis')
    summaries = []
    plot_candidates = []

    for path in tqdm(files, desc='Parsing GET files'):
        raw256 = load_get_raw(path)
        reordered208 = reorder_to_208(raw256)
        signal = dominant_signal(reordered208)

        rel = relative_subject_path(path, args.data_root)
        item = {
            'file': rel,
            'frames': int(raw256.shape[1]),
            'raw_shape': [int(v) for v in raw256.shape],
            'reordered_shape': [int(v) for v in reordered208.shape],
            'raw_mean_abs': float(np.mean(np.abs(raw256))),
            'reordered_mean_abs': float(np.mean(np.abs(reordered208))),
            'signal_std': float(np.std(signal)),
        }
        summaries.append(item)

        if args.export_npz:
            safe_name = rel.replace('/', '__').replace('.get', '.npz')
            np.savez_compressed(
                os.path.join(out_dir, safe_name),
                raw256=raw256.astype(np.float32),
                reordered208=reordered208.astype(np.float32),
                signal=signal.astype(np.float32),
            )

        priority = 0 if args.focus_pattern and args.focus_pattern in rel else 1
        plot_candidates.append((priority, rel, path, raw256, reordered208, signal))

    plot_candidates.sort(key=lambda x: (x[0], x[1]))
    selected = plot_candidates[:max(0, args.max_plots)]
    plot_records = []
    for _, rel, _, raw256, reordered208, signal in selected:
        fig_name = rel.replace('/', '__').replace('.get', '.png')
        fig_path = os.path.join(out_dir, fig_name)
        plot_overview(raw256, reordered208, signal, fig_path)
        plot_records.append({'file': rel, 'figure': fig_name})

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'data_root': args.data_root,
                'num_files': len(files),
                'files': summaries,
                'plots': plot_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f'Output directory: {out_dir}')
    print(f'Parsed files: {len(files)}')


if __name__ == '__main__':
    main()
