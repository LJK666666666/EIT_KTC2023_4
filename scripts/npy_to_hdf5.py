"""
Convert existing .npy/.npz files into HDF5 format.

Supports:
  - Single-sample .npy files (e.g., gt_ztm_000000.npy with shape (256,256))
  - Batch .npy files (e.g., shape (N, 256, 256))
  - Batch .npz files with keys: gt, measurements, reco
  - Appending to existing .h5 files (resizes datasets along axis 0)

Usage:
    # Convert all data types for a level directory
    python scripts/npy_to_hdf5.py --input-dir dataset/level_1 --output dataset/level_1/data.h5

    # Convert only ground truth
    python scripts/npy_to_hdf5.py --input-dir dataset/level_1 --output data.h5 --data-type gt

    # Convert measurements only
    python scripts/npy_to_hdf5.py --input-dir dataset/level_1 --output data.h5 --data-type measurements

    # Specify dataset-dir for paths relative to project (for Colab)
    python scripts/npy_to_hdf5.py --input-dir /content/drive/MyDrive/dataset/level_1 --output /content/drive/MyDrive/dataset/level_1/data.h5

    # Merge .npz batch files (from generate_data.py multiprocess)
    python scripts/npy_to_hdf5.py --input-dir dataset/level_1/_batches --output dataset/level_1/data.h5 --npz
"""

import argparse
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

# (subdir, h5_key, file_glob, expected_single_ndim, dtype)
DATA_TYPE_MAP = {
    'gt': ('gt', 'gt', '*.npy', 2, 'float32'),
    'measurements': ('measurements', 'measurements', '*.npy', 1, 'float64'),
    'reco': ('gm_reco', 'reco', '*.npy', 3, 'float32'),
}


def load_npy_auto(file_path, expected_single_ndim):
    """Load .npy, return as (N, ...) batch array.

    Auto-detects single-sample vs batch:
      - If ndim == expected_single_ndim, treat as single sample → add batch dim
      - Otherwise, treat as batch (ndim == expected_single_ndim + 1)
    """
    arr = np.load(file_path)
    if arr.ndim == expected_single_ndim:
        arr = arr[np.newaxis, ...]
    return arr


def append_to_h5(h5_file, key, data, dtype):
    """Append data array to an HDF5 dataset, creating if needed."""
    if key in h5_file:
        ds = h5_file[key]
        old_len = ds.shape[0]
        ds.resize(old_len + data.shape[0], axis=0)
        ds[old_len:] = data.astype(dtype)
    else:
        item_shape = data.shape[1:]
        h5_file.create_dataset(
            key, data=data.astype(dtype),
            maxshape=(None, *item_shape),
            chunks=(1, *item_shape))


def convert_npy_dir(input_dir, h5_file, data_type):
    """Convert .npy files from a subdirectory into HDF5."""
    subdir, h5_key, file_glob, single_ndim, dtype = DATA_TYPE_MAP[data_type]
    npy_dir = os.path.join(input_dir, subdir)

    if not os.path.isdir(npy_dir):
        print(f'  Skipping {data_type}: {npy_dir} not found')
        return 0

    files = sorted(glob.glob(os.path.join(npy_dir, file_glob)))
    if not files:
        print(f'  Skipping {data_type}: no files in {npy_dir}')
        return 0

    count = 0
    for f in tqdm(files, desc=f'  {data_type}'):
        batch = load_npy_auto(f, single_ndim)
        append_to_h5(h5_file, h5_key, batch, dtype)
        count += batch.shape[0]

    return count


def convert_npz_dir(input_dir, h5_file):
    """Convert .npz batch files into HDF5 (keys: gt, measurements, reco)."""
    files = sorted(glob.glob(os.path.join(input_dir, '*.npz')))
    if not files:
        print(f'  No .npz files in {input_dir}')
        return 0

    # dtype mapping for npz keys
    dtype_map = {'gt': 'float32', 'measurements': 'float64', 'reco': 'float32'}
    count = 0

    for f in tqdm(files, desc='  npz batches'):
        data = np.load(f)
        for key in data.files:
            dtype = dtype_map.get(key, 'float32')
            append_to_h5(h5_file, key, data[key], dtype)
        # Count from first key
        first_key = data.files[0]
        count += data[first_key].shape[0]

    return count


def main():
    parser = argparse.ArgumentParser(
        description='Convert .npy/.npz files to HDF5 format')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing data '
                             '(e.g., dataset/level_1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .h5 file path '
                             '(default: {input-dir}/data.h5)')
    parser.add_argument('--data-type', type=str, default='all',
                        choices=['gt', 'measurements', 'reco', 'all'],
                        help='Data type(s) to convert (default: all)')
    parser.add_argument('--npz', action='store_true',
                        help='Input directory contains .npz batch files '
                             '(from generate_data.py multiprocess)')
    args = parser.parse_args()

    output = args.output or os.path.join(args.input_dir, 'data.h5')
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    import h5py

    mode = 'a' if os.path.exists(output) else 'w'
    existing_info = ''
    if mode == 'a':
        with h5py.File(output, 'r') as f:
            existing_info = ', '.join(
                f'{k}: {v.shape}' for k, v in f.items())
        if existing_info:
            existing_info = f' (existing: {existing_info})'

    print(f'Output: {output}{existing_info}')
    print(f'Input:  {args.input_dir}')

    with h5py.File(output, mode) as h5f:
        if args.npz:
            count = convert_npz_dir(args.input_dir, h5f)
            print(f'Merged {count} samples from .npz files')
        else:
            types = list(DATA_TYPE_MAP.keys()) \
                if args.data_type == 'all' else [args.data_type]
            for dt in types:
                count = convert_npy_dir(args.input_dir, h5f, dt)
                if count > 0:
                    print(f'  {dt}: {count} samples written')

        # Print final summary
        print(f'\nFinal HDF5 contents:')
        for k, v in h5f.items():
            size_mb = v.dtype.itemsize
            for s in v.shape:
                size_mb *= s
            size_mb /= 1024 * 1024
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}, '
                  f'size={size_mb:.1f}MB')

    print('Done.')


if __name__ == '__main__':
    main()
