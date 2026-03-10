"""
Training dataset classes for KTC2023 EIT reconstruction.

Provides:
  - FCUNetTrainingData: Raw measurements + GT for FCUNet direct mapping
  - SimData: 5-channel initial reconstructions + GT (file-based)
  - MmapDataset: Memory-mapped version for large datasets

Adapted from: KTC2023_SubmissionFiles/ktc_training/src/dataset/SimDataset.py
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class FCUNetTrainingData(Dataset):
    """Dataset for FCUNet training: raw measurements + ground truth.

    Each sample returns:
      - measurements: (2356,) float tensor of voltage differences with noise
      - gt_onehot: (3, 256, 256) float tensor, one-hot encoded segmentation

    Args:
        Uref: Reference voltage measurements (ndarray).
        InvLn: Noise precision matrix for measurement noise augmentation.
        base_path: Path to dataset directory containing gt/ and measurements/.
        indices: If not None, only include files whose numeric index (from
            filename like gt_ztm_000042.npy -> 42) is in this list.
        augment_noise: If True (default), add random noise to reference
            measurements as data augmentation. Set False for deterministic
            evaluation (val/test).
    """

    @staticmethod
    def _extract_index(filename):
        """Extract numeric index from filename: gt_ztm_000042.npy -> 42."""
        stem = filename.rsplit('.', 1)[0]   # gt_ztm_000042
        return int(stem.split('_')[-1])     # 42

    def __init__(self, Uref, InvLn, base_path='dataset',
                 indices=None, augment_noise=True):
        self.base_path = base_path
        self.augment_noise = augment_noise

        # List and sort by numeric index for deterministic ordering
        all_files = [
            f for f in os.listdir(os.path.join(self.base_path, 'gt'))
            if f.endswith('.npy')
        ]
        all_files.sort(key=self._extract_index)

        # Filter by indices if specified
        if indices is not None:
            index_set = set(indices)
            all_files = [f for f in all_files
                         if self._extract_index(f) in index_set]

        self.file_list = all_files
        self.length = len(self.file_list)
        self.Uref = Uref
        self.InvLn = InvLn
        print(f'FCUNetTrainingData: {self.length} samples from {base_path}')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gt_name = self.file_list[idx]
        u_name = gt_name.replace('gt', 'u')

        gt_np = np.load(os.path.join(self.base_path, 'gt', gt_name))
        measurements = np.load(
            os.path.join(self.base_path, 'measurements', u_name))

        measurements = np.asarray(measurements).flatten()
        if self.augment_noise:
            # Add noise to reference (data augmentation)
            # Flatten sparse matmul result to 1D to avoid broadcasting
            noise = np.asarray(
                self.InvLn * np.random.randn(self.Uref.shape[0], 1)).flatten()
            measurements = measurements - (self.Uref + noise)
        else:
            measurements = measurements - self.Uref

        # One-hot encode GT
        gt = np.zeros((3, 256, 256), dtype=np.float32)
        gt[0, :, :] = (gt_np == 0)
        gt[1, :, :] = (gt_np == 1)
        gt[2, :, :] = (gt_np == 2)

        return (torch.from_numpy(measurements.astype(np.float32)),
                torch.from_numpy(gt))


class SimData(Dataset):
    """Dataset for PostP/CondD: 5-channel initial reconstructions + GT.

    File-based loading for moderate-sized datasets.

    Args:
        level: Difficulty level (1-7).
        base_path: Path to dataset dir. Data at base_path/level_{level}/.
    """

    def __init__(self, level, base_path='dataset'):
        self.level = level
        self.base_path = os.path.join(base_path, f'level_{self.level}')
        self.file_list = [
            f for f in os.listdir(os.path.join(self.base_path, 'gt'))
            if f.endswith('.npy')
        ]
        self.length = len(self.file_list)
        print(f'SimData: {self.length} samples for level {self.level}')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gt_name = self.file_list[idx]
        reco_name = gt_name.replace('gt', 'recos')

        gt_np = np.load(os.path.join(self.base_path, 'gt', gt_name))
        init_reco = np.load(
            os.path.join(self.base_path, 'gm_reco', reco_name))

        gt = torch.from_numpy(gt_np.astype(np.float32))

        return (torch.from_numpy(init_reco.astype(np.float32)),
                gt,
                torch.tensor(self.level, dtype=torch.float32))


class MmapDataset(Dataset):
    """Memory-mapped dataset for large training sets (PostP/CondD).

    Uses np.load with mmap_mode='r' to avoid loading full dataset into RAM.
    Expects pre-generated .npy files:
      - gt_level={level}_size={num_samples}.npy
      - recos_level={level}_size={num_samples}.npy

    Args:
        level: Difficulty level (1-7).
        num_samples: Number of samples in the mmap files.
        base_path: Path to directory containing mmap .npy files.
    """

    def __init__(self, level, num_samples, base_path='dataset'):
        super().__init__()
        self.level = level

        gt_file = os.path.join(
            base_path, f'gt_level={self.level}_size={num_samples}.npy')
        reco_file = os.path.join(
            base_path, f'recos_level={self.level}_size={num_samples}.npy')

        self.gts = np.load(gt_file, mmap_mode='r')
        self.recos = np.load(reco_file, mmap_mode='r')
        print(f'MmapDataset: {len(self)} samples for level {self.level}')

    def __getitem__(self, item):
        # Handle known corrupted sample
        if self.level == 5 and item == 10506:
            gt_np = np.copy(self.gts[0])
            reco = np.copy(self.recos[0])
        else:
            gt_np = np.copy(self.gts[item])
            reco = np.copy(self.recos[item])

        gt = torch.from_numpy(gt_np.astype(np.float32))

        return (torch.from_numpy(reco.astype(np.float32)),
                gt,
                torch.tensor(self.level, dtype=torch.float32))

    def __len__(self):
        return self.gts.shape[0]
