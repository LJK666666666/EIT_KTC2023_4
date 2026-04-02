"""
Training dataset classes for KTC2023 EIT reconstruction.

Provides:
  - FCUNetTrainingData: Raw measurements + GT for FCUNet direct mapping
  - SimData: 5-channel initial reconstructions + GT (file-based)
  - MmapDataset: Memory-mapped version for large datasets
  - FCUNetHDF5Dataset: HDF5-backed version of FCUNetTrainingData
  - SimHDF5Dataset: HDF5-backed version of SimData

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


# ---------------------------------------------------------------------------
# HDF5-backed dataset classes
# ---------------------------------------------------------------------------

class FCUNetHDF5Dataset(Dataset):
    """HDF5-backed dataset for FCUNet training.

    Same interface as FCUNetTrainingData: returns (measurements, gt_onehot).

    HDF5 datasets expected:
        'gt': (N, 256, 256) float32
        'measurements': (N, 2356) float64

    Args:
        h5_path: Path to .h5 file.
        Uref: Reference voltage measurements (1D ndarray).
        InvLn: Noise precision matrix (sparse).
        indices: Optional subset of sample indices to use.
        augment_noise: If True, add random noise as data augmentation.
    """

    def __init__(self, h5_path, Uref, InvLn,
                 indices=None, augment_noise=True):
        import h5py

        self.h5_path = h5_path
        self.Uref = Uref
        self.InvLn = InvLn
        self.augment_noise = augment_noise

        # Read total length (open briefly)
        with h5py.File(h5_path, 'r') as f:
            total_len = f['gt'].shape[0]

        self.indices = list(indices) if indices is not None \
            else list(range(total_len))
        self._h5_file = None  # Lazy-opened per DataLoader worker
        print(f'FCUNetHDF5Dataset: {len(self)} samples from {h5_path}')

    def _open_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5 = self._open_h5()
        real_idx = self.indices[idx]

        gt_np = h5['gt'][real_idx]                  # (256, 256)
        measurements = h5['measurements'][real_idx]  # (2356,)

        measurements = np.asarray(measurements).flatten()
        if self.augment_noise:
            noise = np.asarray(
                self.InvLn * np.random.randn(
                    self.Uref.shape[0], 1)).flatten()
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

    def __del__(self):
        try:
            if getattr(self, '_h5_file', None) is not None:
                self._h5_file.close()
        except Exception:
            pass


class DCTHDF5Dataset(Dataset):
    """HDF5-backed dataset for DCT predictor training.

    Returns (measurements, gt_indices).
    """

    def __init__(self, h5_path, Uref, InvLn, indices=None, augment_noise=True):
        import h5py

        self.h5_path = h5_path
        self.Uref = Uref
        self.InvLn = InvLn
        self.augment_noise = augment_noise

        with h5py.File(h5_path, 'r') as f:
            total_len = f['gt'].shape[0]

        self.indices = list(indices) if indices is not None \
            else list(range(total_len))
        self._h5_file = None
        print(f'DCTHDF5Dataset: {len(self)} samples from {h5_path}')

    def _open_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5 = self._open_h5()
        real_idx = self.indices[idx]

        gt_np = h5['gt'][real_idx]
        measurements = h5['measurements'][real_idx]
        measurements = np.asarray(measurements).reshape(-1)
        if self.augment_noise:
            noise = np.asarray(
                self.InvLn * np.random.randn(self.Uref.shape[0], 1)).reshape(-1)
            measurements = measurements - (self.Uref + noise)
        else:
            measurements = measurements - self.Uref

        return (
            torch.from_numpy(measurements.astype(np.float32)),
            torch.from_numpy(gt_np.astype(np.int64)),
        )

    def __del__(self):
        try:
            if getattr(self, '_h5_file', None) is not None:
                self._h5_file.close()
        except Exception:
            pass


class SimHDF5Dataset(Dataset):
    """HDF5-backed dataset for PostP/CondD.

    Same interface as SimData: returns (reco, gt, level).

    HDF5 datasets expected:
        'gt': (N, 256, 256) float32
        'reco': (N, 5, 256, 256) float32

    Args:
        h5_path: Path to .h5 file.
        level: Difficulty level (1-7).
        indices: Optional subset of sample indices to use.
    """

    def __init__(self, h5_path, level, indices=None):
        import h5py

        self.h5_path = h5_path
        self.level = level

        with h5py.File(h5_path, 'r') as f:
            total_len = f['gt'].shape[0]

        self.indices = list(indices) if indices is not None \
            else list(range(total_len))
        self._h5_file = None
        print(f'SimHDF5Dataset: {len(self)} samples for level {self.level}')

    def _open_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5 = self._open_h5()
        real_idx = self.indices[idx]

        gt_np = h5['gt'][real_idx]    # (256, 256)
        reco = h5['reco'][real_idx]   # (5, 256, 256)

        gt = torch.from_numpy(gt_np.astype(np.float32))
        return (torch.from_numpy(reco.astype(np.float32)),
                gt,
                torch.tensor(self.level, dtype=torch.float32))

    def __del__(self):
        try:
            if getattr(self, '_h5_file', None) is not None:
                self._h5_file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SAE dataset classes
# ---------------------------------------------------------------------------

class GTHDF5Dataset(Dataset):
    """HDF5-backed dataset for SAE training: GT images only.

    Returns (gt_onehot, gt_indices) per sample.
    No measurements needed for autoencoder training.

    Args:
        h5_path: Path to .h5 file.
        indices: Optional subset of sample indices.
    """

    def __init__(self, h5_path, indices=None):
        import h5py

        self.h5_path = h5_path

        with h5py.File(h5_path, 'r') as f:
            total_len = f['gt'].shape[0]

        self.indices = list(indices) if indices is not None \
            else list(range(total_len))
        self._h5_file = None
        print(f'GTHDF5Dataset: {len(self)} samples from {h5_path}')

    def _open_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5 = self._open_h5()
        real_idx = self.indices[idx]

        gt_np = h5['gt'][real_idx]  # (256, 256) uint8 with values 0, 1, 2

        # One-hot encode: (3, 256, 256)
        gt_onehot = np.zeros((3, 256, 256), dtype=np.float32)
        gt_onehot[0] = (gt_np == 0)
        gt_onehot[1] = (gt_np == 1)
        gt_onehot[2] = (gt_np == 2)

        # Class indices for CrossEntropy: (256, 256) long
        gt_indices = gt_np.astype(np.int64)

        return (torch.from_numpy(gt_onehot),
                torch.from_numpy(gt_indices))

    def __del__(self):
        try:
            if getattr(self, '_h5_file', None) is not None:
                self._h5_file.close()
        except Exception:
            pass


class SAEPredictorHDF5Dataset(Dataset):
    """HDF5-backed dataset for SAE predictor training.

    Loads measurements from data.h5 and pre-computed latent codes from
    latent_codes.h5. Supports vincl masking, noise augmentation, and
    rotation augmentation (electrode circular shift).

    Args:
        h5_path: Path to data.h5 (measurements + gt).
        latent_h5_path: Path to latent_codes.h5 from SAE Phase 2.
        Uref: Reference voltage measurements (1D ndarray).
        InvLn: Noise precision matrix (sparse).
        indices: Subset of sample indices (must match latent_codes indices).
        augment_noise: If True, add random noise to measurements.
        augment_rotation: If True, apply circular shift augmentation.
    """

    def __init__(self, h5_path, latent_h5_path, Uref, InvLn,
                 indices=None, augment_noise=True, augment_rotation=True,
                 slot_class_maps=None):
        import h5py

        self.h5_path = h5_path
        self.Uref = Uref
        self.InvLn = InvLn
        self.augment_noise = augment_noise
        self.augment_rotation = augment_rotation
        self.slot_class_maps = slot_class_maps

        # Load latent codes entirely into memory (small: N×65 floats)
        with h5py.File(latent_h5_path, 'r') as f:
            self._all_codes = f['codes'][:]           # (N, 65)
            self._code_indices = f['indices'][:]      # (N,)

        # Build index mapping: data.h5 sample_id → codes array position
        self._code_map = {int(idx): pos
                          for pos, idx in enumerate(self._code_indices)}

        # Read data.h5 total length
        with h5py.File(h5_path, 'r') as f:
            total_len = f['gt'].shape[0]

        if indices is not None:
            self.indices = [i for i in indices if i in self._code_map]
        else:
            self.indices = [i for i in range(total_len) if i in self._code_map]

        self._h5_file = None
        print(f'SAEPredictorHDF5Dataset: {len(self)} samples '
              f'(data: {h5_path}, codes: {latent_h5_path})')

    def _open_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5 = self._open_h5()
        real_idx = self.indices[idx]

        # Load measurements
        measurements = np.asarray(h5['measurements'][real_idx]).flatten()
        if self.augment_noise:
            noise = np.asarray(
                self.InvLn * np.random.randn(
                    self.Uref.shape[0], 1)).flatten()
            measurements = measurements - (self.Uref + noise)
        else:
            measurements = measurements - self.Uref

        # Load target latent code
        code_pos = self._code_map[real_idx]
        target_z = self._all_codes[code_pos].copy()  # (65,)

        # Rotation augmentation: circular shift of electrode channels
        if self.augment_rotation:
            k = np.random.randint(0, 32)
            if k > 0:
                # Reshape to (31 electrodes, 76 patterns), shift, flatten
                meas_2d = measurements.reshape(31, 76)
                meas_2d = np.roll(meas_2d, shift=k, axis=0)
                measurements = meas_2d.flatten()

                # Update angle_xy: rotate by k × (2π/32)
                delta = k * (2 * np.pi / 32)
                cos_old = target_z[63]
                sin_old = target_z[64]
                target_z[63] = (cos_old * np.cos(delta)
                                - sin_old * np.sin(delta))
                target_z[64] = (sin_old * np.cos(delta)
                                + cos_old * np.sin(delta))

        return (torch.from_numpy(measurements.astype(np.float32)),
                torch.from_numpy(target_z.astype(np.float32)))


# ---------------------------------------------------------------------------
# VQ-SAE dataset classes
# ---------------------------------------------------------------------------


class VQGTHDF5Dataset(GTHDF5Dataset):
    """HDF5-backed dataset for VQ SAE training.

    Same return format as GTHDF5Dataset:
      - gt_onehot: (3, 256, 256)
      - gt_indices: (256, 256)
    """

    def __init__(self, h5_path, indices=None):
        super().__init__(h5_path, indices=indices)
        print(f'VQGTHDF5Dataset: {len(self)} samples from {h5_path}')


class VQSAEPredictorHDF5Dataset(Dataset):
    """HDF5-backed dataset for VQ SAE predictor training.

    Returns:
      - measurements: (2356,)
      - target_indices: (num_slots,)
      - target_angle: (2,)
    """

    def __init__(self, h5_path, latent_h5_path, Uref, InvLn,
                 indices=None, augment_noise=True, augment_rotation=True,
                 slot_class_maps=None):
        import h5py

        self.h5_path = h5_path
        self.Uref = np.asarray(Uref).reshape(-1)
        self.InvLn = InvLn
        self.augment_noise = augment_noise
        self.augment_rotation = augment_rotation
        self.slot_class_maps = slot_class_maps

        with h5py.File(latent_h5_path, 'r') as f:
            self._all_slot_indices = f['indices'][:]        # (N, num_slots)
            self._all_angle_xy = f['angle_xy'][:]           # (N, 2)
            self._code_indices = f['sample_indices'][:]     # (N,)

        self._code_map = {int(idx): pos
                          for pos, idx in enumerate(self._code_indices)}

        with h5py.File(h5_path, 'r') as f:
            total_len = f['gt'].shape[0]

        if indices is not None:
            self.indices = [i for i in indices if i in self._code_map]
        else:
            self.indices = [i for i in range(total_len) if i in self._code_map]

        self._h5_file = None
        print(f'VQSAEPredictorHDF5Dataset: {len(self)} samples '
              f'(data: {h5_path}, codes: {latent_h5_path})')

    def _open_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5 = self._open_h5()
        real_idx = self.indices[idx]

        measurements = np.asarray(h5['measurements'][real_idx]).flatten()
        if self.augment_noise:
            noise = np.asarray(
                self.InvLn * np.random.randn(self.Uref.shape[0], 1)).flatten()
            measurements = measurements - (self.Uref + noise)
        else:
            measurements = measurements - self.Uref

        gt_indices = np.asarray(h5['gt'][real_idx]).astype(np.int64)
        code_pos = self._code_map[real_idx]
        target_indices = self._all_slot_indices[code_pos].copy()
        target_angle = self._all_angle_xy[code_pos].copy()
        rot_steps = 0

        if self.slot_class_maps is not None:
            target_indices = np.asarray([
                self.slot_class_maps[slot_idx][int(code)]
                for slot_idx, code in enumerate(target_indices)
            ], dtype=np.int64)

        if self.augment_rotation:
            k = np.random.randint(0, 32)
            if k > 0:
                rot_steps = int(k)
                meas_2d = measurements.reshape(31, 76)
                meas_2d = np.roll(meas_2d, shift=k, axis=0)
                measurements = meas_2d.flatten()

                delta = k * (2 * np.pi / 32)
                cos_old = target_angle[0]
                sin_old = target_angle[1]
                target_angle[0] = (cos_old * np.cos(delta)
                                   - sin_old * np.sin(delta))
                target_angle[1] = (sin_old * np.cos(delta)
                                   + cos_old * np.sin(delta))

        return (torch.from_numpy(measurements.astype(np.float32)),
                torch.from_numpy(target_indices.astype(np.int64)),
                torch.from_numpy(target_angle.astype(np.float32)),
                torch.from_numpy(gt_indices),
                torch.tensor(rot_steps, dtype=torch.long))

    def __del__(self):
        try:
            if getattr(self, '_h5_file', None) is not None:
                self._h5_file.close()
        except Exception:
            pass
