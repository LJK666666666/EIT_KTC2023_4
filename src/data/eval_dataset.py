"""
Evaluation data loader for KTC2023 EIT reconstruction challenge.

Handles loading of:
- Reference measurements (ref.mat) per level
- Measurement data (data1.mat, data2.mat, data3.mat) per level
- Ground truth segmentation maps per level
"""

import os
import numpy as np
import scipy.io as sio


class EvaluationDataLoader:
    """Loader for KTC2023 evaluation datasets and ground truths.

    Directory structure expected:
        eval_data_dir/
            level1/
                ref.mat        (contains Injref, Uelref, Mpat)
                data1.mat      (contains Uel)
                data2.mat      (contains Uel)
                data3.mat      (contains Uel)
            level2/
                ...
            ...
            level7/
                ...
        gt_dir/
            level_1/
                1_true.mat     (contains truth, 256x256)
                2_true.mat
                3_true.mat
            level_2/
                ...
            ...
            level_7/
                ...

    Note the naming difference: evaluation datasets use "level1" (no underscore),
    while ground truths use "level_1" (with underscore).
    """

    def __init__(self,
                 eval_data_dir='KTC2023/EvaluationData/evaluation_datasets',
                 gt_dir='KTC2023/EvaluationData/GroundTruths'):
        """Initialize the evaluation data loader.

        Args:
            eval_data_dir: Relative path to the evaluation datasets directory.
            gt_dir: Relative path to the ground truths directory.
        """
        self.eval_data_dir = eval_data_dir
        self.gt_dir = gt_dir

    def _get_level_dir(self, level):
        """Get evaluation data directory path for a given level.

        Args:
            level: Difficulty level (1-7).

        Returns:
            Path string to the level directory (e.g. 'eval_data_dir/level1').
        """
        return os.path.join(self.eval_data_dir, f'level{level}')

    def _get_gt_level_dir(self, level):
        """Get ground truth directory path for a given level.

        Args:
            level: Difficulty level (1-7).

        Returns:
            Path string to the ground truth level directory (e.g. 'gt_dir/level_1').
        """
        return os.path.join(self.gt_dir, f'level_{level}')

    def load_reference(self, level):
        """Load reference measurement data (ref.mat) for a given level.

        Args:
            level: Difficulty level (1-7).

        Returns:
            dict with keys:
                'Injref': Injection reference pattern matrix
                'Uelref': Reference electrode voltage measurements
                'Mpat': Measurement pattern matrix
        """
        ref_path = os.path.join(self._get_level_dir(level), 'ref.mat')
        y_ref = sio.loadmat(ref_path)
        return {
            'Injref': y_ref['Injref'],
            'Uelref': y_ref['Uelref'],
            'Mpat': y_ref['Mpat'],
        }

    def load_measurements(self, level):
        """Load measurement data (data1.mat, data2.mat, data3.mat) for a given level.

        Args:
            level: Difficulty level (1-7).

        Returns:
            List of 3 Uel arrays, one per measurement dataset.
            Each Uel is a numpy array of electrode voltage measurements.
        """
        level_dir = self._get_level_dir(level)
        uel_list = []
        for i in range(1, 4):
            data_path = os.path.join(level_dir, f'data{i}.mat')
            uel = np.array(sio.loadmat(data_path)['Uel'])
            uel_list.append(uel)
        return uel_list

    def load_ground_truths(self, level):
        """Load ground truth segmentation maps for a given level.

        Each ground truth is a 256x256 array with values in {0, 1, 2}:
            0: background
            1: resistive inclusion (lower conductivity)
            2: conductive inclusion (higher conductivity)

        Args:
            level: Difficulty level (1-7).

        Returns:
            List of 3 numpy arrays, each 256x256, one per ground truth.
        """
        gt_level_dir = self._get_gt_level_dir(level)
        gt_list = []
        for i in range(1, 4):
            gt_path = os.path.join(gt_level_dir, f'{i}_true.mat')
            truth = np.array(sio.loadmat(gt_path)['truth'])
            gt_list.append(truth)
        return gt_list

    def load_all_for_level(self, level):
        """Load all data (reference, measurements, ground truths) for a given level.

        Args:
            level: Difficulty level (1-7).

        Returns:
            Tuple of (ref_dict, uel_list, gt_list):
                ref_dict: dict with 'Injref', 'Uelref', 'Mpat'
                uel_list: list of 3 Uel arrays
                gt_list: list of 3 ground truth 256x256 arrays
        """
        ref = self.load_reference(level)
        measurements = self.load_measurements(level)
        ground_truths = self.load_ground_truths(level)
        return ref, measurements, ground_truths
