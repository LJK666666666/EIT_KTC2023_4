"""
Abstract base pipeline for KTC2023 EIT reconstruction methods.

Defines a common interface that all reconstruction pipelines (FCUNet,
Postprocessing, Conditional Diffusion, etc.) must implement. This ensures
consistent handling of model loading, reconstruction, and evaluation.

The interface is derived from the common patterns found in:
  - programs/ktc2023_fcunet/main.py
  - programs/ktc2023_postprocessing/main.py
  - programs/ktc2023_conditional_diffusion/main.py

Usage example:
    class MyPipeline(BasePipeline):
        def load_model(self, level):
            ...
        def reconstruct(self, Uel, ref_data, level):
            ...

    pipeline = MyPipeline(device='cuda')
    results = pipeline.evaluate_level(level=1, eval_loader=loader, scoring_fn=scoring_function)
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import os


class BasePipeline(ABC):
    """Abstract base class for EIT reconstruction pipelines.

    All concrete pipeline implementations must provide:
      - load_model(level): Load/initialize model weights for a given difficulty level.
      - reconstruct(Uel, ref_data, level): Produce a 256x256 segmentation from measurements.

    Attributes:
        device: PyTorch device string ('cuda' or 'cpu').
        weights_base_dir: Base directory where model weights are stored.
        model: The loaded model (set by subclass in load_model).
    """

    def __init__(self, device='cuda', weights_base_dir='KTC2023_SubmissionFiles'):
        """Initialize the pipeline.

        Args:
            device: Desired compute device. Falls back to 'cpu' if CUDA is
                not available.
            weights_base_dir: Relative path to the directory containing
                pre-trained model weights.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.weights_base_dir = weights_base_dir
        self.model = None

    @abstractmethod
    def load_model(self, level: int) -> None:
        """Load model weights for the given difficulty level.

        This method should set self.model to the loaded and ready-to-use model.
        The model should be in eval mode and moved to self.device.

        Args:
            level: Difficulty level (integer, typically 1-7).
        """
        ...

    @abstractmethod
    def reconstruct(self, Uel: np.ndarray, ref_data: dict, level: int) -> np.ndarray:
        """Reconstruct a segmentation map from voltage measurements.

        This is the core method that each pipeline must implement. It takes
        raw voltage measurements and reference data, and produces a 256x256
        pixel segmentation with integer values {0, 1, 2}.

        Args:
            Uel: Voltage measurements array (shape depends on the measurement
                configuration, typically (Nel-1, n_patterns) or flattened).
            ref_data: Dictionary containing reference data with keys:
                - 'Injref': Injection reference pattern (Nel x n_patterns).
                - 'Uelref': Reference voltage measurements (same shape as Uel).
                - 'Mpat': Measurement pattern matrix.
            level: Difficulty level (integer, typically 1-7). Higher levels
                have more missing electrode data.

        Returns:
            256x256 numpy array with integer values in {0, 1, 2}, where:
                0 = background
                1 = resistive inclusion
                2 = conductive inclusion
        """
        ...

    def evaluate_level(self, level, eval_loader, scoring_fn):
        """Run reconstruction and scoring on all samples for a given level.

        Loads the model for the specified level, then iterates over all
        measurement samples, reconstructs each one, and computes the score
        against the ground truth.

        Args:
            level: Difficulty level (integer, typically 1-7).
            eval_loader: Data loader object that provides:
                - load_reference(level) -> dict with keys 'Injref', 'Uelref', 'Mpat'
                - load_measurements(level) -> list of Uel arrays
                - load_ground_truths(level) -> list of 256x256 ground truth arrays
            scoring_fn: Callable(groundtruth, reconstruction) -> float.
                Either scoring_function or FastScoringFunction from
                src.evaluation.scoring.

        Returns:
            Dictionary with keys:
                - 'scores': List of float scores, one per sample.
                - 'reconstructions': List of 256x256 numpy arrays.
                - 'ground_truths': List of 256x256 ground truth arrays.
                - 'level': The difficulty level (int).
                - 'mean_score': Mean score across all samples (float).
        """
        self.load_model(level)
        ref_data = eval_loader.load_reference(level)
        measurements = eval_loader.load_measurements(level)
        ground_truths = eval_loader.load_ground_truths(level)

        reconstructions = []
        scores = []
        for i, Uel in enumerate(measurements):
            reco = self.reconstruct(Uel, ref_data, level)
            reconstructions.append(reco)
            if i < len(ground_truths):
                score = scoring_fn(ground_truths[i], reco)
                scores.append(score)

        return {
            'scores': scores,
            'reconstructions': reconstructions,
            'ground_truths': ground_truths,
            'level': level,
            'mean_score': float(np.mean(scores)) if scores else 0.0,
        }

    def evaluate_all_levels(self, levels, eval_loader, scoring_fn):
        """Run evaluation across multiple difficulty levels.

        Args:
            levels: List of difficulty levels to evaluate (e.g., [1, 2, ..., 7]).
            eval_loader: Data loader object (see evaluate_level for interface).
            scoring_fn: Scoring function (see evaluate_level).

        Returns:
            Dictionary mapping level -> result dict from evaluate_level.
        """
        results = {}
        for level in levels:
            print(f"Evaluating level {level}...")
            results[level] = self.evaluate_level(level, eval_loader, scoring_fn)
            mean_score = results[level]['mean_score']
            print(f"  Level {level} mean score: {mean_score:.4f}")
        return results

    @staticmethod
    def create_vincl(level, Injref, Nel=32):
        """Create the voltage inclusion matrix for a given difficulty level.

        Removes measurements according to the difficulty level by zeroing out
        rows and columns corresponding to removed electrodes.

        This is a common utility used by FCUNet and other methods that need
        to handle missing electrode data.

        Args:
            level: Difficulty level (1-7). Level 1 removes no electrodes,
                higher levels remove more.
            Injref: Injection reference pattern matrix (Nel x n_patterns).
            Nel: Number of electrodes (default 32).

        Returns:
            Boolean array of shape ((Nel-1), 76) indicating which voltage
            measurements to include.
        """
        vincl_level = np.ones(((Nel - 1), 76), dtype=bool)
        rmind = np.arange(0, 2 * (level - 1), 1)

        for ii in range(0, 75):
            for jj in rmind:
                if Injref[jj, ii]:
                    vincl_level[:, ii] = 0
                vincl_level[jj, :] = 0

        return vincl_level
