"""
Generate pre-computed 5-channel linearised reconstructions of challenge images.

Computes LinearisedRecoFenics reconstructions for the 4 KTC2023 challenge
images (data1-4.mat) across all 7 difficulty levels. These are needed for
PostP/CondD validation during training.

Output:
    ChallengeReconstructions/level_{1-7}/reco{1-4}.npy  (each shape: 5, 256, 256)

Usage:
    python scripts/generate_val_reco.py
    python scripts/generate_val_reco.py --output-dir ChallengeReconstructions
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reconstruction.linearised_reco import LinearisedRecoFenics
from src.utils.measurement import create_vincl
from src.configs.condd_config import LEVEL_TO_ALPHAS


def generate_val_reco(ref_path, data_dir, output_dir, reco_base_path,
                      num_images=4):
    """Generate 5-channel reconstructions for challenge images.

    Args:
        ref_path: Path to ref.mat (reference measurements).
        data_dir: Directory containing data{1-4}.mat.
        output_dir: Output directory for reconstructions.
        reco_base_path: base_path for LinearisedRecoFenics (Jacobian data).
        num_images: Number of challenge images (default 4).
    """
    # Load reference data
    y_ref = loadmat(ref_path)
    Uelref = y_ref['Uelref']
    Mpat = y_ref['Mpat']
    Injref = y_ref['Injref']
    B = Mpat.T

    # Load all challenge measurements
    challenge_data = []
    for i in range(1, num_images + 1):
        data_path = os.path.join(data_dir, f'data{i}.mat')
        Uel = np.array(loadmat(data_path)['Uel'])
        challenge_data.append(Uel)
        print(f'Loaded data{i}.mat: shape {Uel.shape}')

    total = 7 * num_images
    pbar = tqdm(total=total, desc='Generating reconstructions')

    for level in range(1, 8):
        level_dir = os.path.join(output_dir, f'level_{level}')
        os.makedirs(level_dir, exist_ok=True)

        alphas = LEVEL_TO_ALPHAS[level]
        vincl_level = create_vincl(level, Injref)

        reconstructor = LinearisedRecoFenics(
            Uelref, B, vincl_level,
            mesh_name='sparse',
            base_path=reco_base_path,
        )

        for i, Uel in enumerate(challenge_data, start=1):
            t0 = time.time()

            delta_sigma_list = reconstructor.reconstruct_list(Uel, alphas)
            sigma_images = [
                reconstructor.interpolate_to_image(ds)
                for ds in delta_sigma_list
            ]
            sigma_reco = np.stack(sigma_images)  # (5, 256, 256)

            out_path = os.path.join(level_dir, f'reco{i}.npy')
            np.save(out_path, sigma_reco)

            dt = time.time() - t0
            pbar.set_postfix(level=level, image=i, time=f'{dt:.1f}s')
            pbar.update(1)

    pbar.close()
    print(f'Validation reconstructions saved to: {output_dir}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate challenge image reconstructions for validation')
    parser.add_argument(
        '--ref-path', type=str,
        default='KTC2023/Codes_Python/TrainingData/ref.mat',
        help='Path to reference data (ref.mat)')
    parser.add_argument(
        '--data-dir', type=str,
        default='KTC2023/Codes_Python/TrainingData',
        help='Directory containing data{1-4}.mat')
    parser.add_argument(
        '--output-dir', type=str,
        default='ChallengeReconstructions',
        help='Output directory')
    parser.add_argument(
        '--reco-base-path', type=str,
        default='KTC2023_SubmissionFiles/data',
        help='Base path for LinearisedRecoFenics (pre-computed Jacobian)')
    parser.add_argument(
        '--num-images', type=int, default=4,
        help='Number of challenge images')
    return parser.parse_args()


def main():
    args = parse_args()
    generate_val_reco(
        ref_path=args.ref_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reco_base_path=args.reco_base_path,
        num_images=args.num_images,
    )


if __name__ == '__main__':
    main()
