"""
Measurement inclusion mask utilities for KTC2023 EIT reconstruction.

The vincl (voltage inclusion) mask determines which measurements to include
based on the difficulty level. Higher levels remove more electrode data,
making the reconstruction problem harder.
"""

import numpy as np


def create_vincl(level, Injref, Nel=32):
    """Create measurement inclusion mask for a given difficulty level.

    The mask removes measurements associated with electrodes that are
    "dropped out" according to the difficulty level. For level L, the
    first 2*(L-1) electrodes are removed.

    Args:
        level: Difficulty level (1-7). Level 1 keeps all electrodes,
               level 7 removes the most.
        Injref: Injection reference pattern matrix, shape (Nel-1, 76) or similar.
                Encodes which electrodes are used for each injection pattern.
        Nel: Number of electrodes (default 32).

    Returns:
        vincl: boolean array of shape (Nel-1, 76) indicating which
               measurements to include.
    """
    vincl_level = np.ones(((Nel - 1), 76), dtype=bool)
    rmind = np.arange(0, 2 * (level - 1), 1)  # electrodes whose data is removed

    # remove measurements according to the difficulty level
    for ii in range(0, 75):
        for jj in rmind:
            if Injref[jj, ii]:
                vincl_level[:, ii] = 0
            vincl_level[jj, :] = 0

    return vincl_level


def create_vincl_dict(Injref, levels=range(1, 8), Nel=32):
    """Pre-compute measurement inclusion masks for all specified levels.

    Args:
        Injref: Injection reference pattern matrix.
        levels: Iterable of difficulty levels to compute masks for.
                Defaults to range(1, 8) i.e. levels 1-7.
        Nel: Number of electrodes (default 32).

    Returns:
        vincl_dict: dict mapping level -> vincl boolean array of shape (Nel-1, 76).
    """
    vincl_dict = {}
    for level in levels:
        vincl_dict[level] = create_vincl(level, Injref, Nel=Nel)
    return vincl_dict
