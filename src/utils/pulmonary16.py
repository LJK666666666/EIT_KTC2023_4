"""Utilities for 16-electrode pulmonary EIT simulation.

This module adapts the existing 32-electrode KTC mesh and forward solver to a
16-electrode setting that is compatible with the PLOS One Draeger .get data:

  - 16 stimulation patterns
  - 16 adjacent differential measurements per pattern (raw 256)
  - reordered 208-channel format used by DCT-EIT/read_getData.m
"""

from copy import deepcopy

import numpy as np
from pyeit.eit.protocol import PyEITProtocol

from ..ktc_methods.KTCMeshing import ELEMENT, Mesh


def build_adjacent_skip3_inj(nel: int = 16) -> np.ndarray:
    """Build adjacent-pattern current injections with sink offset 3.

    This matches the Draeger/EIDORS example used in DCT-EIT/simple_3D.m.
    """
    inj = np.zeros((nel, nel), dtype=np.float64)
    for col in range(nel):
        inj[col, col] = 1.0
        inj[(col + 3) % nel, col] = -1.0
    return inj


def build_adjacent_cycle_mpat(nel: int = 16) -> np.ndarray:
    """Build adjacent differential measurement operator for all cyclic pairs.

    Returns an Nel x Nel matrix Mpat such that EITFEM outputs Nel adjacent
    voltage differences per stimulation pattern.
    """
    mpat = np.zeros((nel, nel), dtype=np.float64)
    for col in range(nel):
        mpat[col, col] = 1.0
        mpat[(col + 1) % nel, col] = -1.0
    return mpat


def reorder_raw256_to_208(raw256: np.ndarray) -> np.ndarray:
    """Reorder raw 16x16 adjacent measurements to the DCT-EIT 208-channel layout.

    Accepts:
      - (256,) flattened raw vector
      - (16, 16) matrix of [measurement_channel, pattern]
      - (256, T) or (16, 16, T)
    Returns:
      - (208,) / (208, T)
    """
    raw = np.asarray(raw256)
    if raw.ndim == 1:
        if raw.size != 256:
            raise ValueError(f'Expected 256 raw measurements, got {raw.size}')
        raw = raw.reshape(16, 16)
    elif raw.ndim == 2 and raw.shape == (16, 16):
        pass
    elif raw.ndim == 2 and raw.shape[0] == 256:
        raw = raw.reshape(16, 16, raw.shape[1])
    elif raw.ndim == 3 and raw.shape[:2] == (16, 16):
        pass
    else:
        raise ValueError(f'Unsupported raw256 shape: {raw.shape}')

    if raw.ndim == 2:
        chunks = []
        for i in range(16):
            temp = raw[:, i:i + 1]
            if i == 15:
                temp = np.delete(temp, [0, 13, 14], axis=0)
            else:
                temp = np.concatenate([temp, temp[:i + 2, :]], axis=0)
                temp = np.delete(temp, np.arange(0, i + 2), axis=0)
                temp = np.delete(temp, [13, 14, 15], axis=0)
            chunks.append(temp)
        return np.concatenate(chunks, axis=0).reshape(-1)

    chunks = []
    for i in range(16):
        temp = raw[:, i, :]
        if i == 15:
            temp = np.delete(temp, [0, 13, 14], axis=0)
        else:
            temp = np.concatenate([temp, temp[:i + 2, :]], axis=0)
            temp = np.delete(temp, np.arange(0, i + 2), axis=0)
            temp = np.delete(temp, [13, 14, 15], axis=0)
        chunks.append(temp)
    return np.concatenate(chunks, axis=0)


def make_16e_mesh(mesh, mesh2):
    """Subsample the 32-electrode KTC mesh to 16 electrodes by taking every other pad."""
    selected = list(range(0, len(mesh2.elfaces), 2))
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected)}

    def _convert(src_mesh):
        elfaces = [deepcopy(src_mesh.elfaces[idx]) for idx in selected]
        elements = []
        for element in src_mesh.Element:
            topo = np.array(element.Topology, copy=True)
            electrode = []
            if element.Electrode:
                old_idx = int(element.Electrode[0])
                if old_idx in old_to_new:
                    face_idx = np.array(element.Electrode[1], copy=True)
                    electrode = [old_to_new[old_idx], face_idx]
            elements.append(ELEMENT(topo, electrode))
        return Mesh(src_mesh.H, src_mesh.g, elfaces, src_mesh.Node, elements)

    return _convert(mesh), _convert(mesh2)


def raw256_vector_to_matrix(raw_flat: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_flat).reshape(-1)
    if raw.size != 256:
        raise ValueError(f'Expected raw length 256, got {raw.size}')
    return raw.reshape(16, 16)


def build_draeger208_channel_order(nel: int = 16) -> np.ndarray:
    """Return the 208-channel raw adjacent-difference order.

    The returned array contains the raw adjacent-difference channel indices
    (0..15) that are kept for each excitation pattern after converting raw
    `16 x 16` measurements to the Draeger/DCT-EIT 208-channel layout.
    """
    if nel != 16:
        raise ValueError('Draeger 208-channel ordering is defined for 16 electrodes only.')

    channel_order = np.zeros((nel, 13), dtype=np.int64)
    for exc in range(nel):
        channels = [((exc + 2 + k) % nel) for k in range(13)]
        channel_order[exc] = np.asarray(channels, dtype=np.int64)
    return channel_order


def build_draeger208_meas_pairs(nel: int = 16) -> np.ndarray:
    """Build pyEIT-style adjacent measurement pairs for the 208-channel layout.

    Returns a measurement matrix of shape `(16, 13, 2)` compatible with
    ``PyEITProtocol`` where each pair stores `[N, M]` for
    `v_diff = v_N - v_M`.

    The sign convention follows the local raw adjacent-difference operator used
    by ``build_adjacent_cycle_mpat``:

      channel `j` = `V_j - V_{j+1}`

    so each pair becomes `[j, j+1]`.
    """
    channel_order = build_draeger208_channel_order(nel)
    meas_mat = np.zeros((nel, 13, 2), dtype=np.int64)
    for exc in range(nel):
        for idx, ch in enumerate(channel_order[exc]):
            meas_mat[exc, idx] = [ch, (ch + 1) % nel]
    return meas_mat


def build_draeger208_pyeit_protocol(nel: int = 16) -> PyEITProtocol:
    """Create a `pyEIT` protocol matching the local 16-electrode 208-channel data."""
    ex_mat = np.asarray([[i, (i + 3) % nel] for i in range(nel)], dtype=np.int64)
    meas_mat = build_draeger208_meas_pairs(nel)
    keep_ba = np.ones(nel * 13, dtype=bool)
    return PyEITProtocol(ex_mat=ex_mat, meas_mat=meas_mat, keep_ba=keep_ba)
