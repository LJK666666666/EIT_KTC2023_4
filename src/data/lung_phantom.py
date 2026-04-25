"""Thorax-style lung phantom generation for pulmonary EIT experiments."""

import math

import numpy as np


_IMG_SIZE = 256
_PIX_WIDTH = 0.23 / _IMG_SIZE
_PIX_CENTER = np.linspace(
    -0.115 + _PIX_WIDTH / 2, 0.115 - _PIX_WIDTH / 2 + _PIX_WIDTH, _IMG_SIZE)
_X, _Y = np.meshgrid(_PIX_CENTER, _PIX_CENTER, indexing='ij')
_DOMAIN_MASK = (_X ** 2 + _Y ** 2) <= 0.098 ** 2


def _scaled_uniform(rng, center, radius, scale, low=None, high=None):
    radius = max(0.0, float(radius) * float(scale))
    lo = center - radius
    hi = center + radius
    if low is not None:
        lo = max(lo, low)
    if high is not None:
        hi = min(hi, high)
    if hi < lo:
        hi = lo
    return rng.uniform(lo, hi)


def _ellipse_mask(cx, cy, rx, ry, angle_deg):
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x = _X - cx
    y = _Y - cy
    xr = cos_a * x + sin_a * y
    yr = -sin_a * x + cos_a * y
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def _bean_lung(cx, cy, rx, ry, angle_deg, notch_shift=0.008, notch_scale=0.42):
    outer = _ellipse_mask(cx, cy, rx, ry, angle_deg)
    notch = _ellipse_mask(
        cx + notch_shift * np.sign(cx),
        cy,
        rx * notch_scale,
        ry * 0.55,
        angle_deg,
    )
    return outer & (~notch)


def _smooth_blob_field(rng, num_blobs=3, sigma_range=(0.010, 0.030)):
    """Create a smooth random field from a few Gaussian blobs."""
    field = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.float64)
    for _ in range(max(1, int(num_blobs))):
        cx = rng.uniform(-0.060, 0.060)
        cy = rng.uniform(-0.060, 0.060)
        sx = rng.uniform(*sigma_range)
        sy = rng.uniform(*sigma_range)
        amp = rng.uniform(-1.0, 1.0)
        blob = np.exp(-0.5 * (((_X - cx) / sx) ** 2 + ((_Y - cy) / sy) ** 2))
        field += amp * blob
    max_abs = float(np.max(np.abs(field)))
    if max_abs > 1e-8:
        field /= max_abs
    field[~_DOMAIN_MASK] = 0.0
    return field


def create_lung_phantom(rng=None,
                        anatomy_scale=1.0,
                        pathology_scale=1.0,
                        detail_scale=1.0):
    """Create a 3-class thorax phantom.

    Labels:
      0: background / thorax tissue
      1: lungs (low conductivity)
      2: heart or fluid-like high conductivity region
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8)

    # Thorax occupancy region stays implicit; we only label inner structures.
    insp = _scaled_uniform(rng, 1.0, 0.15, anatomy_scale, low=0.70, high=1.35)
    asym = _scaled_uniform(rng, 1.0, 0.08, anatomy_scale, low=0.80, high=1.20)
    lung_angle = _scaled_uniform(
        rng, 0.0, 12.0, anatomy_scale, low=-24.0, high=24.0)

    left_lung = _bean_lung(
        cx=-0.030 + _scaled_uniform(rng, -0.001, 0.003, anatomy_scale),
        cy=0.005 + _scaled_uniform(rng, 0.0, 0.006, anatomy_scale),
        rx=0.026 * insp * asym,
        ry=0.050 * insp,
        angle_deg=lung_angle + _scaled_uniform(rng, 0.0, 4.0, anatomy_scale),
        notch_shift=0.007,
    )
    right_lung = _bean_lung(
        cx=0.030 + _scaled_uniform(rng, 0.001, 0.003, anatomy_scale),
        cy=0.004 + _scaled_uniform(rng, 0.0, 0.006, anatomy_scale),
        rx=0.024 * insp / asym,
        ry=0.048 * insp,
        angle_deg=-lung_angle + _scaled_uniform(rng, 0.0, 4.0, anatomy_scale),
        notch_shift=-0.007,
    )

    # Mild pathology variants.
    pathology = rng.choice(
        ['normal', 'left_collapse', 'right_collapse', 'left_effusion',
         'right_effusion'],
        p=[0.45, 0.12, 0.12, 0.16, 0.15],
    )

    if pathology == 'left_collapse':
        collapse_factor = np.clip(1.0 - 0.25 * (pathology_scale - 1.0), 0.55, 1.25)
        collapse = _ellipse_mask(
            -0.030, 0.0, 0.018 * collapse_factor, 0.040 * collapse_factor, lung_angle)
        left_lung &= collapse
    elif pathology == 'right_collapse':
        collapse_factor = np.clip(1.0 - 0.25 * (pathology_scale - 1.0), 0.55, 1.25)
        collapse = _ellipse_mask(
            0.030, 0.0, 0.017 * collapse_factor, 0.038 * collapse_factor, -lung_angle)
        right_lung &= collapse

    mask[left_lung | right_lung] = 1

    # Heart / mediastinum
    heart = _ellipse_mask(
        cx=-0.004 + _scaled_uniform(rng, 0.0, 0.004, anatomy_scale),
        cy=-0.020 + _scaled_uniform(rng, 0.0, 0.004, anatomy_scale),
        rx=0.020 * _scaled_uniform(rng, 1.025, 0.125, anatomy_scale, low=0.75, high=1.35),
        ry=0.028 * _scaled_uniform(rng, 1.025, 0.125, anatomy_scale, low=0.75, high=1.35),
        angle_deg=_scaled_uniform(rng, 0.0, 18.0, anatomy_scale, low=-30.0, high=30.0),
    )
    mask[heart] = 2

    # Pleural effusion-like dependent high-conductivity region.
    if pathology == 'left_effusion':
        eff_scale = np.clip(pathology_scale, 0.6, 1.8)
        eff = _ellipse_mask(
            -0.040, -0.050, 0.022 * eff_scale, 0.018 * eff_scale,
            _scaled_uniform(rng, 0.0, 10.0, anatomy_scale, low=-18.0, high=18.0))
        mask[eff & left_lung] = 2
    elif pathology == 'right_effusion':
        eff_scale = np.clip(pathology_scale, 0.6, 1.8)
        eff = _ellipse_mask(
            0.040, -0.050, 0.022 * eff_scale, 0.018 * eff_scale,
            _scaled_uniform(rng, 0.0, 10.0, anatomy_scale, low=-18.0, high=18.0))
        mask[eff & right_lung] = 2

    # Add a small conductive vessel/region sometimes to diversify class 2.
    vessel_prob = float(np.clip(0.25 * detail_scale, 0.05, 0.75))
    if rng.random() < vessel_prob:
        vessel = _ellipse_mask(
            _scaled_uniform(rng, 0.001, 0.011, anatomy_scale),
            _scaled_uniform(rng, 0.0275, 0.0125, anatomy_scale, low=0.005, high=0.055),
            _scaled_uniform(rng, 0.009, 0.003, detail_scale, low=0.004, high=0.020),
            _scaled_uniform(rng, 0.014, 0.004, detail_scale, low=0.006, high=0.028),
            _scaled_uniform(rng, 0.0, 35.0, anatomy_scale, low=-45.0, high=45.0),
        )
        mask[vessel & _DOMAIN_MASK] = 2

    mask[~_DOMAIN_MASK] = 0
    return mask


def create_lung_conductivity(mask, rng=None,
                             conductivity_scale=1.0,
                             texture_scale=0.0):
    """Convert thorax labels to conductivity map."""
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.zeros(mask.shape, dtype=np.float64)
    sigma_bg = _scaled_uniform(
        rng, 0.775, 0.125, conductivity_scale, low=0.55, high=1.05)
    sigma_lung = _scaled_uniform(
        rng, 0.135, 0.085, conductivity_scale, low=0.02, high=0.35)
    sigma_high = _scaled_uniform(
        rng, 1.50, 0.40, conductivity_scale, low=0.80, high=2.40)

    sigma[mask == 0] = sigma_bg
    sigma[mask == 1] = sigma_lung
    sigma[mask == 2] = sigma_high

    if texture_scale > 0:
        bg_field = _smooth_blob_field(
            rng, num_blobs=int(np.clip(round(2 + 2 * texture_scale), 1, 8)),
            sigma_range=(0.020, 0.055),
        )
        lung_field = _smooth_blob_field(
            rng, num_blobs=int(np.clip(round(3 + 3 * texture_scale), 1, 10)),
            sigma_range=(0.010, 0.035),
        )
        high_field = _smooth_blob_field(
            rng, num_blobs=int(np.clip(round(2 + 2 * texture_scale), 1, 8)),
            sigma_range=(0.008, 0.028),
        )

        sigma[mask == 0] += (
            0.04 * texture_scale * sigma_bg * bg_field[mask == 0]
        )
        sigma[mask == 1] += (
            0.35 * texture_scale * max(sigma_lung, 0.03) * lung_field[mask == 1]
        )
        sigma[mask == 2] += (
            0.18 * texture_scale * sigma_high * high_field[mask == 2]
        )

        sigma[mask == 0] = np.clip(sigma[mask == 0], 0.40, 1.20)
        sigma[mask == 1] = np.clip(sigma[mask == 1], 0.01, 0.50)
        sigma[mask == 2] = np.clip(sigma[mask == 2], 0.60, 2.80)

    sigma[~_DOMAIN_MASK] = 0.0
    return sigma


def create_lung_pair_phantom(rng=None,
                             anatomy_scale=1.0,
                             pathology_scale=1.0,
                             detail_scale=1.0,
                             normal_prob=0.25):
    """Create paired pulmonary phantoms for time-difference EIT.

    The reference image keeps only the canonical thorax anatomy
    (lungs + heart), while the target image reuses the same anatomy and adds
    a pathology-dependent local change. This matches the clinical intuition of
    time-difference pulmonary EIT: stable anatomy plus evolving local residuals.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Shared anatomy
    insp = _scaled_uniform(rng, 1.0, 0.15, anatomy_scale, low=0.70, high=1.35)
    asym = _scaled_uniform(rng, 1.0, 0.08, anatomy_scale, low=0.80, high=1.20)
    lung_angle = _scaled_uniform(
        rng, 0.0, 12.0, anatomy_scale, low=-24.0, high=24.0)

    left_lung = _bean_lung(
        cx=-0.030 + _scaled_uniform(rng, -0.001, 0.003, anatomy_scale),
        cy=0.005 + _scaled_uniform(rng, 0.0, 0.006, anatomy_scale),
        rx=0.026 * insp * asym,
        ry=0.050 * insp,
        angle_deg=lung_angle + _scaled_uniform(rng, 0.0, 4.0, anatomy_scale),
        notch_shift=0.007,
    )
    right_lung = _bean_lung(
        cx=0.030 + _scaled_uniform(rng, 0.001, 0.003, anatomy_scale),
        cy=0.004 + _scaled_uniform(rng, 0.0, 0.006, anatomy_scale),
        rx=0.024 * insp / asym,
        ry=0.048 * insp,
        angle_deg=-lung_angle + _scaled_uniform(rng, 0.0, 4.0, anatomy_scale),
        notch_shift=-0.007,
    )

    heart = _ellipse_mask(
        cx=-0.004 + _scaled_uniform(rng, 0.0, 0.004, anatomy_scale),
        cy=-0.020 + _scaled_uniform(rng, 0.0, 0.004, anatomy_scale),
        rx=0.020 * _scaled_uniform(rng, 1.025, 0.125, anatomy_scale, low=0.75, high=1.35),
        ry=0.028 * _scaled_uniform(rng, 1.025, 0.125, anatomy_scale, low=0.75, high=1.35),
        angle_deg=_scaled_uniform(rng, 0.0, 18.0, anatomy_scale, low=-30.0, high=30.0),
    )

    ref_mask = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8)
    ref_mask[left_lung | right_lung] = 1
    ref_mask[heart] = 2

    tgt_mask = ref_mask.copy()

    normal_prob = float(np.clip(normal_prob, 0.0, 0.95))
    rem = max(1e-8, 1.0 - normal_prob)
    pathology = rng.choice(
        ['normal', 'left_collapse', 'right_collapse', 'left_effusion',
         'right_effusion', 'vascular'],
        p=[
            normal_prob,
            0.15 / 0.75 * rem,
            0.15 / 0.75 * rem,
            0.17 / 0.75 * rem,
            0.17 / 0.75 * rem,
            0.11 / 0.75 * rem,
        ],
    )

    if pathology == 'left_collapse':
        collapse_factor = np.clip(1.0 - 0.25 * (pathology_scale - 1.0), 0.55, 1.25)
        collapse = _ellipse_mask(
            -0.030, 0.0, 0.018 * collapse_factor, 0.040 * collapse_factor, lung_angle)
        tgt_mask[(left_lung & (~collapse))] = 0
    elif pathology == 'right_collapse':
        collapse_factor = np.clip(1.0 - 0.25 * (pathology_scale - 1.0), 0.55, 1.25)
        collapse = _ellipse_mask(
            0.030, 0.0, 0.017 * collapse_factor, 0.038 * collapse_factor, -lung_angle)
        tgt_mask[(right_lung & (~collapse))] = 0
    elif pathology == 'left_effusion':
        eff_scale = np.clip(pathology_scale, 0.6, 1.8)
        eff = _ellipse_mask(
            -0.040, -0.050, 0.022 * eff_scale, 0.018 * eff_scale,
            _scaled_uniform(rng, 0.0, 10.0, anatomy_scale, low=-18.0, high=18.0))
        tgt_mask[eff & left_lung] = 2
    elif pathology == 'right_effusion':
        eff_scale = np.clip(pathology_scale, 0.6, 1.8)
        eff = _ellipse_mask(
            0.040, -0.050, 0.022 * eff_scale, 0.018 * eff_scale,
            _scaled_uniform(rng, 0.0, 10.0, anatomy_scale, low=-18.0, high=18.0))
        tgt_mask[eff & right_lung] = 2

    vessel_prob = float(np.clip(0.35 * detail_scale, 0.10, 0.85))
    if pathology == 'vascular' or rng.random() < vessel_prob:
        vessel = _ellipse_mask(
            _scaled_uniform(rng, 0.001, 0.011, anatomy_scale),
            _scaled_uniform(rng, 0.0275, 0.0125, anatomy_scale, low=0.005, high=0.055),
            _scaled_uniform(rng, 0.009, 0.003, detail_scale, low=0.004, high=0.020),
            _scaled_uniform(rng, 0.014, 0.004, detail_scale, low=0.006, high=0.028),
            _scaled_uniform(rng, 0.0, 35.0, anatomy_scale, low=-45.0, high=45.0),
        )
        tgt_mask[vessel & _DOMAIN_MASK] = 2

    ref_mask[~_DOMAIN_MASK] = 0
    tgt_mask[~_DOMAIN_MASK] = 0
    return ref_mask, tgt_mask


def create_lung_pair_conductivity(ref_mask, tgt_mask, rng=None,
                                  conductivity_scale=1.0,
                                  texture_scale=0.0):
    """Create paired conductivity maps with shared class-wise conductivity parameters."""
    if rng is None:
        rng = np.random.default_rng()

    sigma_bg = _scaled_uniform(
        rng, 0.775, 0.125, conductivity_scale, low=0.55, high=1.05)
    sigma_lung = _scaled_uniform(
        rng, 0.135, 0.085, conductivity_scale, low=0.02, high=0.35)
    sigma_high = _scaled_uniform(
        rng, 1.50, 0.40, conductivity_scale, low=0.80, high=2.40)

    def _apply(mask):
        sigma = np.zeros(mask.shape, dtype=np.float64)
        sigma[mask == 0] = sigma_bg
        sigma[mask == 1] = sigma_lung
        sigma[mask == 2] = sigma_high
        if texture_scale > 0:
            bg_field = _smooth_blob_field(
                rng, num_blobs=int(np.clip(round(2 + 2 * texture_scale), 1, 8)),
                sigma_range=(0.020, 0.055),
            )
            lung_field = _smooth_blob_field(
                rng, num_blobs=int(np.clip(round(3 + 3 * texture_scale), 1, 10)),
                sigma_range=(0.010, 0.035),
            )
            high_field = _smooth_blob_field(
                rng, num_blobs=int(np.clip(round(2 + 2 * texture_scale), 1, 8)),
                sigma_range=(0.008, 0.028),
            )
            sigma[mask == 0] += (
                0.04 * texture_scale * sigma_bg * bg_field[mask == 0]
            )
            sigma[mask == 1] += (
                0.35 * texture_scale * max(sigma_lung, 0.03) * lung_field[mask == 1]
            )
            sigma[mask == 2] += (
                0.18 * texture_scale * sigma_high * high_field[mask == 2]
            )
            sigma[mask == 0] = np.clip(sigma[mask == 0], 0.40, 1.20)
            sigma[mask == 1] = np.clip(sigma[mask == 1], 0.01, 0.50)
            sigma[mask == 2] = np.clip(sigma[mask == 2], 0.60, 2.80)
        sigma[~_DOMAIN_MASK] = 0.0
        return sigma

    return _apply(ref_mask), _apply(tgt_mask)
