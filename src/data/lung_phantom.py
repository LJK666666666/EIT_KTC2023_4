"""Thorax-style lung phantom generation for pulmonary EIT experiments."""

import math

import numpy as np


_IMG_SIZE = 256
_PIX_WIDTH = 0.23 / _IMG_SIZE
_PIX_CENTER = np.linspace(
    -0.115 + _PIX_WIDTH / 2, 0.115 - _PIX_WIDTH / 2 + _PIX_WIDTH, _IMG_SIZE)
_X, _Y = np.meshgrid(_PIX_CENTER, _PIX_CENTER, indexing='ij')
_DOMAIN_MASK = (_X ** 2 + _Y ** 2) <= 0.098 ** 2


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


def create_lung_phantom(rng=None):
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
    insp = rng.uniform(0.85, 1.15)
    asym = rng.uniform(0.92, 1.08)
    lung_angle = rng.uniform(-12.0, 12.0)

    left_lung = _bean_lung(
        cx=-0.030 + rng.uniform(-0.004, 0.002),
        cy=0.005 + rng.uniform(-0.006, 0.006),
        rx=0.026 * insp * asym,
        ry=0.050 * insp,
        angle_deg=lung_angle + rng.uniform(-4.0, 4.0),
        notch_shift=0.007,
    )
    right_lung = _bean_lung(
        cx=0.030 + rng.uniform(-0.002, 0.004),
        cy=0.004 + rng.uniform(-0.006, 0.006),
        rx=0.024 * insp / asym,
        ry=0.048 * insp,
        angle_deg=-lung_angle + rng.uniform(-4.0, 4.0),
        notch_shift=-0.007,
    )

    # Mild pathology variants.
    pathology = rng.choice(
        ['normal', 'left_collapse', 'right_collapse', 'left_effusion',
         'right_effusion'],
        p=[0.45, 0.12, 0.12, 0.16, 0.15],
    )

    if pathology == 'left_collapse':
        collapse = _ellipse_mask(-0.030, 0.0, 0.018, 0.040, lung_angle)
        left_lung &= collapse
    elif pathology == 'right_collapse':
        collapse = _ellipse_mask(0.030, 0.0, 0.017, 0.038, -lung_angle)
        right_lung &= collapse

    mask[left_lung | right_lung] = 1

    # Heart / mediastinum
    heart = _ellipse_mask(
        cx=-0.004 + rng.uniform(-0.004, 0.004),
        cy=-0.020 + rng.uniform(-0.004, 0.004),
        rx=0.020 * rng.uniform(0.9, 1.15),
        ry=0.028 * rng.uniform(0.9, 1.15),
        angle_deg=rng.uniform(-18.0, 18.0),
    )
    mask[heart] = 2

    # Pleural effusion-like dependent high-conductivity region.
    if pathology == 'left_effusion':
        eff = _ellipse_mask(-0.040, -0.050, 0.022, 0.018, rng.uniform(-10, 10))
        mask[eff & left_lung] = 2
    elif pathology == 'right_effusion':
        eff = _ellipse_mask(0.040, -0.050, 0.022, 0.018, rng.uniform(-10, 10))
        mask[eff & right_lung] = 2

    # Add a small conductive vessel/region sometimes to diversify class 2.
    if rng.random() < 0.25:
        vessel = _ellipse_mask(
            rng.uniform(-0.010, 0.012),
            rng.uniform(0.015, 0.040),
            rng.uniform(0.006, 0.012),
            rng.uniform(0.010, 0.018),
            rng.uniform(-35.0, 35.0),
        )
        mask[vessel & _DOMAIN_MASK] = 2

    mask[~_DOMAIN_MASK] = 0
    return mask


def create_lung_conductivity(mask, rng=None):
    """Convert thorax labels to conductivity map."""
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.zeros(mask.shape, dtype=np.float64)
    sigma_bg = rng.uniform(0.65, 0.90)
    sigma_lung = rng.uniform(0.05, 0.22)
    sigma_high = rng.uniform(1.10, 1.90)

    sigma[mask == 0] = sigma_bg
    sigma[mask == 1] = sigma_lung
    sigma[mask == 2] = sigma_high
    sigma[~_DOMAIN_MASK] = 0.0
    return sigma
