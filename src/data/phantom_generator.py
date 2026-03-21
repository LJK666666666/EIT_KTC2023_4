"""
Phantom image generation utilities for KTC2023 EIT training data.

Generates random 256x256 phantom images with 3 classes:
  0: Background
  1: Resistive inclusion
  2: Conductive inclusion

Shape types (7 styles):
  - polygon    : irregular polygon with 5-8 vertices (legacy)
  - circle     : ellipse with similar rx/ry
  - rectangle  : axis-aligned rectangle
  - wavy       : radial contour with sinusoidal lobes
  - angular    : low-vertex polygon with sharp spiky corners
  - star       : alternating inner/outer radii (concave arms)
  - blob       : multi-harmonic Fourier radial modulation (natural blob)
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import rotate


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
_IMG_SIZE = 256
_PIX_WIDTH = 0.23 / _IMG_SIZE
_PIX_CENTER = np.linspace(
    -0.115 + _PIX_WIDTH / 2, 0.115 - _PIX_WIDTH / 2 + _PIX_WIDTH, _IMG_SIZE)
_X, _Y = np.meshgrid(_PIX_CENTER, _PIX_CENTER, indexing='ij')
_DOMAIN_MASK = (_X ** 2 + _Y ** 2) <= 0.098 ** 2
_DOMAIN_CENTER = _IMG_SIZE // 2
_DOMAIN_RADIUS_PX = int(0.098 / _PIX_WIDTH)  # ~109

# Number of inclusions: 1~5 with specified probabilities
_N_INCLUSIONS_CHOICES = [1, 2, 3, 4, 5]
_N_INCLUSIONS_PROBS = [0.15, 0.30, 0.30, 0.15, 0.10]

# Shape types and their default probabilities
SHAPE_TYPES = ['polygon', 'circle', 'rectangle', 'wavy', 'angular', 'star',
               'blob']
SHAPE_PROBS = [0.20, 0.10, 0.05, 0.25, 0.15, 0.10, 0.15]

# Contour sampling resolution
_N_CONTOUR_PTS = 128


# ---------------------------------------------------------------------------
# Contour generators (return dx, dy arrays relative to center)
# ---------------------------------------------------------------------------

def _contour_wavy(avg_r, n_pts=_N_CONTOUR_PTS):
    """Sinusoidal radial perturbation — organic / amoeba-like contour."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    n_lobes = np.random.randint(2, 6)
    amp = np.random.uniform(0.15, 0.35) * avg_r
    phase = np.random.uniform(0, 2 * np.pi)
    r = avg_r + amp * np.sin(n_lobes * theta + phase)
    # Optional second harmonic
    if np.random.rand() < 0.5:
        n2 = np.random.randint(3, 8)
        amp2 = np.random.uniform(0.05, 0.15) * avg_r
        r += amp2 * np.sin(n2 * theta + np.random.uniform(0, 2 * np.pi))
    return r * np.cos(theta), r * np.sin(theta)


def _contour_angular(avg_r, n_pts=_N_CONTOUR_PTS):
    """Low-vertex polygon with large radius variance — sharp spiky corners."""
    n_verts = np.random.randint(3, 7)
    raw = np.random.uniform(0.5, 1.5, size=n_verts)
    angles_v = np.cumsum(raw / raw.sum() * 2 * np.pi)
    radii_v = avg_r * np.random.uniform(0.45, 1.25, size=n_verts)

    angles_v = np.concatenate([angles_v, [angles_v[0] + 2 * np.pi]])
    radii_v = np.concatenate([radii_v, [radii_v[0]]])
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = np.interp(theta, angles_v, radii_v, period=2 * np.pi)
    return r * np.cos(theta), r * np.sin(theta)


def _contour_star(avg_r, n_pts=_N_CONTOUR_PTS):
    """Alternating inner/outer radii — star / concave arms."""
    n_arms = np.random.randint(3, 8)
    r_outer = avg_r * np.random.uniform(0.8, 1.15)
    r_inner = avg_r * np.random.uniform(0.25, 0.55)

    angles_v = np.linspace(0, 2 * np.pi, 2 * n_arms, endpoint=False)
    radii_v = np.empty(2 * n_arms)
    radii_v[0::2] = r_outer
    radii_v[1::2] = r_inner
    radii_v += np.random.uniform(-0.08, 0.08, size=len(radii_v)) * avg_r

    angles_v = np.concatenate([angles_v, [angles_v[0] + 2 * np.pi]])
    radii_v = np.concatenate([radii_v, [radii_v[0]]])
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = np.interp(theta, angles_v, radii_v, period=2 * np.pi)
    return r * np.cos(theta), r * np.sin(theta)


def _contour_blob(avg_r, n_pts=_N_CONTOUR_PTS):
    """Multi-harmonic Fourier radius modulation — natural blob."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = np.full(n_pts, float(avg_r))
    n_harmonics = np.random.randint(3, 8)
    for k in range(1, n_harmonics + 1):
        amp = np.random.uniform(0.05, 0.25) * avg_r / k
        phase = np.random.uniform(0, 2 * np.pi)
        r += amp * np.sin(k * theta + phase)
    r = np.clip(r, avg_r * 0.3, avg_r * 1.5)
    return r * np.cos(theta), r * np.sin(theta)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_contour_shape(draw, cx, cy, avg_r, contour_fn, fill_val):
    """Generate contour via contour_fn, apply random rotation, and draw."""
    dx, dy = contour_fn(avg_r)
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rx = cos_a * dx - sin_a * dy + cx
    ry = sin_a * dx + cos_a * dy + cy
    vertices = list(zip(rx.tolist(), ry.tolist()))
    draw.polygon(vertices, fill=fill_val)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def create_phantoms(max_iter=120, distance_between=10, p=None):
    """Generate a random 256x256 phantom image with values {0, 1, 2}.

    Places 1-5 non-overlapping inclusions with diverse contour styles
    inside a circular domain mask. Each inclusion is randomly assigned
    class 1 (resistive) or 2 (conductive).

    Args:
        max_iter: Maximum placement attempts before giving up.
        distance_between: Minimum pixel gap between inclusion bounding circles.
        p: Probability weights for shape types. Length must match SHAPE_TYPES.
            Default: [0.20, 0.10, 0.05, 0.25, 0.15, 0.10, 0.15] for
            [polygon, circle, rectangle, wavy, angular, star, blob].

    Returns:
        np.ndarray: 256x256 array with integer values in {0, 1, 2}.
    """
    if p is None:
        p = SHAPE_PROBS

    im = Image.fromarray(np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8))
    draw = ImageDraw.Draw(im)

    num_forms = np.random.choice(_N_INCLUSIONS_CHOICES, p=_N_INCLUSIONS_PROBS)
    placed = []  # (cx, cy, avg_radius)
    iteration = 0

    while len(placed) < num_forms and iteration < max_iter:
        iteration += 1

        avg_radius = np.random.randint(18, 55)

        # Random center inside domain with margin
        margin = avg_radius + 5
        max_offset = _DOMAIN_RADIUS_PX - margin
        if max_offset < 5:
            max_offset = 5

        cx = _DOMAIN_CENTER + np.random.randint(-max_offset, max_offset + 1)
        cy = _DOMAIN_CENTER + np.random.randint(-max_offset, max_offset + 1)

        # Ensure center + radius fits in circular domain
        dist = math.hypot(cx - _DOMAIN_CENTER, cy - _DOMAIN_CENTER)
        if dist + avg_radius > _DOMAIN_RADIUS_PX:
            continue

        # Collision detection (bounding circle + gap)
        collide = False
        for ox, oy, o_r in placed:
            d = math.hypot(cx - ox, cy - oy)
            if d < avg_radius + o_r + distance_between:
                collide = True
                break
        if collide:
            continue

        fill_val = 1 if np.random.rand() < 0.5 else 2
        shape_type = np.random.choice(SHAPE_TYPES, p=p)

        if shape_type == 'rectangle':
            # Axis-aligned rectangle inscribed in bounding circle
            w = np.random.randint(max(10, avg_radius // 2), avg_radius * 2)
            h = np.random.randint(max(10, avg_radius // 2), avg_radius * 2)
            draw.rectangle(
                [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
                fill=fill_val)
        elif shape_type == 'circle':
            # Ellipse with similar rx/ry
            rx = avg_radius * np.random.uniform(0.7, 1.0)
            ry = avg_radius * np.random.uniform(0.7, 1.0)
            draw.ellipse(
                [cx - rx, cy - ry, cx + rx, cy + ry], fill=fill_val)
        elif shape_type == 'polygon':
            num_vertices = np.random.randint(5, 9)
            vertices = generate_polygon(
                center=(cx, cy), avg_radius=avg_radius,
                irregularity=0.4, spikiness=0.3,
                num_vertices=num_vertices)
            draw.polygon(vertices, fill=fill_val)
        elif shape_type == 'wavy':
            _draw_contour_shape(draw, cx, cy, avg_radius,
                                _contour_wavy, fill_val)
        elif shape_type == 'angular':
            _draw_contour_shape(draw, cx, cy, avg_radius,
                                _contour_angular, fill_val)
        elif shape_type == 'star':
            _draw_contour_shape(draw, cx, cy, avg_radius,
                                _contour_star, fill_val)
        elif shape_type == 'blob':
            _draw_contour_shape(draw, cx, cy, avg_radius,
                                _contour_blob, fill_val)

        placed.append((cx, cy, avg_radius))

    sigma_pix = np.array(np.asarray(im))
    sigma_pix[~_DOMAIN_MASK] = 0
    angle = np.random.randint(0, 360)
    sigma_pix = np.round(
        rotate(sigma_pix, angle, mode='constant', cval=0.0,
               reshape=False, order=0))
    sigma_pix[~_DOMAIN_MASK] = 0

    return sigma_pix


# ---------------------------------------------------------------------------
# Legacy polygon helper (kept for 'polygon' shape type)
# ---------------------------------------------------------------------------

def generate_polygon(center, avg_radius, irregularity, spikiness,
                     num_vertices):
    """Generate a random polygon with the given parameters.

    Reference: https://stackoverflow.com/questions/8997099

    Args:
        center: (x, y) center of the polygon.
        avg_radius: Average distance of vertices from center.
        irregularity: Variance of angular spacing (0-1).
        spikiness: Variance of radial distance (0-1).
        num_vertices: Number of polygon vertices.

    Returns:
        List of (x, y) tuples for the polygon vertices.
    """
    if irregularity < 0 or irregularity > 1:
        raise ValueError('Irregularity must be between 0 and 1.')
    if spikiness < 0 or spikiness > 1:
        raise ValueError('Spikiness must be between 0 and 1.')

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = _random_angle_steps(num_vertices, irregularity)

    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = _clip(random.gauss(avg_radius, spikiness), 0,
                       2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points


def _random_angle_steps(steps, irregularity):
    """Generate random angular divisions for polygon vertices."""
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for _ in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def _clip(value, lower, upper):
    """Clip value to [lower, upper]."""
    return min(upper, max(value, lower))
