"""
Advanced phantom generation for KTC2023 EIT training data.

Contour styles for individual inclusions:
  - smooth  : ellipse / circle (smooth arc boundary)
  - wavy    : radial function with sinusoidal perturbation (organic contour)
  - angular : low-vertex polygon with high spikiness (sharp corners)
  - star    : alternating inner/outer radii (concave edges)
  - perlin  : Perlin-noise level-set boundary (natural blob)

Additional techniques:
  - Elastic Deformation - realistic shape perturbation
  - Perlin background perturbation - smooth conductivity variation

Output format: 256x256 array with values {0, 1, 2}.
"""

import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import rotate, gaussian_filter, map_coordinates


# ---------------------------------------------------------------------------
# Domain constants (same as phantom_generator.py)
# ---------------------------------------------------------------------------
_IMG_SIZE = 256
_PIX_WIDTH = 0.23 / _IMG_SIZE
_PIX_CENTER = np.linspace(
    -0.115 + _PIX_WIDTH / 2, 0.115 - _PIX_WIDTH / 2 + _PIX_WIDTH, _IMG_SIZE)
_X, _Y = np.meshgrid(_PIX_CENTER, _PIX_CENTER, indexing='ij')
_DOMAIN_MASK = (_X ** 2 + _Y ** 2) <= 0.098 ** 2

# Domain boundary in pixel coordinates
_DOMAIN_CENTER = _IMG_SIZE // 2   # 128
_DOMAIN_RADIUS_PX = int(0.098 / _PIX_WIDTH)  # ~109 pixels


# ===================================================================
# 1. Perlin Noise
# ===================================================================

def _perlin_noise_2d(shape, res, rng=None):
    """Generate 2D Perlin noise array.

    Args:
        shape: (H, W) output shape. Must be divisible by res.
        res: (res_y, res_x) number of gradient grid cells.
        rng: numpy random Generator.

    Returns:
        np.ndarray of shape (H, W) with values roughly in [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid_y = np.arange(0, res[0], delta[0])[:shape[0]]
    grid_x = np.arange(0, res[1], delta[1])[:shape[1]]
    grid = np.stack(np.meshgrid(grid_y, grid_x, indexing='ij'), axis=-1)

    # Random gradient angles on the lattice
    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    # Integer and fractional parts
    grid_int = grid.astype(int)
    g0y, g0x = grid_int[:, :, 0], grid_int[:, :, 1]
    g1y, g1x = g0y + 1, g0x + 1
    fy = grid[:, :, 0] - g0y
    fx = grid[:, :, 1] - g0x

    # Smoothstep (quintic)
    def fade(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    uy, ux = fade(fy), fade(fx)

    # Dot products at four corners
    def dot_grad(gy, gx, dy, dx):
        g = gradients[gy % (res[0] + 1), gx % (res[1] + 1)]
        return g[:, :, 0] * dy + g[:, :, 1] * dx

    n00 = dot_grad(g0y, g0x, fy, fx)
    n10 = dot_grad(g1y, g0x, fy - 1, fx)
    n01 = dot_grad(g0y, g1x, fy, fx - 1)
    n11 = dot_grad(g1y, g1x, fy - 1, fx - 1)

    # Bilinear interpolation
    n0 = n00 * (1 - uy) + n10 * uy
    n1 = n01 * (1 - uy) + n11 * uy
    return n0 * (1 - ux) + n1 * ux


def perlin_noise_octaves(shape=(256, 256), octaves=4, persistence=0.5,
                         base_res=(4, 4), rng=None):
    """Generate multi-octave Perlin noise (fractal Brownian motion).

    Args:
        shape: Output array shape.
        octaves: Number of noise octaves.
        persistence: Amplitude decay per octave.
        base_res: Base grid resolution.
        rng: numpy random Generator.

    Returns:
        np.ndarray normalized to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = np.zeros(shape)
    amplitude = 1.0
    total_amp = 0.0
    res = list(base_res)

    for _ in range(octaves):
        # Ensure shape is divisible by resolution
        r = (min(res[0], shape[0]), min(res[1], shape[1]))
        noise += amplitude * _perlin_noise_2d(shape, r, rng)
        total_amp += amplitude
        amplitude *= persistence
        res[0] *= 2
        res[1] *= 2

    noise /= total_amp
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
    return noise


def create_perlin_phantom(threshold_high=0.6, threshold_low=0.4,
                          octaves=4, persistence=0.5,
                          base_res=None, rng=None):
    """Generate phantom using Perlin noise thresholding.

    Regions above threshold_high become conductive (class 2),
    regions below threshold_low become resistive (class 1),
    the rest is background (class 0).

    Args:
        threshold_high: Noise value above which → class 2.
        threshold_low: Noise value below which → class 1.
        octaves: Number of noise octaves.
        persistence: Amplitude decay per octave.
        base_res: Base grid resolution, default random from [(3,3),(4,4),(5,5)].
        rng: numpy random Generator.

    Returns:
        np.ndarray: 256x256 with values {0, 1, 2}.
    """
    if rng is None:
        rng = np.random.default_rng()
    if base_res is None:
        r = rng.choice([3, 4, 5])
        base_res = (r, r)

    noise = perlin_noise_octaves(
        (_IMG_SIZE, _IMG_SIZE), octaves, persistence, base_res, rng)

    mask = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8)
    mask[noise > threshold_high] = 2
    mask[noise < threshold_low] = 1

    # Apply circular domain
    mask[~_DOMAIN_MASK] = 0

    # Filter out tiny regions (< 50 pixels)
    from scipy.ndimage import label
    for cls in [1, 2]:
        binary = (mask == cls)
        labeled, n_comp = label(binary)
        for i in range(1, n_comp + 1):
            if np.sum(labeled == i) < 50:
                mask[labeled == i] = 0

    return mask


# ===================================================================
# 2. Contour-based Shape Generation (non-overlapping inclusions)
# ===================================================================

# Available contour styles and their default weights
CONTOUR_STYLES = ['smooth', 'wavy', 'angular', 'star', 'perlin']
CONTOUR_WEIGHTS = [0.2, 0.25, 0.25, 0.15, 0.15]


def _contour_smooth(n_pts, avg_r, rng):
    """Ellipse / circle contour (smooth arc boundary)."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    rx = avg_r * rng.uniform(0.6, 1.0)
    ry = avg_r * rng.uniform(0.6, 1.0)
    return rx * np.cos(theta), ry * np.sin(theta)


def _contour_wavy(n_pts, avg_r, rng):
    """Radial function with sinusoidal perturbation (organic contour)."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    # 2-5 wave lobes with random amplitude
    n_lobes = rng.integers(2, 6)
    amp = rng.uniform(0.15, 0.35) * avg_r
    phase = rng.uniform(0, 2 * np.pi)
    r = avg_r + amp * np.sin(n_lobes * theta + phase)
    # Add a second harmonic for more variety
    if rng.random() < 0.5:
        n2 = rng.integers(3, 8)
        amp2 = rng.uniform(0.05, 0.15) * avg_r
        r += amp2 * np.sin(n2 * theta + rng.uniform(0, 2 * np.pi))
    return r * np.cos(theta), r * np.sin(theta)


def _contour_angular(n_pts, avg_r, rng):
    """Low-vertex polygon with high spikiness (sharp corners)."""
    n_verts = rng.integers(3, 7)
    # Random angular steps (irregular spacing)
    raw = rng.uniform(0.5, 1.5, size=n_verts)
    angles_v = np.cumsum(raw / raw.sum() * 2 * np.pi)
    # Random radii per vertex (spiky)
    radii_v = avg_r * rng.uniform(0.5, 1.2, size=n_verts)

    # Interpolate between vertices to get n_pts
    angles_v = np.concatenate([angles_v, [angles_v[0] + 2 * np.pi]])
    radii_v = np.concatenate([radii_v, [radii_v[0]]])
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = np.interp(theta, angles_v, radii_v, period=2 * np.pi)
    return r * np.cos(theta), r * np.sin(theta)


def _contour_star(n_pts, avg_r, rng):
    """Alternating inner/outer radii (star / concave edges)."""
    n_arms = rng.integers(3, 8)
    r_outer = avg_r * rng.uniform(0.8, 1.1)
    r_inner = avg_r * rng.uniform(0.25, 0.55)

    # 2*n_arms control points
    angles_v = np.linspace(0, 2 * np.pi, 2 * n_arms, endpoint=False)
    radii_v = np.empty(2 * n_arms)
    radii_v[0::2] = r_outer
    radii_v[1::2] = r_inner
    # Add slight randomness
    radii_v += rng.uniform(-0.08, 0.08, size=len(radii_v)) * avg_r

    angles_v_ext = np.concatenate([angles_v, [angles_v[0] + 2 * np.pi]])
    radii_v_ext = np.concatenate([radii_v, [radii_v[0]]])
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = np.interp(theta, angles_v_ext, radii_v_ext, period=2 * np.pi)
    return r * np.cos(theta), r * np.sin(theta)


def _contour_perlin(n_pts, avg_r, rng):
    """Perlin-noise modulated radius (natural blob)."""
    # Generate 1D Perlin-like smooth noise via random Fourier components
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = np.full(n_pts, float(avg_r))
    n_harmonics = rng.integers(3, 8)
    for k in range(1, n_harmonics + 1):
        amp = rng.uniform(0.05, 0.25) * avg_r / k
        phase = rng.uniform(0, 2 * np.pi)
        r += amp * np.sin(k * theta + phase)
    r = np.clip(r, avg_r * 0.3, avg_r * 1.5)
    return r * np.cos(theta), r * np.sin(theta)


_CONTOUR_FN = {
    'smooth': _contour_smooth,
    'wavy': _contour_wavy,
    'angular': _contour_angular,
    'star': _contour_star,
    'perlin': _contour_perlin,
}


def _draw_shape(cx, cy, avg_r, style, fill_val, rng, n_pts=128):
    """Rasterize a single contour-based shape onto a 256x256 image.

    Args:
        cx, cy: Center in pixel coordinates.
        avg_r: Average radius in pixels.
        style: One of CONTOUR_STYLES.
        fill_val: Fill value (1 or 2).
        rng: numpy random Generator.
        n_pts: Number of contour sampling points.

    Returns:
        np.ndarray: 256x256 uint8 image with the shape filled.
    """
    dx, dy = _CONTOUR_FN[style](n_pts, avg_r, rng)

    # Random rotation
    angle = rng.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rx = cos_a * dx - sin_a * dy + cx
    ry = sin_a * dx + cos_a * dy + cy

    # Convert to polygon vertex list
    vertices = list(zip(rx.tolist(), ry.tolist()))

    im = Image.fromarray(np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8))
    draw = ImageDraw.Draw(im)
    draw.polygon(vertices, fill=int(fill_val))
    return np.array(im)


def create_phantoms_advanced(n_inclusions=None, min_inclusions=1,
                             max_inclusions=5, style_weights=None,
                             radius_range=(18, 55), min_gap=8,
                             max_attempts=120, rng=None):
    """Generate a phantom with 1-5 non-overlapping inclusions.

    Each inclusion has a randomly chosen contour style (smooth, wavy,
    angular, star, or perlin-blob). Inclusions are placed one by one
    with collision detection to prevent overlap.

    Args:
        n_inclusions: Exact number of inclusions (overrides min/max).
        min_inclusions: Minimum number of inclusions.
        max_inclusions: Maximum number of inclusions (exclusive upper bound).
        style_weights: Probability weights for contour styles.
        radius_range: (min, max) average radius in pixels.
        min_gap: Minimum pixel gap between shape bounding circles.
        max_attempts: Maximum placement attempts before giving up.
        rng: numpy random Generator.

    Returns:
        np.ndarray: 256x256 with values {0, 1, 2}.
    """
    if rng is None:
        rng = np.random.default_rng()
    if style_weights is None:
        style_weights = CONTOUR_WEIGHTS
    if n_inclusions is None:
        n_inclusions = rng.integers(min_inclusions, max_inclusions + 1)

    mask = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8)
    placed = []  # list of (cx, cy, r) for collision detection

    attempts = 0
    while len(placed) < n_inclusions and attempts < max_attempts:
        attempts += 1

        avg_r = rng.integers(radius_range[0], radius_range[1] + 1)

        # Random center inside domain (with margin for radius)
        margin = avg_r + 5
        max_offset = _DOMAIN_RADIUS_PX - margin
        if max_offset < 5:
            max_offset = 5
        for _ in range(20):
            cx = _DOMAIN_CENTER + rng.integers(-max_offset, max_offset + 1)
            cy = _DOMAIN_CENTER + rng.integers(-max_offset, max_offset + 1)
            # Check inside circular domain
            dist_from_center = math.hypot(cx - _DOMAIN_CENTER,
                                          cy - _DOMAIN_CENTER)
            if dist_from_center + avg_r <= _DOMAIN_RADIUS_PX:
                break
        else:
            continue

        # Collision detection (bounding circle)
        collide = False
        for ox, oy, o_r in placed:
            d = math.hypot(cx - ox, cy - oy)
            if d < avg_r + o_r + min_gap:
                collide = True
                break
        if collide:
            continue

        # Choose contour style and class
        style = rng.choice(CONTOUR_STYLES, p=style_weights)
        cls = rng.choice([1, 2])

        shape_img = _draw_shape(cx, cy, avg_r, style, cls, rng)

        # Verify no pixel overlap with existing inclusions
        overlap = (shape_img > 0) & (mask > 0)
        if overlap.any():
            continue

        mask[shape_img > 0] = shape_img[shape_img > 0]
        placed.append((cx, cy, avg_r))

    # Apply circular domain mask
    mask[~_DOMAIN_MASK] = 0

    return mask


# ===================================================================
# 3. Elastic Deformation
# ===================================================================

def elastic_deform(image, alpha=15.0, sigma=4.0, rng=None):
    """Apply random elastic deformation to a 2D image.

    Generates a smooth random displacement field and warps the image.

    Args:
        image: 2D numpy array (any dtype).
        alpha: Displacement magnitude (pixels).
        sigma: Gaussian smoothing sigma for displacement field.
        rng: numpy random Generator.

    Returns:
        Deformed image (same shape, same dtype).
    """
    if rng is None:
        rng = np.random.default_rng()

    shape = image.shape
    # Random displacement field
    dx = gaussian_filter(rng.standard_normal(shape), sigma) * alpha
    dy = gaussian_filter(rng.standard_normal(shape), sigma) * alpha

    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    coords = [np.clip(y + dy, 0, shape[0] - 1),
              np.clip(x + dx, 0, shape[1] - 1)]

    return map_coordinates(image.astype(np.float64), coords,
                           order=0, mode='constant', cval=0).astype(image.dtype)


# ===================================================================
# 4. Perlin Noise Background Perturbation
# ===================================================================

def add_perlin_background(sigma_map, amplitude=0.05, rng=None):
    """Add Perlin noise perturbation to a continuous conductivity map.

    This creates smooth spatial variation in the background conductivity,
    simulating real-world inhomogeneities (temperature, ion concentration).

    Args:
        sigma_map: 256x256 float conductivity map.
        amplitude: Relative perturbation strength (e.g. 0.05 = ±5%).
        rng: numpy random Generator.

    Returns:
        Perturbed conductivity map.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = perlin_noise_octaves(sigma_map.shape, octaves=3,
                                 persistence=0.5, base_res=(3, 3), rng=rng)
    # Center noise around 0 with range [-amplitude, +amplitude]
    perturbation = (noise * 2 - 1) * amplitude
    return sigma_map * (1.0 + perturbation)


# ===================================================================
# 5. Combined Pipeline (The Ultimate Recipe)
# ===================================================================

def create_phantoms_v2(method=None, elastic_prob=0.3,
                       rng=None):
    """Advanced phantom generation following the combined pipeline.

    Pipeline:
      1. Shape generation: contour-based (60%), Perlin threshold (20%),
         or legacy polygons (20%).
      2. Optional elastic deformation (30% chance by default).
      3. Domain mask and cleanup.

    Args:
        method: Force a specific method ('contour', 'perlin', 'legacy').
                If None, randomly selected with weights [0.6, 0.2, 0.2].
        elastic_prob: Probability of applying elastic deformation.
        rng: numpy random Generator.

    Returns:
        np.ndarray: 256x256 with values {0, 1, 2}.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Select generation method
    if method is None:
        method = rng.choice(['contour', 'perlin', 'legacy'],
                            p=[0.6, 0.2, 0.2])

    if method == 'contour':
        mask = create_phantoms_advanced(rng=rng)
    elif method == 'perlin':
        th_high = rng.uniform(0.55, 0.7)
        th_low = rng.uniform(0.3, 0.45)
        mask = create_perlin_phantom(
            threshold_high=th_high, threshold_low=th_low, rng=rng)
    elif method == 'legacy':
        from src.data.phantom_generator import create_phantoms
        mask = create_phantoms().astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 2: Optional elastic deformation
    if rng.random() < elastic_prob:
        alpha = rng.uniform(8.0, 20.0)
        sigma = rng.uniform(3.0, 6.0)
        mask = elastic_deform(mask, alpha=alpha, sigma=sigma, rng=rng)

    # Step 3: Apply domain mask and ensure valid classes
    mask = np.clip(np.round(mask), 0, 2).astype(np.uint8)
    mask[~_DOMAIN_MASK] = 0

    # Ensure at least one inclusion exists
    if mask.max() == 0:
        mask = create_phantoms_advanced(n_inclusions=2, rng=rng)

    return mask


def create_conductivity_map(mask, sigma_bg=0.804,
                            sigma_resistive_range=(0.025, 0.125),
                            sigma_conductive_range=(4.0, 6.0),
                            perlin_bg=False, perlin_bg_amp=0.05,
                            rng=None):
    """Convert a class mask to a continuous conductivity map.

    Args:
        mask: 256x256 array with values {0, 1, 2}.
        sigma_bg: Background conductivity (S/m).
        sigma_resistive_range: (min, max) for resistive inclusions.
        sigma_conductive_range: (min, max) for conductive inclusions.
        perlin_bg: Whether to add Perlin noise to background.
        perlin_bg_amp: Amplitude of background perturbation.
        rng: numpy random Generator.

    Returns:
        np.ndarray: 256x256 float conductivity map.
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_map = np.full((_IMG_SIZE, _IMG_SIZE), sigma_bg, dtype=np.float64)

    sigma_r = rng.uniform(*sigma_resistive_range)
    sigma_c = rng.uniform(*sigma_conductive_range)

    sigma_map[mask == 1] = sigma_r
    sigma_map[mask == 2] = sigma_c

    if perlin_bg:
        # Only perturb background region
        bg_mask = (mask == 0) & _DOMAIN_MASK
        noise = perlin_noise_octaves(
            (_IMG_SIZE, _IMG_SIZE), octaves=3, persistence=0.5,
            base_res=(3, 3), rng=rng)
        perturbation = (noise * 2 - 1) * perlin_bg_amp
        sigma_map[bg_mask] *= (1.0 + perturbation[bg_mask])

    sigma_map[~_DOMAIN_MASK] = 0.0
    return sigma_map


# ===================================================================
# CLI for batch visualization / testing
# ===================================================================

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Advanced phantom generation with visualization')
    parser.add_argument('--method', choices=['contour', 'perlin', 'legacy',
                                             'mixed'],
                        default='mixed',
                        help='Generation method (default: mixed)')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save generated images')
    parser.add_argument('--elastic-prob', type=float, default=0.3,
                        help='Probability of elastic deformation')
    parser.add_argument('--perlin-bg', action='store_true',
                        help='Add Perlin noise to background conductivity')
    parser.add_argument('--show-conductivity', action='store_true',
                        help='Show continuous conductivity map instead of mask')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    samples = []
    methods_used = []
    for i in range(args.num_samples):
        method = args.method if args.method != 'mixed' else None
        mask = create_phantoms_v2(method=method,
                                  elastic_prob=args.elastic_prob,
                                  rng=rng)
        if args.show_conductivity:
            sample = create_conductivity_map(
                mask, perlin_bg=args.perlin_bg, rng=rng)
        else:
            sample = mask
        samples.append(sample)
        # Track which method was actually used
        if args.method == 'mixed':
            methods_used.append('mixed')
        else:
            methods_used.append(args.method)

    # Save if requested
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for i, s in enumerate(samples):
            np.save(os.path.join(args.save_dir, f'phantom_{i:04d}.npy'), s)
        print(f'Saved {len(samples)} phantoms to {args.save_dir}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plot')
        return

    ncols = min(4, args.num_samples)
    nrows = math.ceil(args.num_samples / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, s in enumerate(samples):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        if args.show_conductivity:
            im = ax.imshow(s.T, cmap='viridis', origin='lower')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(s.T, cmap='gray', origin='lower', vmin=0, vmax=2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(len(samples), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(args.save_dir, 'overview.png') if args.save_dir \
        else 'phantom_overview.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved plot to {out_path}')


if __name__ == '__main__':
    main()
