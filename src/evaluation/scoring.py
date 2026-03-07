"""
Scoring module for KTC2023 EIT reconstruction evaluation.

Contains two scoring implementations:
1. scoring_function        - Official SSIM-based scorer (uses scipy.signal.convolve2d, accurate but slow)
2. FastScoringFunction     - Fast scorer using separable 1D convolutions (suitable for training-time evaluation)

Both scorers compute the average SSIM of two binary masks:
  - Class 2 mask (conductive inclusions)
  - Class 1 mask (resistive inclusions)
using a large Gaussian kernel (r=80) as the SSIM weighting window.

Additionally provides:
  - Otsu / Otsu2: Otsu thresholding for 2-class / 3-class segmentation
  - segment_recon_otsu: Segment a continuous reconstruction into {0, 1, 2} classes
"""

import numpy as np
import scipy.signal as sps
from scipy.ndimage import convolve1d


# ---------------------------------------------------------------------------
# Otsu thresholding helpers
# ---------------------------------------------------------------------------

def Otsu(image, nvals, figno=None):
    """Binary Otsu's method for finding a single segmentation threshold.

    Args:
        image: Input image (any shape, will be flattened).
        nvals: Number of histogram bins.
        figno: Unused (kept for API compatibility).

    Returns:
        level: Index of the optimal threshold in the histogram bin edges.
        x: Histogram bin edges array.
    """
    histogramCounts, x = np.histogram(image.ravel(), nvals)

    total = np.sum(histogramCounts)
    top = 256
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.dot(np.arange(top), histogramCounts)
    for ii in range(1, top):
        wF = total - wB
        if wB > 0 and wF > 0:
            mF = (sum1 - sumB) / wF
            val = wB * wF * (((sumB / wB) - mF) ** 2)
            if val >= maximum:
                level = ii
                maximum = val
        wB = wB + histogramCounts[ii]
        sumB = sumB + (ii - 1) * histogramCounts[ii]

    return level, x


def Otsu2(image, nvals, figno=None):
    """Three-class Otsu's method to find two segmentation thresholds.

    Args:
        image: Input image (any shape, will be flattened).
        nvals: Number of histogram bins.
        figno: Unused (kept for API compatibility).

    Returns:
        level: List of two indices [threshold1, threshold2] in the histogram bin centers.
        x: Histogram bin centers array.
    """
    histogramCounts, tx = np.histogram(image.ravel(), nvals)
    x = (tx[0:-1] + tx[1:]) / 2

    maximum = 0.0
    level = [0, 0]
    muT = np.dot(np.arange(1, nvals + 1), histogramCounts) / np.sum(histogramCounts)
    for ii in range(1, nvals):
        for jj in range(1, ii):
            w1 = np.sum(histogramCounts[:jj])
            w2 = np.sum(histogramCounts[jj:ii])
            w3 = np.sum(histogramCounts[ii:])
            if w1 > 0 and w2 > 0 and w3 > 0:
                mu1 = np.dot(np.arange(1, jj + 1), histogramCounts[:jj]) / w1
                mu2 = np.dot(np.arange(jj + 1, ii + 1), histogramCounts[jj:ii]) / w2
                mu3 = np.dot(np.arange(ii + 1, nvals + 1), histogramCounts[ii:]) / w3

                val = (w1 * ((mu1 - muT) ** 2)
                       + w2 * ((mu2 - muT) ** 2)
                       + w3 * ((mu3 - muT) ** 2))
                if val >= maximum:
                    level = [jj - 1, ii - 1]
                    maximum = val

    return level, x


def segment_recon_otsu(recon):
    """Segment a continuous-valued reconstruction into three classes {0, 1, 2}
    using three-class Otsu thresholding.

    The background class is identified as the one with the most pixels,
    and is assigned value 0. The remaining two classes are assigned 1 and 2
    depending on their position relative to the thresholds.

    Args:
        recon: 2D numpy array with continuous values.

    Returns:
        Segmented 2D numpy array with integer values in {0, 1, 2}.
    """
    level, x = Otsu2(recon.flatten(), 256, 7)

    deltareco_pixgrid_segmented = np.zeros_like(recon)

    ind0 = recon < x[level[0]]
    ind1 = np.logical_and(recon >= x[level[0]], recon <= x[level[1]])
    ind2 = recon > x[level[1]]
    inds = [np.count_nonzero(ind0), np.count_nonzero(ind1), np.count_nonzero(ind2)]
    bgclass = inds.index(max(inds))  # background class

    if bgclass == 0:
        deltareco_pixgrid_segmented[ind1] = 2
        deltareco_pixgrid_segmented[ind2] = 2
    elif bgclass == 1:
        deltareco_pixgrid_segmented[ind0] = 1
        deltareco_pixgrid_segmented[ind2] = 2
    elif bgclass == 2:
        deltareco_pixgrid_segmented[ind0] = 1
        deltareco_pixgrid_segmented[ind1] = 1

    return deltareco_pixgrid_segmented


# ---------------------------------------------------------------------------
# Official SSIM computation (using scipy.signal.convolve2d)
# ---------------------------------------------------------------------------

def _ssim_official(truth, reco):
    """Compute the structural similarity index between two binary images
    using the official KTC2023 method (full 2D Gaussian convolution via
    scipy.signal.convolve2d).

    Args:
        truth: 2D numpy array (binary, values 0 or 1).
        reco: 2D numpy array (binary, values 0 or 1).

    Returns:
        Mean SSIM score (float).
    """
    c1 = 1e-4
    c2 = 9e-4
    r = 80

    ws = int(np.ceil(2 * r))
    wr = np.arange(-ws, ws + 1)
    X, Y = np.meshgrid(wr, wr)
    ker = (1 / np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * np.divide((np.square(X) + np.square(Y)), r ** 2)
    )
    correction = sps.convolve2d(np.ones(truth.shape), ker, mode='same')

    gt = np.divide(sps.convolve2d(truth, ker, mode='same'), correction)
    gr = np.divide(sps.convolve2d(reco, ker, mode='same'), correction)

    mu_t2 = np.square(gt)
    mu_r2 = np.square(gr)
    mu_t_mu_r = np.multiply(gt, gr)

    sigma_t2 = np.divide(
        sps.convolve2d(np.square(truth), ker, mode='same'), correction
    ) - mu_t2
    sigma_r2 = np.divide(
        sps.convolve2d(np.square(reco), ker, mode='same'), correction
    ) - mu_r2
    sigma_tr = np.divide(
        sps.convolve2d(np.multiply(truth, reco), ker, mode='same'), correction
    ) - mu_t_mu_r

    num = np.multiply((2 * mu_t_mu_r + c1), (2 * sigma_tr + c2))
    den = np.multiply((mu_t2 + mu_r2 + c1), (sigma_t2 + sigma_r2 + c2))
    ssimimage = np.divide(num, den)

    score = np.mean(ssimimage)
    return score


def scoring_function(groundtruth, reconstruction):
    """Official KTC2023 scoring function.

    Computes the average SSIM over two binary masks (class 1 and class 2)
    between ground truth and reconstruction. Uses full 2D Gaussian
    convolution (accurate but slower).

    Args:
        groundtruth: 256x256 numpy array with integer values in {0, 1, 2}.
        reconstruction: 256x256 numpy array with integer values in {0, 1, 2}.

    Returns:
        Score (float) in [0, 1]. Returns 0 if reconstruction shape is invalid.

    Raises:
        Exception: If groundtruth shape is not (256, 256).
    """
    if np.any(groundtruth.shape != np.array([256, 256])):
        raise Exception('The shape of the given ground truth is not 256 x 256!')

    if np.any(reconstruction.shape != np.array([256, 256])):
        return 0

    # Class 2 (conductive inclusions) binary mask
    truth_c = np.zeros(groundtruth.shape)
    truth_c[np.abs(groundtruth - 2) < 0.1] = 1
    reco_c = np.zeros(reconstruction.shape)
    reco_c[np.abs(reconstruction - 2) < 0.1] = 1
    score_c = _ssim_official(truth_c, reco_c)

    # Class 1 (resistive inclusions) binary mask
    truth_d = np.zeros(groundtruth.shape)
    truth_d[np.abs(groundtruth - 1) < 0.1] = 1
    reco_d = np.zeros(reconstruction.shape)
    reco_d[np.abs(reconstruction - 1) < 0.1] = 1
    score_d = _ssim_official(truth_d, reco_d)

    score = 0.5 * (score_c + score_d)
    return score


# ---------------------------------------------------------------------------
# Fast SSIM computation (using separable 1D convolutions)
# ---------------------------------------------------------------------------

def _convolve2d_separable(img, kernel, mode='constant'):
    """Fast 2D convolution using two separable 1D convolutions.

    The 2D Gaussian kernel is decomposed into the product of two 1D kernels,
    so that the convolution can be computed as two sequential 1D passes
    (along axis 0 and axis 1). This is significantly faster than full 2D
    convolution for large kernels.

    Args:
        img: 2D numpy array.
        kernel: 2D numpy array of shape (1, K) representing a row vector kernel.
        mode: Boundary mode passed to scipy.ndimage.convolve1d.

    Returns:
        2D numpy array (same shape as img).
    """
    return convolve1d(
        convolve1d(img, kernel[0], axis=0, mode=mode),
        kernel[0], axis=1, mode=mode
    )


def _ssim_fast(truth, reco):
    """Compute SSIM using separable 1D Gaussian convolutions (fast version).

    Uses a 1D Gaussian kernel and applies it along each axis separately,
    which is mathematically equivalent to convolving with a separable 2D
    Gaussian but much faster.

    Note: The fast version uses 'constant' boundary mode (zero-padding)
    instead of 'same' mode from the official scorer. Results may differ
    very slightly near boundaries, but are practically identical for scoring.

    Args:
        truth: 2D numpy array (binary, values 0 or 1).
        reco: 2D numpy array (binary, values 0 or 1).

    Returns:
        Mean SSIM score (float).
    """
    c1 = 1e-4
    c2 = 9e-4
    r = 80

    ws = int(np.ceil(2 * r))
    wr = np.arange(-ws, ws + 1)
    ker = ((1 / np.sqrt(2 * np.pi))
           * np.exp(-0.5 * np.divide(wr ** 2, r ** 2)))[None]

    correction = _convolve2d_separable(np.ones(truth.shape), ker, mode='constant')

    gt = np.divide(_convolve2d_separable(truth, ker, mode='constant'), correction)
    gr = np.divide(_convolve2d_separable(reco, ker, mode='constant'), correction)

    mu_t2 = np.square(gt)
    mu_r2 = np.square(gr)
    mu_t_mu_r = np.multiply(gt, gr)

    sigma_t2 = np.divide(
        _convolve2d_separable(np.square(truth), ker, mode='constant'), correction
    ) - mu_t2
    sigma_r2 = np.divide(
        _convolve2d_separable(np.square(reco), ker, mode='constant'), correction
    ) - mu_r2
    sigma_tr = np.divide(
        _convolve2d_separable(np.multiply(truth, reco), ker, mode='constant'),
        correction
    ) - mu_t_mu_r

    num = np.multiply((2 * mu_t_mu_r + c1), (2 * sigma_tr + c2))
    den = np.multiply((mu_t2 + mu_r2 + c1), (sigma_t2 + sigma_r2 + c2))
    ssimimage = np.divide(num, den)

    score = np.mean(ssimimage)
    return score


def FastScoringFunction(groundtruth, reconstruction):
    """Fast KTC2023 scoring function for training-time evaluation.

    Computes the same metric as scoring_function but uses separable 1D
    Gaussian convolutions instead of full 2D convolution, resulting in
    significant speedup. Suitable for evaluating during training loops.

    Args:
        groundtruth: 256x256 numpy array with integer values in {0, 1, 2}.
        reconstruction: 256x256 numpy array with integer values in {0, 1, 2}.

    Returns:
        Score (float) in [0, 1]. Returns 0 if reconstruction shape is invalid.

    Raises:
        Exception: If groundtruth shape is not (256, 256).
    """
    if np.any(groundtruth.shape != np.array([256, 256])):
        raise Exception('The shape of the given ground truth is not 256 x 256!')

    if np.any(reconstruction.shape != np.array([256, 256])):
        return 0

    # Class 2 (conductive inclusions) binary mask
    truth_c = np.zeros(groundtruth.shape)
    truth_c[np.abs(groundtruth - 2) < 0.1] = 1
    reco_c = np.zeros(reconstruction.shape)
    reco_c[np.abs(reconstruction - 2) < 0.1] = 1
    score_c = _ssim_fast(truth_c, reco_c)

    # Class 1 (resistive inclusions) binary mask
    truth_d = np.zeros(groundtruth.shape)
    truth_d[np.abs(groundtruth - 1) < 0.1] = 1
    reco_d = np.zeros(reconstruction.shape)
    reco_d[np.abs(reconstruction - 1) < 0.1] = 1
    score_d = _ssim_fast(truth_d, reco_d)

    score = 0.5 * (score_c + score_d)
    return score
