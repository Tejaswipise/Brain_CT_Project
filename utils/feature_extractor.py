# Purpose: Extract all 25 radiomic features from one CT slice

import numpy as np
import cv2
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

from utils.dicom_loader import apply_brain_window


# ─────────────────────────────────────────────────────────────────
# GROUP A — INTENSITY FEATURES (8)
# ─────────────────────────────────────────────────────────────────

def extract_intensity_features(hu_array, brain_mask):
    brain_hu = hu_array[brain_mask].astype(np.float64)

    if len(brain_hu) == 0:
        return {k: 0.0 for k in ['hu_mean','hu_std','hu_skewness',
                                   'hu_kurtosis','hu_p10','hu_p25',
                                   'hu_p75','hu_p90']}

    return {
        'hu_mean':     float(np.mean(brain_hu)),
        'hu_std':      float(np.std(brain_hu)),
        'hu_skewness': float(stats.skew(brain_hu)),
        'hu_kurtosis': float(stats.kurtosis(brain_hu)),
        'hu_p10':      float(np.percentile(brain_hu, 10)),
        'hu_p25':      float(np.percentile(brain_hu, 25)),
        'hu_p75':      float(np.percentile(brain_hu, 75)),
        'hu_p90':      float(np.percentile(brain_hu, 90)),
    }


# ─────────────────────────────────────────────────────────────────
# GROUP B — GLCM TEXTURE FEATURES (6)
# ─────────────────────────────────────────────────────────────────

def extract_glcm_features(hu_array, brain_mask):
    windowed = apply_brain_window(hu_array)

    # Mask non-brain pixels
    masked = windowed.copy()
    masked[~brain_mask] = 0

    # Quantise to 64 grey levels
    quantised = (masked * 63).astype(np.uint8)

    # Check we have enough variation for GLCM
    unique_vals = np.unique(quantised[brain_mask])
    if len(unique_vals) < 2:
        return {k: 0.0 for k in ['glcm_contrast','glcm_homogeneity',
                                   'glcm_energy','glcm_correlation',
                                   'glcm_dissimilarity','glcm_asm']}

    # Compute GLCM at 4 angles, distance=1
    glcm = graycomatrix(
        quantised,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=64,
        symmetric=True,
        normed=True
    )

    return {
        'glcm_contrast':      float(graycoprops(glcm, 'contrast').mean()),
        'glcm_homogeneity':   float(graycoprops(glcm, 'homogeneity').mean()),
        'glcm_energy':        float(graycoprops(glcm, 'energy').mean()),
        'glcm_correlation':   float(graycoprops(glcm, 'correlation').mean()),
        'glcm_dissimilarity': float(graycoprops(glcm, 'dissimilarity').mean()),
        'glcm_asm':           float(graycoprops(glcm, 'ASM').mean()),
    }


# ─────────────────────────────────────────────────────────────────
# GROUP C — HISTOGRAM FEATURES (4)
# ─────────────────────────────────────────────────────────────────

def extract_histogram_features(hu_array, brain_mask):
    brain_hu = hu_array[brain_mask].astype(np.float64)

    if len(brain_hu) == 0:
        return {'hist_entropy': 0.0, 'hist_energy': 0.0,
                'hist_peak_hu': 0.0, 'hist_iqr': 0.0}

    hist, bin_edges = np.histogram(brain_hu, bins=128, range=(-100, 200))
    hist_norm = hist / (hist.sum() + 1e-10)

    epsilon      = 1e-10
    hist_entropy = float(-np.sum(hist_norm * np.log(hist_norm + epsilon)))
    hist_energy  = float(np.sum(hist_norm ** 2))

    peak_idx     = int(np.argmax(hist_norm))
    hist_peak_hu = float((bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2)

    hist_iqr     = float(np.percentile(brain_hu, 75) -
                         np.percentile(brain_hu, 25))

    return {
        'hist_entropy':  hist_entropy,
        'hist_energy':   hist_energy,
        'hist_peak_hu':  hist_peak_hu,
        'hist_iqr':      hist_iqr,
    }


# ─────────────────────────────────────────────────────────────────
# GROUP D — TISSUE FRACTION FEATURES (5)
# ─────────────────────────────────────────────────────────────────

def extract_tissue_fraction_features(hu_array, brain_mask):
    """
    5 features: proportion of brain pixels in each tissue HU range.

    high_hu_fraction is computed on a further-eroded inner mask
    to avoid skull-adjacent partial-volume artifacts contributing
    false positives in the 55-90 HU blood range.
    """
    from skimage.morphology import erosion, disk

    total = brain_mask.sum()
    if total == 0:
        return {k: 0.0 for k in ['air_fraction', 'csf_fraction',
                                   'brain_fraction', 'high_hu_fraction',
                                   'bone_fraction']}

    brain_hu = hu_array[brain_mask]

    # For high_hu_fraction specifically, use an inner mask
    # with additional 3-pixel erosion to exclude skull-adjacent pixels
    # This is the most important feature and must not be contaminated
    inner_mask = erosion(brain_mask, disk(3))
    inner_total = inner_mask.sum()

    if inner_total > 500:
        inner_hu = hu_array[inner_mask]
        high_hu_fraction = float(
            np.sum((inner_hu >= 55) & (inner_hu <= 90)) / inner_total
        )
    else:
        # Inner mask too small — fall back to full mask
        high_hu_fraction = float(
            np.sum((brain_hu >= 55) & (brain_hu <= 90)) / total
        )

    return {
        'air_fraction':      float(np.sum(brain_hu < -900)                      / total),
        'csf_fraction':      float(np.sum((brain_hu >= -10) & (brain_hu < 15))  / total),
        'brain_fraction':    float(np.sum((brain_hu >= 15)  & (brain_hu < 55))  / total),
        'high_hu_fraction':  high_hu_fraction,
        'bone_fraction':     float(np.sum(brain_hu > 150)                       / total),
    }
# ─────────────────────────────────────────────────────────────────
# GROUP E — IMAGE QUALITY FEATURES (2)
# ─────────────────────────────────────────────────────────────────

def extract_quality_features(hu_array, brain_mask):
    """
    2 features capturing scanner-specific image characteristics.

    sharpness:      Laplacian variance on brain-windowed image
    noise_estimate: std of HU in pure air background

    Pure air in a CT scan = HU range -1100 to -900
    Using < -900 catches both air and the transition zone
    Using < -800 is stricter and avoids partial-volume pixels
    After sentinel fix (values < -1500 replaced with -1000),
    pure air pixels cluster tightly around -1000 HU
    Expected noise_estimate: 5–30 HU for a clean air region
    """
    # Sharpness: Laplacian variance on brain-windowed image
    windowed       = apply_brain_window(hu_array)
    windowed_uint8 = (windowed * 255).astype(np.uint8)
    laplacian      = cv2.Laplacian(windowed_uint8, cv2.CV_64F)
    sharpness      = float(laplacian.var())

    # Noise: std of pixels in PURE air background
    # Strict range: -1100 to -900 HU
    # Excludes transition zone pixels and any remaining artifacts
    # Also excludes the -3024 sentinel pixels (now replaced with -1000,
    # but use strict range to be safe)
    air_mask = (
        (~brain_mask) &
        (hu_array < -900) &
        (hu_array > -1100)
    )

    if air_mask.sum() > 200:
        noise_estimate = float(np.std(hu_array[air_mask]))
    else:
        # Not enough clean air pixels in this slice
        # Use a fallback: std of all pixels below -900 outside brain
        fallback_mask = (~brain_mask) & (hu_array < -900) & (hu_array > -1500)
        if fallback_mask.sum() > 50:
            noise_estimate = float(np.std(hu_array[fallback_mask]))
        else:
            noise_estimate = 0.0

    return {
        'sharpness':       sharpness,
        'noise_estimate':  noise_estimate,
    }


# ─────────────────────────────────────────────────────────────────
# COMBINED EXTRACTOR
# ─────────────────────────────────────────────────────────────────

def extract_all_features(hu_array, brain_mask):
    features = {}
    features.update(extract_intensity_features(hu_array, brain_mask))
    features.update(extract_glcm_features(hu_array, brain_mask))
    features.update(extract_histogram_features(hu_array, brain_mask))
    features.update(extract_tissue_fraction_features(hu_array, brain_mask))
    features.update(extract_quality_features(hu_array, brain_mask))
    return features