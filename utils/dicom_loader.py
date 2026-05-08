# utils/dicom_loader.py

import pydicom
import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import erosion, disk


def load_hu_array(file_path):
    """
    Load a DICOM file and return Hounsfield Unit array (float32).
    Fixes GE sentinel values: out-of-FOV pixels set to -3024 → replaced with -1000 (air).
    """
    ds = pydicom.dcmread(str(file_path), force=True)

    if not hasattr(ds, 'PixelData'):
        raise ValueError(f"No pixel data in {file_path}")

    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', -1024.0))

    hu = ds.pixel_array.astype(np.float32) * slope + intercept

    # GE scanners mark out-of-FOV pixels with very negative values (-3024 HU)
    # These are not real tissue — replace with -1000 (standard air HU)
    # Without this fix, noise_estimate picks up -3024 pixels and shows
    # impossibly large standard deviations (200–1000 instead of 5–25)
    hu[hu < -1500] = -1000

    return hu


def get_brain_mask(hu_array, min_brain_pixels=8000):
    """
    Extract binary brain mask from a CT slice.

    Steps:
    1. Threshold -20 to 150 HU (captures brain tissue and acute blood)
    2. Fill holes to make brain a solid region
    3. Keep largest connected component
    4. Remove bone pixels (>200 HU) that got included during hole-filling
    5. Erode 2 pixels to remove skull-adjacent partial-volume artifacts
    6. Reject slices with too few brain pixels (top/bottom of volume)
    """
    # Step 1: Soft tissue + blood threshold
    mask = (hu_array > -20) & (hu_array < 150)

    # Step 2: Fill holes
    mask = ndimage.binary_fill_holes(mask)

    # Step 3: Largest connected component
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return None

    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    mask = (labeled == largest.label)

    # Step 4: Remove bone pixels included by hole-filling
    bone_mask = hu_array > 200
    mask = mask & (~bone_mask)

    # Step 5: Erode to clean edges
    mask = erosion(mask, disk(5))

    # Step 6: Skip near-empty slices
    if mask.sum() < min_brain_pixels:
        return None

    return mask


def apply_brain_window(hu_array, window_center=40, window_width=80):
    """
    Apply standard brain CT window and normalise to [0, 1].
    Window: center=40 HU, width=80 HU → clips to [0, 80] HU.
    Used only for GLCM texture features.
    """
    window_min = window_center - window_width / 2   # 0 HU
    window_max = window_center + window_width / 2   # 80 HU
    clipped    = np.clip(hu_array, window_min, window_max)
    normalised = (clipped - window_min) / (window_max - window_min)
    return normalised.astype(np.float32)