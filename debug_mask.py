# debug_mask.py
# Run this before full Stage 1 to verify masks look correct

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from utils.dicom_loader import load_hu_array, get_brain_mask

df = pd.read_csv('data/validated_metadata_clean.csv')

# Pick one slice from BRIVO Normal (middle of volume, not edge)
brivo_slices = df[df['scanner_model'].str.contains('BRIVO')].sort_values(
    ['patient_id', 'instance_number'])
# Take a middle slice from CT20015788
brivo_mid = df[(df['patient_id'] == 'CT20015788') &
               (df['instance_number'] == 15)]['file_path'].values
if len(brivo_mid) == 0:
    brivo_mid = df[df['patient_id'] == 'CT20015788']['file_path'].values[14:15]

# Pick one slice from Haemorrhage Revolution
haem_mid = df[(df['patient_id'] == 'CT3225-12-2025') &
              (df['instance_number'] == 15)]['file_path'].values
if len(haem_mid) == 0:
    haem_mid = df[df['patient_id'] == 'CT3225-12-2025']['file_path'].values[14:15]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for row_idx, (fpath_arr, label) in enumerate([
    (brivo_mid, 'BRIVO Normal CT20015788'),
    (haem_mid,  'Revolution Haemorrhage CT3225')
]):
    if len(fpath_arr) == 0:
        print(f"Could not find slice for {label}")
        continue

    fpath = fpath_arr[0]
    hu    = load_hu_array(fpath)
    mask  = get_brain_mask(hu)

    # Panel 1: Raw HU clipped to brain window
    axes[row_idx, 0].imshow(np.clip(hu, 0, 80), cmap='gray')
    axes[row_idx, 0].set_title(f'{label}\nRaw HU (0-80 window)')
    axes[row_idx, 0].axis('off')

    if mask is not None:
        # Panel 2: Brain mask
        axes[row_idx, 1].imshow(mask, cmap='gray')
        axes[row_idx, 1].set_title(f'Brain mask\n{mask.sum()} pixels')
        axes[row_idx, 1].axis('off')

        # Panel 3: Masked HU — what features are computed on
        masked_hu = hu.copy()
        masked_hu[~mask] = -1000
        axes[row_idx, 2].imshow(np.clip(masked_hu, 0, 80), cmap='gray')

        # Highlight pixels in high_hu range (55-90 HU) in red
        high_hu_pixels = (masked_hu >= 55) & (masked_hu <= 90)
        overlay = np.zeros((*hu.shape, 3))
        overlay[..., 0] = np.clip(masked_hu, 0, 80) / 80  # grey base
        overlay[..., 1] = np.clip(masked_hu, 0, 80) / 80
        overlay[..., 2] = np.clip(masked_hu, 0, 80) / 80
        overlay[high_hu_pixels, 0] = 1.0  # red for high HU
        overlay[high_hu_pixels, 1] = 0.0
        overlay[high_hu_pixels, 2] = 0.0

        axes[row_idx, 2].imshow(overlay)
        n_high = high_hu_pixels.sum()
        frac   = n_high / mask.sum() if mask.sum() > 0 else 0
        axes[row_idx, 2].set_title(f'Masked HU (red=55-90 HU)\n'
                                    f'high_hu pixels: {n_high} ({frac:.4f})')
        axes[row_idx, 2].axis('off')

        print(f"\n{label}:")
        print(f"  Mask pixels: {mask.sum()}")
        print(f"  High HU (55-90) pixels: {high_hu_pixels.sum()}")
        print(f"  high_hu_fraction: {frac:.4f}")
        brain_in_mask = hu[mask]
        print(f"  HU range in mask: {brain_in_mask.min():.0f} to {brain_in_mask.max():.0f}")
        print(f"  HU p90 in mask: {np.percentile(brain_in_mask, 90):.1f}")
    else:
        axes[row_idx, 1].set_title('Mask: None (skipped)')
        axes[row_idx, 2].set_title('N/A')

plt.tight_layout()
plt.savefig('results/plots/debug_masks.png', dpi=100, bbox_inches='tight')
print("\nSaved: results/plots/debug_masks.png")
print("Open this image to visually verify the brain masks look correct")
print("Red pixels = high HU (55-90 HU) — should be MORE red in Haemorrhage slice")