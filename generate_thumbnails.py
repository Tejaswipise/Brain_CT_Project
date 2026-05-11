import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from config import VALIDATED_METADATA, DATA_DIR
from utils.dicom_loader import load_hu_array, get_brain_mask

# Output folder
THUMBNAIL_DIR = Path('results/thumbnails')
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# LOAD METADATA
# ─────────────────────────────────────────────────────────────────

df = pd.read_csv(VALIDATED_METADATA)

# Filter to soft-tissue series only
df_soft = df[
    df['kernel_class'].str.lower().str.contains('soft', na=False)
].copy()

print(f"Loaded {len(df_soft)} soft-tissue slices from "
      f"{df_soft['patient_id'].nunique()} patients")

# ─────────────────────────────────────────────────────────────────
# HELPER: Save a CT slice as PNG
# ─────────────────────────────────────────────────────────────────

def save_slice_png(hu_array, filepath, mask=None, figsize=(3, 3)):
    """
    Save a CT slice as a PNG image.
    Applies brain window (0-80 HU).
    If mask provided, applies mask to show only brain region.
    """
    if mask is not None:
        display = hu_array.copy()
        display[~mask] = -1000  # set non-brain to air
    else:
        display = hu_array.copy()

    # Apply brain window
    display = np.clip(display, 0, 80)
    display = (display - 0) / (80 - 0)  # normalise to 0-1

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filepath, dpi=100, bbox_inches='tight',
                pad_inches=0, facecolor='black')
    plt.close()


# ─────────────────────────────────────────────────────────────────
# GENERATE THUMBNAILS FOR EACH PATIENT
# ─────────────────────────────────────────────────────────────────

patients = df_soft['patient_id'].unique()
normal_pixel_arrays = []  # collect normal patient arrays for composite

print("\nGenerating thumbnails...")

for patient_id in patients:
    print(f"  Processing: {patient_id}")

    patient_slices = df_soft[
        df_soft['patient_id'] == patient_id
    ].sort_values('instance_number')

    # Pick the middle slice of the volume
    n = len(patient_slices)
    middle_idx = n // 2
    middle_row = patient_slices.iloc[middle_idx]

    try:
        # Load HU array
        hu = load_hu_array(middle_row['file_path'])

        # Get brain mask
        mask = get_brain_mask(hu)

        if mask is None:
            # Middle slice rejected — try adjacent slices
            for offset in [1, -1, 2, -2, 3, -3]:
                try_idx = middle_idx + offset
                if 0 <= try_idx < n:
                    try_row = patient_slices.iloc[try_idx]
                    hu = load_hu_array(try_row['file_path'])
                    mask = get_brain_mask(hu)
                    if mask is not None:
                        print(f"    Used slice at offset {offset} from middle")
                        break

        if mask is None:
            print(f"    WARNING: Could not get valid mask for {patient_id}")
            continue

        # Save raw slice thumbnail
        raw_path = THUMBNAIL_DIR / f"{patient_id}_slice.png"
        save_slice_png(hu, raw_path)
        print(f"    Saved: {raw_path.name}")

        # Save masked slice thumbnail
        masked_path = THUMBNAIL_DIR / f"{patient_id}_masked.png"
        save_slice_png(hu, masked_path, mask=mask)
        print(f"    Saved: {masked_path.name}")

        # Collect Normal patient arrays for composite
        class_label = patient_slices['class_label'].iloc[0]
        if class_label == 0:  # Normal
            # Store the windowed, masked array for averaging
            windowed = np.clip(hu, 0, 80)
            windowed = (windowed - 0) / 80.0
            windowed[~mask] = 0  # zero out non-brain
            normal_pixel_arrays.append(windowed)

    except Exception as e:
        print(f"    ERROR on {patient_id}: {e}")
        continue

# ─────────────────────────────────────────────────────────────────
# GENERATE AVERAGE NORMAL BRAIN COMPOSITE
# ─────────────────────────────────────────────────────────────────

print(f"\nGenerating average normal brain composite "
      f"from {len(normal_pixel_arrays)} Normal patients...")

if len(normal_pixel_arrays) >= 2:
    # All arrays are 512x512 — stack and average
    stacked = np.stack(normal_pixel_arrays, axis=0)
    average = np.mean(stacked, axis=0)

    composite_path = THUMBNAIL_DIR / 'average_normal.png'

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(average, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(composite_path, dpi=100, bbox_inches='tight',
                pad_inches=0, facecolor='black')
    plt.close()

    print(f"Saved: {composite_path.name}")
else:
    print("WARNING: Not enough Normal patients to generate composite")

# ─────────────────────────────────────────────────────────────────
# VERIFY ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("VERIFICATION")
print("=" * 50)

expected_files = []
for pid in patients:
    expected_files.append(f"{pid}_slice.png")
    expected_files.append(f"{pid}_masked.png")
expected_files.append("average_normal.png")

all_ok = True
for fname in expected_files:
    fpath = THUMBNAIL_DIR / fname
    exists = fpath.exists()
    size   = fpath.stat().st_size if exists else 0
    status = "OK" if (exists and size > 1000) else "MISSING or TOO SMALL"
    print(f"  {status:25s} {fname}")
    if status != "OK":
        all_ok = False

print()
if all_ok:
    print("All thumbnails generated successfully.")
    print(f"Location: {THUMBNAIL_DIR.resolve()}")
    print("\nShare the entire results/thumbnails/ folder with Tejaswi.")
else:
    print("Some files missing. Check errors above.")