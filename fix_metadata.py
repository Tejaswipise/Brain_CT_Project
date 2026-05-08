# fix_metadata.py
# Run this once. It produces a clean validated_metadata_clean.csv
# Use this file for all subsequent stages instead of validated_metadata.csv

import pandas as pd
from pathlib import Path

# Load original
df = pd.read_csv('data/validated_metadata.csv')
print(f"Original shape: {df.shape}")
print(f"Original slice counts per patient:")
print(df.groupby('patient_id')['file_path'].count().to_string())

# ─────────────────────────────────────────────────────────────────
# FIX 1: Remove Scout images
# These are localiser images, not diagnostic slices
# Identified by: missing SliceLocation AND file path contains 'Scout'
# ─────────────────────────────────────────────────────────────────

scout_mask = df['file_path'].str.contains('Scout', case=False, na=False)
print(f"\nRemoving {scout_mask.sum()} Scout slices")
df = df[~scout_mask].copy()

# ─────────────────────────────────────────────────────────────────
# FIX 2: Remove duplicate series for CT20015788
# Keep Plain-5mm2, remove Plain-5mm3
# Both series are identical scans of the same patient
# ─────────────────────────────────────────────────────────────────

duplicate_mask = (
    (df['patient_id'] == 'CT20015788') &
    (df['file_path'].str.contains('Plain-5mm3', case=False, na=False))
)
print(f"Removing {duplicate_mask.sum()} duplicate series slices (Plain-5mm3 for CT20015788)")
df = df[~duplicate_mask].copy()

# ─────────────────────────────────────────────────────────────────
# FIX 3: Assign kernel_class = soft for BRIVO patients
# Their DICOM headers do not store ConvolutionKernel
# but we know from folder names and clinical context these are
# standard soft-tissue brain series
# ─────────────────────────────────────────────────────────────────

brivo_missing_mask = (
    (df['scanner_model'].str.contains('BRIVO', case=False, na=False)) &
    (df['convolution_kernel'] == 'MISSING')
)
print(f"\nAssigning kernel_class=soft to {brivo_missing_mask.sum()} BRIVO slices")
df.loc[brivo_missing_mask, 'convolution_kernel'] = 'STND'
df.loc[brivo_missing_mask, 'kernel_class'] = 'soft'

# ─────────────────────────────────────────────────────────────────
# VERIFY the fix
# ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"AFTER FIXES")
print(f"{'='*60}")
print(f"\nNew shape: {df.shape}")

print(f"\nSlice counts per patient (should all be 28-38):")
counts = df.groupby(['patient_id','class_name',
                     'scanner_model'])['file_path'].count()
print(counts.reset_index(name='slice_count').to_string(index=False))

print(f"\nKernel check (no MISSING should remain):")
print(df.groupby(['class_name','convolution_kernel'])
        ['patient_id'].nunique()
        .reset_index(name='patient_count')
        .to_string(index=False))

print(f"\nMissing SliceLocation check:")
missing_loc = df[df['slice_location'].isna()]
print(f"Slices with missing SliceLocation: {len(missing_loc)}")

print(f"\nScanner distribution:")
print(df.groupby(['scanner_model','class_name'])
        ['patient_id'].nunique()
        .reset_index(name='patient_count')
        .to_string(index=False))

# ─────────────────────────────────────────────────────────────────
# SAVE clean version
# ─────────────────────────────────────────────────────────────────

output_path = 'data/validated_metadata_clean.csv'
df.to_csv(output_path, index=False)
print(f"\n{'='*60}")
print(f"Saved clean metadata: {output_path}")
print(f"Total slices: {len(df)}")
print(f"{'='*60}")