import pandas as pd
import numpy as np
import sys
import traceback
from pathlib import Path

from config import VALIDATED_METADATA, FEATURES_SLICES
from utils.dicom_loader import load_hu_array, get_brain_mask
from utils.feature_extractor import extract_all_features


def run_stage1():

    print("=" * 60)
    print("STAGE 1 — FEATURE EXTRACTION")
    print("=" * 60)

    # ── Load metadata ──────────────────────────────────────────────
    df_meta = pd.read_csv(VALIDATED_METADATA)
    print(f"\nLoaded metadata: {len(df_meta)} slices, "
          f"{df_meta['patient_id'].nunique()} patients")

    # ── Filter to soft-tissue series only ─────────────────────────
    df_soft = df_meta[
        df_meta['kernel_class'].str.lower().str.contains('soft', na=False)
    ].copy()

    print(f"Soft-tissue slices to process: {len(df_soft)}")
    print(f"\nSlices per patient:")
    print(df_soft.groupby(['patient_id','class_name','scanner_model'])
                 ['file_path'].count()
                 .reset_index(name='slices')
                 .to_string(index=False))

    # ── Feature extraction loop ────────────────────────────────────
    all_records = []
    skipped     = 0
    errors      = 0

    total = len(df_soft)
    print(f"\nExtracting features from {total} slices...")
    print("Progress: ", end="", flush=True)

    for idx, (_, row) in enumerate(df_soft.iterrows()):

        # Progress indicator every 20 slices
        if idx % 20 == 0:
            print(f"{idx}/{total}", end=" ", flush=True)

        try:
            # Load HU array
            hu = load_hu_array(row['file_path'])

            # Get brain mask
            brain_mask = get_brain_mask(hu)

            if brain_mask is None:
                # Slice has too few brain pixels — near top/bottom of volume
                skipped += 1
                continue

            # Extract all 25 features
            features = extract_all_features(hu, brain_mask)

            # Add metadata columns needed for grouping in Stage 2
            features['file_path']          = row['file_path']
            features['patient_id']         = row['patient_id']
            features['class_label']        = int(row['class_label'])
            features['class_name']         = row['class_name']
            features['scanner_model']      = row['scanner_model']
            features['scanner_label']      = int(row['scanner_label'])
            features['instance_number']    = int(row['instance_number'])
            features['slice_location']     = row.get('slice_location', np.nan)
            features['convolution_kernel'] = row['convolution_kernel']

            all_records.append(features)

        except Exception as e:
            errors += 1
            print(f"\n  ERROR on slice {idx} "
                  f"({row.get('patient_id','?')}): {e}")
            continue

    print(f"\n\nExtraction complete.")
    print(f"  Slices processed successfully: {len(all_records)}")
    print(f"  Slices skipped (low brain content): {skipped}")
    print(f"  Errors: {errors}")

    if len(all_records) == 0:
        print("\nERROR: No features extracted. Check file paths in metadata.")
        return None

    # ── Build dataframe ────────────────────────────────────────────
    df_features = pd.DataFrame(all_records)

    # ── Sanity check ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    print(f"\nOutput shape: {df_features.shape}")
    print(f"Patients:     {df_features['patient_id'].nunique()}")

    print("\nSlices per patient (after skipping low-content slices):")
    print(df_features.groupby(['patient_id','class_name'])
                     ['file_path'].count()
                     .reset_index(name='slices')
                     .to_string(index=False))

    print("\nKey feature summary by class:")
    print("(high_hu_fraction should be higher in Haemorrhage)")
    summary_features = ['high_hu_fraction', 'hu_p90',
                        'noise_estimate', 'sharpness']
    print(df_features.groupby('class_name')[summary_features]
                     .mean().round(4).to_string())

    print("\nKey feature summary by scanner:")
    print("(noise_estimate and sharpness should differ between scanners)")
    scanner_features = ['noise_estimate', 'sharpness',
                        'glcm_homogeneity', 'glcm_contrast']
    print(df_features.groupby('scanner_model')[scanner_features]
                     .mean().round(4).to_string())

    print("\nNaN check:")
    feature_cols = [c for c in df_features.columns
                    if c not in ['file_path','patient_id','class_label',
                                 'class_name','scanner_model','scanner_label',
                                 'instance_number','slice_location',
                                 'convolution_kernel']]
    nan_counts = df_features[feature_cols].isna().sum()
    nan_total  = nan_counts.sum()
    if nan_total == 0:
        print("  No NaN values found ✓")
    else:
        print(f"  WARNING: {nan_total} NaN values found")
        print(nan_counts[nan_counts > 0])

    # ── Save output ────────────────────────────────────────────────
    df_features.to_csv(FEATURES_SLICES, index=False)

    print("\n" + "=" * 60)
    print(f"SAVED: {FEATURES_SLICES}")
    print(f"Shape: {df_features.shape}")
    print("=" * 60)

    return df_features


if __name__ == '__main__':
    run_stage1()