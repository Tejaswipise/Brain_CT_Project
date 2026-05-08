import pandas as pd
import numpy as np
from config import FEATURES_SLICES, FEATURES_PATIENTS

FEATURE_NAMES = [
    'hu_mean', 'hu_std', 'hu_skewness', 'hu_kurtosis',
    'hu_p10', 'hu_p25', 'hu_p75', 'hu_p90',
    'glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
    'glcm_correlation', 'glcm_dissimilarity', 'glcm_asm',
    'hist_entropy', 'hist_energy', 'hist_peak_hu', 'hist_iqr',
    'air_fraction', 'csf_fraction', 'brain_fraction',
    'high_hu_fraction', 'bone_fraction',
    'sharpness', 'noise_estimate'
]


def run_stage2():

    print("=" * 60)
    print("STAGE 2 — PATIENT-LEVEL AGGREGATION")
    print("=" * 60)

    df = pd.read_csv(FEATURES_SLICES)
    print(f"\nLoaded: {len(df)} slices from {df['patient_id'].nunique()} patients")

    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"\nERROR: Missing feature columns: {missing}")
        return None

    # ── Slice quality filter ───────────────────────────────────────
    # Remove slices with extreme high_hu_fraction values
    # Any slice where > 40% of brain pixels are in the 55-90 HU range
    # is almost certainly a bad edge slice or artifact, not real pathology
    # Even massive haemorrhage would not occupy 40% of the entire brain
    before = len(df)
    df = df[df['high_hu_fraction'] <= 0.40].copy()
    after = len(df)
    print(f"Slice quality filter: removed {before - after} slices "
          f"with high_hu_fraction > 0.40")
    print(f"Remaining slices: {after}")

    # Also remove slices where brain_fraction is extremely low
    # (less than 5% of mask pixels are in normal brain range)
    # These are transitional slices with mostly CSF or artifact
    before = after
    df = df[df['brain_fraction'] >= 0.05].copy()
    after = len(df)
    print(f"Brain content filter: removed {before - after} slices "
          f"with brain_fraction < 0.05")
    print(f"Remaining slices: {after}")

    print(f"\nSlices per patient after filtering:")
    print(df.groupby(['patient_id', 'class_name', 'scanner_model'])
            ['high_hu_fraction'].count()
            .reset_index(name='slices')
            .to_string(index=False))

    # ── Aggregate per patient ──────────────────────────────────────
    patient_records = []

    for patient_id, group in df.groupby('patient_id'):

        record = {}

        record['patient_id']    = patient_id
        record['class_label']   = int(group['class_label'].iloc[0])
        record['class_name']    = group['class_name'].iloc[0]
        record['scanner_model'] = group['scanner_model'].iloc[0]
        record['scanner_label'] = int(group['scanner_label'].iloc[0])
        record['n_slices']      = len(group)

        for feat in FEATURE_NAMES:
            vals = group[feat].dropna().values

            if len(vals) == 0:
                record[f'{feat}_mean'] = np.nan
                record[f'{feat}_max']  = np.nan
                continue

            record[f'{feat}_mean'] = float(np.mean(vals))

            # For max aggregation use 95th percentile instead of true max
            # This avoids single outlier slices dominating the patient value
            # 95th percentile = typical value of the top 5% of slices
            # For a patient with 28 slices, this is roughly the top 1-2 slices
            record[f'{feat}_max']  = float(np.percentile(vals, 95))

        patient_records.append(record)

    df_patients = pd.DataFrame(patient_records)

    # ── Sanity check ───────────────────────────────────────────────
    feature_cols = [c for c in df_patients.columns
                    if c.endswith('_mean') or c.endswith('_max')]

    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    print(f"\nOutput shape: {df_patients.shape}")
    print(f"Patients:     {len(df_patients)}")
    print(f"Features:     {len(feature_cols)}")

    print("\nPatient summary:")
    print(df_patients[['patient_id', 'class_name', 'scanner_model',
                        'n_slices']].to_string(index=False))

    print("\n--- high_hu_fraction_mean by patient ---")
    print(df_patients[['patient_id', 'class_name', 'scanner_model',
                        'high_hu_fraction_mean']]
          .sort_values('high_hu_fraction_mean', ascending=False)
          .to_string(index=False))

    print("\n--- high_hu_fraction_max (95th pct) by patient ---")
    print(df_patients[['patient_id', 'class_name', 'scanner_model',
                        'high_hu_fraction_max']]
          .sort_values('high_hu_fraction_max', ascending=False)
          .to_string(index=False))

    print("\n--- hu_p90_max (95th pct) by patient ---")
    print(df_patients[['patient_id', 'class_name', 'scanner_model',
                        'hu_p90_max']]
          .sort_values('hu_p90_max', ascending=False)
          .to_string(index=False))

    print("\n--- Scanner fingerprint features ---")
    print(df_patients.groupby('scanner_model')[
        ['sharpness_mean', 'noise_estimate_mean',
         'glcm_contrast_mean', 'glcm_homogeneity_mean']
    ].mean().round(4).to_string())

    print("\n--- NaN check ---")
    nan_total = df_patients[feature_cols].isna().sum().sum()
    if nan_total == 0:
        print("No NaN values found ✓")
    else:
        print(f"WARNING: {nan_total} NaN values")
        print(df_patients[feature_cols].isna().sum()[
            df_patients[feature_cols].isna().sum() > 0
        ])

    # ── Final go/no-go ─────────────────────────────────────────────
    print("\n--- GO / NO-GO ---")

    checks = []

    # Check 1: correct number of patients
    checks.append(("10 patients in output",
                   len(df_patients) == 10))

    # Check 2: no NaN
    checks.append(("No NaN values",
                   nan_total == 0))

    # Check 3: high_hu_fraction_max values are reasonable (below 0.5)
    max_val = df_patients['high_hu_fraction_max'].max()
    checks.append((f"high_hu_fraction_max <= 0.5 (actual max={max_val:.3f})",
                   max_val <= 0.5))

    # Check 4: scanner features differ
    brivo_sharp = df_patients[
        df_patients['scanner_model'].str.contains('BRIVO')
    ]['sharpness_mean'].mean()
    rev_sharp = df_patients[
        df_patients['scanner_model'].str.contains('Revolution')
    ]['sharpness_mean'].mean()
    checks.append((f"Sharpness differs by scanner "
                   f"(BRIVO={brivo_sharp:.0f}, Rev={rev_sharp:.0f})",
                   abs(brivo_sharp - rev_sharp) > 50))

    all_pass = True
    for label, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {label}")
        if not result:
            all_pass = False

    if all_pass:
        print("\n✅ Stage 2 PASSED. Ready for Stage 3.")
    else:
        print("\n❌ Issues found. Check above before Stage 3.")

    # ── Save ───────────────────────────────────────────────────────
    df_patients.to_csv(FEATURES_PATIENTS, index=False)
    print(f"\nSAVED: {FEATURES_PATIENTS}")
    print(f"Shape: {df_patients.shape}")

    return df_patients


if __name__ == '__main__':
    run_stage2()