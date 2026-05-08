"""
utils/augmentation.py
Synthetic BRIVO-Haemorrhage vector generation for Experiment 6.
"""
import numpy as np
import pandas as pd

FRACTION_FEATURES = [
    'air_fraction_mean', 'air_fraction_max',
    'csf_fraction_mean', 'csf_fraction_max',
    'brain_fraction_mean', 'brain_fraction_max',
    'high_hu_fraction_mean', 'high_hu_fraction_max',
    'bone_fraction_mean', 'bone_fraction_max'
]


def generate_synthetic_brivo_haem(df, feature_cols, scanner_offset_path, n_synthetic=5, random_state=42):
    """
    For each Revolution Haemorrhage patient, add the BRIVO-Revolution
    scanner offset to create a synthetic BRIVO-style Haemorrhage vector.
    
    Returns: synthetic_df with same columns as df (feature_cols + metadata)
    """
    offset_df = pd.read_csv(scanner_offset_path)
    offset_dict = dict(zip(offset_df['feature'], offset_df['scanner_offset']))

    # Base: Revolution Haemorrhage patients only
    rev_haem = df[(df['scanner_label'] == 1) & (df['class_label'] == 1)].copy()
    rng = np.random.RandomState(random_state)

    synthetic_rows = []
    for i, (_, row) in enumerate(rev_haem.iterrows()):
        base = row[feature_cols].values.astype(float).copy()
        # Add scanner offset (BRIVO - Revolution) to simulate BRIVO acquisition
        for j, feat in enumerate(feature_cols):
            base[j] += offset_dict.get(feat, 0.0)
        # Clip fraction features to valid [0, 1] range
        for feat in FRACTION_FEATURES:
            if feat in feature_cols:
                idx = list(feature_cols).index(feat)
                base[idx] = np.clip(base[idx], 0.0, 1.0)
        
        synth_row = dict(zip(feature_cols, base))
        synth_row['patient_id'] = f'SYNTH_{i:03d}'
        synth_row['class_label'] = 1
        synth_row['class_name'] = 'Haemorrhage'
        synth_row['scanner_model'] = 'BRIVO_synthetic'
        synth_row['scanner_label'] = 0
        synth_row['n_slices'] = row['n_slices']
        synthetic_rows.append(synth_row)

    synth_df = pd.DataFrame(synthetic_rows)
    print(f"  Generated {len(synth_df)} synthetic BRIVO-Haemorrhage vectors "
          f"(one per Revolution Haem patient)")
    return synth_df
