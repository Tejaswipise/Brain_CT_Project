# run_before_zip.py
import os
required = [
    'config.py',
    'stage1_feature_extraction.py',
    'stage2_patient_aggregation.py',
    'utils/__init__.py',
    'utils/dicom_loader.py',
    'utils/feature_extractor.py',
    'data/validated_metadata_clean.csv',
    'data/features_slices.csv',
    'data/features_patients.csv',
    'requirements.txt',
]
all_ok = True
for f in required:
    exists = os.path.exists(f)
    print(f"{'✓' if exists else '✗'} {f}")
    if not exists:
        all_ok = False
print('\n✅ All files present.' if all_ok else '\n❌ Missing files. Do not zip yet.')