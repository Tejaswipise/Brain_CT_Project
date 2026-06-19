import os

required = [
    'stage5_streamlit_app.py',
    'models/exp2_RandomForest_full.pkl',
    'models/exp3_RevOnly_RandomForest.pkl',
    'results/experiment_results.csv',
    'results/shap_stability_table.csv',
    'results/plots/shap_summary_bar.png',
    'results/plots/shap_stability_bar.png',
    'results/thumbnails/CT180-01-2026_slice.png',
    'results/thumbnails/CT20015788_slice.png',
    'results/thumbnails/CT3225-12-2025_slice.png',
    'data/data/features_patients.csv',
]

print("=== Tejaswi File Verification ===\n")
all_ok = True
for f in required:
    exists = os.path.exists(f)
    print(f"{'OK     ' if exists else 'MISSING'} {f}")
    if not exists:
        all_ok = False

print()
if all_ok:
    print("All files present. Safe to run the app.")
else:
    print("Some files missing. Fix the MISSINGs before running.")