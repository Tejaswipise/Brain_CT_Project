import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

Path('results').mkdir(exist_ok=True)

df = pd.read_csv('data/data/features_patients.csv')
feat_cols = [c for c in df.columns if c.endswith('_mean') or c.endswith('_max')]

def extract_class1_shap(shap_values):
    """Handle both old (list) and new (3D array) SHAP output formats."""
    if isinstance(shap_values, list):
        # Old format: list of [class0_array, class1_array]
        return shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        # New format: single 3D array (n_samples, n_features, n_classes)
        return shap_values[:, :, 1]
    else:
        # Already 2D
        return shap_values

# ── Full dataset SHAP ──────────────────────────────────────────
print("Computing SHAP for full dataset model...")
X_full = df[feat_cols].values
y_full = df['class_label'].values

rf_full = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_full.fit(X_full, y_full)

explainer_full = shap.TreeExplainer(rf_full)
raw_full = explainer_full.shap_values(X_full)
sv_full = extract_class1_shap(raw_full)

print(f"  SHAP values shape: {sv_full.shape}")  # should be (10, 50)

df_shap_full = pd.DataFrame(sv_full, columns=feat_cols)
df_shap_full.insert(0, 'patient_id', df['patient_id'].values)
df_shap_full.to_csv('results/shap_values_full.csv', index=False)
print(f"  Saved: results/shap_values_full.csv")

# ── Revolution-only SHAP ───────────────────────────────────────
print("Computing SHAP for Revolution-only model...")
df_rev = df[df['scanner_label'] == 1].copy()
X_rev = df_rev[feat_cols].values
y_rev = df_rev['class_label'].values

rf_rev = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_rev.fit(X_rev, y_rev)

explainer_rev = shap.TreeExplainer(rf_rev)
raw_rev = explainer_rev.shap_values(X_rev)
sv_rev = extract_class1_shap(raw_rev)

print(f"  SHAP values shape: {sv_rev.shape}")  # should be (7, 50)

df_shap_rev = pd.DataFrame(sv_rev, columns=feat_cols)
df_shap_rev.insert(0, 'patient_id', df_rev['patient_id'].values)
df_shap_rev.to_csv('results/shap_values_revolution.csv', index=False)
print(f"  Saved: results/shap_values_revolution.csv")

print("\nDone.")