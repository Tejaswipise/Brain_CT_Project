"""
stage4_shap_analysis.py  —  Lahari
SHAP stability analysis across LOOCV folds.

Requires Stage 3 to have run first (needs features_patients.csv and
models/exp2_RandomForest_full.pkl).

Usage (from project root):
    python stage4_shap_analysis.py
"""
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES_PATIENTS, RESULTS_DIR, MODELS_DIR, DATA_DIR, SCANNER_OFFSET_VECTOR
from utils.evaluation import get_classifiers


# ══════════════════════════════════════════════════════════════════
# STEP 4.1 — SHAP stability across LOOCV folds
# ══════════════════════════════════════════════════════════════════
def run_shap_stability(df, feature_cols, top_n=10):
    """
    For each LOOCV fold:
      - Train RF on n-1 patients
      - Compute SHAP values for the held-out patient
      - Record the top_n most important features for that fold

    After all folds:
      - Count how often each feature appears in the top_n
      - Features appearing in ≥70% of folds → ROBUST
      - Features appearing in 40–69% of folds → MODERATE
      - Features appearing in <40% of folds → UNSTABLE

    Returns: stability_df
    """
    print("\n" + "="*60)
    print("STAGE 4: SHAP Stability Analysis")
    print(f"  Running LOOCV SHAP on {len(df)} patients, top {top_n} features per fold")
    print("="*60)

    feature_cols = list(feature_cols)  # ensure plain list for indexing
    X = df[feature_cols].values
    y = df['class_label'].values
    patient_ids = df['patient_id'].values
    n_folds = len(df)

    # feature_count[i] = number of folds where feature i was in top_n
    feature_count = np.zeros(len(feature_cols), dtype=int)
    fold_shap_records = []  # for detailed output

    loo = LeaveOneOut()
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train_enc)

        # SHAP TreeExplainer (fast for RF)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)

        # Handle both old API (list of 2D arrays) and new API (3D array)
        if isinstance(shap_values, list):
            # Old: list[class0, class1], each shape (n_samples, n_features)
            sv = shap_values[1]   # class 1 = Haemorrhage
            abs_shap = np.abs(sv[0])
        elif shap_values.ndim == 3:
            # New: shape (n_samples, n_features, n_classes)
            abs_shap = np.abs(shap_values[0, :, 1])
        else:
            abs_shap = np.abs(shap_values[0])
        top_indices = np.argsort(abs_shap)[::-1][:top_n]
        feature_count[top_indices] += 1

        top_feat_names = [feature_cols[int(i)] for i in top_indices]
        top_feat_vals  = [float(abs_shap[int(i)]) for i in top_indices]
        print(f"  Fold {fold+1:2d}  patient={patient_ids[test_idx[0]]:20s}  "
              f"top feature: {top_feat_names[0]} ({top_feat_vals[0]:.4f})")

        fold_shap_records.append({
            'fold': fold,
            'patient_id': patient_ids[test_idx[0]],
            'top_features': top_feat_names,
            'top_shap_vals': top_feat_vals
        })

    # ── Build stability table ──────────────────────────────────────
    stability_rows = []
    for i, feat in enumerate(feature_cols):
        count = feature_count[i]
        pct = count / n_folds
        if pct >= 0.70:
            label = 'ROBUST'
        elif pct >= 0.40:
            label = 'MODERATE'
        else:
            label = 'UNSTABLE'
        stability_rows.append({
            'feature': feat,
            'fold_count': count,
            'fold_pct': round(pct, 3),
            'stability': label
        })

    stability_df = pd.DataFrame(stability_rows).sort_values('fold_count', ascending=False)
    out_path = RESULTS_DIR / 'shap_stability_table.csv'
    stability_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    print(f"\n  Feature Stability Summary:")
    print(f"  ROBUST   (≥70% folds): {(stability_df['stability']=='ROBUST').sum()} features")
    print(f"  MODERATE (40-69%):     {(stability_df['stability']=='MODERATE').sum()} features")
    print(f"  UNSTABLE (<40%):       {(stability_df['stability']=='UNSTABLE').sum()} features")
    print(f"\n  Top 15 most stable features:")
    print(stability_df.head(15)[['feature','fold_count','fold_pct','stability']].to_string(index=False))

    return stability_df, fold_shap_records


# ══════════════════════════════════════════════════════════════════
# STEP 4.2 — Scanner vs Haemorrhage feature overlap
# ══════════════════════════════════════════════════════════════════
def scanner_haem_overlap(stability_df, top_scanner_n=10):
    """
    Compare ROBUST haemorrhage features vs top scanner fingerprint features.
    Overlap = bias evidence.
    """
    print("\n" + "="*60)
    print("STEP 4.2: Scanner-Haemorrhage Feature Overlap")
    print("="*60)

    if not SCANNER_OFFSET_VECTOR.exists():
        print("  WARNING: scanner_offset_vector.csv not found. Run Stage 3 first.")
        return

    offset_df = pd.read_csv(SCANNER_OFFSET_VECTOR)
    robust_feats   = set(stability_df[stability_df['stability'] == 'ROBUST']['feature'])
    top_scanner    = set(offset_df.nlargest(top_scanner_n, 'abs_offset')['feature'])
    overlap        = robust_feats & top_scanner

    print(f"  Robust haemorrhage features : {len(robust_feats)}")
    print(f"  Top {top_scanner_n} scanner fingerprint features: {len(top_scanner)}")
    print(f"  OVERLAP                     : {len(overlap)}")

    if overlap:
        print(f"\n  ⚠ Overlapping features (evidence of scanner bias):")
        for f in sorted(overlap):
            print(f"    {f}")
    else:
        print(f"\n  ✓ No overlap — robust haemorrhage features differ from scanner fingerprint")

    return overlap


# ══════════════════════════════════════════════════════════════════
# STEP 4.3 — SHAP summary bar plot (full dataset model)
# ══════════════════════════════════════════════════════════════════
def plot_shap_summary(df, feature_cols):
    """
    Train RF on all 10 patients, compute SHAP on training data,
    save a mean |SHAP| bar chart.
    """
    print("\n  Generating SHAP summary bar plot...")
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df['class_label'].values)

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    clf.fit(X, y)

    explainer  = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    elif shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    mean_abs = np.abs(sv).mean(axis=0)
    feat_importance = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    feat_importance.head(20).plot(kind='barh', ax=ax, color='steelblue')
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Top 20 Features by Mean |SHAP| (Full Dataset RF)')
    plt.tight_layout()
    out = RESULTS_DIR / 'plots' / 'shap_summary_bar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════
# STEP 4.4 — Stability bar chart
# ══════════════════════════════════════════════════════════════════
def plot_stability_chart(stability_df):
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = stability_df.head(20).copy()
    colors = top20['stability'].map({'ROBUST': '#2ecc71', 'MODERATE': '#f39c12', 'UNSTABLE': '#e74c3c'})
    ax.barh(top20['feature'], top20['fold_pct'], color=colors)
    ax.axvline(0.70, color='green', linestyle='--', label='ROBUST threshold (70%)')
    ax.axvline(0.40, color='orange', linestyle='--', label='MODERATE threshold (40%)')
    ax.invert_yaxis()
    ax.set_xlabel('Fraction of LOOCV folds in top-10')
    ax.set_title('SHAP Feature Stability Across LOOCV Folds (Top 20)')
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / 'plots' / 'shap_stability_bar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "#"*60)
    print("  STAGE 4 — SHAP STABILITY ANALYSIS")
    print("#"*60)

    df = pd.read_csv(FEATURES_PATIENTS)
    feature_cols = list(df.columns[df.columns.str.endswith('_mean') | df.columns.str.endswith('_max')])
    print(f"\nLoaded {len(df)} patients, {len(feature_cols)} features")

    stability_df, fold_records = run_shap_stability(df, feature_cols, top_n=10)
    scanner_haem_overlap(stability_df, top_scanner_n=10)
    plot_shap_summary(df, feature_cols)
    plot_stability_chart(stability_df)

    print("\nStage 4 complete.")
    print(f"  Outputs:")
    print(f"    results/shap_stability_table.csv")
    print(f"    results/plots/shap_summary_bar.png")
    print(f"    results/plots/shap_stability_bar.png")
