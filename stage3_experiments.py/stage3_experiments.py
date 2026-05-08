"""
stage3_experiments.py  —  Lahari
Run all 6 bias-aware experiments and save results.

Usage (from project root):
    python stage3_experiments.py
"""
import sys
import os
import numpy as np
import pandas as pd
import joblib
import copy
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ── project imports ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES_PATIENTS, RESULTS_DIR, MODELS_DIR, DATA_DIR, SCANNER_OFFSET_VECTOR
from utils.evaluation import get_classifiers, run_loocv, compute_bootstrap_auc, evaluate_and_print
from utils.augmentation import generate_synthetic_brivo_haem

# ── ensure output dirs exist ───────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'confusion_matrices').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'plots').mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
def load_patient_data():
    df = pd.read_csv(FEATURES_PATIENTS)
    feature_cols = [c for c in df.columns if c.endswith('_mean') or c.endswith('_max')]
    print(f"\nLoaded {len(df)} patients, {len(feature_cols)} features")
    print(f"Class distribution: {df['class_name'].value_counts().to_dict()}")
    print(f"Scanner distribution:")
    print(df.groupby(['scanner_model', 'class_name']).size().to_string())
    return df, feature_cols


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Scanner-Only Baseline (rule-based, no ML)
# ══════════════════════════════════════════════════════════════════
def experiment1_scanner_baseline(df):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Scanner-Only Baseline")
    print("Confound status: CONFOUNDED (intentional ceiling)")
    print("="*60)

    y_true = df['class_label'].values
    # Rule: BRIVO (scanner_label=0) → Normal (0)
    #       Revolution (scanner_label=1) → Haemorrhage (1)
    y_pred = df['scanner_label'].values
    y_prob = y_pred.astype(float)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    print(f"  Rule: BRIVO→Normal, Revolution→Haemorrhage")
    print(f"  Accuracy: {acc:.3f}   AUC: {auc:.3f}")
    print(f"\n  Per-patient:")
    for _, row in df.iterrows():
        pred = int(row['scanner_label'])
        true = int(row['class_label'])
        marker = "✓" if pred == true else "✗"
        print(f"    {marker}  {row['patient_id']:20s}  scanner={row['scanner_model']:15s}  "
              f"true={true}  pred={pred}")

    print(f"\n  INTERPRETATION: Classifiers that don't substantially beat "
          f"AUC={auc:.3f} on the full dataset are dominated by scanner shortcuts.")

    return [{
        'experiment': 'Exp1_Scanner_Baseline',
        'confound_status': 'CONFOUNDED',
        'n_patients': len(df),
        'classifier': 'RuleBase',
        'auc_mean': auc, 'ci_lower': np.nan, 'ci_upper': np.nan,
        'sensitivity': np.nan, 'specificity': np.nan,
        'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0
    }]


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Full Dataset Classifier (all 10 patients)
# ══════════════════════════════════════════════════════════════════
def experiment2_full_classifier(df, feature_cols):
    print("\n" + "="*60)
    print("EXPERIMENT 2: Full Dataset Classifier")
    print("Confound status: CONFOUNDED")
    print("="*60)

    X = df[feature_cols].values
    y = df['class_label'].values
    patient_ids = df['patient_id'].values
    all_results = []

    for clf_name, clf in get_classifiers().items():
        print(f"\n  --- {clf_name} ---")
        use_scaler = (clf_name == 'SVM')
        fold_results, y_true_all, y_prob_all = run_loocv(X, y, patient_ids, clf, use_scaler)
        metrics = evaluate_and_print(
            f"Exp2 Full Dataset - {clf_name}", "CONFOUNDED",
            y_true_all, y_prob_all, patient_ids, fold_results
        )
        # Save model trained on full data
        _save_model(X, y, clf_name, f'exp2_{clf_name}_full', use_scaler)

        result = {'experiment': f'Exp2_Full_{clf_name}', 'confound_status': 'CONFOUNDED',
                  'n_patients': len(df), 'classifier': clf_name}
        result.update(metrics)
        all_results.append(result)

    return all_results


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Revolution-Only Classifier (7 patients)
# ══════════════════════════════════════════════════════════════════
def experiment3_revolution_only(df, feature_cols):
    print("\n" + "="*60)
    print("EXPERIMENT 3: Revolution-Only Classifier")
    print("Confound status: REDUCED CONFOUND")
    print("WARNING: Only 2 Revolution Normal patients — specificity unreliable")
    print("="*60)

    df_rev = df[df['scanner_label'] == 1].copy()
    print(f"  Using {len(df_rev)} patients: "
          f"{(df_rev['class_label']==0).sum()} Normal, "
          f"{(df_rev['class_label']==1).sum()} Haemorrhage")

    X = df_rev[feature_cols].values
    y = df_rev['class_label'].values
    patient_ids = df_rev['patient_id'].values
    all_results = []

    for clf_name, clf in get_classifiers().items():
        print(f"\n  --- {clf_name} ---")
        use_scaler = (clf_name == 'SVM')
        fold_results, y_true_all, y_prob_all = run_loocv(X, y, patient_ids, clf, use_scaler)
        metrics = evaluate_and_print(
            f"Exp3 Revolution-Only - {clf_name}", "REDUCED CONFOUND",
            y_true_all, y_prob_all, patient_ids, fold_results
        )
        result = {'experiment': f'Exp3_RevOnly_{clf_name}', 'confound_status': 'REDUCED_CONFOUND',
                  'n_patients': len(df_rev), 'classifier': clf_name}
        result.update(metrics)
        all_results.append(result)

    return all_results


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Scanner Fingerprint + Offset Vector
# ══════════════════════════════════════════════════════════════════
def experiment4_scanner_fingerprint(df, feature_cols):
    print("\n" + "="*60)
    print("EXPERIMENT 4: Scanner Fingerprint Classifier")
    print("Confound status: CLEAN (Normal patients only, no pathology)")
    print("="*60)

    df_normal = df[df['class_label'] == 0].copy()
    print(f"  Using {len(df_normal)} Normal patients: "
          f"{(df_normal['scanner_label']==0).sum()} BRIVO, "
          f"{(df_normal['scanner_label']==1).sum()} Revolution")

    X = df_normal[feature_cols].values
    y = df_normal['scanner_label'].values   # TARGET = scanner, not disease
    patient_ids = df_normal['patient_id'].values

    # ── compute scanner offset vector ─────────────────────────────
    brivo_feats = df_normal[df_normal['scanner_label'] == 0][feature_cols].values
    rev_feats   = df_normal[df_normal['scanner_label'] == 1][feature_cols].values
    brivo_mean = np.mean(brivo_feats, axis=0)
    rev_mean   = np.mean(rev_feats,   axis=0)
    scanner_offset = brivo_mean - rev_mean

    offset_df = pd.DataFrame({
        'feature': feature_cols,
        'brivo_mean': brivo_mean,
        'revolution_mean': rev_mean,
        'scanner_offset': scanner_offset,
        'abs_offset': np.abs(scanner_offset)
    }).sort_values('abs_offset', ascending=False)

    offset_df.to_csv(SCANNER_OFFSET_VECTOR, index=False)
    print(f"\n  Top 5 features with largest scanner offset:")
    print(offset_df[['feature', 'brivo_mean', 'revolution_mean', 'scanner_offset']].head(5).to_string(index=False))
    print(f"  Saved: {SCANNER_OFFSET_VECTOR}")

    # ── RF only (stable at n=5) ───────────────────────────────────
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    print(f"\n  --- RandomForest (scanner target) ---")
    fold_results, y_true_all, y_prob_all = run_loocv(X, y, patient_ids, clf, use_scaler=False)
    metrics = evaluate_and_print(
        "Exp4 Scanner Fingerprint - RF", "CLEAN",
        y_true_all, y_prob_all, patient_ids, fold_results
    )

    result = {'experiment': 'Exp4_ScannerFingerprint_RF', 'confound_status': 'CLEAN',
              'n_patients': len(df_normal), 'classifier': 'RandomForest'}
    result.update(metrics)
    return [result], offset_df


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — Z-score Normalisation (repeat Exp 2, 3, 4)
# ══════════════════════════════════════════════════════════════════
def experiment5_normalisation(df, feature_cols):
    print("\n" + "="*60)
    print("EXPERIMENT 5: Z-Score Normalisation")
    print("Applies StandardScaler across ALL patients, then repeats Exp 2+3+4")
    print("="*60)

    # Scale across full dataset (this is the normalisation experiment —
    # we scale globally to remove inter-patient intensity shift)
    scaler_global = StandardScaler()
    df_norm = df.copy()
    df_norm[feature_cols] = scaler_global.fit_transform(df[feature_cols])

    all_results = []

    # ── 5a: Full dataset normalised (Exp 2 repeat) ────────────────
    print("\n  [5a] Full dataset — normalised")
    X = df_norm[feature_cols].values
    y = df_norm['class_label'].values
    patient_ids = df_norm['patient_id'].values

    for clf_name, clf in get_classifiers().items():
        print(f"\n  --- {clf_name} ---")
        use_scaler = (clf_name == 'SVM')
        fold_results, y_true_all, y_prob_all = run_loocv(X, y, patient_ids, clf, use_scaler)
        metrics = evaluate_and_print(
            f"Exp5a Normalised Full - {clf_name}", "CONFOUNDED_NORMALISED",
            y_true_all, y_prob_all, patient_ids, fold_results
        )
        result = {'experiment': f'Exp5a_NormFull_{clf_name}',
                  'confound_status': 'CONFOUNDED_NORMALISED',
                  'n_patients': len(df_norm), 'classifier': clf_name}
        result.update(metrics)
        all_results.append(result)

    # ── 5b: Revolution-only normalised (Exp 3 repeat) ────────────
    print("\n  [5b] Revolution-only — normalised")
    df_rev_norm = df_norm[df_norm['scanner_label'] == 1].copy()
    X = df_rev_norm[feature_cols].values
    y = df_rev_norm['class_label'].values
    patient_ids = df_rev_norm['patient_id'].values

    for clf_name, clf in get_classifiers().items():
        print(f"\n  --- {clf_name} ---")
        use_scaler = (clf_name == 'SVM')
        fold_results, y_true_all, y_prob_all = run_loocv(X, y, patient_ids, clf, use_scaler)
        metrics = evaluate_and_print(
            f"Exp5b Normalised RevOnly - {clf_name}", "REDUCED_CONFOUND_NORMALISED",
            y_true_all, y_prob_all, patient_ids, fold_results
        )
        result = {'experiment': f'Exp5b_NormRevOnly_{clf_name}',
                  'confound_status': 'REDUCED_CONFOUND_NORMALISED',
                  'n_patients': len(df_rev_norm), 'classifier': clf_name}
        result.update(metrics)
        all_results.append(result)

    # ── 5c: Scanner fingerprint normalised (Exp 4 repeat) ─────────
    print("\n  [5c] Scanner fingerprint — normalised")
    df_norm_normal = df_norm[df_norm['class_label'] == 0].copy()
    X = df_norm_normal[feature_cols].values
    y = df_norm_normal['scanner_label'].values
    patient_ids = df_norm_normal['patient_id'].values
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    fold_results, y_true_all, y_prob_all = run_loocv(X, y, patient_ids, clf, use_scaler=False)
    metrics = evaluate_and_print(
        "Exp5c Normalised ScannerFP - RF", "CLEAN_NORMALISED",
        y_true_all, y_prob_all, patient_ids, fold_results
    )
    result = {'experiment': 'Exp5c_NormScannerFP_RF', 'confound_status': 'CLEAN_NORMALISED',
              'n_patients': len(df_norm_normal), 'classifier': 'RandomForest'}
    result.update(metrics)
    all_results.append(result)

    return all_results


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — Synthetic Augmentation
# ══════════════════════════════════════════════════════════════════
def experiment6_augmentation(df, feature_cols):
    print("\n" + "="*60)
    print("EXPERIMENT 6: Synthetic BRIVO-Haemorrhage Augmentation")
    print("Confound status: REDUCED_CONFOUND_SYNTHETIC")
    print("="*60)

    # Generate synthetic vectors (one per Revolution Haem patient)
    synth_df = generate_synthetic_brivo_haem(df, feature_cols, SCANNER_OFFSET_VECTOR)
    synth_df.to_csv(DATA_DIR / 'synthetic_features.csv', index=False)
    print(f"  Saved synthetic features to data/synthetic_features.csv")

    # Revolution-only base (same as Exp 3)
    df_rev = df[df['scanner_label'] == 1].copy()

    # LOOCV with augmentation injected into TRAINING only
    X_rev = df_rev[feature_cols].values
    y_rev = df_rev['class_label'].values
    patient_ids_rev = df_rev['patient_id'].values

    X_synth = synth_df[feature_cols].values
    y_synth = synth_df['class_label'].values

    all_results = []

    for clf_name, clf_template in get_classifiers().items():
        print(f"\n  --- {clf_name} ---")
        use_scaler = (clf_name == 'SVM')
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import LabelEncoder

        loo = LeaveOneOut()
        fold_results = []
        y_true_all = []
        y_prob_all = []

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_rev)):
            X_train = np.vstack([X_rev[train_idx], X_synth])
            y_train = np.concatenate([y_rev[train_idx], y_synth])
            X_test  = X_rev[test_idx]
            y_test  = y_rev[test_idx]

            if use_scaler:
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test  = sc.transform(X_test)

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)

            clf = copy.deepcopy(clf_template)
            clf.fit(X_train, y_train_enc)

            if hasattr(clf, 'predict_proba'):
                y_prob = clf.predict_proba(X_test)[0][1]
            else:
                y_prob = clf.decision_function(X_test)[0]

            true_label = int(y_test[0])
            y_true_all.append(true_label)
            y_prob_all.append(float(y_prob))
            fold_results.append({
                'fold': fold,
                'patient_id': patient_ids_rev[test_idx[0]],
                'true_label': true_label,
                'pred_prob': float(y_prob),
                'pred_label': int(y_prob >= 0.5)
            })

        y_true_arr = np.array(y_true_all)
        y_prob_arr = np.array(y_prob_all)
        metrics = evaluate_and_print(
            f"Exp6 Augmented RevOnly - {clf_name}", "REDUCED_CONFOUND_SYNTHETIC",
            y_true_arr, y_prob_arr, patient_ids_rev, fold_results
        )
        result = {'experiment': f'Exp6_Augmented_{clf_name}',
                  'confound_status': 'REDUCED_CONFOUND_SYNTHETIC',
                  'n_patients': len(df_rev), 'classifier': clf_name}
        result.update(metrics)
        all_results.append(result)

    return all_results


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _save_model(X, y, clf_name, filename, use_scaler=False):
    from sklearn.preprocessing import LabelEncoder
    clf = get_classifiers()[clf_name]
    le  = LabelEncoder()
    y_enc = le.fit_transform(y)
    if use_scaler:
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        clf.fit(X_sc, y_enc)
        joblib.dump({'model': clf, 'scaler': sc}, MODELS_DIR / f'{filename}.pkl')
    else:
        clf.fit(X, y_enc)
        joblib.dump(clf, MODELS_DIR / f'{filename}.pkl')


def print_summary_table(results_df):
    print("\n\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    cols = ['experiment', 'confound_status', 'n_patients', 'classifier',
            'auc_mean', 'ci_lower', 'ci_upper']
    available = [c for c in cols if c in results_df.columns]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(results_df[available].to_string(index=False))

    # Key comparisons
    rf_rows = results_df[results_df['classifier'] == 'RandomForest']
    exp2_rf = rf_rows[rf_rows['experiment'].str.startswith('Exp2_Full_RF')]
    exp3_rf = rf_rows[rf_rows['experiment'].str.startswith('Exp3_RevOnly_RF')]
    exp6_rf = rf_rows[rf_rows['experiment'].str.startswith('Exp6_Augmented_RF')]

    print("\n" + "="*50)
    print("KEY METRICS (Random Forest)")
    print("="*50)
    if len(exp2_rf) > 0 and len(exp3_rf) > 0:
        auc2 = exp2_rf['auc_mean'].values[0]
        auc3 = exp3_rf['auc_mean'].values[0]
        bias = auc2 - auc3
        print(f"  Exp2 (Confounded) AUC     : {auc2:.3f}")
        print(f"  Exp3 (Honest)     AUC     : {auc3:.3f}")
        print(f"  Bias magnitude (Exp2-Exp3): {bias:.3f}")
    if len(exp3_rf) > 0 and len(exp6_rf) > 0:
        auc3 = exp3_rf['auc_mean'].values[0]
        auc6 = exp6_rf['auc_mean'].values[0]
        aug  = auc6 - auc3
        print(f"  Exp6 (Augmented)  AUC     : {auc6:.3f}")
        print(f"  Augmentation effect (6-3) : {aug:+.3f}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "#"*60)
    print("  STAGE 3 — SIX EXPERIMENTS")
    print("  Bias-Aware Brain CT Haemorrhage Classification")
    print("#"*60)

    df, feature_cols = load_patient_data()
    all_results = []

    # Run all 6 experiments
    all_results.extend(experiment1_scanner_baseline(df))
    all_results.extend(experiment2_full_classifier(df, feature_cols))
    all_results.extend(experiment3_revolution_only(df, feature_cols))
    exp4_results, offset_df = experiment4_scanner_fingerprint(df, feature_cols)
    all_results.extend(exp4_results)
    all_results.extend(experiment5_normalisation(df, feature_cols))
    all_results.extend(experiment6_augmentation(df, feature_cols))

    # Save all results
    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / 'experiment_results.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\n\nAll results saved to {out_path}")

    print_summary_table(results_df)
    print("\nStage 3 complete.\n")
