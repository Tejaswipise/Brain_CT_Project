"""
utils/evaluation.py
Shared evaluation utilities for all Stage 3 experiments.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_classifiers():
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            scale_pos_weight=1,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        ),
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }


def run_loocv(X, y, patient_ids, classifier, use_scaler=False):
    """
    Leave-One-Patient-Out CV.
    Returns fold_results list, y_true_all array, y_prob_all array.
    """
    from sklearn.preprocessing import LabelEncoder
    loo = LeaveOneOut()
    fold_results = []
    y_true_all = []
    y_prob_all = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale if needed (SVM)
        if use_scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # XGBoost needs 0/1 labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        import copy
        clf = copy.deepcopy(classifier)
        clf.fit(X_train, y_train_enc)

        # Predict probability for positive class
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_test)[0][1]
        else:
            y_prob = clf.decision_function(X_test)[0]

        true_label = int(y_test[0])
        pred_label = int(y_prob >= 0.5)

        y_true_all.append(true_label)
        y_prob_all.append(float(y_prob))
        fold_results.append({
            'fold': fold,
            'patient_id': patient_ids[test_idx[0]],
            'true_label': true_label,
            'pred_prob': float(y_prob),
            'pred_label': pred_label
        })

    return fold_results, np.array(y_true_all), np.array(y_prob_all)


def compute_bootstrap_auc(y_true, y_prob, n_bootstrap=1000, random_state=42):
    """AUC with 95% bootstrap CI. Returns (mean, ci_lower, ci_upper)."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.RandomState(random_state)
    bootstrap_aucs = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        yt = y_true[indices]
        yp = y_prob[indices]
        if len(np.unique(yt)) < 2:
            continue
        try:
            bootstrap_aucs.append(roc_auc_score(yt, yp))
        except Exception:
            continue

    if len(bootstrap_aucs) < 100:
        auc_mean = roc_auc_score(y_true, y_prob)
        return float(auc_mean), np.nan, np.nan

    return (float(np.mean(bootstrap_aucs)),
            float(np.percentile(bootstrap_aucs, 2.5)),
            float(np.percentile(bootstrap_aucs, 97.5)))


def evaluate_and_print(exp_name, confound_status, y_true, y_prob, patient_ids, fold_results):
    """Print standardised evaluation block and return metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"  Confound status: {confound_status}")
    print(f"  N patients: {len(y_true)}")
    print(f"{'='*60}")

    if len(np.unique(y_true)) < 2:
        print("  WARNING: Only one class in subset. AUC undefined.")
        return {'auc_mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'sensitivity': np.nan, 'specificity': np.nan,
                'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

    auc_mean, ci_lower, ci_upper = compute_bootstrap_auc(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    print(f"  AUC: {auc_mean:.3f}  [{ci_lower:.3f}, {ci_upper:.3f}]  95% CI")
    print(f"  Sensitivity: {sensitivity:.3f}   Specificity: {specificity:.3f}")
    print(f"  Confusion matrix  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"\n  Per-patient predictions:")
    for r in fold_results:
        marker = "✓" if r['true_label'] == r['pred_label'] else "✗"
        print(f"    {marker}  {r['patient_id']:20s}  true={r['true_label']}  "
              f"prob={r['pred_prob']:.3f}  pred={r['pred_label']}")

    return {
        'auc_mean': auc_mean, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'sensitivity': sensitivity, 'specificity': specificity,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
