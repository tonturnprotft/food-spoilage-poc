"""Isolation‑Forest for 10‑step window stats."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score

def _stats_from_npz(npz_path: str):
    data = np.load(npz_path)
    X_seq, y = data["X"], data["y"]  # X_seq shape: [N, 10, 2]
    # features: mean & std per channel across the 10‑step window
    means = X_seq.mean(axis=1)      # [N, 2]
    stds  = X_seq.std(axis=1)
    X = np.hstack([means, stds])    # [N, 4]
    return X, y

def train_iforest_npz(npz_train: str, contamination: float = 0.1):
    X_tr, y_tr = _stats_from_npz(npz_train)
    # Use only fresh windows (label 0) for unsupervised training
    X_fresh = X_tr[y_tr == 0]
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    ).fit(X_fresh)
    return model

def evaluate_npz(model, npz_path: str, cost_fn: float = 10.0):
    X_te, y_te = _stats_from_npz(npz_path)
    scores = -model.decision_function(X_te)       # higher = more anomalous

    # ── Handle single-class test set ──────────────────────────────
    if len(np.unique(y_te)) > 1:
        auc = roc_auc_score(y_te, scores)
    else:
        auc = None  # undefined when only one class is present

    tau = np.percentile(scores, 95)               # 95th-percentile threshold
    y_pred = (scores >= tau).astype(int)
    f1 = f1_score(y_te, y_pred)

    fn = ((y_pred == 0) & (y_te == 1)).sum()
    fp = ((y_pred == 1) & (y_te == 0)).sum()
    cost = fn * cost_fn + fp * 1.0

    return dict(f1=f1, auc=auc, threshold=tau,
                fn=int(fn), fp=int(fp), cost=cost)