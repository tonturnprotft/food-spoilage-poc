from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

def train_cv(X, y, n_splits: int = 5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores, boosters = [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        model = LGBMClassifier(
            n_estimators=500,
            class_weight={0: 1, 1: 10},
            random_state=42 + fold,
            learning_rate=0.05,
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        f1 = f1_score(y.iloc[val_idx], model.predict(X.iloc[val_idx]))
        scores.append(f1); boosters.append(model.booster_)
        print(f"• fold {fold}: F1 = {f1:.3f}")

    print(f"✓ mean F1 = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    # return the model with best fold score just to have one artefact
    best_idx = int(np.argmax(scores))
    return boosters[best_idx], float(np.mean(scores))