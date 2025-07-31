import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
import pandas as pd

# ------------------------------------------------------------
# Helper: broadcast session‑level image probability to windows
# ------------------------------------------------------------

def prepare_features(X_df, img_df):
    # convert image day strings → int64 days-since-epoch to match sensor df
    img_df = img_df.copy()
    img_df["day"] = (
        pd.to_datetime(img_df["day"])
        .view("int64") // 86_400_000_000_000
    )

    sess_prob = (
        img_df.groupby(["day", "session"])["img_prob"].mean()
        .rename("img_prob")
    )
    merged = X_df.join(sess_prob, on=["day", "session"])
    img_vec = merged["img_prob"].fillna(0).values
    X_mat = merged.drop(columns=["day", "session", "img_prob"]).values
    return X_mat, img_vec

# ------------------------------------------------------------
# Train / Test wrappers
# ------------------------------------------------------------

def train_fusion(X_sens, img_vec, y):
    X = np.column_stack([X_sens, img_vec])
    return lgb.LGBMClassifier(n_estimators=200, random_state=42).fit(X, y)

def test_fusion(model, X_sens, img_vec, y):
    X = np.column_stack([X_sens, img_vec])
    return f1_score(y, model.predict(X))