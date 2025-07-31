import pathlib, yaml, pandas as pd

def load_dataset(name: str, cfg_file="datasets.yml"):
    cfg  = yaml.safe_load(open(cfg_file))[name]
    root = pathlib.Path(cfg["path"])

    df = pd.concat(
        [pd.read_csv(p) for p in (root/"sensor").glob("*tuna*.csv")],
        ignore_index=True
    )

    labels = pd.read_csv(root/cfg["label_file"])
    labels["label"] = labels["score"].map(cfg["freshness_map"]).dropna()
    labels = labels.dropna(subset=["label"]).astype({"label":int})

    df = df.merge(labels[["day","session","species","label"]],
                  on=["day","session","species"])
    X = df.drop(columns=["label","timestamp"])
    y = df["label"]
    X["day"] = pd.to_datetime(X["day"]).view("int64") // 86_400_000_000_000  # days since epoch
    X = X.drop(columns=["species"])
    return X, y


# ----------------------------------------------------------------------
# Hold‑out split: newest 20 % of sessions become the test set
# ----------------------------------------------------------------------
from typing import Tuple


def load_dataset_split(
    name: str,
    cfg_file: str = "datasets.yml",
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Return X_train, y_train, X_test, y_test.

    Sessions are identified by (day, session).  We sort sessions
    chronologically by day then session and take the newest `test_frac`
    proportion as the hold‑out test set.  This prevents any temporal
    leakage while keeping class balance reasonable.
    """
    # full dataset
    X, y = load_dataset(name, cfg_file)
    df = X.copy()
    df["label"] = y

    # unique sessions in chronological order
    sessions = (
        df[["day", "session"]]
        .drop_duplicates()
        .sort_values(["day", "session"])
    )
    # head + tail split: take oldest ½·test_frac and newest ½·test_frac sessions
    n_each = max(1, int(len(sessions) * test_frac / 2))
    test_sessions = pd.concat([sessions.head(n_each), sessions.tail(n_each)])

    # mask rows that belong to test sessions
    is_test = df.set_index(["day", "session"]).index.isin(
        test_sessions.set_index(["day", "session"]).index
    )

    X_train, y_train = X.loc[~is_test], y.loc[~is_test]
    X_test, y_test = X.loc[is_test], y.loc[is_test]
    return X_train, y_train, X_test, y_test