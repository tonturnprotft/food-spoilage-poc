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
    return X, y