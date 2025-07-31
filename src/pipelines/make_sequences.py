import pandas as pd, numpy as np, pathlib, typer, yaml

app = typer.Typer()

def make_sequences(root: pathlib.Path, win: int = 10):
    csvs = sorted((root/"sensor").glob("*tuna*.csv"))
    dfs  = [pd.read_csv(c) for c in csvs]
    df   = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
    # ─── 80/20 session split (newest sessions for test set) ─────────
    sessions = (
        df[["day", "session"]]
        .drop_duplicates()
        .sort_values(["day", "session"])
    )
    n_each = max(1, int(len(sessions) * 0.1))  # 10 % from each end
    test_sessions = pd.concat([sessions.head(n_each), sessions.tail(n_each)])
    test_key = set(zip(test_sessions.day, test_sessions.session))
    cfg = yaml.safe_load(open("datasets.yml"))["dafif"]
    labels = pd.read_csv(root /cfg["label_file"])
    labels["label"] = labels["score"].map(cfg["freshness_map"])
    labels = labels.dropna(subset=["label"]).astype({"label": int})
    df = df.merge(
        labels[["day", "session", "species", "label"]],
        on=["day", "session", "species"],
    )

    seq_X_train, seq_y_train, seq_X_test, seq_y_test = [], [], [], []
    for _, g in df.groupby(["day", "session"]):
        g = g.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(g) - win):
            window = g.loc[i:i+win-1, ["mq1", "tgs1"]].to_numpy()
            label  = g.loc[i+win-1, "label"]          # label at end of window
            target_X, target_y = (
                (seq_X_test, seq_y_test)
                if (g.loc[0, "day"], g.loc[0, "session"]) in test_key
                else (seq_X_train, seq_y_train)
            )
            target_X.append(window)
            target_y.append(label)

    out_train = root / f"seq_win{win}_train.npz"
    out_test  = root / f"seq_win{win}_test.npz"
    np.savez_compressed(out_train, X=np.array(seq_X_train), y=np.array(seq_y_train))
    np.savez_compressed(out_test,  X=np.array(seq_X_test),  y=np.array(seq_y_test))
    print(f"✓ sequences: {len(seq_X_train)} train windows → {out_train}")
    print(f"✓ sequences: {len(seq_X_test)}  test windows → {out_test}")

@app.command()
def main(dataset: str = "dafif", win: int = 10):
    cfg  = yaml.safe_load(open("datasets.yml"))[dataset]
    root = pathlib.Path(cfg["path"])
    out_train = root / f"seq_win{win}_train.npz"
    out_test  = root / f"seq_win{win}_test.npz"
    make_sequences(root, win)

if __name__ == "__main__":
    app()