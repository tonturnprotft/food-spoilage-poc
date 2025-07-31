"""
Scan tuna image folders, extract day/session, join with labels,
and write `image_index.csv` for CNN training.

$ python -m src.pipelines.index_images --dataset dafif
"""

import pathlib, re, yaml, pandas as pd, typer

app = typer.Typer(help="Build image-to-label index CSV")

FOLDER_RE = re.compile(r"day(\d{2})_session(\d)_tuna", re.I)

def build_index(root: pathlib.Path, cfg: dict) -> pd.DataFrame:
    label_df = pd.read_csv(root / cfg["label_file"])
    label_df["label"] = label_df["score"].map(cfg["freshness_map"])
    label_df = label_df.dropna(subset=["label"]).astype({"label": int})

    rows = []
    for img_path in (root / "images").rglob("*.jpg"):
        m = FOLDER_RE.search(str(img_path.parent))
        if not m:
            continue
        day_num, session = m.groups()
        # map dayXX → real date string from sensor filenames
        # sensors use dates like 19-01-2024; get mapping from existing CSVs
        rows.append(
            dict(
                img_path=str(img_path.relative_to(root)),
                day=int(day_num),  # 1-based
                session=int(session),
            )
        )

    df = pd.DataFrame(rows)
    # convert day index back to actual date by joining on sensor files
    day_rows = []
    for p in (root / "sensor").glob("*tuna*.csv"):
        df_head = pd.read_csv(p, nrows=1)
        if "day" in df_head.columns:
            day_rows.append(df_head[["day"]])
    if not day_rows:
        raise ValueError("No sensor CSV contained a 'day' column")
    sensor_days = (
        pd.concat(day_rows)
        .drop_duplicates()
        .sort_values("day")
        .reset_index(drop=True)
        .assign(day_idx=lambda d: d.index + 1)
    )
    df = df.merge(sensor_days, left_on="day", right_on="day_idx").drop(columns="day_idx")
    # After the merge, we have two `day` columns:
    #   - `day_x` (integer day index from folder name)
    #   - `day_y` (actual date string from sensor CSV)
    # We want the real date string to be our canonical `day`.
    if "day_x" in df.columns and "day_y" in df.columns:
        df = (
            df.rename(columns={"day_x": "day_idx", "day_y": "day"})
              .drop(columns="day_idx")  # keep the real date string
        )
    df = df.merge(label_df[["day", "session", "species", "label"]],
                  on=["day", "session"], how="left")
    return df.dropna(subset=["label"]).reset_index(drop=True)

@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", help="Dataset key in datasets.yml")
):
    cfg = yaml.safe_load(open("datasets.yml"))[dataset]
    root = pathlib.Path(cfg["path"])
    out_csv = root / "image_index.csv"
    idx = build_index(root, cfg)
    idx.to_csv(out_csv, index=False)
    print(f"✓ indexed {len(idx):,} images → {out_csv}")

if __name__ == "__main__":
    app()