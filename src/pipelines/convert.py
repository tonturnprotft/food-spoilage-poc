# src/pipelines/convert_xlsx.py
"""
Convert every *.xlsx in <root>/sensors/ to tidy CSV.

Usage
-----
# from repo root
python -m src.pipelines.convert_xlsx --root data/raw/dafif
"""

import argparse, pathlib, re, datetime as dt
import pandas as pd


PAT = re.compile(r"(?P<date>\d{2}-\d{2}-\d{4})_session(?P<sess>\d)_(?P<species>\w+)", re.I)


def convert_one(xlsx_path: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    m = PAT.match(xlsx_path.stem)
    if not m:
        raise ValueError(f"Filename does not match pattern: {xlsx_path.name}")

    date_raw  = m.group("date")                     # 19-01-2024
    day_iso   = dt.datetime.strptime(date_raw, "%d-%m-%Y").strftime("%Y-%m-%d")
    session   = int(m.group("sess"))
    species   = m.group("species").lower()

    df = pd.read_excel(xlsx_path)

    # merge Date + Time → timestamp   (skip if they’re already combined)
    if {"Date", "Time"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
        df = df.drop(columns=["Date", "Time"])

    # standardise column names: lower-case, no spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # add day / session / species columns so the loader can merge with labels
    df["day"]     = day_iso
    df["session"] = session
    df["species"] = species

    out_path = out_dir / (xlsx_path.stem + ".csv")
    df.to_csv(out_path, index=False)
    return out_path


def main(root: str = "data/raw/dafif"):
    root = pathlib.Path(root)
    src_dir = root / "sensors"
    out_dir = root / "sensor"          # ← matches datasets.yml sensor_glob
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for xlsx in src_dir.glob("*.xlsx"):
        out_file = convert_one(xlsx, out_dir)
        converted.append(out_file.name)

    print(f"✓ Converted {len(converted)} files → {out_dir}")
    # (optional) uncomment to delete originals
    # for x in src_dir.glob("*.xlsx"): x.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/dafif",
                        help="dataset root containing sensors/ folder")
    args = parser.parse_args()
    main(root=args.root)