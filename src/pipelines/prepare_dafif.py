import typer, zipfile, pathlib, pandas as pd, re, shutil, yaml, os, sys
from datetime import datetime

app = typer.Typer()

def unzip_all(src_dir: pathlib.Path, dst: pathlib.Path):
    for z in src_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)

def tidy_sensor(xlsx: pathlib.Path, day: int, sess: int, dst_dir: pathlib.Path):
    m = re.search(r'\((\w+)', xlsx.stem, re.I)
    species = m.group(1).lower() if m else "unknown"
    df = pd.read_excel(xlsx)
    df["timestamp"] = pd.to_datetime(df["Date"].astype(str)+" "+df["Time"])
    df = df.drop(columns=["Date","Time"]).rename(columns=str.lower)
    df["species"] = species; df["day"] = day; df["session"] = sess
    out = dst_dir / f"day{day:02d}_session{sess}_{species}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

@app.command()
def main(config: str = "datasets.yml", dataset: str = "dafif"):
    cfg = yaml.safe_load(open(config))[dataset]
    root = pathlib.Path(cfg["path"])
    # 1) unzip
    unzip_all(root/"zips", root)
    # 2) walk folders Day*/Session*/<fish>/
    for day_dir in root.glob("Day *"):
        day = int(day_dir.name.split()[-1])
        for sess_dir in day_dir.glob("Session *"):
            sess = int(sess_dir.name.split()[-1])
            for item in sess_dir.iterdir():
                if item.is_dir():  # images
                    dst = root/"images"/f"day{day:02d}_session{sess}_{item.name.lower()}"
                    dst.mkdir(parents=True, exist_ok=True)
                    for img in item.glob("*.jpg"):
                        shutil.move(img, dst/img.name)
                elif item.suffix == ".xlsx":  # sensor sheet
                    tidy_sensor(item, day, sess, root/"sensor")
    # 3) log summary
    summary = root/"meta"
    summary.mkdir(exist_ok=True)
    (summary/"README.txt").write_text(
        f"Prepared {datetime.now().isoformat()}\n" +
        f"Tuna sensor files: {len(list((root/'sensor').glob('*tuna*.csv')))}\n" )
    print("✓ prepare_dafif done")

if __name__ == "__main__":
    app()