import pathlib, json, typer, yaml, torch
from .pipelines.build_dataset import load_dataset
from .models import gbm, lstm  # make sure lstm.py exists

app = typer.Typer()

@app.command()
def main(dataset: str, model_name: str):
    """
    Run `python -m src.cli <dataset> <model>`
    where <model> is 'gbm' or 'lstm'.
    """
    if model_name == "gbm":
        # ─── GBM with 5-fold CV ───────────────────────────────────────
        X, y = load_dataset(dataset)
        gbm_model, f1 = gbm.train_cv(X, y)

        out_dir = pathlib.Path("models/artifacts") / dataset / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        gbm_model.save_model(out_dir / "model.txt")
        json.dump({"cv_f1_mean": f1}, open(out_dir / "metrics.json", "w"))
        print(f"✓ GBM trained — mean CV F1 = {f1:.3f}")

    elif model_name == "lstm":
        # ─── sequence model ──────────────────────────────────────────
        cfg = yaml.safe_load(open("datasets.yml"))[dataset]
        seq_file = pathlib.Path(cfg["path"]) / "seq_win10.npz"
        if not seq_file.exists():
            typer.echo("❌ seq_win10.npz not found. Run make seq DATASET=dafif first.")
            raise typer.Exit(1)

        model_obj = lstm.train_npz(str(seq_file))
        out_dir = pathlib.Path("models/artifacts") / dataset / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_obj.state_dict(), out_dir / "model.pt")
        print("✓ LSTM trained and saved")

    else:
        raise typer.BadParameter("MODEL must be 'gbm' or 'lstm'")

if __name__ == "__main__":
    app()