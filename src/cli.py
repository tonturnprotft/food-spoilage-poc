import pathlib, json, typer
from .pipelines.build_dataset import load_dataset
from .models import gbm

app = typer.Typer()

@app.command()
def main(dataset: str, model_name: str):
    if model_name != "gbm":
        raise typer.BadParameter("Only MODEL=gbm implemented.")
    X, y = load_dataset(dataset)
    gbm_model, f1 = gbm.train_cv(X, y)

    out_dir = pathlib.Path("models/artifacts") / dataset / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    gbm_model.save_model(out_dir / "model.txt")
    json.dump({"cv_f1_mean": f1}, open(out_dir / "metrics.json", "w"))
    print(f"✓ GBM trained — F1 = {f1:.3f}")

if __name__ == "__main__":
    app()