import pathlib, json, typer
from pipelines.build_dataset import load_dataset
from models import gbm

app = typer.Typer()

@app.command()
def main(dataset: str, model: str):
    if model != "gbm":
        raise typer.BadParameter("Only MODEL=gbm implemented.")
    X, y = load_dataset(dataset)
    m, f1 = gbm.train(X, y)

    out = pathlib.Path("models/artifacts")/dataset/model
    out.mkdir(parents=True, exist_ok=True)
    m.booster_.save_model(out/"model.txt")
    json.dump({"f1": f1}, open(out/"metrics.json","w"))
    print(f"✓ GBM trained — F1 = {f1:.3f}")

if __name__ == "__main__":
    app()