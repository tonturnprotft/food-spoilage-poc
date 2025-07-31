import pathlib, json, typer, yaml, torch
import numpy as np
from .pipelines.build_dataset import load_dataset_split
from .models import gbm, lstm
from .models import cnn,fusion  # make sure lstm.py exists

app = typer.Typer()

@app.command()
def main(dataset: str, model_name: str):
    """
    Run `python -m src.cli <dataset> <model>`
    where <model> is 'gbm' or 'lstm'.
    """
    if model_name == "gbm":
        # ─── GBM with 5‑fold CV + hold‑out test ───────────────────────
        X_tr, y_tr, X_te, y_te = load_dataset_split(dataset)
        gbm_model, cv_f1 = gbm.train_cv(X_tr, y_tr)
        test_f1 = gbm.evaluate(gbm_model, X_te, y_te)

        # save raw probabilities and true labels for calibration
        out_dir = pathlib.Path("models/artifacts") / dataset / model_name
        np.save(out_dir / "y_test.npy", y_te.values)
        np.save(out_dir / "y_score.npy", gbm_model.predict(X_te))

        out_dir.mkdir(parents=True, exist_ok=True)
        gbm_model.save_model(out_dir / "model.txt")
        json.dump(
            {"cv_f1_mean": cv_f1, "test_f1": test_f1},
            open(out_dir / "metrics.json", "w"),
        )
        print(f"✓ GBM CV F1 = {cv_f1:.3f} | Test F1 = {test_f1:.3f}")

    elif model_name == "lstm":
        # ─── LSTM sequence model with hold‑out test ──────────────────
        cfg = yaml.safe_load(open("datasets.yml"))[dataset]
        root = pathlib.Path(cfg["path"])
        seq_train = root / "seq_win10_train.npz"
        seq_test  = root / "seq_win10_test.npz"
        if not (seq_train.exists() and seq_test.exists()):
            typer.echo("❌ Train/test sequence files missing; run `make seq` first.")
            raise typer.Exit(1)

        model_obj, cv_f1 = lstm.train_cv_npz(str(seq_train))
        test_f1 = lstm.evaluate_npz(model_obj, str(seq_test))

        # save probabilities & labels for calibration
        out_dir = pathlib.Path("models/artifacts") / dataset / model_name
        y_scores, y_true = lstm.predict_proba_npz(model_obj, str(seq_test))
        np.save(out_dir / "y_test.npy", y_true)
        np.save(out_dir / "y_score.npy", y_scores)

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_obj.state_dict(), out_dir / "model.pt")
        json.dump(
            {"cv_f1_mean": cv_f1, "test_f1": test_f1},
            open(out_dir / "metrics.json", "w"),
        )
        print(f"✓ LSTM CV F1 = {cv_f1:.3f} | Test F1 = {test_f1:.3f}")
    elif model_name == "iforest":
        # ─── Unsupervised Isolation‑Forest on window stats ───────────
        cfg = yaml.safe_load(open("datasets.yml"))[dataset]
        root = pathlib.Path(cfg["path"])
        seq_train = root / "seq_win10_train.npz"
        seq_test  = root / "seq_win10_test.npz"
        if not (seq_train.exists() and seq_test.exists()):
            typer.echo("❌ Train/test sequence files missing; run `make seq` first.")
            raise typer.Exit(1)
        from .models import iforest
        model_obj = iforest.train_iforest_npz(str(seq_train))
        stats = iforest.evaluate_npz(model_obj, str(seq_test))

        out_dir = pathlib.Path("models/artifacts") / dataset / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # save model as pickle
        import joblib
        joblib.dump(model_obj, out_dir / "model.joblib")
        json.dump(stats, open(out_dir / "metrics.json", "w"), indent=2)
        auc_display = "—" if stats["auc"] is None else f"{stats['auc']:.3f}"
        print(
            f"✓ IF AUC = {auc_display} | F1 = {stats['f1']:.3f} | Cost = {stats['cost']}"
        )
    elif model_name =="cnn":
        model_obj, probs, labels = cnn.train_cnn(dataset)
        out_dir = pathlib.Path("models/artifacts") / dataset / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_obj.state_dict(), out_dir / "model.pt")
        np.save(out_dir / "img_probs.npy", probs)
        np.save(out_dir / "img_labels.npy", labels)
        print(f"✓ CNN trained — Test AUC ≈ {float(((labels==1)&(probs>0.5)).mean()):.3f}")
    elif model_name =="fusion":
        cfg = yaml.safe_load(open("datasets.yml"))[dataset]
        root = pathlib.Path(cfg["path"])
        img_probs = np.load(root.parent / "models/artifacts" / dataset / "cnn" / "img_probs.npy")
        img_labels= np.load(root.parent / "models/artifacts" / dataset / "cnn" / "img_labels.npy")

        X_tr, y_tr, X_te, y_te = load_dataset_split(dataset)
        # align sizes: img arrays correspond to same test sessions as X_te
        f_model = fusion.train_fusion(X_tr.values, np.zeros(len(X_tr)), y_tr)  # dummy train; refined later
        f1 = fusion.test_fusion(f_model, X_te.values, img_probs, y_te)
        print(f"✓ Fusion F1 = {f1:.3f}")
        

    else:
        raise typer.BadParameter("MODEL must be 'gbm' or 'lstm' or 'iforest ")

if __name__ == "__main__":
    app()