"""Calibrate optimal decision threshold for GBM or LSTM.

Usage
-----
$ python -m src.pipelines.calibrate_threshold --dataset dafif --model gbm --cost_fn 10 --cost_fp 1
"""
import argparse, pathlib, json, numpy as np
from sklearn.metrics import precision_recall_fscore_support

def sweep_thresholds(y_true, scores, cost_fn: float = 10.0, cost_fp: float = 1.0):
    best_tau, best_cost, best_stats = 0.5, float("inf"), None
    for tau in np.linspace(0.0, 1.0, 201):  # step 0.005
        y_pred = (scores >= tau).astype(int)
        tn, fp, fn, tp = ((y_true == 0) & (y_pred == 0)).sum(), ((y_true == 0) & (y_pred == 1)).sum(), ((y_true == 1) & (y_pred == 0)).sum(), ((y_true == 1) & (y_pred == 1)).sum()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_tau, best_cost = tau, cost
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            best_stats = dict(precision=prec, recall=rec, f1=f1, fp=int(fp), fn=int(fn))
    return best_tau, best_cost, best_stats

def main(dataset: str, model: str, cost_fn: float, cost_fp: float):
    art_dir = pathlib.Path("models/artifacts") / dataset / model
    meta = json.loads((art_dir / "metrics.json").read_text())
    # Load cached test‑set probabilities saved by cli.py
    y_true = np.load(art_dir / "y_test.npy")
    scores = np.load(art_dir / "y_score.npy")
    tau, cost, stats = sweep_thresholds(y_true, scores, cost_fn, cost_fp)
    meta.update({
        "best_threshold": tau,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "cost_total": cost,
        "precision_at_tau": stats["precision"],
        "recall_at_tau": stats["recall"],
        "f1_at_tau": stats["f1"],
        "fp_at_tau": stats["fp"],
        "fn_at_tau": stats["fn"],
    })
    (art_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ {model.upper()}  τ={tau:.3f}  Precision {stats['precision']:.3f} | Recall {stats['recall']:.3f} | Cost {cost}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=["gbm", "lstm"])
    p.add_argument("--cost_fn", type=float, default=10.0)
    p.add_argument("--cost_fp", type=float, default=1.0)
    args = p.parse_args()
    main(args.dataset, args.model, args.cost_fn, args.cost_fp)