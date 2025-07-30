# Food‑Spoilage‑PoC

Predicting imminent spoilage events & remaining shelf‑life from IoT cold‑chain sensor streams.

---

## 🚀 Quick‑start

```bash
# clone & install
$ git clone https://github.com/tonturnprotft/old-food-spoilage-poc.git
$ cd old-food-spoilage-poc
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# build features (3 rolling windows + engineered deg‑hours‑over‑8)
$ python -m features.build_features \
    --raw data/interim/sim_labeled.parquet \
    --out data/processed/train_feats.parquet \
    --auto_thr_q 0.20                 # auto pick RSL threshold at 20‑th pct

# train Gradient Boosting baseline (group‑aware split, cost‑optimised)
$ python models/train_gbm.py \ 
    --group_col sensor_id \
    --min_pos_per_split 15 \
    --save_splits \
    --val_calibrate \
    --thr_table --per_group_metrics \
    --calibrate_probs isotonic \
    --cv_folds 5 \
    --fp_cost 1 --fn_cost 10 \
    --extra_drop_cols ts food_type \
    --debug_checks
```

> **Tip** : add `--no_plots` on headless servers and enable GPU LightGBM (`pip install lightgbm==4.0.0 --install-option=--gpu`) for large runs.

---

## 📂 Project layout

```
├── data/                 # raw ↔ interim ↔ processed datasets
├── features/             # rolling‑window & leakage‑proof feature builders
├── models/               # train_gbm.py, train_lstm.py, train_iforest.py …
├── scripts/              # CLI helpers / batch jobs
├── notebooks/            # exploratory analysis (ignored in CI)
└── requirements.txt      # locked deps
```

---

## 🔧 Feature engineering

| block             | description                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| **Rolling stats** | mean/std/slope over 60‑, 180‑, 360‑min windows for temp, RH, ethylene, VOC, shock‑g, microbe\_idx          |
| **Domain rules**  | degree‑hours > 8 °C, cumulative over‑8 minutes                                                             |
| **Timestamp**     | converted to Unix seconds (tz stripped) — beware pandas tz‑aware → naïve conversion                        |
| **Leakage guard** | any column containing `label`,`cls`,`spoil`,`event`,`deg_hours_over8`,`remaining_hours`,`rsl` auto‑dropped |

---

## 📈 Latest GBM results

**Cross‑val (5‑fold stratified‑group)**

| metric            | mean        |
| ----------------- | ----------- |
| ROC‑AUC           | **0.99998** |
| Average‑Precision | **0.99997** |
| Brier             | 0.00153     |
| LogLoss           | 0.00751     |

**Hold‑out (random stratified)**

| split | pos‑rate | Brier   | best‑F1 | cost‑optim thr |
| ----- | -------- | ------- | ------- | -------------- |
| Val   | 0.200    | 0.00008 | 0.99977 | 0.10 (cost=9)  |
| Test  | 0.200    | 0.00067 | 0.99977 | 0.10 (cost=9)  |

*Group breakdown* (val): sensors S000‑S008 are all‑zero (cold rooms never breached threshold) → precision/recall 0. Others ≈ 1.0.\
We therefore **force low‑variance / zero‑positive groups into *****train*** to stabilise val/test rates.

---

## 🤖 Other models

| model                | status | notes                                                                                                     |
| -------------------- | ------ | --------------------------------------------------------------------------------------------------------- |
| **LSTM**             | 🚧     | uses 4‑hour sequences @ 10‑min granularity; needs GPU; current ROC‑AUC ≈ 0.993 but overfits small sensors |
| **Isolation‑Forest** | ✅      | unsupervised anomaly score fitted on RSL > 180 h data, used as ensemble feature                           |

---

## 🛠️ Common pitfalls & how we fixed them

1. **Zero‑positive validation folds** → use `min_pos_per_split` + force groups.
2. **Leaking engineered labels** (`remaining_hours`, `deg_hours*`) → regex drop.
3. **Over‑confident 0/1 probabilities** → isotonic calibration.
4. **tz‑aware timestamps crash **`` → always `tz_localize(None)`.
5. **Threshold selection mismatch (F1 vs cost)** → store both `ml_prob_threshold` & `ml_prob_threshold_cost`.

---

## 📌 Next steps

-

---

## 📜 License

MIT

---

*Last updated: 2025‑07‑30*

