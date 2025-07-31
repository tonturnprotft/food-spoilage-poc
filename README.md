# Food‑Spoilage‑PoC

**End‑to‑end proof‑of‑concept that predicts tuna spoilage by fusing real‑time sensor streams and fish images.**

> *IoT gas sensors + Computer Vision + LightGBM → early‑warning alerts & lower food waste.*

---

## 🌟 Key Features

| Component                 | Model               | Metric        | Notes                        |
| ------------------------- | ------------------- | ------------- | ---------------------------- |
| Gas‑sensors               | **LightGBM**        | **F1 = 1.00** | mq1, tgs1, encoded day       |
| Time‑series               | **LSTM**            | F1 ≈ 0.997    | 10‑step windows              |
| Images                    | **EfficientNet‑B0** | AUC ≈ 0.61    | fine‑tuned on session photos |
| **Sensor × Image Fusion** | GBM + `img_prob`    | **F1 = 1.00** | best of both worlds          |

*Cost‑based threshold optimisation yields ****0 false positives & 0 false negatives**** on held‑out test data.*

---

## 📂 Repository Structure

```
├── data/                  # raw & processed datasets (NOT in Git)
│   └── raw/dafif/...
├── models/
│   └── artifacts/         # trained weights & metrics
├── notebooks/             # analysis & visualisations
├── sample/                # tiny demo dataset (csv + 2 images)
├── src/                   # python package
│   ├── models/            # gbm.py, lstm.py, cnn.py, fusion.py
│   └── pipelines/         # prepare_dafif.py, make_sequences.py, ...
├── demo_predict.py        # <–– one‑shot prediction script
├── Makefile               # common CLI targets
└── README.md              # you are here
```

---

## ⚡ Quick Start

```bash
# 1) clone & install
$ git clone https://github.com/<your_handle>/food-spoilage-poc.git
$ cd food-spoilage-poc
$ poetry install

# 2) run the 5‑second demo on tiny sample data
$ poetry run python demo_predict.py sample/sensor.csv sample/images
mean image prob = 0.50
mean sensor‑fusion prob = 1.00
Verdict: ❌ SPOILED
```

> The demo loads pretrained models from `models/artifacts/dafif/*` and prints a session‑level verdict (✅ FRESH / ❌ SPOILED).

---

## 🛠  Reproduce Training Pipeline

```bash
# preprocess raw Dafif dataset\$ make prepare DATASET=dafif
# create LSTM windows\$ make seq DATASET=dafif
# train / evaluate models
$ make run DATASET=dafif MODEL=gbm   # gas‑sensor GBM
$ make run DATASET=dafif MODEL=lstm  # time‑series LSTM
$ make run_cnn  DATASET=dafif        # image CNN
$ make run_fusion DATASET=dafif      # late fusion GBM
# calibrate thresholds against cost fn
$ make calibrate DATASET=dafif MODEL=gbm
```

All artefacts are saved under `models/artifacts/<dataset>/<model>/`.

---

## 📊 Exploratory Notebooks

| Notebook                     | What it shows                              |
| ---------------------------- | ------------------------------------------ |
| **01\_sensor\_eda.ipynb**    | gas‑sensor patterns across days & sessions |
| **02\_model\_metrics.ipynb** | ROC / PR curves, confusion matrices        |
| **03\_case\_study.ipynb**    | side‑by‑side sensor trace + fish photos    |

Run with:

```bash
poetry run jupyter notebook
```

---

## 💡 Future Work

- Larger image dataset → stronger CNN
- Real‑time streaming demo (MQTT + FastAPI)
- Edge deployment on Raspberry Pi + Coral TPU

---

## 📜 License

MIT — free to use for research & education. Attribution appreciated.

---

## 🙏 Acknowledgements

- **DaFiF Tuna Spoilage Dataset** – please cite the following when using the data:
  - Prasetyo *et al.* (2024), *“DaFiF: A Complete Dataset for Fish's Freshness Problems,”* **Data in Brief**.
  - Prasetyo *et al.* (2024), *“Standardizing the fish freshness class during ice storage using clustering approach,”* **Ecological Informatics**, 80, DOI: [https://doi.org/10.1016/j.ecoinf.2024.102533](https://doi.org/10.1016/j.ecoinf.2024.102533)
- EfficientNet implementation from **torchvision**.
- Inspiration from Google’s “Produce Insight” white‑paper.

