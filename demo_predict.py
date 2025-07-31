#!/usr/bin/env python
"""demo_predict.py – spoilage verdict for a new tuna session"""
import argparse, pathlib, json, numpy as np, pandas as pd, torch, lightgbm as lgb
from torchvision import transforms as T, models

from PIL import Image

# --- SimpleEffNet import or fallback -----------------------------------
try:
    from src.models.cnn import SimpleEffNet  # original class from training
except (ImportError, ModuleNotFoundError):
    # Fallback wrapper matching the checkpoint structure
    import torch.nn as nn
    from torchvision import models as tv_models

    class SimpleEffNet(nn.Module):
        def __init__(self, num_classes: int = 1):
            super().__init__()
            m = tv_models.efficientnet_b0(weights=None)
            # replace last layer
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
            self.net = m

        def forward(self, x):
            return self.net(x)

parser = argparse.ArgumentParser()
parser.add_argument("sensor_csv", help="CSV with mq1,tgs1,day,session")
parser.add_argument("image_dir", help="Folder with JPGs of the fish")
args = parser.parse_args()

# ── load artefacts ─────────────────────────────────────────────
root   = pathlib.Path(__file__).parent
art    = root/"models"/"artifacts"/"dafif"

# sensor GBM
s_gbm  = lgb.Booster(model_file=art/"gbm"/"model.txt")


# ── CNN checkpoint -----------------------------------------------------
ckpt_dir  = art / "cnn"
ckpt_path = ckpt_dir / "model.pt"   # training saved via Lightning

if not ckpt_path.exists():
    raise FileNotFoundError(f"Expected CNN checkpoint at {ckpt_path}")

# Lightning checkpoints store a dict with key 'state_dict'
state = torch.load(ckpt_path, map_location="cpu")
state_dict = state["state_dict"] if "state_dict" in state else state

# remove potential 'net.' prefix used in LightningModule
new_state = {k.replace("net.", ""): v for k, v in state_dict.items()}

cnn = SimpleEffNet(num_classes=1)
cnn.load_state_dict(new_state, strict=False)
cnn.eval()

tr = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ── image probability ─────────────────────────────────────────
imgs = list(pathlib.Path(args.image_dir).glob("*.jpg"))
probs=[]
with torch.no_grad():
    for p in imgs:
        img = tr(Image.open(p).convert("RGB"))[None]
        probs.append(torch.sigmoid(cnn(img)).item())
img_prob = float(np.mean(probs))
print(f"mean image prob = {img_prob:.3f}")

# ── sensor probability ───────────────────────────────────────
df = pd.read_csv(args.sensor_csv)

# replicate training‑time feature engineering
X = df[["mq1", "tgs1", "day"]].copy()
# convert day to “days since epoch” int, just like in training
X["day"] = pd.to_datetime(X["day"]).view("int64") // 86_400_000_000_000

# add the fused image probability
X["img_prob"] = img_prob  # fusion feature

pred = s_gbm.predict(X, predict_disable_shape_check=True).mean()
print(f"mean sensor‑fusion prob = {pred:.3f}")
print("Verdict:", "❌ SPOILED" if pred>0.5 else "✅ FRESH")