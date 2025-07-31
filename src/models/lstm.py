import numpy as np
import torch, pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

class LSTMNet(pl.LightningModule):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
        self.head  = nn.Sequential(nn.Flatten(), nn.Linear(hidden, 1), nn.Sigmoid())
        self.lossf = nn.BCELoss()

    def forward(self, x):      # x: [B, T, 2]
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

    def training_step(self, batch, _):
        x, y = batch
        yhat = self(x).squeeze()
        loss = self.lossf(yhat, y.float())
        self.log("train_loss", loss); return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def _train_on_subset(npz_path: str, epochs: int = 30):
    data = np.load(npz_path)
    X = torch.tensor(data["X"]).float()
    y = torch.tensor(data["y"]).float()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    model = LSTMNet()
    trainer = pl.Trainer(max_epochs=epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, dl)
    # Evaluate on training data for F1 score
    model.eval()
    with torch.no_grad():
        yhat = model(X).squeeze()
        preds = (yhat > 0.5).int().numpy()
        true = y.int().numpy()
        f1 = f1_score(true, preds)
    return model, f1

def train_cv_npz(npz_path: str, epochs: int = 30):
    """
    Train on full data, return model and F1 on training set.
    """
    return _train_on_subset(npz_path, epochs)

def evaluate(model, X_test, y_test):
    """
    Compute F1 on a hold‑out set.
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).float()
        yhat = model(X_test_tensor).squeeze()
        preds = (yhat > 0.5).int().numpy()
    return f1_score(y_test, preds)

# ----------------------------------------------------------------------
# Hold‑out evaluation on a saved sequence npz file
# ----------------------------------------------------------------------
def evaluate_npz(model: LSTMNet, npz_path: str) -> float:
    """
    Load X/y from `npz_path` and compute F1 with the trained LSTM model.
    """
    data = np.load(npz_path)
    X = torch.tensor(data["X"]).float()
    y = data["y"]
    model.eval()
    with torch.no_grad():
        preds = (model(X).squeeze() > 0.5).int().numpy()
    return f1_score(y, preds)

def predict_proba_npz(model: LSTMNet, npz_path: str):
    data = np.load(npz_path)
    X = torch.tensor(data["X"]).float()
    model.eval()
    with torch.no_grad():
        return model(X).squeeze().numpy(), data["y"]