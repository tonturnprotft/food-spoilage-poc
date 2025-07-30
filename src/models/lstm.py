import numpy as np
import torch, pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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

def train_npz(npz_path: str, epochs: int = 30):
    data = np.load(npz_path)
    X = torch.tensor(data["X"]).float()
    y = torch.tensor(data["y"]).float()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    model = LSTMNet()
    trainer = pl.Trainer(max_epochs=epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, dl)
    return model