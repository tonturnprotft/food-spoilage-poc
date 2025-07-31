import pathlib, yaml, torch, pytorch_lightning as pl
from torchvision import models
from torch.utils.data import DataLoader
from ..datasets.vision import TunaImageDataset

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class CNNLit(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.net.classifier[1] = torch.nn.Linear(self.net.classifier[1].in_features, 1)
        self.lossf = torch.nn.BCEWithLogitsLoss()

    def forward(self, x): return self.net(x).squeeze()

    def step(self, batch):
        x,y = batch
        y_hat = self(x)
        loss = self.lossf(y_hat, y)
        return loss, torch.sigmoid(y_hat)

    def training_step(self, batch, _): loss,_ = self.step(batch); self.log("loss", loss); return loss
    def validation_step(self, batch, _): loss,prob = self.step(batch); self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def train_cnn(dataset: str, epochs=5, bs=32):
    cfg = yaml.safe_load(open("datasets.yml"))[dataset]
    root = pathlib.Path(cfg["path"])
    index_csv = root / "image_index.csv"
    df = pd.read_csv(index_csv)

    # stratified 80 / 20 split to guarantee both classes in test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(df["img_path"], df["label"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    # save temporary csv files so TunaImageDataset can load them unchanged
    tmp_train_csv = root / "image_index_train.csv"
    tmp_test_csv  = root / "image_index_test.csv"
    df_train.to_csv(tmp_train_csv, index=False)
    df_test.to_csv(tmp_test_csv,  index=False)

    ds_tr = TunaImageDataset(tmp_train_csv, root, train=True)
    ds_te = TunaImageDataset(tmp_test_csv,  root, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=4)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=4)

    model = CNNLit()
    trainer = pl.Trainer(max_epochs=epochs, logger=False, enable_checkpointing=False, devices="auto")
    trainer.fit(model, dl_tr, dl_te)

    # get probs on test set
    model.eval(); probs, labels = [], []
    with torch.no_grad():
        for x,y in dl_te:
            probs.append(torch.sigmoid(model(x)).cpu())
            labels.append(y.cpu())
    probs = torch.cat(probs).numpy(); labels = torch.cat(labels).numpy()
    return model, probs, labels