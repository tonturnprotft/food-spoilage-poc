import pathlib, pandas as pd, torch
from torchvision import transforms as T
from PIL import Image

class TunaImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: pathlib.Path, root: pathlib.Path, train: bool):
        self.df = pd.read_csv(csv_path)
        self.root = root
        # simple 80 / 20 split by session order
        sessions = (
            self.df[["day", "session"]]
            .drop_duplicates()
            .sort_values(["day", "session"])
        )
        n_train = int(len(sessions) * 0.8)
        keep = sessions.head(n_train) if train else sessions.tail(len(sessions) - n_train)
        self.df = self.df.merge(keep, on=["day", "session"])
        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row.img_path).convert("RGB")
        return self.tf(img), torch.tensor(row.label, dtype=torch.float32)