# src/plastic_id/models/cnn.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Simple1DCNN(nn.Module):
    def __init__(self, n_channels: int = 8, n_classes: int = 10, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(p_drop),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → 64 × 1
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):             # x shape: (B, 8)
        return self.net(x.unsqueeze(1))


class CNNClassifier:
    """Scikit‑learn style wrapper around the above network."""
    def __init__(self, epochs: int = 200, lr: float = 1e-3, batch: int = 64):
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.le = LabelEncoder()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y):
        y = self.le.fit_transform(y)
        n_classes = len(self.le.classes_)
        self.model = Simple1DCNN(X.shape[1], n_classes).to(self.device)

        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        dl = DataLoader(ds, batch_size=self.batch, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss_fn(self.model(xb), yb).backward()
                opt.step()

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.tensor(X, dtype=torch.float32).to(self.device)
            )
        preds = logits.argmax(dim=1).cpu().numpy()
        return self.le.inverse_transform(preds)
