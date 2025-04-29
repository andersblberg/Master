# ──────────────────────────────────────────────────────────────────────────────
# src/plastic_id/models/cnn.py
# A tiny fully-connected “CNN” (really an MLP) for the 8-band spectra.
# It follows the scikit-learn API so the rest of the code-base can call
# .fit / .predict / .predict_proba just like with RF, SVM, …
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


N_FEATURES: Final[int] = 8            # 8 wavelength bands
N_CLASSES:  Final[int] = 6            # HDPE, LDPE, …, PVC


# ───────────────────────────── network definition ────────────────────────────
class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-D convolution along the wavelength axis
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)

        # flatten → linear classifier
        self.out   = nn.Linear(32 * N_FEATURES, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (batch, channels=1, 8)
        x = F.relu(self.conv1(x))   # (batch, 16, 8)
        x = F.relu(self.conv2(x))   # (batch, 32, 8)
        x = x.view(x.size(0), -1)   # flatten: (batch, 32*8)
        return self.out(x)          # logits for N_CLASSES


# ───────────────────────── wrapper: scikit-learn style ───────────────────────
class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Small PyTorch net wrapped in scikit-learn API.

    Parameters
    ----------
    lr : float
        Adam learning-rate.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    weight_decay : float
        L2 regularisation factor.
    patience : int
        Early-stopping patience (epochs without val-loss improvement).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        patience: int = 20,
        seed: int = 42,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed

    # ────────────── scikit-learn mandatory methods ──────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        g = torch.Generator().manual_seed(self.seed)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(self._encode_labels(y), dtype=torch.long)

        ds      = TensorDataset(X, y)
        n_val   = max(1, int(0.1 * len(ds)))        # 10 % for val-loss
        ds_train, ds_val = torch.utils.data.random_split(ds, [len(ds)-n_val, n_val], generator=g)

        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True,  generator=g)
        val_loader   = DataLoader(ds_val,   batch_size=self.batch_size, shuffle=False)

        self._net = _Net()
        opt  = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.CrossEntropyLoss()

        best_val = float("inf")
        since_improved = 0

        for epoch in range(self.epochs):
            self._net.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = crit(self._net(xb), yb)
                loss.backward()
                opt.step()

            # ── validation loss for early-stopping ──
            self._net.eval()
            with torch.no_grad():
                val_loss = sum(
                    crit(self._net(xb), yb).item() * len(xb) for xb, yb in val_loader
                ) / len(ds_val)

            if val_loss < best_val - 1e-4:   # tiny tolerance
                best_val = val_loss
                since_improved = 0
                self._best_state = self._net.state_dict()  # save best params
            else:
                since_improved += 1
                if since_improved >= self.patience:
                    break  # early stop

        # load best weights
        self._net.load_state_dict(self._best_state)
        self._net.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "_net")
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self._net(X)
            return self._decode_labels(torch.argmax(logits, dim=1).cpu().numpy())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "_net")
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self._net(X)
            return F.softmax(logits, dim=1).cpu().numpy()

    # ────────────── helper: label ↔ integer mapping ─────────────
    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_classes_"):
            self._classes_ = np.unique(y)
            self._class_to_idx = {c: i for i, c in enumerate(self._classes_)}
        return np.vectorize(self._class_to_idx.get)(y)

    def _decode_labels(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(self._classes_)[idx]
