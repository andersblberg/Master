# ──────────────────────────────────────────────────────────────────────────────
# src/plastic_id/models/cnn.py
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

# ─── GPU / CPU selector ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N_FEATURES: Final[int] = 8  # 8 wavelength bands
N_CLASSES: Final[int] = 6  # HDPE, LDPE, …, PVC


# ───────────────────────────── network definition ────────────────────────────
class _Net(nn.Module):
    def __init__(self, n_filters: int, k_size: int, dropout: float):
        super().__init__()                   # ←  IMPORTANT
        pad = k_size // 2                    # same-length padding (odd k)
        # 1-D convolution along the wavelength axis
        self.conv1   = nn.Conv1d(1,  n_filters,          kernel_size=k_size, padding=pad)
        self.conv2   = nn.Conv1d(n_filters, 2*n_filters, kernel_size=k_size, padding=pad)
        self.dropout = nn.Dropout(p=dropout)

        # flatten → linear classifier
        self.out     = nn.Linear(2 * n_filters * N_FEATURES, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, channels=1, 8)
        x = F.relu(self.conv1(x))  # (batch, 16, 8)
        x = F.relu(self.conv2(x))          # (batch, 2·n_filters, 8)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten: (batch, 32*8)
        return self.out(x)  # logits for N_CLASSES


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
        n_filters: int = 16,
        k_size: int = 3,
        dropout: float = 0.25,
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
        self.n_filters = n_filters
        self.k_size    = k_size
        self.dropout   = dropout

    # ────────────── scikit-learn mandatory methods ──────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        g = torch.Generator().manual_seed(self.seed)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(self._encode_labels(y), dtype=torch.long)

        # expose labels in the scikit-learn-standard place
        self.classes_ = self._classes_

        ds = TensorDataset(X, y)
        n_val = max(1, int(0.1 * len(ds)))  # 10 % for val-loss
        ds_train, ds_val = torch.utils.data.random_split(
            ds, [len(ds) - n_val, n_val], generator=g
        )

        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            ds_train, batch_size=self.batch_size, shuffle=True,
            generator=g, pin_memory=pin
        )
        val_loader = DataLoader(
            ds_val, batch_size=self.batch_size, shuffle=False,
            pin_memory=pin
        )

        self._net = _Net(self.n_filters, self.k_size, self.dropout)
        self._net.to(DEVICE)
        opt = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        crit = nn.CrossEntropyLoss()

        best_val = float("inf")
        since_improved = 0

        for epoch in range(self.epochs):
            self._net.train()
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                loss = crit(self._net(xb), yb)
                loss.backward()
                opt.step()

            # ── validation loss for early-stopping ──
            self._net.eval()
            with torch.no_grad():
                val_loss = 0.0
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)
                    val_loss += crit(self._net(xb), yb).item() * len(xb)
                val_loss /= len(ds_val)

            if val_loss < best_val - 1e-4:  # tiny tolerance
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
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits = self._net(X)
            return self._decode_labels(torch.argmax(logits, dim=1).cpu().numpy())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "_net")
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
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
