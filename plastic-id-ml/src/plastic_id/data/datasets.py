from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    'nm940', 'nm1050', 'nm1200', 'nm1300', 'nm1450',
    'nm1550', 'nm1650', 'nm1720',
]
TARGET_COLUMN = 'Category'

# class PlasticDataset:
#     def __init__(self, csv_path: Path, test_size: float = 0.2, seed: int = 42):
#         self.df = pd.read_csv(csv_path)
#         self.X = self.df[FEATURE_COLUMNS].values
#         self.y = self.df[TARGET_COLUMN].values
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             self.X, self.y, test_size=test_size, stratify=self.y, random_state=seed)
        
CHANNEL_IDX = {940:0, 1050:1, 1200:2, 1300:3, 1450:4, 1550:5, 1650:6, 1720:7}

class PlasticDataset:
    def __init__(
        self,
        csv_path: Path | str,
        test_size: float = 0.2,
        seed: int = 42,
        drop_channels: list[int] | None = None,   # NEW
    ):
        self.df = pd.read_csv(csv_path)
        self.X = self.df[FEATURE_COLUMNS].values
        self.y = self.df[TARGET_COLUMN].values

        # --- optional ablation -------------------------------------------------
        if drop_channels:
            idx = [CHANNEL_IDX[w] for w in drop_channels]
            self.X = np.delete(self.X, idx, axis=1)
        # ----------------------------------------------------------------------

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=seed,
        )