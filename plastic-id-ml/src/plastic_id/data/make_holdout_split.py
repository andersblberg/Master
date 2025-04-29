"""
Make one stratified 15 % test split and save row-indices to disk
(run ONCE – commit the resulting .npy file to the repo).
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

RAW = Path("data/interim/combined_DB22_measurements_sorted_clean.csv")

df = pd.read_csv(RAW)
y  = df["label"].values                     # <-- change if your label column differs
sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(sss.split(df, y))

np.save("data/splits/test_idx.npy",  test_idx)
np.save("data/splits/dev_idx.npy",   train_idx)
print("Saved index arrays to data/splits/")
