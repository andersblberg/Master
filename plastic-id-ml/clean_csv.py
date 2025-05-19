import pandas as pd
from pathlib import Path

# ---------- settings ----------------------------------------------------------
LABEL_COL = "PlasticType"
labels_to_drop = [
    "unknown",
    "reference",
    "Other",
]
SRC_CSV = Path("data/interim/combined_DB22_measurements_sorted.csv")
DST_CSV = SRC_CSV.with_name(SRC_CSV.stem + "_clean.csv")
# -----------------------------------------------------------------------------


raw = pd.read_csv(SRC_CSV)

clean = raw[~raw[LABEL_COL].str.lower().isin([x.lower() for x in labels_to_drop])]

clean.to_csv(DST_CSV, index=False)
print("Saved →", DST_CSV.resolve())
print("\nRemaining class counts:\n", clean[LABEL_COL].value_counts())
