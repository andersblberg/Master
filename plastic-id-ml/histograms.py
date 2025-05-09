#!/usr/bin/env python3
"""
Save one histogram per PlasticType to <output_dir>.

Usage (hard-coded path):
    python make_histograms.py

Usage (command-line path):
    python make_histograms.py results/plots
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. CONFIGURATION ---------------------------------
# --------------------------------------------------
csv_path = "data/interim/combined_DB22_measurements_sorted_clean.csv"
output_dir = sys.argv[1] if len(sys.argv) > 1 else "plots/histograms"
img_format = "png"  # png | pdf | svg …

# --------------------------------------------------
# 2. PREP ------------------------------------------
# --------------------------------------------------
os.makedirs(output_dir, exist_ok=True)  # create folder if missing
df = pd.read_csv(csv_path)

raw_cols = [
    c
    for c in df.columns
    if c.startswith("nm") and not (c.endswith("_snv") or c.endswith("_norm"))
]
plastic_types = df["PlasticType"].unique()

# --------------------------------------------------
# 3. MAKE & SAVE PLOTS -----------------------------
# --------------------------------------------------
for ptype in plastic_types:
    subset = df.loc[df["PlasticType"] == ptype, raw_cols]
    values = subset.values.ravel()

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=50, edgecolor="black")
    plt.title(f"Raw spectral values – {ptype}")
    plt.xlabel("Reflectance")
    plt.ylabel("Frequency")
    plt.tight_layout()

    fname = f"{ptype}_hist.{img_format}".replace(" ", "_")
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=300)
    plt.close()  # release memory

    print(f"✔ saved {fpath}")

print("\nDone – all histograms are in", os.path.abspath(output_dir))
