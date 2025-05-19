"""
One raw-spectra histogram per PlasticType
==============================================================

Plots match the colour/opacity used in the pooled absorbance histogram.

Example:
    python make_histograms.py               # saves to plots/histograms
    python make_histograms.py results/plots # custom folder
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

# exact palette from the pooled absorbance histogram
fixed_colors = {
    "PET": "#7e57c2",  # purple
    "HDPE": "#43a047",  # green
    "PVC": "#000000",  # black
    "LDPE": "#29b6f6",  # light-blue
    "PP": "#e53935",  # red
    "PS": "#fdd835",  # yellow
}

# --------------------------------------------------
# 2. PREP ------------------------------------------
# --------------------------------------------------
os.makedirs(output_dir, exist_ok=True)
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
    plt.hist(
        values,
        bins=50,
        histtype="stepfilled",  # same look as pooled plot
        alpha=0.45,  # same transparency
        color=fixed_colors.get(ptype, "grey"),
        edgecolor="black",
        linewidth=0.8,
    )
    plt.title(f"Raw spectral values – {ptype}")
    plt.xlabel("Reflectance")
    plt.ylabel("Frequency")
    plt.tight_layout()

    fname = f"{ptype}_hist.{img_format}".replace(" ", "_")
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=300)
    plt.close()

    print(f"✔ saved {fpath}")

print("\nDone – all histograms are in", os.path.abspath(output_dir))
