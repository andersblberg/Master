"""
Pooled spectral histogram coloured by PlasticType
=========================================================================
* Loads the cleaned DB22 CSV.
* Optionally converts raw reflectance to absorbance (0–1).
* Collects every value of every wavelength column into one long vector
  per class, then overlays their histograms.
* Saves the figure and (with --show) displays it.

Example
-------
poetry run python histogram_by_class.py --csv data/interim/combined_DB22_measurements_sorted_clean.csv --out figs/class_hist.png --absorbance
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------- argument parsing
ap = argparse.ArgumentParser(description="Overlayed histogram by PlasticType")
ap.add_argument("--csv", required=True, help="input CSV file")
ap.add_argument("--out", default="class_hist.png", help="output image path")
ap.add_argument(
    "--absorbance",
    action="store_true",
    help="plot absorbance (1 - R/R_max) instead of raw reflectance",
)
ap.add_argument("--show", action="store_true", help="display the window")
ap.add_argument("--bins", type=int, default=60, help="number of histogram bins")
args = ap.parse_args()

# -------------------------------------------------- I/O
df = pd.read_csv(args.csv)
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------- spectral columns
raw_cols = sorted(
    [
        c
        for c in df.columns
        if c.startswith("nm") and not (c.endswith("_snv") or c.endswith("_norm"))
    ],
    key=lambda nm: int(nm[2:]),
)

# -------------------------------------------------- reflectance → absorbance
if args.absorbance:
    R_max = df[raw_cols].max().replace(0, np.nan)
    df[raw_cols] = 1 - df[raw_cols].divide(R_max)
    y_label = "Absorbance (0–1)"
else:
    y_label = "Reflectance (raw counts)"

# -------------------------------------------------- colour palette & class order
fixed_colors = {
    "PET": "#7e57c2",  # purple
    "HDPE": "#43a047",  # green
    "PVC": "#000000",  # black
    "LDPE": "#29b6f6",  # light-blue
    "PP": "#e53935",  # red
    "PS": "#fdd835",  # yellow
}
class_order = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]

# -------------------------------------------------- plot
plt.figure(figsize=(7, 4.5))

for ptype in class_order:
    subset = df.loc[df["PlasticType"] == ptype, raw_cols].values.ravel()
    plt.hist(
        subset,
        bins=args.bins,
        histtype="stepfilled",
        alpha=0.45,
        density=True,
        color=fixed_colors[ptype],
        label=ptype,
    )

plt.xlabel(y_label.split()[0])  # “Absorbance” or “Reflectance”
plt.ylabel("Density")
title_suffix = "absorbance" if args.absorbance else "reflectance"
# plt.title(f"Pooled {title_suffix} distribution by PlasticType")
plt.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

plt.legend(frameon=False, ncol=6, loc="upper right")
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"histogram saved to {out_path.resolve()}")

if args.show:
    plt.show()
else:
    plt.close()
