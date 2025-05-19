#!/usr/bin/env python3
"""
histogram_per_class.py – individual density histograms
======================================================
* Same x– and y–axis meaning as the pooled histogram script:
  - density=True  → bar areas integrate to 1 for each class
  - identical colours, opacity, and binning
* Writes one PNG per PlasticType.

Example
-------
poetry run python histogram_per_class.py \
      --csv data/interim/combined_DB22_measurements_sorted_clean.csv \
      --outdir figs/per_class \
      --absorbance
      poetry run python make_histograms.py --csv data/interim/combined_DB22_measurements_sorted_clean.csv --outdir figs/per_class --absorbance
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------- CLI
ap = argparse.ArgumentParser(description="One density histogram per PlasticType")
ap.add_argument("--csv", required=True, help="input CSV file")
ap.add_argument(
    "--outdir",
    default="figs/per_class",
    help="directory to save PNGs (default: figs/per_class)",
)
ap.add_argument(
    "--absorbance",
    action="store_true",
    help="plot absorbance (1 - R/R_max) instead of raw reflectance",
)
ap.add_argument("--bins", type=int, default=60, help="number of histogram bins")
ap.add_argument("--show", action="store_true", help="display each window")
args = ap.parse_args()

out_dir = Path(args.outdir)
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------- load data
df = pd.read_csv(args.csv)
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
    x_label = "Absorbance (0–1)"
    suffix = "abs"
else:
    x_label = "Reflectance (raw counts)"
    suffix = "refl"

# -------------------------------------------------- colour palette & order
fixed_colors = {
    "PET": "#7e57c2",
    "HDPE": "#43a047",
    "PVC": "#000000",
    "LDPE": "#29b6f6",
    "PP": "#e53935",
    "PS": "#fdd835",
}
class_order = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]


# ---------- 1. GLOBAL Y-MAX (add this) -------------------------------
global_ymax = 0.0
for ptype in class_order:
    vals = df.loc[df["PlasticType"] == ptype, raw_cols].values.ravel()
    counts, _ = np.histogram(vals, bins=args.bins, density=True)
    global_ymax = max(global_ymax, counts.max())
global_ymax *= 1.05  # +5 % head-room
# ---------------------------------------------------------------------

# -------------------------------------------------- per-class plots
for ptype in class_order:
    values = df.loc[df["PlasticType"] == ptype, raw_cols].values.ravel()

    plt.figure(figsize=(7, 4.5))
    plt.hist(
        values,
        bins=args.bins,
        histtype="stepfilled",
        alpha=0.45,
        density=True,  # ← same as pooled plot
        color=fixed_colors[ptype],
    )
    plt.xlabel(x_label)
    plt.ylabel("Density")
    plt.title(f"{ptype} – {'Absorbance' if args.absorbance else 'Reflectance'}")
    plt.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    plt.ylim(0, global_ymax)
    plt.tight_layout()

    fpath = out_dir / f"{ptype}_hist_{suffix}.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()
    print(f"✔ saved {fpath.resolve()}")

print("\nDone – individual histograms written to", out_dir.resolve())
