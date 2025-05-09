#!/usr/bin/env python3
"""
plot_mean_absorbance.py – mean absorbance (0‒0.8) per PlasticType
=================================================================
* Converts raw reflectance to absorbance on a 0–1 scale:
  A = 1 − R / R_max (R_max = global max at each wavelength).
* Plots each class in a fixed colour.
* Legend is a single horizontal row above the plot.
* Y-axis limited to 0–0.8.
* Light grid lines at every major tick make it easy to read values.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------- argument parsing
ap = argparse.ArgumentParser(description="Mean absorbance (0‒0.8) per Plastic Type")
ap.add_argument("--csv", required=True, help="input CSV file")
ap.add_argument("--out", default="mean_absorbance.png", help="output image path")
ap.add_argument("--show", action="store_true", help="display the window")
args = ap.parse_args()

# -------------------------------------------------- load data
df = pd.read_csv(args.csv)
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------- discover raw wavelength columns
raw_cols = sorted(
    [c for c in df.columns if c.startswith("nm")
     and not (c.endswith("_snv") or c.endswith("_norm"))],
    key=lambda nm: int(nm[2:])
)
wavelengths = np.array([int(c[2:]) for c in raw_cols])  # e.g. [940, 1050, …]

# -------------------------------------------------- reflectance -> absorbance (0–1) then mean
R_max = df[raw_cols].max().replace(0, np.nan)
absorbance_df = 1 - df[raw_cols].divide(R_max)

mean_abs = (
    absorbance_df.join(df["PlasticType"])
    .groupby("PlasticType")[raw_cols]     # no key column → no deprecation warning
    .mean()
)

# -------------------------------------------------- fixed colours
fixed_colors = {
    "PET":  "#7e57c2",  # purple
    "HDPE": "#43a047",  # green
    "PVC":  "#000000",  # black
    "LDPE": "#29b6f6",  # light-blue
    "PP":   "#e53935",  # red
    "PS":   "#fdd835",  # yellow
}
missing = set(mean_abs.index) - fixed_colors.keys()
if missing:
    raise KeyError(f"No colour specified for: {', '.join(missing)}")

# -------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(7, 4.5))

for ptype, series in mean_abs.iterrows():
    ax.plot(
        wavelengths,
        series.values,
        label=ptype,
        color=fixed_colors[ptype],
        linewidth=1.8,
    )

# axis labels & limits
ax.set(
    xlabel="Wavelength (nm)",
    ylabel="Absorbance (0‒0.8)",
    title=None,
    ylim=(0, 0.8),
)
ax.set_xticks(wavelengths, [str(w) for w in wavelengths])

# --------------- grid lines -------------------------
ax.grid(
    True,                    # turn grid on
    which="major",           # major ticks only
    axis="both",
    linestyle=":",           # dotted
    linewidth=0.6,
    alpha=0.7,
)

# --------------- legend above the plot --------------
order = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]      # desired order
h, l = ax.get_legend_handles_labels()                   # current order
handles = [h[l.index(k)] for k in order]                # re-index handles
labels  = order       

fig.legend(
    handles, labels,
    title=None,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(mean_abs),
    frameon=False,
)
fig.subplots_adjust(top=0.80)   # leave room for legend

# -------------------------------------------------- save / show
fig.tight_layout()
fig.savefig(out_path, dpi=300, bbox_inches="tight")  # no clipping
print(f"✔ saved {out_path.resolve()}")

if args.show:
    plt.show()
else:
    plt.close(fig)
