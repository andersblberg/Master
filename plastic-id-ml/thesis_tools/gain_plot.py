"""
Bar charts of model mean accuracy at three noise gains (×3, ×6, ×9).

"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# ── configuration ─────────────────────────────────────────────────────────
METRIC_KEY = "accuracy_mean"
METRIC_LABEL = "Mean Accuracy"

# Colour palette (same order you used originally)
COLOUR_PALETTE: Sequence[str] = (
    "#e15759",
    "#edc948",
    "#f28e2b",
    "#9c755f",
    "#b07aa1",
    "#4e79a7",
    "#59a14f",
)

# Canonical IDs in the order that maps to the palette above
CANONICAL = [
    "et",  # red
    "rf",  # yellow
    "xgb",  # orange
    "svm",  # brown-grey
    "cnn",  # purple
    "mlp",  # blue
    "psmlp",  # green  (Plastic-Scanner MLP)
]

# Friendly names and colour look-up
LEGEND = {
    "et": ("ET", COLOUR_PALETTE[6]),
    "rf": ("RF", COLOUR_PALETTE[5]),
    "xgb": ("XGB", COLOUR_PALETTE[4]),
    "svm": ("SVM (SNV)", COLOUR_PALETTE[3]),
    "cnn": ("1D-CNN", COLOUR_PALETTE[2]),
    "mlp": ("MLP", COLOUR_PALETTE[1]),
    "psmlp": ("MLP Plastic Scanner", COLOUR_PALETTE[0]),
}

ALPHA, EDGEWIDTH = 0.7, 1.2


# ── helpers ───────────────────────────────────────────────────────────────
def _canonical_id(path: Path) -> str | None:
    """
    Map a folder name to its model ID.
    """
    # drop leading timestamp
    name = re.sub(r"^\d{8}T\d{6}_", "", path.name)

    # normalise: convert '_' → ' ', then remove trailing '_<digit>'
    name = re.sub(r"_\d+$", "", name).replace("_", " ")
    low = name.lower().strip()  # e.g. "mlp cv rere"

    if "et" in low and "rf" not in low:
        return "et"
    if low.startswith("rf"):
        return "rf"
    if "xgb" in low:
        return "xgb"
    if low.startswith("svm"):
        return "svm"
    if low.startswith("cnn"):
        return "cnn"

    # Plastic-Scanner MLP variant (matches either “… rere” or “… ps”)
    if "mlp" in low and ("rere" in low or "ps" in low):
        return "psmlp"

    if low.startswith("mlp"):
        return "mlp"
    return None


def collect_scores(gain_dir: Path) -> Dict[str, float]:
    """Return {canonical_id → accuracy_mean} for one GAINx* directory."""
    out: Dict[str, float] = {}
    for js in gain_dir.rglob("cv_mean_std.json"):
        try:
            acc = float(json.loads(js.read_text())[METRIC_KEY])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        cid = _canonical_id(js.parent)
        if cid:
            out[cid] = acc
        else:
            sys.stderr.write(f"unknown folder skipped: {js.parent.name}\n")
    if not out:
        raise RuntimeError(f"No usable {METRIC_KEY} in {gain_dir}")
    return out


def make_plot(
    scores: Dict[str, float], title: str, output: Path, figsize=(6, 6)
) -> None:
    """Draw a single-metric bar chart and save to output"""
    # sort bars low → high for readability
    items = sorted(scores.items(), key=lambda kv: kv[1])
    ids, vals = zip(*items)
    idx = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=figsize)
    for i, cid in enumerate(ids):
        label, colour = LEGEND[cid]
        ax.bar(
            idx[i],
            vals[i],
            0.6,
            label=label,
            color=colour,
            edgecolor=colour,
            linewidth=EDGEWIDTH,
            alpha=ALPHA,
        )

    ax.set_xticks(idx)
    ax.set_xticklabels([LEGEND[cid][0] for cid in ids], rotation=45, ha="right")
    ax.set_ylabel(METRIC_LABEL)
    ax.set_ylim(0.1, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(axis="y", which="major", ls="--", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Plot saved → {output}")


# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Plot accuracy at noise gains.")
    ap.add_argument(
        "results_root", type=Path, help="directory that holds GAINx3/, GAINx6/, GAINx9/"
    )
    ap.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("."),
        help="output directory (created if missing)",
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for gain in ("GAINx3", "GAINx6", "GAINx9"):
        gdir = args.results_root / gain
        if not gdir.is_dir():
            sys.stderr.write(f"⚠ {gdir} not found – skipped\n")
            continue
        data = collect_scores(gdir)
        make_plot(
            data,
            f"Accuracy at Noise {gain[-2:]}",
            args.outdir / f"accuracy_{gain.lower()}.png",
        )


if __name__ == "__main__":
    main()
