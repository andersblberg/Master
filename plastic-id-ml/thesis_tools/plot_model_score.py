"""
Bar charts of model mean accuracy at three noise gains (×3, ×6, ×9).

Directory layout expected
-------------------------
<results_root>/
    GAINx3/
        20250418T120401_rf_cv/        <-- contains cv_mean_std.json
        20250418T120402_svm_cv/
        ...
    GAINx6/
        ...
    GAINx9/
        ...

Each inner directory must include a *cv_mean_std.json* with an
"accuracy_mean" entry.

Usage
-----
python plot_noise_gains.py <results_root> [-o OUTPUT_DIR]

Creates:
    OUTPUT_DIR/accuracy_gain_x3.png
    OUTPUT_DIR/accuracy_gain_x6.png
    OUTPUT_DIR/accuracy_gain_x9.png
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# ── configuration ─────────────────────────────────────────────────────────
METRIC_KEY = "accuracy_mean"
METRIC_LABEL = "Mean Accuracy"
COLOUR_PALETTE: Sequence[str] = (
    "#e15759",
    "#edc948",
    "#f28e2b",
    "#9c755f",
    "#b07aa1",
    "#4e79a7",
    "#59a14f",
)
LEGEND_LABELS = {
    "mlp cv ReReMeter Units": "MLP Plastic Scanner",
    "svm snv cv tuned": "SVM (SNV)",
    "rf cv": "RF",
    "mlp tuned cv": "MLP",
    "xgb cv tuned": "XGB",
    "et cv tuned": "ET",
    "cnn cv tuned": "1D-CNN",
}
ALPHA, EDGEWIDTH = 0.7, 1.2


# ── helpers ───────────────────────────────────────────────────────────────
def _label_from_dir(d: Path) -> str:
    m = re.match(r"^\d{8}T\d{6}_(.*)$", d.name)
    return (m.group(1) if m else d.name).replace("_", " ")


def collect_scores(base: Path) -> Dict[str, float]:
    """Return {model label → accuracy_mean} for one gain folder."""
    out: Dict[str, float] = {}
    for js in base.rglob("cv_mean_std.json"):
        try:
            val = float(json.loads(js.read_text())[METRIC_KEY])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        out[_label_from_dir(js.parent)] = val
    if not out:
        raise RuntimeError(f"No usable cv_mean_std.json in {base}")
    return out


def make_plot(scores: Dict[str, float], title: str, output: Path, figsize=(6, 6)):
    """Single-metric bar chart (one bar per model)."""
    models = sorted(scores.items(), key=lambda kv: kv[1])  # low → high
    labels, vals = zip(*models)

    idx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)

    for i, (lab, val) in enumerate(models):
        c = COLOUR_PALETTE[i % len(COLOUR_PALETTE)]
        ax.bar(
            idx[i],
            val,
            0.6,
            label=LEGEND_LABELS.get(lab, lab),
            color=c,
            edgecolor=c,
            linewidth=EDGEWIDTH,
            alpha=ALPHA,
        )

    ax.set_xticks(idx)
    ax.set_xticklabels(
        [LEGEND_LABELS.get(l, l) for l in labels], rotation=45, ha="right"
    )
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
def main():
    ap = argparse.ArgumentParser(description="Plot accuracy at noise gains.")
    ap.add_argument(
        "results_root",
        type=Path,
        help="directory that holds GAINx3/, GAINx6/, GAINx9/ …",
    )
    ap.add_argument(
        "-o", "--outdir", type=Path, default=Path("."), help="output directory"
    )
    args = ap.parse_args()

    for gain in ("GAINx3", "GAINx6", "GAINx9"):
        gain_dir = args.results_root / gain
        if not gain_dir.is_dir():
            sys.stderr.write(f"⚠  {gain_dir} not found – skipped\n")
            continue
        scores = collect_scores(gain_dir)
        out_png = args.outdir / f"accuracy_{gain.lower()}.png"
        make_plot(scores, f"Accuracy at Noise {gain[-2:]}", out_png)


if __name__ == "__main__":
    main()
