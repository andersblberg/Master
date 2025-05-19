from __future__ import annotations
from typing import Sequence, Mapping

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def compute_metrics(y_true, y_pred) -> Mapping[str, float]:
    """Return a dict of the core classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
    }


def pretty_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=3)


def precision_recall_per_class(
    y_true,
    y_pred,
    *,
    zero_division: int | float = 0,
) -> pd.DataFrame:
    """
    Return a DataFrame with precision, recall, f1 and support **per class**.

    Columns:  precision | recall | f1 | support
    Index  :  class labels (same dtype as `y_true` / `y_pred`)
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=zero_division,
    )

    return pd.DataFrame(
        {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": supp,
        },
        index=labels,
    )


def make_cm_plot(
    y_true,
    y_pred,
    *,
    labels=None,
    title: str | None = None,
    out_path: Path | str | None = None,
    dpi: int = 200,
):
    """
    Render and save a confusion-matrix plot.

    Parameters
    ----------
    y_true, y_pred : array-likes
    labels         : explicit ordering of class labels
    title          : str – figure title
    out_path       : if given, image is written there (PNG)
    dpi            : resolution for saved image
    Returns
    -------

    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        xticks_rotation="vertical",
        colorbar=False,
        ax=ax,
    )
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

    return fig
