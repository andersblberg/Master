from __future__ import annotations
from typing import Sequence, Mapping

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
)

def compute_metrics(y_true, y_pred) -> Mapping[str, float]:
    """Return a dict of the core classification metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall":    recall_score(y_true, y_pred, average="macro"),
    }

def pretty_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=3)
