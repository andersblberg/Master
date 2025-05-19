from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
)

# ── default axis labels for 8-channel spectra ──────────────────────────────
DEFAULT_WAVE_LABELS = [
    "940 nm",
    "1050 nm",
    "1200 nm",
    "1300 nm",
    "1450 nm",
    "1550 nm",
    "1650 nm",
    "1720 nm",
]


from .metrics import compute_metrics, precision_recall_per_class

ARTIFACTS_ROOT = Path("artifacts")


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def _run_dir(tag: str) -> Path:
    """
    Return a unique directory such as  artifacts/20250412T101212_et/
    and make sure it exists.  Keeps runs isolated so files never overwrite.
    """
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    p = ARTIFACTS_ROOT / f"{ts}_{tag}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _make_cm_plot(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
):
    """Return a matplotlib *Figure* with a nicely labelled confusion matrix."""
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        colorbar=False,
    )
    disp.ax_.set_xticklabels(disp.ax_.get_xticklabels(), rotation=45, ha="right")

    # annotate each cell with the count
    for (i, j), v in np.ndenumerate(disp.confusion_matrix):
        disp.ax_.text(
            j,
            i,
            int(v),
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )
    plt.tight_layout()
    return disp.figure_


def _maybe_save_feature_importance(
    model, run_dir: Path, *, tag: str, wave_labels: list[str]
):
    """Save bar-plot if the model exposes .feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        return

    imp = model.feature_importances_
    order = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(wave_labels)[order], imp[order], color="steelblue")
    ax.set_xlabel("Mean decrease in impurity (relative importance)")
    ax.set_title(f"{tag.upper()} feature importance")
    plt.tight_layout()
    fig.savefig(run_dir / "feature_importance.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# public helpers (used from cli.py)
# --------------------------------------------------------------------------- #
def save_model(model, tag: str, *, run_dir: Path | None = None) -> Path:
    if run_dir is None:  # NEW
        run_dir = _run_dir(tag)  # NEW
    path = run_dir / "model.joblib"
    joblib.dump(model, path)
    return path


def save_reports(
    y_true,
    y_pred,
    tag: str,
    run_dir: Path | None = None,
    *,
    model=None,
    X_test: np.ndarray | None = None,
    wave_labels: list[str] | None = None,
) -> None:
    """
    Persist all evaluation artefacts for one run in its dedicated folder:

    ├─ confusion_matrix.png
    ├─ metrics.json               (overall accuracy / macro-F1 …)
    ├─ per_class_metrics.csv      (precision / recall per label)
    ├─ feature_importance.png     (tree models only)
    └─ PR_curve_<class>.png       (one per class if predict_proba available)
    """
    run_dir = run_dir or _run_dir(tag)

    # 1) confusion matrix ------------------------------------------------------
    cm_fig = _make_cm_plot(y_true, y_pred, labels=np.unique(y_true))
    cm_fig.savefig(run_dir / "confusion_matrix.png", dpi=300)
    plt.close(cm_fig)

    # 2) overall metrics -------------------------------------------------------
    overall = compute_metrics(y_true, y_pred)
    (run_dir / "metrics.json").write_text(json.dumps(overall, indent=2))

    # 3) per-class metrics -----------------------------------------------------
    cls_df = precision_recall_per_class(y_true, y_pred)
    cls_df.to_csv(run_dir / "per_class_metrics.csv", index=False)

    # 4) feature importance (if available) ------------------------------------
    # if model is not None:
    #     _maybe_save_feature_importance(
    #         model,
    #         run_dir,
    #         tag=tag,
    #         wave_labels=[
    #             "940 nm",
    #             "1050 nm",
    #             "1200 nm",
    #             "1300 nm",
    #             "1450 nm",
    #             "1550 nm",
    #             "1650 nm",
    #             "1720 nm",
    #         ],
    #     )
    if model is not None:
        _maybe_save_feature_importance(
            model,
            run_dir,
            tag=tag,
            wave_labels=wave_labels or DEFAULT_WAVE_LABELS,
        )

    # 5) PR curves (needs predict_proba) --------------------------------------
    if model is not None and X_test is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        labels = np.unique(y_true)
        for i, lab in enumerate(labels):
            p, r, _ = precision_recall_curve(
                (np.asarray(y_true) == lab).astype(int), proba[:, i]
            )
            plt.step(r, p, where="post")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR curve – {lab}")
            plt.tight_layout()
            plt.savefig(run_dir / f"PR_curve_{lab}.png", dpi=300)
            plt.close()


# --------------------------------------------------------------------------- #
# make the run-directory helper importable from cli.py
# --------------------------------------------------------------------------- #
__all__ = ["_run_dir", "save_model", "save_reports"]
