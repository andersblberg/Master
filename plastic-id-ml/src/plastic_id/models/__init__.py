# src/plastic_id/models/__init__.py
"""Central registry for all classification models."""
from __future__ import annotations
from typing import Callable, Dict, Any

from sklearn.pipeline import make_pipeline
from plastic_id.preprocessing import RowNormalizer, RowSNV, make_pca

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# ---------- optional back‑ends ------------------------------------------------
try:
    from .cnn import CNNClassifier          # needs torch
except Exception:
    CNNClassifier = None                    # noqa: N816  (keep ALL_CAPS style)

try:
    from xgboost import XGBClassifier       # optional
except ModuleNotFoundError:
    XGBClassifier = None
# ------------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[[dict[str, Any]], object]] = {
    "rf":  lambda cfg: RandomForestClassifier(**cfg),
    "svm": lambda cfg: SVC(probability=True, **cfg),
    "mlp": lambda cfg: MLPClassifier(**cfg),
    "et":  lambda cfg: ExtraTreesClassifier(**cfg),
}

REGISTRY["svm_raw"]  = lambda c: SVC(probability=True, **c)
REGISTRY["svm_norm"] = lambda c: make_pipeline(RowNormalizer(), SVC(probability=True, **c))
REGISTRY["svm_snv"]  = lambda c: make_pipeline(RowSNV(),       SVC(probability=True, **c))
REGISTRY["svm_pca2"]  = lambda c: make_pipeline(make_pca(2),  SVC(probability=True, **c))
REGISTRY["svm_pca4"]  = lambda c: make_pipeline(make_pca(4),  SVC(probability=True, **c))
REGISTRY["svm_pca8"]  = lambda c: make_pipeline(make_pca(8),  SVC(probability=True, **c))

def _balanced_svc(cfg: dict[str, Any]) -> SVC:
    """SVC with class_weight='balanced' + user overrides."""
    return SVC(probability=True, class_weight="balanced", **cfg)

REGISTRY["svm_bal"] = _balanced_svc
REGISTRY["svm_pca8_bal"] = lambda c: make_pipeline(
    make_pca(8),
    SVC(probability=True, class_weight="balanced", **c)
)


if XGBClassifier:
    REGISTRY["xgb"] = lambda cfg: XGBClassifier(use_label_encoder=False, **cfg)
if CNNClassifier:
    REGISTRY["cnn"] = lambda cfg: CNNClassifier(**cfg)


def get_model(name: str, cfg: dict[str, Any]):
    try:
        return REGISTRY[name](cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}") from exc

