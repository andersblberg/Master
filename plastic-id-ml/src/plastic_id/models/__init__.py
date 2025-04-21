"""Central registry for all classification models."""
from __future__ import annotations
from typing import Dict, Callable, Any

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# ---------- optional imports ------------------------------------------
try:
    from .cnn import CNNClassifier          # needs torch
except Exception:                           # ImportError, SyntaxError, ...
    CNNClassifier = None

try:
    from xgboost import XGBClassifier       # optional dependency
except ModuleNotFoundError:
    XGBClassifier = None
# ----------------------------------------------------------------------

REGISTRY: Dict[str, Callable[[dict[str, Any]], object]] = {
    "rf":  lambda cfg: RandomForestClassifier(**cfg),
    "svm": lambda cfg: SVC(probability=True, **cfg),
    "mlp": lambda cfg: MLPClassifier(**cfg),
    "et":  lambda cfg: ExtraTreesClassifier(**cfg),
}

if XGBClassifier is not None:
    REGISTRY["xgb"] = lambda cfg: XGBClassifier(use_label_encoder=False, **cfg)

if CNNClassifier is not None:
    REGISTRY["cnn"] = lambda cfg: CNNClassifier(**cfg)


def get_model(name: str, cfg: dict[str, Any]):
    """Instantiate a model by key. Raises ValueError on unknown key."""
    try:
        return REGISTRY[name](cfg)
    except KeyError:  # pragma: no cover – defensive guard
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(REGISTRY)}"
        ) from None