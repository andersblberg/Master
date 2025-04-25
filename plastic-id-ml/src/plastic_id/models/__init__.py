# ───────────────────────────────────────────────────────────────
# src/plastic_id/models/__init__.py
# Central registry: maps a short string → callable that returns a
# fully-configured estimator or sklearn.pipeline.Pipeline.
# ───────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Callable, Dict

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from plastic_id.preprocessing import RowNormalizer, RowSNV, make_pca

# -------- optional back-ends -----------------------------------
try:                                          # PyTorch installed?
    from .cnn import CNNClassifier            # noqa: D
except Exception:                             # any import failure → disable
    CNNClassifier = None                      # type: ignore

try:                                          # XGBoost installed?
    from xgboost import XGBClassifier         # noqa: D
except ModuleNotFoundError:
    XGBClassifier = None                      # type: ignore
# ---------------------------------------------------------------

REGISTRY: Dict[str, Callable[[dict[str, Any]], object]] = {
    # plain baselines ---------------------------------------------------------
    "rf":  lambda cfg: RandomForestClassifier(**cfg),
    "et":  lambda cfg: ExtraTreesClassifier(**cfg),
    "mlp": lambda cfg: MLPClassifier(**cfg),
    "svm": lambda cfg: SVC(probability=True, **cfg),          # default SVM
}

# ─── SVM variants (pipelines) ──────────────────────────────────
REGISTRY.update(
    {
        "svm_raw":   lambda c: SVC(probability=True, **c),
        "svm_norm":  lambda c: make_pipeline(RowNormalizer(), SVC(probability=True, **c)),
        "svm_snv":   lambda c: make_pipeline(RowSNV(),       SVC(probability=True, **c)),
        "svm_pca2":  lambda c: make_pipeline(make_pca(2),    SVC(probability=True, **c)),
        "svm_pca4":  lambda c: make_pipeline(make_pca(4),    SVC(probability=True, **c)),
        "svm_pca8":  lambda c: make_pipeline(make_pca(8),    SVC(probability=True, **c)),
        "svm_bal":   lambda c: SVC(probability=True, class_weight="balanced", **c),
        "svm_pca8_bal": lambda c: make_pipeline(
            make_pca(8),
            SVC(probability=True, class_weight="balanced", **c),
        ),
    }
)

# ─── optional models (only if libs available) ─────────────────
if XGBClassifier is not None:
    REGISTRY["xgb"] = lambda cfg: XGBClassifier(use_label_encoder=False, **cfg)

if CNNClassifier is not None:
    REGISTRY["cnn"] = lambda cfg: CNNClassifier(**cfg)

# ---------------------------------------------------------------------------
def get_model(name: str, cfg: dict[str, Any]):
    """
    Return an *unfitted* model/pipeline by key, or raise a helpful error.
    """
    try:
        return REGISTRY[name](cfg)
    except KeyError as exc:
        raise ValueError(
            f"Unknown model key '{name}'. "
            f"Available: {', '.join(sorted(REGISTRY))}"
        ) from exc


# at the very bottom of src/plastic_id/models/__init__.py
if CNNClassifier:                       # <- class is truthy
    REGISTRY["cnn"] = lambda cfg: CNNClassifier(**cfg)
