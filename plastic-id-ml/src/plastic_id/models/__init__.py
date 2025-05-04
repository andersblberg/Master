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
try:  # PyTorch installed?
    from .cnn import CNNClassifier  # noqa: D
except Exception:  # any import failure → disable
    CNNClassifier = None  # type: ignore

try:  # XGBoost installed?
    from xgboost import XGBClassifier  # noqa: D
except ModuleNotFoundError:
    XGBClassifier = None  # type: ignore
# ---------------------------------------------------------------

REGISTRY: Dict[str, Callable[[dict[str, Any]], object]] = {
    # plain baselines ---------------------------------------------------------
    "rf": lambda cfg: RandomForestClassifier(**cfg),
    "rf_par": lambda cfg: RandomForestClassifier(n_jobs=-1, **cfg),
    "et": lambda cfg: ExtraTreesClassifier(**cfg),
    "mlp": lambda cfg: MLPClassifier(**cfg),
    "mlp_tuned": lambda cfg: MLPClassifier(
        hidden_layer_sizes=(64, 64),
        learning_rate_init=7.314912068413965e-4,
        alpha=7.4903600713776925e-3,
        batch_size=32,
        **cfg,
    ),
    "svm": lambda cfg: SVC(probability=True, **cfg),  # default SVM
}

# ─── SVM variants (pipelines) ──────────────────────────────────
REGISTRY.update(
    {
        "svm_raw": lambda c: SVC(probability=True, **c),
        "svm_norm": lambda c: make_pipeline(
            RowNormalizer(), SVC(probability=True, **c)
        ),
        "svm_snv": lambda c: make_pipeline(RowSNV(), SVC(probability=True, **c)),
        "svm_pca2": lambda c: make_pipeline(make_pca(2), SVC(probability=True, **c)),
        "svm_pca4": lambda c: make_pipeline(make_pca(4), SVC(probability=True, **c)),
        "svm_pca8": lambda c: make_pipeline(make_pca(8), SVC(probability=True, **c)),
        "svm_bal": lambda c: SVC(probability=True, class_weight="balanced", **c),
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
    REGISTRY["cnn_tuned"] = lambda cfg: CNNClassifier(
        n_filters=16,
        k_size=3,
        dropout=0.07874344760988262,
        lr=2.397945525670308e-4,
        batch_size=32,
        **cfg,
    )

# ---------------------------------------------------------------------------
_SUFFIX_MAP = {
    "_norm": RowNormalizer(),
    "_snv": RowSNV(),
}


def get_model(name: str, cfg: dict[str, Any]):
    """
    Return an *unfitted* model or model-pipeline by key.

    ─ accepted keys ───────────────────────────────────────────────
    rf, svm, et, xgb, …                ← raw spectra
    svm_norm, rf_snv, cnn_norm, …      ← any <model> + _norm / _snv
    ----------------------------------------------------------------
    """
    # --- 1. does the key carry a preprocessing suffix? -------------
    for suf, transformer in _SUFFIX_MAP.items():
        if name.endswith(suf):
            base_name = name[: -len(suf)]
            if base_name not in REGISTRY:
                raise ValueError(
                    f"Unknown base model '{base_name}' " f"(derived from '{name}')"
                )
            base_model = REGISTRY[base_name](cfg)
            return make_pipeline(transformer, base_model)

    # --- 2. plain model --------------------------------------------
    try:
        return REGISTRY[name](cfg)
    except KeyError as exc:
        raise ValueError(
            f"Unknown model key '{name}'. " f"Available: {', '.join(sorted(REGISTRY))}"
        ) from exc
