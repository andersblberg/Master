"""
One-off hyper-parameter tuner for any registry model
----------------------------------------------------
Usage example for all models:
-----------------------------
poetry run python -m plastic_id.tools.hyperparameter_tuning --model rf --trials 100 --outfile tuned_rf.yaml --logdir artifacts\hyperpar_logs

or

poetry run python -m plastic_id.tools.hyperparameter_tuning --model svm_snv --trials 100 --outfile configs/experiment/svm_snv_tuned.yaml --logdir artifacts\hyperpar_logs

or

poetry run python -m plastic_id.tools.hyperparameter_tuning --model et --trials 100 --outfile configs/experiment/et_tuned.yaml --logdir artifacts\hyperpar_logs

or

poetry run python -m plastic_id.tools.hyperparameter_tuning --model xgb --trials 100 --outfile configs/experiment/xgb_tuned.yaml --logdir artifacts\hyperpar_logs

or

poetry run python -m plastic_id.tools.hyperparameter_tuning --model cnn --trials 20 --outfile configs/experiment/cnn_tuned.yaml --logdir artifacts\hyperpar_logs

or

poetry run python -m plastic_id.tools.hyperparameter_tuning --model mlp --trials 40 --outfile configs/experiment/mlp_tuned.yaml --logdir artifacts\hyperpar_logs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna
import yaml
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from plastic_id.data.datasets import PlasticDataset
from plastic_id.models import get_model  # registry helper

# ------------------------------------------------------------------ #
# Argument parsing
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv", default="data/interim/combined_DB22_measurements_sorted_clean.csv"
)
parser.add_argument(
    "--model", required=True, help="registry key, e.g. rf, et, svm_pca8"
)
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--outfile", type=Path, required=True)
parser.add_argument(
    "--logdir",
    type=Path,
    default=Path("logs"),  # <── all logs end up here
    help="Directory for *.log files (will be created if missing).",
)
args = parser.parse_args()


# ------------------------------------------------------------------ #
# Logging helper
# console  and  a file  logs/<outfile-stem>.log
args.logdir.mkdir(parents=True, exist_ok=True)
log_path = args.logdir / (args.outfile.stem + ".log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w"),  # to terminal and file
    ],
)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------ #
# Data + CV splitter
# ------------------------------------------------------------------ #
# ds = PlasticDataset(Path(args.csv))
# X, y = ds.X, ds.y
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

ds = PlasticDataset(Path(args.csv))
X = ds.X
le = LabelEncoder()
y = le.fit_transform(ds.y)  # y is now 0 … 5 instead of strings


# ------------------------------------------------------------------ #
# Search spaces for a few common models
# ------------------------------------------------------------------ #
def rf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        # allow “no limit” by explicitly listing None
        "max_depth": trial.suggest_categorical(
            "max_depth", [None] + list(range(5, 55, 5))
        ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }


def et_space(trial):
    return {
        # trees
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        # allow *either* an integer depth or unlimited (None)
        "max_depth": trial.suggest_categorical(
            "max_depth", [None] + list(range(5, 55, 5))  # None, 5, 10, … 50
        ),
        # standard choices
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": "balanced",
        "random_state": 42,
    }


def svm_space(trial):
    """
    log-scale grid for C and gamma, kernel kept at default "rbf".
    """
    C = 10 ** trial.suggest_float("logC", -1, 3)  # 0.1 … 1000
    gamma = 10 ** trial.suggest_float("logGamma", -4, 0)  # 1e-4 … 1
    return {
        "C": C,
        "gamma": gamma,
    }


def xgb_space(trial):
    return {
        # capacity / complexity -------------------------------------------------
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        # regularisation & pruning ---------------------------------------------
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        # bookkeeping — **keep everything except use_label_encoder**
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }


def mlp_space(trial):
    """Reasonable grid for sklearn.neural_network.MLPClassifier"""
    return {
        # network size ----------------------------------------------------------
        "hidden_layer_sizes": tuple(
            trial.suggest_categorical(
                "layer_sizes", [(64,), (128,), (64, 64), (128, 64), (128, 128)]
            )
        ),
        # optimisation ----------------------------------------------------------
        "learning_rate_init": trial.suggest_float("lr_init", 1e-4, 3e-2, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),  # ℓ2
        "batch_size": trial.suggest_categorical("batch", [32, 64, 128]),
        "max_iter": 200,  # ← keep short for CV
        "early_stopping": True,
        "random_state": 42,
    }


def cnn_space(trial):
    return {
        "n_filters": trial.suggest_categorical("n_filters", [8, 16, 32]),
        "k_size": trial.suggest_categorical("k_size", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch", [16, 32, 64]),
        # epochs / patience left at default (150/20) – they don’t affect the net’s shape
    }


SPACE = {
    "rf": rf_space,
    "rf_par": rf_space,
    "et": et_space,
    "svm_snv": svm_space,
    "xgb": xgb_space,
    "mlp": mlp_space,
    "cnn": cnn_space,
}

if args.model not in SPACE:
    raise ValueError(
        f"No search-space defined for '{args.model}'. "
        "Add one in hyperparameter_tuning.py:SPACE."
    )


# ------------------------------------------------------------------ #
# Optuna objective
# ------------------------------------------------------------------ #
def objective(trial):
    params = SPACE[args.model](trial)
    clf = get_model(args.model, params)
    score = cross_val_score(clf, X, y, cv=kf, scoring="accuracy", n_jobs=-1).mean()
    return score


def log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Write one line per finished trial to the log file *and* console."""
    if trial.value is not None and not np.isnan(trial.value):
        logger.info(
            f"trial={trial.number:03d}  " f"acc={trial.value:.4f}  " f"{trial.params}"
        )
    else:  # failed / nan trial – still record params
        logger.info(f"trial={trial.number:03d}  acc=FAIL  {trial.params}")


study = optuna.create_study(direction="maximize")
study.optimize(
    objective,
    n_trials=args.trials,
    show_progress_bar=True,
    callbacks=[log_callback],
)

best_params = study.best_trial.params
logger.info(f"Best params: {best_params}")
logger.info(f"Best score : {study.best_value:0.4f}")
# ------------------------------------------------------------------ #
# Write YAML so it can be re-used as an experiment config
# ------------------------------------------------------------------ #
out_dict = {
    "data": {"csv_path": str(Path(args.csv))},
    "eval_cv": {"n_splits": 10, "random_state": 42},
    "model": {"name": args.model, "params": best_params},
}
args.outfile.write_text(yaml.dump(out_dict, sort_keys=False))
print(f"wrote tuned config to {args.outfile}")
