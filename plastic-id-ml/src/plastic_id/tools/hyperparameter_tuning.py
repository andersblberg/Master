#!/usr/bin/env python
"""
One-off hyper-parameter tuner for any registry model
----------------------------------------------------
Usage example
-------------
poetry run python -m plastic_id.tools.hyperparameter_tuning --model rf --trials 50 --outfile tuned_rf.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
args = parser.parse_args()

# ------------------------------------------------------------------ #
# Data + CV splitter
# ------------------------------------------------------------------ #
ds = PlasticDataset(Path(args.csv))
X, y = ds.X, ds.y
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


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


SPACE = {
    "rf": rf_space,
    "rf_par": rf_space,
    "et": et_space,
    "svm_snv": svm_space,
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


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

best_params = study.best_trial.params
print("Best params:", best_params)

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
