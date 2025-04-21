"""
Typer CLI for the plastic‑ID ML toolkit
Commands:
    run   – train/evaluate any model in the registry
    noise – same, but with Gaussian noise levels
"""

from __future__ import annotations

import json
import yaml
import typer

from pathlib import Path
from typing import List, Optional

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score        

from plastic_id.data.datasets import PlasticDataset
from plastic_id.models import get_model
from plastic_id.utils.timer import timed

app = typer.Typer(help="Plastic‑ID ML toolkit", add_completion=False)


def _fill_params_if_empty(cfg: dict):
    """Populate cfg.model.params from configs/model/<name>.yaml if missing/None."""
    if cfg["model"].get("params"):
        return  
    cfg["model"]["params"] = {}  
    param_file = Path(f"configs/model/{cfg['model']['name']}.yaml")
    if param_file.exists():
        cfg["model"]["params"].update(
            yaml.safe_load(param_file.read_text()) or {}
        )


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    """If no sub‑command is given, run the baseline RF."""
    if ctx.invoked_subcommand is None:
        cfg = yaml.safe_load(Path("configs/experiment/baseline.yaml").read_text())
        acc = _run_core(cfg)
        typer.echo(json.dumps({"model": "rf", "accuracy": acc}, indent=2))


# --------------------------------------------------------------------------- #
# internal helper                                                             #
# --------------------------------------------------------------------------- #
def _run_core(cfg: dict, *, noise_pct: float | None = None) -> float:
    """Train & evaluate once, return accuracy."""
    ds = PlasticDataset(Path(cfg["data"]["csv_path"]))
    X_train, X_test = ds.X_train, ds.X_test

    if noise_pct is not None:
        from plastic_id.evaluation.noise import add_gaussian_noise
        X_train = add_gaussian_noise(X_train, noise_pct)
        X_test = add_gaussian_noise(X_test, noise_pct)

    model = get_model(cfg["model"]["name"], cfg["model"]["params"])

    if cfg["model"]["name"] == "xgb":
        # encode your string classes into 0..n_classes-1
        le = LabelEncoder().fit(ds.y_train)
        y_train_enc = le.transform(ds.y_train)

        with timed("fit"):
            model.fit(X_train, y_train_enc)

        with timed("predict"):
            y_pred_enc = model.predict(X_test)

        # turn the numeric predictions back into your original strings
        y_pred = le.inverse_transform(y_pred_enc)
    else:
        with timed("fit"):
            model.fit(X_train, ds.y_train)
        with timed("predict"):
            y_pred = model.predict(X_test)

    return accuracy_score(ds.y_test, y_pred)  # now defined!


# --------------------------------------------------------------------------- #
# commands                                                                    #
# --------------------------------------------------------------------------- #
@app.command()
def run(
    #                      ↓  Option → Argument  (positional, no --model flag)
    model: Optional[str] = typer.Argument(
        None,  # default = use whatever the YAML says
        help="Model key (rf, svm, mlp, et, xgb). "
             "If omitted, model will be read from the YAML."
    ),
    cfg_path: Path = typer.Option(
        Path("configs/experiment/baseline.yaml"), exists=True
    ),
):
    cfg = yaml.safe_load(cfg_path.read_text())
    if model:
        cfg["model"]["name"] = model
        _fill_params_if_empty(cfg)

    acc = _run_core(cfg)
    typer.echo(json.dumps({"model": cfg['model']['name'], "accuracy": acc}, indent=2))


@app.command()
def noise(
    pct: List[float] = typer.Option(
        [0.5, 1, 2], help="Noise pct list, e.g. --pct 0.5 1 2"
    ),
    model: str = typer.Option(
        "rf", help="Model key (rf, svm, mlp, et, xgb if installed)"
    ),
    cfg_path: Path = typer.Option(
        Path("configs/experiment/baseline.yaml"), exists=True
    ),
):
    """Run baseline + several noise levels; print Δ‑accuracy."""
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["model"]["name"] = model

    base = _run_core(cfg)
    typer.echo(f"Baseline {model}: {base:.3f}")
    for p in pct:
        acc = _run_core(cfg, noise_pct=p)
        typer.echo(f"±{p}% → {acc:.3f}  (Δ {acc - base:+.3f})")


if __name__ == "__main__":
    app()
