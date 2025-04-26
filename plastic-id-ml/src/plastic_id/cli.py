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

import matplotlib.pyplot as plt
import numpy as np
from plastic_id.preprocessing import RowNormalizer, RowSNV, make_pca

from pathlib import Path
from typing import List, Optional

from sklearn.preprocessing import LabelEncoder

from plastic_id.evaluation import save_model, save_reports
from plastic_id.evaluation.metrics import compute_metrics, pretty_report
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
def _run_core(cfg: dict, *, noise_pct: float | None = None) -> dict:
    """Train & evaluate once, return a dict of metrics."""
    ds = PlasticDataset(Path(cfg["data"]["csv_path"]))
    X_train, X_test = ds.X_train, ds.X_test

    if noise_pct is not None:
        from plastic_id.evaluation.noise import add_gaussian_noise
        X_train = add_gaussian_noise(X_train, noise_pct)
        X_test  = add_gaussian_noise(X_test,  noise_pct)

    model = get_model(cfg["model"]["name"], cfg["model"]["params"])

    # -------- fit / predict ---------------------------------------------------
    if cfg["model"]["name"].startswith("xgb"):
        le = LabelEncoder().fit(ds.y_train)
        y_train_enc = le.transform(ds.y_train)
        with timed("fit"):
            model.fit(X_train, y_train_enc)
        with timed("predict"):
            y_pred_enc = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
    else:
        with timed("fit"):
            model.fit(X_train, ds.y_train)
        with timed("predict"):
            y_pred = model.predict(X_test)
    # -------------------------------------------------------------------------

    metrics = compute_metrics(ds.y_test, y_pred)
    typer.echo(pretty_report(ds.y_test, y_pred))      # nice table

    tag = cfg["model"]["name"]           # e.g. "rf", "xgb", …
    save_model(model, tag)               # artifacts/<tag>.joblib
    save_reports(                     
        ds.y_test,
        y_pred,
        tag,
        model=model,
        X_test=X_test,
    )
    return metrics



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

    res = _run_core(cfg)
    typer.echo(json.dumps({"model": cfg["model"]["name"], **res}, indent=2))



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

    base = _run_core(cfg)["accuracy"]
    typer.echo(f"Baseline {model}: {base:.3f}")

    for p in pct:
        acc = _run_core(cfg, noise_pct=p)["accuracy"]
        typer.echo(f"±{p}% → {acc:.3f}  (Δ {acc - base:+.3f})")

@app.command()
def grid(
    models: List[str] = typer.Option(
        ["rf", "svm", "mlp", "et", "xgb", "cnn"],
        "--model",
        "-m",
        help="One or more model keys",
    ),
    cfg_path: Path = typer.Option(Path("configs/experiment/baseline.yaml"), exists=True),
):
    """Train/eval several models and print a summary table."""
    import pandas as pd
    rows = []
    for m in models:
        typer.echo(f"\n── {m.upper()} ─────────────────────────────────")
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["model"]["name"] = m
        _fill_params_if_empty(cfg)
        res = _run_core(cfg)
        res["model"] = m
        rows.append(res)
    df = pd.DataFrame(rows).set_index("model").sort_values("accuracy", ascending=False)
    typer.echo(df.to_markdown())

# --------------------------------------------------------------------------- #
# visualisation                                                               #
# --------------------------------------------------------------------------- #
@app.command()
def viz(
    type: str = typer.Option(
        "raw",
        help="raw | norm | snv | pca20 | pca40 | pca80"
    ),
    rows: int = typer.Option(
        30, help="How many random spectra to draw"
    ),
    csv_path: Path = typer.Option(
        Path("data/interim/combined_DB22_measurements_sorted.csv"),
        exists=True
    ),
):
    """Quick look at transformed spectra / PCA scores."""
    ds = PlasticDataset(csv_path)
    X = ds.X_train[:rows, :]      # take first N for speed

    if type == "norm":
        X = RowNormalizer().fit_transform(X)
    elif type == "snv":
        X = RowSNV().fit_transform(X)
    elif type.startswith("pca"):
        n = int(type[3:])          # grabs 20 / 40 / 80
        X = make_pca(n).fit_transform(X)
    elif type != "raw":
        typer.echo(f"Unknown --type {type}", err=True)
        raise typer.Exit(code=1)

    plt.figure(figsize=(8, 4))
    plt.plot(X.T, alpha=0.5)
    plt.title(f"{type.upper()} – {rows} spectra")
    plt.xlabel("feature index")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
