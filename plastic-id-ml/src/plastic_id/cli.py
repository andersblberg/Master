"""
Typer CLI for the plastic‑ID ML toolkit
Commands:
    run   – train/evaluate any model in the registry
    noise – same, but with Gaussian noise levels
"""

from __future__ import annotations

import pandas as pd, json, numpy as np
import yaml
import typer

from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
from plastic_id.preprocessing import RowNormalizer, RowSNV, make_pca

from pathlib import Path
from typing import List, Optional

from sklearn.model_selection import StratifiedKFold
from itertools import combinations

from plastic_id.evaluation import _run_dir, save_model, save_reports
from plastic_id.evaluation.metrics import compute_metrics, pretty_report
from plastic_id.data.datasets import PlasticDataset
from plastic_id.models import get_model
from plastic_id.utils.timer import timed
from plastic_id.evaluation.noise import add_gaussian_noise

app = typer.Typer(help="Plastic‑ID ML toolkit", add_completion=False)


def _fill_params_if_empty(cfg: dict):
    """Populate cfg.model.params from configs/model/<name>.yaml if missing/None."""
    if cfg["model"].get("params"):
        return
    cfg["model"]["params"] = {}
    param_file = Path(f"configs/model/{cfg['model']['name']}.yaml")
    if param_file.exists():
        cfg["model"]["params"].update(yaml.safe_load(param_file.read_text()) or {})


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    """If no sub‑command is given, run the baseline RF."""
    if ctx.invoked_subcommand is None:
        cfg = yaml.safe_load(Path("configs/experiment/baseline.yaml").read_text())
        acc = _run_core(cfg)
        typer.echo(json.dumps({"model": "rf", "accuracy": acc}, indent=2))


# ------------------------------------------------------------------ #
# single run (hold-out  or  k-fold CV)
# ------------------------------------------------------------------ #
def _run_core(cfg: dict) -> dict:
    """
    • If the YAML contains an `eval_cv:` stanza → k-fold CV.
    • Otherwise fall back to the classic 80 / 20 split.

    Semantics shared by both modes
    ──────────────────────────────
    – test-noise is injected **only** into the current test partition
    – ablation (dropping wavelengths) is applied to BOTH train & test
    """

    # ––––––––––––– load the full dataset once –––––––––––––––––––––––
    ds = PlasticDataset(Path(cfg["data"]["csv_path"]))

    # =================================================================
    # A) ------------- k-fold cross-validation branch ------------------
    # =================================================================
    if "eval_cv" in cfg:
        cv_cfg = cfg["eval_cv"] or {}
        k = cv_cfg.get("n_splits", 10)
        kf = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=cv_cfg.get("random_state"),
        )

        # collectors ---------------------------------------------------
        fold_acc, fold_f1, fold_prec, fold_rec = [], [], [], []
        y_true_all, y_pred_all = [], []

        # ------------- iterate over CV folds --------------------------
        for tr_idx, te_idx in kf.split(ds.X, ds.y):
            X_train, X_test = ds.X[tr_idx], ds.X[te_idx]
            y_train, y_test = ds.y[tr_idx], ds.y[te_idx]

            # ---- optional wavelength ablation ------------------------
            if "eval_ablation" in cfg:
                from plastic_id.evaluation.ablation import (
                    drop_channels,
                    CHANNEL_IDX,
                )

                chans = cfg["eval_ablation"]["channels"]
                if chans == "SINGLES":
                    chans = [[c] for c in CHANNEL_IDX]
                elif chans == "PAIRS":
                    chans = list(combinations(CHANNEL_IDX, 2))

                X_train = drop_channels(X_train, chans)
                X_test = drop_channels(X_test, chans)

            # ---- optional gaussian noise on TEST only ----------------
            if "eval_noise" in cfg:
                from plastic_id.evaluation.noise import add_gaussian_noise

                pct = cfg["eval_noise"].get("pct", 0.0)
                seed = cfg["eval_noise"].get("rng", None)
                X_test = add_gaussian_noise(X_test, pct, seed)

            # ---- fit / predict ---------------------------------------
            model = get_model(cfg["model"]["name"], cfg["model"]["params"])
            if cfg["model"]["name"].startswith("xgb"):
                le = LabelEncoder().fit(y_train)
                model.fit(X_train, le.transform(y_train))
                y_pred = le.inverse_transform(model.predict(X_test))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # ---- collect fold-wise stuff ------------------------------
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

            fold_acc.append((y_pred == y_test).mean())
            fold_f1.append(f1_score(y_test, y_pred, average="macro"))
            fold_prec.append(precision_score(y_test, y_pred, average="macro"))
            fold_rec.append(recall_score(y_test, y_pred, average="macro"))

        # ========== pooled report (all folds together) ================
        y_true_all = np.asarray(y_true_all)
        y_pred_all = np.asarray(y_pred_all)

        typer.echo(pretty_report(y_true_all, y_pred_all))

        # ---------- artefacts: SINGLE folder --------------------------
        tag = cfg["model"]["name"] + "_cv"
        run_dir = _run_dir(tag)

        # pooled CM + metrics, per-class etc.
        save_reports(
            y_true_all,
            y_pred_all,
            tag,
            model=None,  # no single fold model
            X_test=None,
            run_dir=run_dir,
        )

        # fold-level CSV + JSON
        pd.DataFrame(
            {
                "fold": np.arange(1, k + 1),
                "accuracy": fold_acc,
                "f1_macro": fold_f1,
                "precision": fold_prec,
                "recall": fold_rec,
            }
        ).to_csv(run_dir / "cv_fold_scores.csv", index=False)

        (run_dir / "cv_mean_std.json").write_text(
            json.dumps(
                {
                    "accuracy_mean": float(np.mean(fold_acc)),
                    "accuracy_std": float(np.std(fold_acc)),
                    "f1_macro_mean": float(np.mean(fold_f1)),
                    "f1_macro_std": float(np.std(fold_f1)),
                    "precision_mean": float(np.mean(fold_prec)),
                    "precision_std": float(np.std(fold_prec)),
                    "recall_mean": float(np.mean(fold_rec)),
                    "recall_std": float(np.std(fold_rec)),
                },
                indent=2,
            )
        )

        # ---------- feature-importance & model.joblib -----------------
        # (fit once on the *full* data – optionally after ablation)
        full_X, full_y = ds.X.copy(), ds.y.copy()
        if "eval_ablation" in cfg:
            from plastic_id.evaluation.ablation import drop_channels, CHANNEL_IDX

            chans = cfg["eval_ablation"]["channels"]
            if chans == "SINGLES":
                chans = [[c] for c in CHANNEL_IDX]
            elif chans == "PAIRS":
                chans = list(combinations(CHANNEL_IDX, 2))
            full_X = drop_channels(full_X, chans)

        full_model = get_model(cfg["model"]["name"], cfg["model"]["params"])
        if cfg["model"]["name"].startswith("xgb"):
            le = LabelEncoder().fit(full_y)
            full_model.fit(full_X, le.transform(full_y))
        else:
            full_model.fit(full_X, full_y)

        save_model(full_model, tag, run_dir=run_dir)

        # tree models → feature importance plot
        from plastic_id.evaluation import (
            _maybe_save_feature_importance,
            DEFAULT_WAVE_LABELS,
        )

        _maybe_save_feature_importance(
            full_model,
            run_dir,
            tag=tag,
            wave_labels=DEFAULT_WAVE_LABELS,
        )

        # ---------- console summary -----------------------------------
        acc_arr = np.asarray(fold_acc)
        f1_arr = np.asarray(fold_f1)
        prec_arr = np.asarray(fold_prec)
        rec_arr = np.asarray(fold_rec)

        typer.echo(
            f"\n{k}-fold CV summary\n"
            f"  accuracy    : {acc_arr.mean():.3f} ± {acc_arr.std():.3f}\n"
            f"  macro-F1    : {f1_arr .mean():.3f} ± {f1_arr .std():.3f}\n"
            f"  macro-Prec. : {prec_arr.mean():.3f} ± {prec_arr.std():.3f}\n"
            f"  macro-Recall: {rec_arr.mean():.3f} ± {rec_arr.std():.3f}\n"
        )

        # numbers returned to interactive_runner’s summary table
        return {
            "accuracy": acc_arr.mean(),
            "f1_macro": f1_arr.mean(),
            "precision": prec_arr.mean(),
            "recall": rec_arr.mean(),
        }

    # =================================================================
    # B) -------------- classic 80 / 20 hold-out split -----------------
    # =================================================================
    X_train, X_test = ds.X_train, ds.X_test
    y_train, y_test = ds.y_train, ds.y_test

    # optional ablation
    if "eval_ablation" in cfg:
        from plastic_id.evaluation.ablation import drop_channels, CHANNEL_IDX

        chans = cfg["eval_ablation"]["channels"]
        if chans == "SINGLES":
            chans = [[c] for c in CHANNEL_IDX]
        elif chans == "PAIRS":
            chans = list(combinations(CHANNEL_IDX, 2))
        X_train = drop_channels(X_train, chans)
        X_test = drop_channels(X_test, chans)

    # optional noise on TEST only
    if "eval_noise" in cfg:
        from plastic_id.evaluation.noise import add_gaussian_noise

        pct = cfg["eval_noise"].get("pct", 0.0)
        seed = cfg["eval_noise"].get("rng", None)
        X_test = add_gaussian_noise(X_test, pct, seed)

    # fit & predict
    model = get_model(cfg["model"]["name"], cfg["model"]["params"])
    if cfg["model"]["name"].startswith("xgb"):
        le = LabelEncoder().fit(y_train)
        model.fit(X_train, le.transform(y_train))
        y_pred = le.inverse_transform(model.predict(X_test))
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # single-split artefacts
    typer.echo(pretty_report(y_test, y_pred))

    tag = cfg["model"]["name"]
    save_model(model, tag)
    save_reports(
        y_test,
        y_pred,
        tag,
        model=model,
        X_test=X_test,
    )
    return compute_metrics(y_test, y_pred)


# --------------------------------------------------------------------------- #
# commands                                                                    #
# --------------------------------------------------------------------------- #
@app.command()
def run(
    #                      ↓  Option → Argument  (positional, no --model flag)
    model: Optional[str] = typer.Argument(
        None,  # default = use whatever the YAML says
        help="Model key (rf, svm, mlp, et, xgb). "
        "If omitted, model will be read from the YAML.",
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
    cfg_path: Path = typer.Option(
        Path("configs/experiment/baseline.yaml"), exists=True
    ),
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
    type: str = typer.Option("raw", help="raw | norm | snv | pca20 | pca40 | pca80"),
    rows: int = typer.Option(30, help="How many random spectra to draw"),
    csv_path: Path = typer.Option(
        Path("data/interim/combined_DB22_measurements_sorted.csv"), exists=True
    ),
):
    """Quick look at transformed spectra / PCA scores."""
    ds = PlasticDataset(csv_path)
    X = ds.X_train[:rows, :]  # take first N for speed

    if type == "norm":
        X = RowNormalizer().fit_transform(X)
    elif type == "snv":
        X = RowSNV().fit_transform(X)
    elif type.startswith("pca"):
        n = int(type[3:])  # grabs 20 / 40 / 80
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
