#!/usr/bin/env python
"""
Interactive wrapper for `pid grid`

• Lists the dataset config files it can find
• Shows all model keys registered in plastic_id.models.REGISTRY
• Builds and executes the exact `poetry run pid grid …` command
-----------------------------------------------------------------
Place this file anywhere outside src/, e.g.  tools/interactive_runner.py
Run with:  poetry run python tools/interactive_runner.py
-----------------------------------------------------------------
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import typer            # already a dependency of plastic-id-ml
import yaml             # already installed via PyYAML

# --------------------------------------------------------------
# 1‒  discover datasets (YAML files)
# --------------------------------------------------------------
_DEFAULT_CFG_DIR   = Path("configs/experiment")
_OPTIONAL_REG_FILE = Path("configs/interactive_registry.yaml")
#   If you create `interactive_registry.yaml`, it should look like:
#   ----------------------------------------------
#   baseline_clean : configs/experiment/dataset_clean.yaml
#   baseline_full  : configs/experiment/baseline.yaml
#   ----------------------------------------------

def _list_datasets() -> dict[str, str]:
    if _OPTIONAL_REG_FILE.exists():
        # user-maintained registry wins
        mapping: dict[str, str] = yaml.safe_load(_OPTIONAL_REG_FILE.read_text())
        return {k: str(Path(v)) for k, v in mapping.items()}
    # fallback → take every YAML inside configs/experiment
    return {p.stem: str(p) for p in _DEFAULT_CFG_DIR.glob("*.yaml")}

# --------------------------------------------------------------
# 2‒  discover model names from code
# --------------------------------------------------------------
from plastic_id.models import REGISTRY as _MODEL_REGISTRY    # noqa: E402

_MODEL_KEYS = sorted(_MODEL_REGISTRY.keys())

# --------------------------------------------------------------
# 3‒  interactive flow (uses Typer prompts)
# --------------------------------------------------------------
app = typer.Typer(add_completion=False, help="Interactive front-end for pid grid")

@app.command()
def run() -> None:
    """Ask user → launch experiment."""
    datasets = _list_datasets()
    if not datasets:
        typer.secho("No dataset configs found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # ---------- dataset choice ------------------------------------------------
    typer.echo("\n  Available datasets")
    for i, name in enumerate(datasets, 1):
        typer.echo(f"  [{i}]  {name}")
    ds_idx = typer.prompt("Select dataset number", type=int)
    try:
        ds_name  = list(datasets.keys())[ds_idx - 1]
        cfg_file = datasets[ds_name]
    except (IndexError, ValueError):
        typer.secho("Invalid selection.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # ---------- model choice --------------------------------------------------
    typer.echo("\n Available models")
    for i, m in enumerate(_MODEL_KEYS, 1):
        typer.echo(f"  [{i}]  {m}")
    raw = typer.prompt(
        "Select one *or more* model numbers (comma-separated, e.g. 1,5,8)"
    )
    try:
        indices = [int(s.strip()) - 1 for s in raw.split(",")]
        models  = [_MODEL_KEYS[i] for i in indices]
    except Exception:
        typer.secho("Invalid model list.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # ---------- build command -------------------------------------------------
    cmd: list[str] = (
        ["poetry", "run", "pid", "grid", "--cfg-path", cfg_file] +
        sum([["-m", m] for m in models], [])
    )

    typer.echo("\n Executing:\n   " + " ".join(cmd) + "\n")
    # inherit stdout/stderr
    subprocess.run(cmd, check=True)    # will raise if pid exits non-zero


if __name__ == "__main__":
    app()
