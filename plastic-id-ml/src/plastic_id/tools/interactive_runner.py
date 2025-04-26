#!/usr/bin/env python
"""
Interactive wrapper around `pid grid`

• Lists available dataset-config YAMLs
• Lets you choose 1-N registered model keys
• Lets you choose ONE pre-processing (raw / norm / snv)
• Builds and launches the exact `poetry run pid grid …` command
-----------------------------------------------------------------
Run with:  poetry run python src/plastic_id/tools/interactive_runner.py
-----------------------------------------------------------------
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer            # Typer is already a dependency
import yaml             # installed via PyYAML

# ───────────────────────────────────────────────────────────────
# 1  Discover datasets
# ───────────────────────────────────────────────────────────────
_CFG_DIR   = Path("configs/experiment")
_USER_REG  = Path("configs/interactive_registry.yaml")


def _list_datasets() -> dict[str, str]:
    """Return {pretty_name: yaml_path}."""
    if _USER_REG.exists():
        mapping: dict[str, str] = yaml.safe_load(_USER_REG.read_text()) or {}
        return {k: str(Path(v)) for k, v in mapping.items()}

    return {p.stem: str(p) for p in _CFG_DIR.glob("*.yaml")}


# ───────────────────────────────────────────────────────────────
# 2  Discover model keys
# ───────────────────────────────────────────────────────────────
from plastic_id.models import REGISTRY as _REGISTRY  # noqa: E402

_MODEL_KEYS = sorted(_REGISTRY.keys())

# ───────────────────────────────────────────────────────────────
# 3  CLI
# ───────────────────────────────────────────────────────────────
app = typer.Typer(add_completion=False, help="Interactive front-end for pid grid")


@app.command()
def run() -> None:
    datasets = _list_datasets()
    if not datasets:
        typer.secho("No dataset configs found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # ── dataset prompt ─────────────────────────────────────────
    typer.echo("\n  Available datasets")
    for i, name in enumerate(datasets, 1):
        typer.echo(f"  [{i}]  {name}")
    ds_idx = typer.prompt("Select dataset number", type=int)

    try:
        ds_name  = list(datasets.keys())[ds_idx - 1]
        cfg_file = datasets[ds_name]
    except (IndexError, ValueError):
        typer.secho(" Invalid selection.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # ── model prompt ───────────────────────────────────────────
    typer.echo("\n  Available models")
    for i, m in enumerate(_MODEL_KEYS, 1):
        typer.echo(f"  [{i}]  {m}")
    raw = typer.prompt(
        "Select one *or more* model numbers (comma-separated, e.g. 1,5,8)"
    )

    try:
        indices      = [int(s.strip()) - 1 for s in raw.split(",")]
        models_base  = [_MODEL_KEYS[i] for i in indices]
    except Exception:
        typer.secho(" Invalid model list.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # ── pre-processing prompt ──────────────────────────────────
    typer.echo("\n  Pre-processing")
    typer.echo("  [1]  raw   (no scaling)")
    typer.echo("  [2]  norm  (row normaliser)")
    typer.echo("  [3]  snv   (row SNV)")
    prep = input("Select pre-processing number: ").strip()

    suffix_map = {"1": "", "2": "_norm", "3": "_snv"}
    suffix = suffix_map.get(prep)
    if suffix is None:
        typer.secho(" Invalid choice – aborted.", fg=typer.colors.RED)
        raise typer.Exit(1)

    models = [m + suffix for m in models_base]

    # ── build & launch command ─────────────────────────────────
    cmd: list[str] = (
        ["poetry", "run", "pid", "grid", "--cfg-path", cfg_file] +
        sum([["-m", m] for m in models], [])
    )

    typer.echo("\n Executing:\n   " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)   # raises if pid exits with error


if __name__ == "__main__":
    app()
