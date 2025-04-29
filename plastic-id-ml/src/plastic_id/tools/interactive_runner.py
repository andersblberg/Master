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


# ─── helper to create a throw-away YAML for ablation ────────────
import tempfile, json, shutil

_ABL_WAVELENGTHS = [940, 1050, 1200, 1300, 1450, 1550, 1650, 1720]

def _prepare_ablation_cfg(base_cfg_path: str) -> str:
    """
    Interactively ask which channels to drop, clone the YAML into a
    temp file, inject the chosen list under `eval_ablation.channels`,
    and return the *new* path.
    """
    typer.echo("\n  Ablation - choose wavelengths to drop")
    typer.echo("  [0]  type your own comma-separated list")
    for i, wv in enumerate(_ABL_WAVELENGTHS, 1):
        typer.echo(f"  [{i}]  {wv}")
    typer.echo("  [9]  every single channel (loop)")
    typer.echo("  [10] every pair of channels (loop)")
    choice = typer.prompt("Select option", type=int)

    # ── decide the list/loop spec ───────────────────────────────
    if choice == 0:
        # chans = [int(x) for x in typer.prompt("Comma-separated wavelengths").split(",")]
        raw = [int(x) for x in typer.prompt(
            "Comma-separated list (index 1-8 or wavelength)").split(",")]
        # translate 1-8 → wavelengths; leave real wavelengths untouched

        chans: list[int] = []
        for val in raw:
            if 1 <= val <= 8:                   # menu index → wavelength
                chans.append(_ABL_WAVELENGTHS[val - 1])
            elif val in _ABL_WAVELENGTHS:       # already a wavelength
                chans.append(val)
            else:
                # ── graceful error and exit ───────────────────────────
                typer.secho(
                    f"  ✘  '{val}' is neither 1-8 nor a valid wavelength "
                    f"({_ABL_WAVELENGTHS}).  Aborted.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(1)
    elif 1 <= choice <= 8:
        chans = [_ABL_WAVELENGTHS[choice - 1]]
    elif choice == 9:
        chans = "SINGLES"
    elif choice == 10:
        chans = "PAIRS"
    else:
        typer.secho(" Invalid choice – aborted.", fg=typer.colors.RED); raise typer.Exit(1)

    # ── clone YAML and inject field ─────────────────────────────
    cfg = yaml.safe_load(Path(base_cfg_path).read_text())
    cfg["eval_ablation"] = {"channels": chans}
    tmp = Path(tempfile.mkstemp(suffix=".yaml", prefix="abl_")[1])
    tmp.write_text(yaml.dump(cfg, sort_keys=False))
    return str(tmp)



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
    typer.echo("\n  Configurations")
    for i, name in enumerate(datasets, 1):
        typer.echo(f"  [{i}]  {name}")
    ds_idx = typer.prompt("Select configuration number", type=int)

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

    # ── ablation: replace cfg file on the fly if needed ─────────
    if ds_name == "ablation":
        cfg_file = _prepare_ablation_cfg(cfg_file)

    # ── build & launch command ─────────────────────────────────
    cmd: list[str] = (
        ["poetry", "run", "pid", "grid", "--cfg-path", cfg_file] +
        sum([["-m", m] for m in models], [])
    )

    typer.echo("\n Executing:\n   " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)   # raises if pid exits with error


if __name__ == "__main__":
    app()
