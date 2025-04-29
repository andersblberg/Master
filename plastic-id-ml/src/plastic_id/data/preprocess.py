#!/usr/bin/env python
"""
Script to merge & preprocess raw spectroscopic data.
"""

# ─── Path‑hack so that `src/` is on sys.path when run directly ────────────────
import os, sys
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import yaml
import pandas as pd
from src.data.loaders import load_raw
from src.plastic_id.utils.timer import Timer


def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    with Timer("Merge DataFrames", verbose=True):
        return pd.concat(dfs, ignore_index=True)


def drop_missing_spectra(df: pd.DataFrame, spectrum_col: str) -> pd.DataFrame:
    with Timer("Drop Missing Spectra", verbose=True):
        before = len(df)
        df = df[df[spectrum_col].notna()].copy()
        print(f"Dropped {before - len(df)} rows lacking '{spectrum_col}'")
        return df


def normalize_spectra(df: pd.DataFrame, spectrum_cols: list[str]) -> pd.DataFrame:
    with Timer("Normalize Spectra", verbose=True):
        for col in spectrum_cols:
            mn, mx = df[col].min(), df[col].max()
            df[col] = (df[col] - mn) / (mx - mn)
        return df


def organize_categories(
    df: pd.DataFrame,
    plastic_col:  str,
    other_labels: list[str] = ["reference", "calibration"]
) -> pd.DataFrame:
    with Timer("Organize & Sort Categories", verbose=True):
        mask_other = df[plastic_col].str.lower().isin(other_labels)
        plastics = sorted(df.loc[~mask_other, plastic_col].unique())
        df['Category'] = df[plastic_col].where(~mask_other, 'Other')
        df['Category'] = pd.Categorical(
            df['Category'],
            categories=plastics + ['Other'],
            ordered=True
        )
        return df.sort_values('Category').reset_index(drop=True)


def preprocess_all(
    dfs:             list[pd.DataFrame],
    spectrum_cols:   list[str],
    plastic_col:     str,
    spectrum_col:    str
) -> pd.DataFrame:
    df = merge_dataframes(dfs)
    df = drop_missing_spectra(df, spectrum_col)
    df = normalize_spectra(df, spectrum_cols)
    df = organize_categories(df, plastic_col)
    return df


def main(config_path: str):
    # 1) Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 2) Load raw CSVs
    dfs = load_raw(cfg)

    # 3) Preprocess
    processed = preprocess_all(
        dfs,
        spectrum_cols=cfg['data']['spectrum_cols'],
        plastic_col=cfg['data']['plastic_col'],
        spectrum_col=cfg['data']['spectrum_col']
    )

    # 4) Save processed data
    out_csv     = cfg['data']['processed_path']
    summary_csv = cfg['data']['summary_path']
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    processed.to_csv(out_csv, index=False)
    print(f"Saved processed data to {out_csv}")

    # 5) Save summary counts
    counts = (
        processed['Category']
        .value_counts()
        .rename_axis('Category')
        .reset_index(name='Count')
    )
    counts.to_csv(summary_csv, index=False)
    print(f"Saved summary counts to {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge & preprocess raw data")
    parser.add_argument(
        "-c", "--config",
        default="configs/baseline.yml",
        help="Path to preprocessing YAML config"
    )
    args = parser.parse_args()
    main(args.config)
