#!/usr/bin/env python3
"""
---------------------------------------------------------
Exploratory Data Analysis for combined_DB22_measurements
---------------------------------------------------------

• Basic info & missing‐value scan
• Descriptive stats of numeric features
• Z‐score table (saved)
• Correlation matrix & heat-map (saved)
• Class-balance bar plot (saved)
• SNV check (mean ≈ 0, std ≈ 1)
• Range check of *_norm columns
• Std-dev of each raw wavelength •per PlasticType* (printed & saved)

Run:
    python eda_plastics.py --csv data/interim/combined_DB22_measurements_sorted_clean.csv \
                           --out results/eda
"""
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="EDA for plastics NIR data")
    ap.add_argument("--csv", required=True, help="Path to input CSV file")
    ap.add_argument("--out", default="eda_results", help="Output directory")
    return ap.parse_args()


# ----------------------------------------------------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {csv_path} – shape = {df.shape}")
    return df


# ----------------------------------------------------------------------
def basic_overview(df: pd.DataFrame, out_dir: Path):
    print("\n--- BASIC INFO -------------------------")
    print(df.info())

    miss = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values per column (top 10):")
    print(miss.head(10))
    miss.to_csv(out_dir / "missing_values.csv")


# ----------------------------------------------------------------------
def identify_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    raw_cols = [
        c
        for c in numeric_cols
        if c.startswith("nm") and not (c.endswith("_snv") or c.endswith("_norm"))
    ]
    snv_cols = [c for c in df.columns if c.endswith("_snv")]
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    return numeric_cols, raw_cols, snv_cols, norm_cols


# ----------------------------------------------------------------------
def describe_numeric(df: pd.DataFrame, numeric_cols, out_dir):
    desc = df[numeric_cols].describe().T
    print("\n--- DESCRIPTIVE STATISTICS (numeric) ---")
    print(desc.head())
    desc.to_csv(out_dir / "descriptive_stats.csv")


# ----------------------------------------------------------------------
def zscore_table(df, numeric_cols, out_dir):
    zscores = df[numeric_cols].apply(stats.zscore, nan_policy="omit")
    zscores.to_csv(out_dir / "zscores_table.csv", index=False)
    print(f"\nZ-score table saved ({zscores.shape[0]}×{zscores.shape[1]})")


# ----------------------------------------------------------------------
def correlation(df, numeric_cols, out_dir):
    corr = df[numeric_cols].corr()
    corr.to_csv(out_dir / "correlation_matrix.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws=dict(label="ρ"))
    plt.title("Correlation Heat-map (numeric features)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=300)
    plt.close()
    print("Correlation heat-map saved.")


# ----------------------------------------------------------------------
def class_balance(df, out_dir):
    counts = df["PlasticType"].value_counts().sort_index()
    counts.to_csv(out_dir / "class_counts.csv")
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("PlasticType distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=300)
    plt.close()
    print("Class-balance plot saved.")


# ----------------------------------------------------------------------
def snv_and_norm_checks(df, snv_cols, norm_cols, out_dir):
    if snv_cols:
        snv_stats = df[snv_cols].agg(["mean", "std"]).T
        snv_stats.to_csv(out_dir / "snv_stats.csv")
        print("\nSNV sanity check (mean≈0, std≈1):")
        print(snv_stats.head())

    if norm_cols:
        norm_range = df[norm_cols].agg(["min", "max"]).T
        norm_range.to_csv(out_dir / "norm_ranges.csv")
        print("\nNormalised feature ranges:")
        print(norm_range.head())


# ----------------------------------------------------------------------
def std_per_wavelength_per_class(df, raw_cols, out_dir):
    std_tbl = df.groupby("PlasticType")[raw_cols].std()
    std_tbl.to_csv(out_dir / "std_per_wavelength_per_class.csv")
    print("\n--- Std-dev of each wavelength per PlasticType ---")
    print(std_tbl.head())


# ----------------------------------------------------------------------
def pca_preview(df, raw_cols, out_dir):
    """Optional quick PCA to gauge dimensionality (saved plot)."""
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X = scaler.fit_transform(df[raw_cols])
    pca = PCA().fit(X)
    expl = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(expl) + 1), expl, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA scree plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "pca_scree.png", dpi=300)
    plt.close()
    print("PCA scree plot saved.")


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.csv))
    basic_overview(df, out_dir)

    numeric_cols, raw_cols, snv_cols, norm_cols = identify_columns(df)
    describe_numeric(df, numeric_cols, out_dir)
    zscore_table(df, numeric_cols, out_dir)
    correlation(df, numeric_cols, out_dir)
    class_balance(df, out_dir)
    snv_and_norm_checks(df, snv_cols, norm_cols, out_dir)
    std_per_wavelength_per_class(df, raw_cols, out_dir)
    pca_preview(df, raw_cols, out_dir)

    print(f"\nEDA complete — results in: {out_dir.resolve()}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
