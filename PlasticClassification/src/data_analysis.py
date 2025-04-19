import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def data_analysis():
    # Determine the absolute path to this script's directory (Core/)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define key paths
    data_path = os.path.join(script_dir, "Data", "combined_measurements_cleaned.csv")
    plot_base_dir = os.path.join(script_dir, "Data", "plots")
    hist_dir = os.path.join(plot_base_dir, "histograms")
    corr_dir = os.path.join(plot_base_dir, "correlation")

    # Create folders if they don't exist
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)

    # 1) Load dataset
    if not os.path.exists(data_path):
        print(f"ERROR: File not found at '{data_path}'")
        return

    df = pd.read_csv(data_path)
    print("-------------------------------------------------")
    print("LOADING DATA")
    print("-------------------------------------------------")
    print(f"Data loaded from: {data_path}")
    print(f"Shape: {df.shape}")

    print("\n-------------------------------------------------")
    print("COLUMNS & DATA TYPES")
    print("-------------------------------------------------")
    print(df.dtypes)

    print("\n-------------------------------------------------")
    print("HEAD (first 5 rows)")
    print("-------------------------------------------------")
    print(df.head())

    print("\n-------------------------------------------------")
    print("MISSING VALUES")
    print("-------------------------------------------------")
    print(df.isnull().sum())

    print("\n-------------------------------------------------")
    print("DESCRIPTIVE STATISTICS (NUMERIC COLUMNS)")
    print("-------------------------------------------------")
    print(df.describe())

    # Classify columns
    all_cols = df.columns.tolist()
    raw_cols = [c for c in all_cols if c.startswith("nm") and "_snv" not in c and "_norm" not in c]
    snv_cols = [c for c in all_cols if "_snv" in c]
    norm_cols = [c for c in all_cols if "_norm" in c]

    print("\n-------------------------------------------------")
    print("COLUMN CLASSIFICATION")
    print("-------------------------------------------------")
    print("Raw columns:", raw_cols)
    print("SNV columns:", snv_cols)
    print("Normalized columns:", norm_cols)

    # Histograms
    print("\n-------------------------------------------------")
    print("PLOTTING HISTOGRAMS FOR NUMERIC COLUMNS")
    print("-------------------------------------------------")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        plt.figure()
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Distribution of {col}")
        outpath = os.path.join(hist_dir, f"{col}_hist.png")
        plt.savefig(outpath)
        plt.close()
        print(f"Saved histogram -> {outpath}")

    # Correlation heatmap
    print("\n-------------------------------------------------")
    print("CORRELATION HEATMAP")
    print("-------------------------------------------------")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap of Numeric Columns")
        corr_path = os.path.join(corr_dir, "correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(corr_path)
        plt.close()
        print(f"Saved correlation heatmap -> {corr_path}")
    else:
        print("Not enough numeric columns to produce a correlation heatmap.")

    # PlasticType distribution
    if "PlasticType" in df.columns:
        print("\n-------------------------------------------------")
        print("PLASTIC TYPE DISTRIBUTION")
        print("-------------------------------------------------")
        print(df["PlasticType"].value_counts())

    # SNV and norm checks
    if snv_cols:
        print("\n-------------------------------------------------")
        print("CHECKING SNV COLUMNS (MEAN, STD)")
        print("-------------------------------------------------")
        for col in snv_cols:
            print(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    if norm_cols:
        print("\n-------------------------------------------------")
        print("CHECKING NORMALIZED COLUMNS (MIN, MAX)")
        print("-------------------------------------------------")
        for col in norm_cols:
            print(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}")

    print("\nDONE: Basic Data Analysis completed.")

def main():
    data_analysis()

if __name__ == "__main__":
    main()
