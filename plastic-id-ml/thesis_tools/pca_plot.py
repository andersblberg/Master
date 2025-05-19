"""

Run PCA on raw spectral data columns
(`nm940` – `nm1720`) and visualise the first three components in a 3‑D
scatter plot. Uses the PlasticType column as the default label.

Parameters
----------
--file, -f       Path to the CSV dataset. (Required)
--components, -c Number of principal components to compute. Default is 3.
--target, -t     Column containing class labels (default: ``PlasticType``).
--no-show        Save figure to *pca_plot.png* instead of opening a window.

Example
-------
python pca_plot.py -f C:/Users/Blomberg/Desktop/Master/plastic-id-ml/data/interim/combined_DB22_measurements_sorted_clean.csv -c 6
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – keeps 3‑D backend
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


###############################################################################
# Argument parsing
###############################################################################


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run PCA on the raw nm940‑nm1720 spectral range and plot the first 3 PCs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to CSV file containing your data.",
    )
    parser.add_argument(
        "-c",
        "--components",
        type=int,
        default=3,
        help="Number of principal components to calculate (≥3 recommended for 3‑D plot).",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="PlasticType",
        help="Column to use for colouring / class labels (default: PlasticType).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure as pca_plot.png without opening an interactive window.",
    )
    return parser.parse_args(argv)


###############################################################################
# Main routine
###############################################################################


def main(argv=None):
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    csv_path = Path(args.file)
    if not csv_path.is_file():
        sys.exit(f"Error: {csv_path} does not exist.")

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # 2. Select raw spectral columns (nm940 – nm1720, *without* _norm)
    # ------------------------------------------------------------------
    spectral_pattern = re.compile(r"^nm(\d{3,4})$")
    spectral_cols = [
        col
        for col in df.columns
        if (m := spectral_pattern.match(col)) and 940 <= int(m.group(1)) <= 1720
    ]

    if not spectral_cols:
        sys.exit("Error: No raw spectral columns (nm940–nm1720) found in the file.")

    X = df[spectral_cols]

    # ------------------------------------------------------------------
    # 3. Handle the target / label column
    # ------------------------------------------------------------------
    target_col = args.target if args.target else None
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
    elif target_col:
        print(
            f"Warning: target column '{target_col}' not found – proceeding unlabeled."
        )

    # ------------------------------------------------------------------
    # 4. Pre‑processing: impute & scale
    # ------------------------------------------------------------------
    X = X.fillna(X.mean())  # simple mean imputation for missing wavelengths
    X_scaled = StandardScaler().fit_transform(X)

    # ------------------------------------------------------------------
    # 5. PCA
    # ------------------------------------------------------------------
    n_components = max(1, args.components)
    pca = PCA(n_components=n_components, random_state=0)
    components = pca.fit_transform(X_scaled)

    # Explained variance output
    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)
    print("Explained variance ratio per component:")
    for idx, var in enumerate(explained, start=1):
        print(f"  PC{idx}: {var:.2%} (cumulative {cum_explained[idx-1]:.2%})")

    # ------------------------------------------------------------------
    # 6. Prepare 3‑D scatter data (pad with zeros if <3 PCs)
    # ------------------------------------------------------------------
    plot_data = np.zeros((components.shape[0], 3))
    plot_data[:, : min(components.shape[1], 3)] = components[:, :3]

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if y is not None:
        labels = pd.Categorical(y)
        scatter = ax.scatter(
            plot_data[:, 0],
            plot_data[:, 1],
            plot_data[:, 2],
            c=labels.codes,
            cmap="tab10",
            alpha=0.8,
            s=40,
        )
        legend1 = ax.legend(
            *scatter.legend_elements(),
            loc="upper right",
            title=target_col,
            bbox_to_anchor=(1.05, 1),
        )
        ax.add_artist(legend1)
    else:
        ax.scatter(
            plot_data[:, 0],
            plot_data[:, 1],
            plot_data[:, 2],
            alpha=0.8,
            s=40,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3‑D PCA Scatter Plot (raw 940–1720 nm)")

    plt.tight_layout()

    if args.no_show:
        fig.savefig("pca_plot.png", dpi=300)
        print("Plot saved to pca_plot.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
