"""This script is combining the PlasticScanner datasets and sorts the readings by PlasticTypes, then prints out how many there are of each resin and outputs a csv file"""

import os
import pandas as pd

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output paths
input_dir = os.path.join(script_dir, "DBv22")
output_path = os.path.join(script_dir, "combined_DB22_measurements_sorted.csv")
summary_path = os.path.join(script_dir, "summary_counts.csv")

# Name of the column indicating plastic type (adjust if needed)
plastic_col = "PlasticType"

# CSV filenames to combine
csv_files = [
    "measurement1.csv",
    "measurement2.csv",
    "measurement3.csv",
    "measurement4.csv",
]

dataframes = []
for file in csv_files:
    full_path = os.path.join(input_dir, file)
    if os.path.exists(full_path):
        print(f"Loading: {file}")
        df = pd.read_csv(full_path)
        df["SourceFile"] = file  # track origin
        dataframes.append(df)
    else:
        print(f"Warning: File not found - {file}")

# Abort if path is wrong / no data is loaded
if not dataframes:
    print("No files were loaded. Nothing to combine.")
    exit(1)

# Concatenate all measurements
combined_df = pd.concat(dataframes, ignore_index=True)

# Identify readings that are reference or calibration (case-insensitive)
other_mask = combined_df[plastic_col].str.lower().isin(["reference", "calibration"])

# Dynamically determine all plastic types (excluding reference/calibration)
plastic_types = sorted(combined_df.loc[~other_mask, plastic_col].unique())

# Assign each row to its plastic type or 'Other'
combined_df["Category"] = combined_df[plastic_col].where(~other_mask, "Other")

# Define an ordered categorical for sorting: plastic types first (alphabetically), then Other
combined_df["Category"] = pd.Categorical(
    combined_df["Category"], categories=plastic_types + ["Other"], ordered=True
)

# Sort by category so plastics come first in order, then Other at bottom
combined_df.sort_values("Category", inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# Save the organized dataset
combined_df.to_csv(output_path, index=False)
print(f"Sorted dataset saved to: {output_path}")

# Generate summary counts for report
summary = (
    combined_df["Category"]
    .value_counts()
    .reindex(plastic_types + ["Other"], fill_value=0)
    .rename_axis("Category")
    .reset_index(name="Count")
)

# Print table to console
print("\nSummary of readings by category:")
print(summary.to_string(index=False))

# Save summary to CSV for later use
summary.to_csv(summary_path, index=False)
print(f"Summary counts saved to: {summary_path}")
