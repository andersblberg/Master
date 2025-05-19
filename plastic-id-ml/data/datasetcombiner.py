"""This script is combining the PlasticScanner datasets and sorts the readings by PlasticTypes, then prints out how many there are of each resin and outputs a csv file"""

import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(script_dir, "DBv22")
output_path = os.path.join(script_dir, "combined_DB22_measurements_sorted.csv")
summary_path = os.path.join(script_dir, "summary_counts.csv")

plastic_col = "PlasticType"

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
        df["SourceFile"] = file
        dataframes.append(df)
    else:
        print(f"Warning: File not found - {file}")

if not dataframes:
    print("No files were loaded. Nothing to combine.")
    exit(1)

combined_df = pd.concat(dataframes, ignore_index=True)

other_mask = combined_df[plastic_col].str.lower().isin(["reference", "calibration"])

plastic_types = sorted(combined_df.loc[~other_mask, plastic_col].unique())

combined_df["Category"] = combined_df[plastic_col].where(~other_mask, "Other")

combined_df["Category"] = pd.Categorical(
    combined_df["Category"], categories=plastic_types + ["Other"], ordered=True
)

combined_df.sort_values("Category", inplace=True)
combined_df.reset_index(drop=True, inplace=True)

combined_df.to_csv(output_path, index=False)
print(f"Sorted dataset saved to: {output_path}")

summary = (
    combined_df["Category"]
    .value_counts()
    .reindex(plastic_types + ["Other"], fill_value=0)
    .rename_axis("Category")
    .reset_index(name="Count")
)

print("\nSummary of readings by category:")
print(summary.to_string(index=False))

summary.to_csv(summary_path, index=False)
print(f"Summary counts saved to: {summary_path}")
