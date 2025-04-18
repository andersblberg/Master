import os
import pandas as pd

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the folder where the DBv22 files are located
input_dir = os.path.join(script_dir, "DBv22")
output_path = os.path.join(script_dir, "combined_DB22_measurements.csv")

# List of CSV filenames
csv_files = [
    "measurement1.csv",
    "measurement2.csv",
    "measurement3.csv",
    "measurement4.csv"
]

# Combine CSVs
dataframes = []
for file in csv_files:
    full_path = os.path.join(input_dir, file)
    if os.path.exists(full_path):
        print(f"Loading: {file}")
        df = pd.read_csv(full_path)
        df["SourceFile"] = file  # Optional: track origin
        dataframes.append(df)
    else:
        print(f"Warning: File not found - {file}")

# Concatenate and save
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"✅ Combined dataset saved to: {output_path}")
else:
    print("❌ No files were loaded. Nothing to combine.")
