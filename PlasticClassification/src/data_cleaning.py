import os
import sys

# Ensure project root is on the Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.datacleaner import DataCleaner
from Core.Timer.timer import Timer

"""
Script: Core/data_cleaning.py
Orchestrates the data cleaning pipeline with timing:
 1. Load raw data
 2. Drop unused meta-data columns
 3. Drop missing values
 4. Remove outliers (raw intensities only)
 5. Save cleaned dataset to CSV
Each step is wrapped in a Timer context manager.
"""

def main():
    # Define file paths
    data_dir = os.path.join(script_dir, "Data")
    raw_csv = os.path.join(data_dir, "combined_measurements.csv")
    cleaned_csv = os.path.join(data_dir, "combined_measurements_cleaned.csv")

    # Initialize the DataCleaner
    cleaner = DataCleaner(raw_csv)

    # 1) Load raw data
    with Timer("Load Data") as t:
        cleaner.load_data()
    print(f"[Load Data] completed in {t.elapsed:.4f}s")

    # 2) Drop unused meta-data columns
    with Timer("Drop Unused Columns") as t:
        cleaner.drop_unused_columns()
    print(f"[Drop Unused Columns] completed in {t.elapsed:.4f}s")

    # 3) Drop rows with missing values
    with Timer("Drop Missing Values") as t:
        cleaner.drop_missing()
    print(f"[Drop Missing Values] completed in {t.elapsed:.4f}s")

    # 4) Remove outliers based on raw intensity columns
    with Timer("Remove Outliers") as t:
        before_shape = cleaner.get_data().shape
        cleaner.remove_outliers()
        after_shape = cleaner.get_data().shape
    print(f"[Remove Outliers] completed in {t.elapsed:.4f}s")
    print(f"Shape before outlier removal: {before_shape}, after: {after_shape}")

    # Retrieve cleaned data
    cleaned_df = cleaner.get_data()

    # 5) Save cleaned dataset
    with Timer("Save Cleaned Data") as t:
        cleaned_df.to_csv(cleaned_csv, index=False)
    print(f"[Save Cleaned Data] completed in {t.elapsed:.4f}s")
    print(f"Cleaned dataset saved to: {cleaned_csv}")

if __name__ == "__main__":
    main()
