import os
import sys

# Ensure the project root is on sys.path so we can use absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from src.Timer.timer import Timer

class DataCleaner:
    """
    DataCleaner handles loading and cleaning of the dataset, with timing for each step.
    It provides methods to:
      - load_data: loads CSV into memory, timed
      - drop_unused_columns: drops meta-data columns, timed
      - drop_missing: removes any rows with missing values, timed
      - remove_outliers: removes outliers based on z-score for raw intensities, timed
      - get_data: returns the cleaned DataFrame

    Note: This class only holds data in memory. To persist the cleaned dataset,
    call get_data() and save to CSV in a separate script (e.g., Core/data_cleaning.py).
    """

    # Default columns to drop (meta-data not used for modeling)
    DEFAULT_DROP_COLUMNS = [
        "Reading",
        "Name",
        "Color",
        "MeasurementType",
        "SourceFile"
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """
        Load CSV from filepath into DataFrame. Drops no columns here.
        Returns:
          pd.DataFrame: Loaded dataset, accessible via get_data().
        Timing: recorded via Timer("DataCleaner.load_data").
        """
        with Timer("DataCleaner.load_data") as t:
            self.data = pd.read_csv(self.filepath)
        print(f"[load_data] completed in {t.elapsed:.4f}s; shape={self.data.shape}")
        return self.data

    def drop_unused_columns(self, cols_to_drop=None):
        """
        Drop meta-data columns that are not features for modeling.
        If cols_to_drop is None, DEFAULT_DROP_COLUMNS is used.
        Timing: recorded via Timer("DataCleaner.drop_unused_columns").
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if cols_to_drop is None:
            cols_to_drop = self.DEFAULT_DROP_COLUMNS
        with Timer("DataCleaner.drop_unused_columns") as t:
            self.data.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        print(f"[drop_unused_columns] completed in {t.elapsed:.4f}s; shape={self.data.shape}")
        return self.data

    def drop_missing(self):
        """
        Drop rows with any missing values.
        Timing: recorded via Timer("DataCleaner.drop_missing").
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        with Timer("DataCleaner.drop_missing") as t:
            self.data.dropna(inplace=True)
        print(f"[drop_missing] completed in {t.elapsed:.4f}s; shape={self.data.shape}")
        return self.data

    def remove_outliers(self, z_threshold=3.0):
        """
        Remove rows considered outliers based on z-score of raw intensity columns only.
        Only columns starting with 'nm' and without '_snv' or '_norm' are used.
        Timing: recorded via Timer("DataCleaner.remove_outliers").
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        from scipy.stats import zscore
        raw_cols = [col for col in self.data.columns 
                    if col.startswith("nm") and "_snv" not in col and "_norm" not in col]
        with Timer("DataCleaner.remove_outliers") as t:
            z = np.abs(zscore(self.data[raw_cols]))
            mask = (z < z_threshold).all(axis=1)
            before_shape = self.data.shape
            self.data = self.data[mask]
        print(f"[remove_outliers] completed in {t.elapsed:.4f}s; shape from {before_shape} to {self.data.shape}")
        return self.data

    def get_data(self):
        """
        Return the current cleaned DataFrame.
        """
        if self.data is None:
            raise ValueError("Data not loaded/cleaned. Call load_data() first.")
        return self.data

if __name__ == '__main__':
    # Standalone test block
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Core", "Data", "combined_measurements.csv")
    cleaner = DataCleaner(filepath)
    cleaner.load_data()
    cleaner.drop_unused_columns()
    cleaner.drop_missing()
    cleaner.remove_outliers()
    final_df = cleaner.get_data()
    print(f"Final cleaned data shape: {final_df.shape}")
