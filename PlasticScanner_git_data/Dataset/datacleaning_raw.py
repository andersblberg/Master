import pandas as pd

# Load the CSV files
df1 = pd.read_csv('C:/Users/Blomberg/Downloads/measurement_data/measurement1.csv')
df2 = pd.read_csv('C:/Users/Blomberg/Downloads/measurement_data/measurement2.csv')
df3 = pd.read_csv('C:/Users/Blomberg/Downloads/measurement_data/measurement3.csv')

# Add a source column to track the origin
df1['SourceFile'] = 'measurement1.csv'
df2['SourceFile'] = 'measurement2.csv'
df3['SourceFile'] = 'measurement3.csv'

# Combine the datasets
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Exclude calibration measurements
combined_df = combined_df[combined_df['MeasurementType'].str.lower() != 'calibration']

# Optionally, fill missing values in categorical columns with a placeholder (e.g., "unknown")
for col in ['Name', 'Color']:
    combined_df[col] = combined_df[col].fillna("unknown")

# Identify columns to drop:
# 1. 'DateTime' and 'MeasurementType' (now redundant)
# 2. All SNV-transformed columns (ending with '_snv')
# 3. All normalized measurement columns (ending with '_norm')
columns_to_drop = ['DateTime', 'MeasurementType']
columns_to_drop += [col for col in combined_df.columns if col.endswith('_snv')]
columns_to_drop += [col for col in combined_df.columns if col.endswith('_norm')]

# Drop the identified columns
combined_df = combined_df.drop(columns=columns_to_drop)

# Remove duplicates if any
combined_df = combined_df.drop_duplicates()

# Save the cleaned, combined dataset to a new CSV file
output_csv_path = 'C:/Users/Blomberg/Downloads/measurement_data/combined_measurements_cleaned_raw.csv'
combined_df.to_csv(output_csv_path, index=False)

print("Cleaned dataset (DateTime, MeasurementType, SNV, and normalized columns excluded) saved to:", output_csv_path)
