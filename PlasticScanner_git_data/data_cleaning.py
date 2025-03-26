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

# Optionally, handle missing values in categorical columns: fill with a placeholder (e.g., "unknown")
for col in ['Name', 'Color']:
    combined_df[col] = combined_df[col].fillna("unknown")

# Drop the DateTime column since it's not needed
combined_df = combined_df.drop(columns=['DateTime'])

# Remove duplicates if any
combined_df.drop_duplicates(inplace=True)

# Save the cleaned, combined dataset (without calibration data and DateTime column) to a new CSV file
combined_csv_path = 'C:/Users/Blomberg/Downloads/measurement_data/combined_measurements_clean.csv'
combined_df.to_csv(combined_csv_path, index=False)

print("Combined dataset (calibration data excluded and DateTime column dropped) saved to:", combined_csv_path)