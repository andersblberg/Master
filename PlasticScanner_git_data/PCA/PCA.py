import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load the cleaned dataset
df = pd.read_csv('C:/Users/Blomberg/Downloads/measurement_data/combined_measurements_clean.csv')

# Define the 8 raw wavelength measurement columns
wavelength_columns = ['nm940', 'nm1050', 'nm1200', 'nm1300', 'nm1450', 'nm1550', 'nm1650', 'nm1720']

# Extract the raw measurement data
X = df[wavelength_columns].values

# Scale the data to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 principal components (adjust n_components if desired)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Print the explained variance ratio of each principal component
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Create a DataFrame for the PCA results and add the PlasticType
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['PlasticType'] = df['PlasticType'].values

# Save the PCA-transformed data to a CSV file
pca_csv_path = 'C:/Users/Blomberg/Downloads/measurement_data/pca_transformed_data.csv'
pca_df.to_csv(pca_csv_path, index=False)
print("PCA-transformed data saved to:", pca_csv_path)

# Get unique plastic types and assign a unique color to each
unique_types = pca_df['PlasticType'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_types)))

# Plot the PCA results, grouping by PlasticType
plt.figure(figsize=(8,6))
for plastic, color in zip(unique_types, colors):
    subset = pca_df[pca_df['PlasticType'] == plastic]
    plt.scatter(subset['PC1'], subset['PC2'], label=plastic, color=color, alpha=0.7)
    
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Raw Wavelength Measurements')
plt.legend(title='Plastic Type')
plt.show()
