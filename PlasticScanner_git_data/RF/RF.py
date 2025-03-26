# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load the CSV Data
df = pd.read_csv("C:/Users/Blomberg/Downloads/measurement_data/measurement3.csv")  # Update the file path if necessary
print("Initial Data:")
print(df.head())
print(df.info())

# 3. Data Cleaning: Filter Out Calibration Rows
df = df[df["MeasurementType"] == "regular"].reset_index(drop=True)

# 4. Choose the Target and Feature Columns
# Target: PlasticType (e.g., 'PMMA', 'PS', etc.)
y = df["PlasticType"]

# Features: Sensor measurement columns (using raw sensor values in this example)
sensor_columns = ['nm940', 'nm1050', 'nm1200', 'nm1300', 'nm1450', 'nm1550', 'nm1650', 'nm1720']
X = df[sensor_columns]
print("Features Preview:")
print(X.head())

# 5. Encode the Target Variable (since PlasticType is categorical)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("Target Classes:", encoder.classes_)

# 6. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 7. Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 8. Evaluate the Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. (Optional) Plot Feature Importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [sensor_columns[i] for i in indices], rotation=45)
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
