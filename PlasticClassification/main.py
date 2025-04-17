# main.py

import pandas as pd
from SVM.SVM_model import train_svm_model

def main():
    # 1) Load final cleaned CSV
    data_path = "C:/Users/Blomberg/Desktop/Master/PlasticClassification/Core/Data/final_cleaned_measurements.csv"
    df = pd.read_csv(data_path)
    print("Loaded final cleaned dataset:", df.shape)
    print("Columns:", df.columns.tolist())

    # 2) Identify features & target based on your CSV
    #    From your EDA, you have: [PlasticType, nm940, nm1050, nm1200, nm1300, nm1450, nm1550, nm1650, nm1720]
    #    We dropped [Reading, Name, Color, SourceFile], so columns should be: [PlasticType, nm940, nm1050, nm1200, nm1300, nm1450, nm1550, nm1650, nm1720]
    feature_cols = ["nm940", "nm1050", "nm1200", "nm1300", "nm1450", "nm1550", "nm1650", "nm1720"]
    target_col = "PlasticType"

    # Quick check
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Check your cleaning steps or the CSV columns.")

    X = df[feature_cols]
    y = df[target_col]

    # 3) Train SVM
    results = train_svm_model(X, y)
    print("SVM training complete. Accuracy:", results["accuracy"])

if __name__ == "__main__":
    main()
