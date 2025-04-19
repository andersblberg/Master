# Experiments/Experiments.py

import os
import time
import itertools
import numpy as np
import pandas as pd

from PlasticClassification.src.models.SVM import train_svm_model
from PlasticClassification.src.models.RF import train_rf_model
from PlasticClassification.src.models.NN import train_nn_model

# Dictionary of models and their training functions.
MODELS = {
    "SVM": train_svm_model,
    "RF": train_rf_model,
    "NN": train_nn_model
}

def run_baseline_experiments(X, y):
    results = []
    for model_name, train_func in MODELS.items():
        start = time.time()
        res = train_func(X, y)
        run_time = time.time() - start
        results.append({
            "model": model_name,
            "experiment": "Baseline",
            "accuracy": res["accuracy"],
            "time": run_time
        })
    return results

def run_noise_experiments(X, y, noise_factors=[0.1, 0.5, 1.0, 2.0, 5.0]):
    results = []
    for noise in noise_factors:
        X_noisy = X * noise
        for model_name, train_func in MODELS.items():
            start = time.time()
            res = train_func(X_noisy, y)
            run_time = time.time() - start
            results.append({
                "model": model_name,
                "experiment": f"Noise={noise}",
                "accuracy": res["accuracy"],
                "time": run_time
            })
    return results

def run_wavelength_removal_experiments(X, y, feature_cols):
    results = []
    # Remove 1, 2, 3, and 4 wavelengths
    for r in range(1, 5):
        for combo in itertools.combinations(feature_cols, r):
            remaining_features = [f for f in feature_cols if f not in combo]
            X_subset = X[remaining_features]
            removal_str = f"Remove={'/'.join(combo)}"
            for model_name, train_func in MODELS.items():
                start = time.time()
                res = train_func(X_subset, y)
                run_time = time.time() - start
                results.append({
                    "model": model_name,
                    "experiment": removal_str,
                    "accuracy": res["accuracy"],
                    "time": run_time
                })
    return results

def run_all_experiments():
    # Load final cleaned dataset; make sure the path is correct relative to the project root.
    df = pd.read_csv("PlasticClassification/Core/Data/final_cleaned_measurements.csv")
    feature_cols = ["nm940", "nm1050", "nm1200", "nm1300", "nm1450", "nm1550", "nm1650", "nm1720"]
    target_col = "PlasticType"
    X = df[feature_cols]
    y = df[target_col]
    
    all_results = []
    all_results.extend(run_baseline_experiments(X, y))
    all_results.extend(run_noise_experiments(X, y))
    all_results.extend(run_wavelength_removal_experiments(X, y, feature_cols))
    
    # Save the combined experiment results
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/all_experiments.csv", index=False)
    
    return all_results

def main():
    results = run_all_experiments()
    print("Experiments complete. Results saved in 'results/all_experiments.csv'.")

if __name__ == "__main__":
    main()
