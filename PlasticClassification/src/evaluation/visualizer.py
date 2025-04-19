# Visualizer/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import re

def plot_accuracy_comparison(results, out_path="results/accuracy_comparison.png"):
    """
    Plots a grouped bar chart comparing accuracy across models and experiments.
    Expects results to be a list of dictionaries with keys: 'model', 'experiment', 'accuracy'.
    """
    df = pd.DataFrame(results)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="model", y="accuracy", hue="experiment", ci=None)
    plt.title("Accuracy Comparison Across Experiments")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0,1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved accuracy comparison plot to {out_path}")

def plot_time_comparison(results, out_path="results/time_comparison.png"):
    """
    Plots a grouped bar chart comparing processing time across models and experiments.
    Expects each result to include a 'time' field (in seconds).
    """
    df = pd.DataFrame(results)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="model", y="time", hue="experiment", ci=None)
    plt.title("Time Comparison Across Experiments")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Model")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved time comparison plot to {out_path}")

def plot_noise_resilience(results, out_path="results/noise_resilience.png"):
    """
    Plots a line chart showing how accuracy changes with different noise factors.
    Expects experiment strings in the form "Noise=X" where X is a float.
    """
    data = []
    for r in results:
        exp = r.get("experiment", "")
        if "Noise=" in exp:
            match = re.search(r"Noise=([0-9\.]+)", exp)
            if match:
                noise = float(match.group(1))
                data.append({"model": r["model"], "noise": noise, "accuracy": r["accuracy"]})
    if not data:
        print("No noise experiment data found.")
        return
    df = pd.DataFrame(data)
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x="noise", y="accuracy", hue="model", marker="o")
    plt.title("Noise Resilience: Accuracy vs. Noise Factor")
    plt.xlabel("Noise Factor")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved noise resilience plot to {out_path}")

def plot_wv_removal(results, out_path="results/wv_removal.png"):
    """
    Plots a bar chart showing accuracy versus removed wavelengths.
    Expects experiment strings like "Remove=nm940/nm1050" indicating which wavelengths were removed.
    """
    data = []
    for r in results:
        exp = r.get("experiment", "")
        if "Remove=" in exp:
            removed = exp.split("=")[1]
            data.append({"model": r["model"], "removed": removed, "accuracy": r["accuracy"]})
    if not data:
        print("No wavelength removal data found.")
        return
    df = pd.DataFrame(data)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="removed", y="accuracy", hue="model", ci=None)
    plt.title("Wavelength Removal: Accuracy vs. Removed Wavelength(s)")
    plt.xlabel("Removed Wavelength(s)")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved wavelength removal plot to {out_path}")

if __name__ == "__main__":
    # Demo with dummy data if run standalone
    dummy_data = [
        {"model": "SVM", "experiment": "Baseline", "accuracy": 0.85, "time": 0.6},
        {"model": "RF", "experiment": "Baseline", "accuracy": 0.88, "time": 0.4},
        {"model": "NN", "experiment": "Baseline", "accuracy": 0.80, "time": 0.5},
        {"model": "SVM", "experiment": "Noise=2.0", "accuracy": 0.78, "time": 0.65},
        {"model": "RF", "experiment": "Noise=2.0", "accuracy": 0.80, "time": 0.42},
        {"model": "NN", "experiment": "Noise=2.0", "accuracy": 0.79, "time": 0.55},
        {"model": "SVM", "experiment": "Remove=nm940/nm1050", "accuracy": 0.81, "time": 0.55},
        {"model": "RF", "experiment": "Remove=nm940/nm1050", "accuracy": 0.83, "time": 0.38},
        {"model": "NN", "experiment": "Remove=nm940/nm1050", "accuracy": 0.80, "time": 0.50}
    ]
    plot_accuracy_comparison(dummy_data, out_path="results/dummy_accuracy.png")
    plot_time_comparison(dummy_data, out_path="results/dummy_time.png")
    plot_noise_resilience(dummy_data, out_path="results/dummy_noise.png")
    plot_wv_removal(dummy_data, out_path="results/dummy_wv_removal.png")
