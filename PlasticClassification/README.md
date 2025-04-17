## Project Structure


PlasticClassification/                <-- Project Root
│   main.py                           <-- (Optional) Simple entry point or baseline run
│   README.md                         <-- Overall documentation and usage notes
│   requirements.txt (optional)       <-- Python dependencies (scikit-learn, pandas, etc.)
│   
├── Core/                             <-- Generic or "core" utilities for data loading/cleaning
│   ├── __init__.py
│   ├── data_analysis.py             <-- Exploratory data analysis (plots, stats)
│   ├── data_cleaning.py             <-- Script that orchestrates cleaning & saves final CSV
│   ├── datacleaner.py               <-- Class or functions to load/clean data
│   └── Data/
│       ├── Original_PlasticScanner_measurements/
│       │    ├── measurement1.csv
│       │    ├── measurement2.csv
│       │    └── measurement3.csv
│       ├── combined_measurements_cleaned_raw_only.csv
│       ├── final_cleaned_measurements.csv         <-- Output of data_cleaning.py
│       └── (other CSVs as needed)
│
├── Experiments/                     <-- Scripts that run all your experimental scenarios
│   ├── __init__.py
│   └── Experiments.py               <-- Code for advanced experiments: noise, removing wv, etc.
│
├── NN/
│   ├── __init__.py
│   └── NN_model.py                  <-- Neural Network training code
│
├── PCA/
│   ├── __init__.py
│   └── PCA_module.py                <-- PCA or dimensionality reduction logic
│
├── RF/
│   ├── __init__.py
│   └── RF_model.py                  <-- Random Forest training code
│
├── SVM/
│   ├── __init__.py
│   └── SVM_model.py                 <-- SVM training code
│
├── Visualizer/
│   ├── __init__.py
│   └── visualizer.py                <-- All plotting: accuracy/time/ noise resilience/wv removal
│
└── results/ (or “graphs/”)          <-- Unified location for final plots & metrics
    ├── correlation_heatmap.png
    ├── noise_resilience.png
    ├── time_comparison.png
    └── (etc.)
