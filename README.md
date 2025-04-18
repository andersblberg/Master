## 📁 Project Structure

```text
PlasticClassification/
├── main.py
├── README.md
├── requirements.txt
│
├── Core/
│   ├── datacleaner.py
│   ├── data_analysis.py
│   ├── data_cleaning.py
│   ├── __init__.py
│   │
│   ├── Data/
│   │   ├── combined_measurements.csv
│   │   ├── combined_measurements_cleaned.csv
│   │   ├── combined_measurements_cleaned_raw_only.csv
│   │   │
│   │   └── Original_PlasticScanner_measurements/
│   │       ├── measurement1.csv
│   │       ├── measurement2.csv
│   │       └── measurement3.csv
│   │
│   └── Timer/
│       ├── timer.py
│       └── __init__.py
│
├── Experiments/
│   ├── Experiments.py
│   └── __init__.py
│
├── NN/
│   ├── NN_model.py
│   └── __init__.py
│
├── PCA/
│   ├── PCA_module.py
│   └── __init__.py
│
├── RF/
│   ├── RF_model.py
│   └── __init__.py
│
├── SVM/
│   ├── SVM_model.py
│   └── __init__.py
│
└── Visualizer/
    ├── visualizer.py
    └── __init__.py
