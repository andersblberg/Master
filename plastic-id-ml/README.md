# Plastic-ID-ML 
*Infrared spectral data classification of polymers with Python & ML*

[![Tests](https://github.com/<your-org>/plastic-id-ml/actions/workflows/pytest.yml/badge.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repo contains all the code, configs and utilities developed and used in my M.Sc.-thesis
work on **classifying polymers (HDPE, LDPE, PET, PP, PS, PVC, …)**
from their near-infra-red / FT-IR spectra derived from the [![Plastic-Scanner](https://github.com/Plastic-Scanner)] project.

| **Highlights** | |
|----------------|--------------------------------------------------------------|
| • Reproducible pipelines powered by Poetry + Hydra | environment & config management |
| • Modular dataset + preprocessing stack | SNV, row-normalisation, PCA, noise injection |
| • Pluggable model zoo | SVM, Random-Forest, XGBoost, MLP, 1-D CNN |
| • One-command experiment runner | `poetry run pid train …` |
| • Rich evaluation artefacts | confusion matrix, PR curves, per-class metrics |
| • Growing test-suite (pytest) | loaders, transforms, registry sanity |

---

## 1 . Quick-start

```bash
# 1. clone
git clone https://github.com/andersblberg/Master.git
cd Master/plastic-id-ml

# 2. create the Poetry env (Python 3.11)
poetry install

# 3. activate shell OR prefix every command with `poetry run`
poetry shell

# 4. run a baseline experiment
pid train --cfg-path configs/experiment/baseline.yaml         # or
pid grid  --cfg-path configs/experiment/dataset_clean.yaml -m xgb rf svm
