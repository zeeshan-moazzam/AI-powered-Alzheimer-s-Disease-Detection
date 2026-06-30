# AI-powered-Alzheimer-s-Disease-Detection

# 🧠 AI-Powered Alzheimer's Disease Detection using Resting-State EEG & Graph Signal Processing

[![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-OpenNeuro%20ds004504-orange.svg)](https://openneuro.org/datasets/ds004504/versions/1.0.6)

> **B.Tech Major Project** — Department of Computer Engineering, Jamia Millia Islamia, New Delhi (2025–26)

---

## 📌 Overview

Alzheimer's disease (AD) affects an estimated **55 million people worldwide**, yet no non-invasive, low-cost, scalable early-screening tool exists in clinical practice. This project presents a complete **EEG-GSP (Graph Signal Processing) framework** that classifies resting-state EEG recordings into **Alzheimer's Disease (AD)**, **Frontotemporal Dementia (FTD)**, and **Healthy Controls (CN)** using graph-theoretic features and classical machine learning.

Instead of treating EEG electrodes as independent or pairwise signals (as conventional spectral/coherence methods do), this project models the **19-electrode array as a graph**, where edges are weighted by inter-electrode coherence. Spectral graph theory is then used to extract interpretable biomarkers that capture the **disrupted functional connectivity** characteristic of neurodegeneration — something flat power-spectral features fail to represent.

### Key Results

| Metric | Multi-class (AD/FTD/CN) | Binary (Dementia/Healthy) |
|---|---|---|
| **Accuracy** | **86.4%** | **91.2%** |
| Weighted F1-Score | 0.850 | 0.901 |
| Mean AUC-ROC | 0.912 | 0.940 |
| LOOCV Accuracy | 83.7% ± 1.9% | 88.4% ± 2.1% |

Random Forest outperformed all other classifiers tested (XGBoost, SVM-RBF, KNN, Logistic Regression, Naïve Bayes) and exceeded the best published 3-class baseline on the same dataset (Miltiadous et al., 2021 — 73–78%).

---

## 🩺 Problem Statement

Clinical symptoms of AD typically appear only after **15–20 years of subclinical neurodegeneration**, by which point irreversible damage has occurred. Existing diagnostic tools each have major drawbacks:

| Method | Strength | Weakness |
|---|---|---|
| Cognitive Tests (MMSE) | Low cost, rapid | Subjective, insensitive early |
| MRI | High spatial resolution | Detects late-stage atrophy only |
| PET (Amyloid/FDG) | Specific biomarker | Expensive, radiation, inaccessible |
| CSF Biomarkers | High specificity | Invasive (lumbar puncture) |
| EEG (Power Spectral) | Non-invasive, cheap | Noise-sensitive, feature-limited |
| **EEG + GSP (this work)** | **Graph connectivity features** | Small dataset |

This project targets the gap left by conventional EEG analysis: the inability to capture **global network topology** disrupted by AD pathology, by applying Graph Signal Processing to resting-state EEG.

---

## 🗂️ Dataset

**[OpenNeuro ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.6)** — a BIDS-formatted, FAIR-compliant open dataset.

- **88 subjects total**
  - 36 Alzheimer's Disease (mean MMSE: 17.75)
  - 23 Frontotemporal Dementia (mean MMSE: 22.17)
  - 29 Healthy Controls (mean MMSE: 30.0)
- 19-channel EEG (10-20 international system), resting-state, eyes closed, 5 minutes
- Sampled at 500 Hz, referenced to linked mastoids

---

## ⚙️ Methodology

```
Raw EEG (.set) → MNE-Python Loader → Bandpass Filter (1–40 Hz) → ICA Artifact Removal
       → Epoching (2s windows) → Coherence Matrix (alpha band, 8–13 Hz)
       → Graph Construction (PyGSP) → Graph Laplacian → GSP Feature Extraction (9 features)
       → RobustScaler → Train/Test Split (70/30, stratified) → ML Classification → Evaluation
```

### 1. Preprocessing
- Zero-phase FIR bandpass filter (1–40 Hz) + 50 Hz notch filter
- FastICA (19 components) for automated ocular artifact rejection
- Epoch rejection at ±100 µV peak-to-peak amplitude

### 2. Graph Construction
- Magnitude-squared coherence computed between all 171 electrode pairs (alpha band, Welch's method)
- Coherence matrix thresholded at 0.1 → sparse 19×19 adjacency matrix `W`
- Normalised graph Laplacian: `L = I − D^(-1/2) W D^(-1/2)`

### 3. GSP Feature Extraction (via PyGSP)

| # | Feature | Category | Description |
|---|---|---|---|
| 1 | **Spectral Entropy** | Graph Spectral | Entropy of Graph Fourier Transform (GFT) coefficient distribution |
| 2 | **Total Variation** | Graph Smoothness | Sum of squared weighted edge differences |
| 3 | **Graph Energy** | Graph Spectral | Sum of squared GFT coefficients |
| 4 | **Tik-norm** | Graph Smoothness | Tikhonov regularisation norm (`xᵀLx`) |
| 5 | **Signal Power** | Temporal | Mean squared EEG amplitude |
| 6 | **Diffusion Distance** | Graph Topology | Heat-kernel diffusion distance between nodes |
| 7 | **Stationary Ratio** | Graph Stationarity | Ratio of GFT energy in stationary components |
| 8 | **Signal Energy** | Temporal | Total signal energy across electrodes |
| 9 | **Average Node Degree** | Graph Structure | Mean coherence-weighted connections per node |

### 4. Classification
Six supervised classifiers were trained and compared under identical preprocessing and cross-validation conditions:
**Random Forest · XGBoost · SVM (RBF) · KNN · Logistic Regression · Naïve Bayes**

Final model: `RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", oob_score=True, random_state=42)`

---

## 📊 Results

### Model Comparison (Multi-class)

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Naïve Bayes | 72.4% | 0.701 | 0.849 |
| Logistic Regression | 76.1% | 0.743 | 0.872 |
| KNN (k=5) | 79.5% | 0.778 | 0.880 |
| SVM (RBF) | 81.3% | 0.796 | 0.901 |
| XGBoost | 83.7% | 0.821 | 0.920 |
| **Random Forest** | **86.4%** | **0.850** | **0.912** |

### Feature Importance (Top Drivers)
**Spectral Entropy (MDI = 0.182)** and **Total Variation (MDI = 0.157)** are the most discriminative features — together with Graph Energy and Tik-norm they account for **59.1% of total classification importance**. This directly reflects the loss of spatially coherent, low-graph-frequency neural oscillations in AD pathology.

### Visualizations Included
- Confusion matrices (multi-class & binary)
- ROC curves (one-vs-rest, per-model comparison)
- UMAP 2D projection of the 9-dimensional GSP feature space
- Feature importance bar chart
- Learning curve (training set size vs. accuracy)
- Power spectral density comparison across diagnostic groups

---

## 🛠️ Tech Stack

| Component | Version |
|---|---|
| Python | 3.10.12 |
| MNE-Python | 1.4.0 |
| PyGSP | 0.5.1 |
| scikit-learn | 1.3.0 |
| XGBoost | 2.0.0 |
| UMAP-learn | 0.5.3 |
| NumPy / Pandas | 1.24.3 / 2.0.2 |
| Matplotlib / Seaborn | 3.7.2 / 0.12.2 |
| Joblib | 1.3.1 |

---

## 📁 Project Structure

```
.
├── data/
│   └── ds004504/                  # OpenNeuro raw EEG (.set) files (BIDS format)
├── notebooks/
│   ├── 01_preprocessing.ipynb     # EEG loading, filtering, ICA, epoching
│   ├── 02_graph_construction.ipynb# Coherence matrix + GSP feature extraction
│   ├── 03_model_training.ipynb    # Classifier training & hyperparameter tuning
│   └── 04_evaluation.ipynb        # Metrics, confusion matrices, UMAP, ROC
├── src/
│   ├── preprocessing.py
│   ├── gsp_features.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── random_forest_model.joblib
├── figures/                       # Generated plots used in the report
├── report/
│   └── Major_project_final_report.pdf
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/alzheimers-eeg-gsp-detection.git
cd alzheimers-eeg-gsp-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
# OpenNeuro ds004504
openneuro-py download --dataset ds004504 --target_dir data/ds004504
```

### 4. Run the pipeline
```bash
python src/preprocessing.py
python src/gsp_features.py
python src/train.py
python src/evaluate.py
```

---

## 📈 Evaluation Metrics Used

Given the small cohort size (n = 88), accuracy alone is insufficient (majority-class baseline = 40.9%). The following metrics were used for rigorous evaluation:
- Per-class Precision, Recall, F1-Score
- Macro & Weighted F1-Score
- AUC-ROC (one-vs-rest)
- Leave-One-Out Cross-Validation (LOOCV)
- Out-of-Bag (OOB) accuracy

---

## ⚠️ Limitations

- **Small cohort size** (88 subjects) — wide confidence intervals, limited generalisability
- **Single-dataset evaluation** — no cross-dataset validation performed
- **No clinical validation** — this is a computational prototype, not a diagnostic tool
- **Cross-sectional only** — no longitudinal EEG tracking

## 🔮 Future Scope

- Replace handcrafted GSP features with **Graph Neural Networks (GNNs)**
- **Multi-modal fusion** with cognitive scores and structural MRI
- **Federated learning** for privacy-preserving multi-site data collection
- **SHAP-based explainability** for per-subject clinical interpretation
- Adaptation for **sparse-electrode wearable EEG** (Muse, OpenBCI, Neurosity Crown)

---

## 📚 Key References

1. Jeong, J. (2004). *EEG dynamics in patients with Alzheimer's disease.* Clinical Neurophysiology, 115(7), 1490–1505.
2. Miltiadous, A., et al. (2021). *Alzheimer's disease and frontotemporal dementia: A robust classification method of EEG signals.* Diagnostics, 11(8), 1437.
3. Stam, C.J., et al. (2006). *Small-world networks and functional connectivity in Alzheimer's disease.* Cerebral Cortex, 17(1), 92–99.
4. Ortega, A., et al. (2018). *Graph signal processing: Overview, challenges, and applications.* Proceedings of the IEEE, 106(5), 808–828.
5. Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32.

Full reference list (28 citations) available in the [project report](report/Major_project_final_report.pdf).

---

## 📄 License

This project is released under the [MIT License](LICENSE). The OpenNeuro ds004504 dataset is used under its own open-data license — see [openneuro.org/datasets/ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.6) for terms.

---

## 🙏 Acknowledgements

We thank the OpenNeuro platform and contributors of the ds004504 dataset, the PyGSP development team, and the Department of Computer Engineering at Jamia Millia Islamia for computational resources and guidance.
