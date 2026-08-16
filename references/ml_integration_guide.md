# Drift-Sense Machine Learning Integration Guide & Benchmark Specifications

---

## 📌 Executive Summary for ML Engineering Teammate

This document specifies the dataset structure, ground-truth label formats, candidate proposal interfaces, and classical baseline benchmarks provided by the **Drift-Sense V1.7 Simulation Engine**.

You do not need to manually collect or annotate wafer SEM images. The Drift-Sense simulation engine generates physics-backed, continuous die layouts (FinFET and DRAM) with exact ground-truth inspection coordinates $(x_{\text{true}}, y_{\text{true}})$.

---

## 📁 Dataset Directory Layout

Datasets exported via `python export_ml_dataset.py` follow standard machine learning split structures:

```text
results/ml_dataset/
├── train/
│   ├── reference/
│   │   ├── ref_0001.png       (1000 x 1000 px, 100x high-mag reference view)
│   │   └── ...
│   ├── search/
│   │   ├── search_0001.png    (1000 x 1000 px, 10x wide-field search view)
│   │   └── ...
│   └── labels.csv             (Ground truth target coordinates & SEM metadata)
│
├── validation/
│   ├── reference/
│   ├── search/
│   └── labels.csv
│
└── test/
    ├── reference/
    ├── search/
    └── labels.csv
```

---

## 📑 Manifest Schema (`labels.csv`)

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `pair_id` | Integer | Unique identifier for image pair |
| `architecture` | String | Layout type: `"FinFET"` or `"DRAM"` |
| `reference_file` | String | Relative path to reference image PNG |
| `search_file` | String | Relative path to search image PNG |
| `true_x` | Float | Ground truth target X coordinate in search image ($0.0$ to $1000.0$) |
| `true_y` | Float | Ground truth target Y coordinate in search image ($0.0$ to $1000.0$) |
| `scale_ratio` | Float | Magnification scale ratio ($9.0\times$ to $11.0\times$) |
| `relative_rotation_deg` | Float | Stage capture relative rotation ($\pm 2.0^\circ$) |
| `blur_sigma` | Float | Beam astigmatic PSF blur ($\sigma$) |
| `shot_dose` | Float | Poisson electron shot noise factor ($\lambda$) |
| `gaussian_std` | Float | Thermal detector readout Gaussian noise ($\sigma$) |
| `charging_strength` | Float | Surface dielectric charging streak amplitude |

---

## 🎯 Recommended ML Model Architectures

### Formulation A: Hybrid Candidate Verification (Recommended)
Rather than training a neural network to predict $(x, y)$ from scratch on raw 1000x1000 images, use **Classical Stage 1 Candidate Proposal (`src/matcher.py`)** to propose the top $K=20$ candidate patch locations $(x_k, y_k)$.

```
   1000x1000 Images (Ref + Search)
                 │
                 ▼
     Stage 1 NCC Candidate Proposal (src/matcher.py)
                 │
   Top 20 Candidate Bounding Boxes / Patches
                 │
                 ▼
     Neural Network Candidate Classifier / Ranker
                 │
   Predicted Best Bounding Box -> (x_pred, y_pred)
```

**Why this works best**:
1. Eliminates $99\%$ of spatial search space using fast classical NCC.
2. Allows the neural network to focus $100\%$ of its capacity on **disambiguating periodic array match candidates**.

### Formulation B: Spatial Probability Heatmap Regression
* Input: 2-channel tensor `torch.cat([ref_img, search_img], dim=0)` (2 x 1000 x 1000).
* Output: 1-channel probability heatmap tensor (1 x 1000 x 1000).
* Loss function: Focal Loss or MSE against a 2D Gaussian heatmap centered at $(x_{\text{true}}, y_{\text{true}})$.

---

## 📊 Benchmark Targets to Beat (Classical V1.7 Baseline)

Your ML model will be evaluated against the classical V1.7 deterministic baseline ([README.md](file:///D:/SemiconFINFETwork/README.md)):

| Metric / Benchmark | Classical V1.7 Baseline Target |
| :--- | :--- |
| **Pass Rate @ 5.0px Threshold** | **`73.3% - 86.7%`** |
| **Pass Rate @ 4.0px Threshold** | **`66.7% - 80.0%`** |
| **Pass Rate @ 2.0px Threshold** | **`30.0% - 43.3%`** |
| **Pass Rate @ 1.0px Threshold** | **`16.7% - 23.3%`** |
| **Mean Euclidean Error** | **`2.08 px - 3.54 px`** |
| **Worst-Case Error Cap** | **`8.68 px`** (Zero periodic spatial jumps) |
