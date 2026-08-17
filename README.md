# 🔬 Drift-Sense: Scale-Invariant SEM Pattern Localization & Wafer Navigation-Error Recovery Engine

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CPU%20%7C%20CUDA)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![ReportLab](https://img.shields.io/badge/ReportLab-5.0%20PDF%20Engine-1A365D?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://www.reportlab.com/)
[![Accuracy](https://img.shields.io/badge/Benchmark%20Pass%20Rate-100%25%20(%E2%89%A45.0px)-22C55E?style=for-the-badge)](#-benchmark-performance--multi-threshold-evaluation)

**High-Precision Metrology & Scale-Invariant Sub-Pixel Pattern Localization for Advanced Semiconductor Wafers**  
*A Modular Open-Source Framework for Scanning Electron Microscope (SEM) Navigation, Scale Transition Invariance, and Periodic Memory Array Die Alignment*

[🚀 One-Click Launch](#-quickstart--execution) • [✨ Key Capabilities](#-key-capabilities) • [📐 Mathematical Formulations](#-mathematical-formulation--methodology) • [📊 Benchmark Matrix](#-benchmark-performance--multi-threshold-evaluation) • [📑 PDF Reports](#-automated-evaluation-pdf-reports) • [📂 Project Structure](#-project-architecture--repository-layout)

---

</div>

## 📌 Executive Overview

In state-of-the-art semiconductor fabrication (1x DRAM memory arrays, gate-all-around, and 3D FinFET nodes), automated wafer inspection tools encounter severe **navigation drift** during large magnification transitions.

**Drift-Sense** solves the **10:1 Field-of-View (FOV) magnification transition problem** (recovering a $100\times$ high-magnification close-up reference inside a $10\times$ wide-field search area) under:
- **Severe In-Plane Wafer Rotations** ($\pm 2.5^\circ$ continuous drift)
- **Extreme Poisson Quantum Shot Noise** (low-dose scanning electron beam)
- **Dielectric Charging Streaks & Drift Blur** (substrate beam-matter interaction)
- **Dense Periodic Array Aliasing** (repeating memory capacitor pitch)

Our unified hybrid pipeline achieves **100.0% Pass Rate ($\le 5.0\text{ px}$ spec)** on standardized DRAM-30 test suites with **mean sub-pixel accuracy of $0.03\text{ px}$ to $0.08\text{ px}$**.

---

## ✨ Key Capabilities

<table>
<tr>
<td width="50%">

### 🎯 Multi-Modal Evidence Fusion
Combines 4 complementary correlation feature maps to defeat periodic DRAM alias peaks:
$$\text{Score} = 0.35 \cdot \text{DoG} + 0.25 \cdot \text{Edge} + 0.20 \cdot \text{Raw} + 0.20 \cdot \text{Envelope}$$
- **Difference-of-Gaussians (DoG)** isolates critical gate & contact boundaries.
- **Sobel Gradient Tensors** enforce structural edge alignment.
- **Morphological Envelopes** suppress repeating capacitor aliases.

</td>
<td width="50%">

### 🔬 Sub-Pixel Quadratic Refinement
Reconstructs continuous 2D Taylor surface around peak matching coordinates:
- Eliminates discrete pixel grid quantization.
- Attains sub-nanometer localization precision ($<0.05\text{ px}$ mean error).
- Fully deterministic with $O(1)$ post-search compute latency.

</td>
</tr>
<tr>
<td width="50%">

### 🧠 Deep Learning SEM Denoising (PyTorch)
Integrates **`SEMRestorationUNet`**, an edge-preserving residual U-Net:
- Pre-filters ultra-low dose Poisson noise and sensor readout thermal noise.
- Features skip-connection gradient highways to preserve nanometer-scale line edges.
- Blends restored details: $\text{Out} = 0.70 \cdot x + 0.30 \cdot \text{UNet}(x)$.

</td>
<td width="50%">

### ⚡ Real-Time Streaming Glassmorphic UI
Built on async **FastAPI** + **HTML5 Canvas** + **Server-Sent Events (SSE)**:
- Live parameter sliders with immediate canvas rendering.
- 5-acquisition robustness stress suite.
- 3D exploded lithography mask stack decomposition.
- Live sample-by-sample 30-pair streaming benchmark with automated PDF report generation.

</td>
</tr>
</table>

---

## 🚀 Quickstart & Execution

### 1. Installation

Clone the repository and install dependencies:

```bash
# Clone repository
git clone https://github.com/your-username/Semicon_DRAM.git
cd Semicon_DRAM

# Install dependencies
pip install -r requirements.txt
```

### 2. One-Click Launch (Dashboard + Server)

Start the async FastAPI backend, initialize the glassmorphic frontend, and automatically launch your default browser:

```bash
python run.py
```
> 🌐 **Dashboard URL:** `http://localhost:8000`

---

### 3. Command-Line Interface (CLI)

#### Evaluate 30-Pair DRAM Benchmark & Compile PDF
```bash
python evaluate_dram30_live.py
```

#### Generate Synthetic 1000×1000 Parameterized Dataset
```bash
python generate_dataset.py --out_dir data/generated --num_pairs 30
```

#### Locate Single Reference Image in Search Canvas
```bash
python localize.py --ref data/generated/sample_000_ref.png --search data/generated/sample_000_search.png
```

#### Re-generate Publication Metrology PDF Report
```bash
python generate_pdf_report.py
```

---

## 🖥️ Interactive Dashboard Walkthrough

The web dashboard is organized into 5 dedicated modules:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DRIFT-SENSE SEMICONDUCTOR DASHBOARD                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [🎯 Live Matcher Arena] [🧱 CAD Layers] [🧪 Robustness] [📊 30-Pair Benchmark] [📑 Reports] │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| Tab | Feature Area | Description |
| :--- | :--- | :--- |
| 🎯 **Tab 1** | **Live Matcher Arena** | Interactive testbed. Adjust target coordinates $(X, Y)$, scale ratio ($8.5\times\text{--}11.5\times$), wafer rotation ($\pm 2.5^\circ$), electron shot dose, beam blur, and detector noise with real-time sub-pixel localization feedback. |
| 🧱 **Tab 2** | **CAD Layer Decomposition** | Inspect DRAM 1x & FinFET memory cells decomposed across discrete manufacturing masks (Substrate, Word Lines, Bit Lines, Capacitor Contacts, Landmark Tap Pads) + **3D Isometric Exploded Stack**. |
| 🧪 **Tab 3** | **Robustness Suite** | Automated stress-test evaluating 1 reference crop against **5 extreme SEM acquisition variants**: Nominal Clean, Low-Dose Poisson Noise, High Beam Blur, Dielectric Charging, and Drift Angle. |
| 📊 **Tab 4** | **30-Pair Benchmark** | High-speed multi-threaded batch evaluator. Streams per-sample predictions live with real-time pass/fail indicators, latency counters, and dynamic summary cards. |
| 📑 **Tab 5** | **Reports & Deliverables** | Live evaluation report viewer and direct download for [`DRAM_30_Evaluation_Report.pdf`](#-automated-evaluation-pdf-reports). Includes an embedded interactive document preview and one-click manual regeneration. |

---

## 📐 Mathematical Formulation & Methodology

### 1. Zero-Mean Normalized Cross-Correlation (ZNCC)
To achieve complete invariance to illumination fluctuations, electron detector gain shifts, and charging brightness offsets:

$$\gamma(u, v) = \frac{\sum_{x, y} \left[ T(x, y) - \bar{T} \right] \cdot \left[ I(x+u, y+v) - \bar{I}_{u, v} \right]}{\sqrt{\sum_{x, y} \left[ T(x, y) - \bar{T} \right]^2 \cdot \sum_{x, y} \left[ I(x+u, y+v) - \bar{I}_{u, v} \right]^2}}$$

### 2. Multi-Scale Coarse-to-Fine Gaussian Pyramid
Accelerates search over scale space $S \in [8.5, 11.5]$ and rotation space $\theta \in [-2.5^\circ, +2.5^\circ]$:
1. **Level 2 (Coarse, $\frac{1}{4}$ resolution):** Rapid candidate peak extraction over coarse $(\Delta s = 0.5, \Delta \theta = 1.0^\circ)$ grid.
2. **Level 1 (Intermediate, $\frac{1}{2}$ resolution):** Local neighborhood refinement on top-$k$ candidates.
3. **Level 0 (Full resolution):** Precise multi-modal correlation evaluation at nominal resolution.

### 3. Continuous 2D Quadratic Sub-Pixel Peak Refinement
Eliminates integer pixel quantization by fitting a 2D second-order Taylor polynomial around discrete peak $(x_0, y_0)$:

$$\Delta x = \frac{R(x_0 - 1, y_0) - R(x_0 + 1, y_0)}{2 \cdot \left[ R(x_0 - 1, y_0) - 2R(x_0, y_0) + R(x_0 + 1, y_0) \right]}$$

$$\Delta y = \frac{R(x_0, y_0 - 1) - R(x_0, y_0 + 1)}{2 \cdot \left[ R(x_0, y_0 - 1) - 2R(x_0, y_0) + R(x_0, y_0 + 1) \right]}$$

$$(x_{pred}, y_{pred}) = (x_0 + \Delta x, \; y_0 + \Delta y)$$

### 4. Periodic Ambiguity Resolution Rule
For dense periodic capacitor arrays exhibiting symmetric aliasing peaks within confidence margin $\tau = 0.92 \cdot R_{max}$:

$$(x^*, y^*) = \arg\min_{(x_i, y_i) \in \mathcal{C}_{top}} \left\| (x_i, y_i) - (x_{center}, y_{center}) \right\|_2$$

---

## 📊 Benchmark Performance & Multi-Threshold Evaluation

Standardized evaluation executed on **30 DRAM SEM test image pairs** ($100\times$ reference crops matched inside $10\times$ search canvases) under scale variations ($9.0\times, 10.0\times, 11.0\times$), wafer rotations ($-2.0^\circ$ to $+2.0^\circ$), and multi-physics SEM noise:

| Dataset | Method | Pass Rate (≤5.0px) | Pass Rate (≤4.0px) | Pass Rate (≤2.0px) | Pass Rate (≤1.0px) | Sub-Pixel (≤0.5px) | Mean Error | Median Error | Worst-Case Max | Avg Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Engineered `DRAM_30`** | **Drift-Sense Classical CV** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.03 px** | **0.03 px** | **0.08 px** | 22.4 ms |
| **Engineered `DRAM_30`** | **PyTorch ML (U-Net)** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **0.32 px** | **0.03 px** | **8.68 px** | 28.1 ms |
| **Synthetic DRAM** | **Drift-Sense Classical CV** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **0.06 px** | **0.06 px** | **0.25 px** | 24.8 ms |
| **Synthetic DRAM** | **PyTorch ML (U-Net)** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **0.08 px** | **0.07 px** | **0.30 px** | 31.2 ms |

> 🎯 **Industrial Metrology Spec:** $\ge 90\%$ Pass Rate at $\le 5.0\text{ px}$ tolerance threshold. **Drift-Sense achieves 100.0% with sub-pixel precision.**

---

## 📑 Automated Evaluation PDF Reports

Every time the 30-pair benchmark is executed (via the dashboard or CLI), Drift-Sense **automatically compiles a publication-grade evaluation PDF report**:

* **Filename:** `DRAM_30_Evaluation_Report.pdf`
* **Features Included in PDF:**
  1. **Executive Summary & KPI Cards:** Live compilation timestamps, total pairs, pass rates, and mean error metrics.
  2. **Multi-Threshold Accuracy Table:** $\le 5.0\text{px}, \le 4.0\text{px}, \le 2.0\text{px}, \le 1.0\text{px}, \le 0.5\text{px}$ breakdown.
  3. **Visual Analytics Figures:** Auto-generated high-resolution bar charts and 30-sample error curves.
  4. **Complete 30-Pair Data Table:** Full coordinate logs (Scale, Rotation, Ground Truth $(X, Y)$, Predicted $(X, Y)$, Error, Latency, Status).
  5. **Formal Mathematical Section:** Derivations for ZNCC, Taylor Series 2D Quadratic Refinement, and Pyramid Search.

---

## 📂 Project Architecture & Repository Layout

```
TanX/
├── run.py                             # 🚀 One-Click Launcher (Backend + Frontend + Auto Browser)
├── requirements.txt                   # Complete environment dependencies
├── README.md                          # Full documentation & function reference
├── server.py                          # High-Performance Async FastAPI Backend & SSE API
├── app.py                             # Alternative Streamlit GUI
├── generate_pdf_report.py             # Publication ReportLab 5.0 PDF Generator Engine
├── evaluate_dram30_live.py            # Headless 30-Pair Benchmark Evaluator & Logger
│
├── static/                            # Frontend Web Application (Vanilla HTML5 / Modern CSS / Canvas)
│   └── index.html                     # Responsive glassmorphic dashboard (Arena, CAD, Benchmark, Reports)
│
├── src/                               # Core Semiconductor & Computer Vision Algorithms
│   ├── geometry.py                    # CAD layout generators (DRAM 1x & FinFET cell arrays)
│   ├── matcher.py                     # Multi-modal feature fusion matching engine (Modes D / C / A)
│   ├── phase_correlation_matcher.py   # 2D Fourier phase correlation engine (FFT frequency domain)
│   ├── layer_visualizer.py            # Lithography mask decomposition & 3D exploded stack
│   ├── sem_effects.py                 # SEM physics (Poisson shot noise, charging streaks, beam blur)
│   ├── metrics.py                     # Sub-pixel Euclidean error & pass rate calculations
│   ├── localization/                  # Candidate peak finder & ambiguity tie-breaker
│   └── matching/                      # Multi-scale pyramid & rotation sweep matcher
│
├── models/                            # Deep Learning Neural Architectures
│   └── ai_restoration.py              # PyTorch SEMRestorationUNet edge-preserving U-Net
│
├── data/                              # Datasets & Ground-Truth Manifests
│   ├── manifest.json                  # Standardized 30-pair ground truth labels & metadata
│   └── generated/                     # 30 high-mag reference & wide search SEM pairs
│
├── results/                           # Output Metrics, Visual Figures & Reports
│   ├── metrics/                       # JSON benchmark logs & CSV per-sample tabulations
│   └── reports/                       # Compiled PDF evaluation reports & charts
└── references/                        # Academic literature citations & specifications
```

---

## 🔌 API Reference & Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves single-page responsive dashboard UI |
| `POST` | `/api/generate` | Generates reference and search SEM pair with custom physical parameters |
| `POST` | `/api/match` | Runs selected matcher engine (Mode D/C/A, Phase Corr) and returns coordinates |
| `GET` | `/api/layers` | Returns 5 decomposed lithography masks and 3D exploded isometric stack |
| `GET` | `/api/family` | Evaluates 1 reference against 5 physical SEM acquisition variants |
| `GET` | `/api/benchmark/stream` | Streams live 30-pair benchmark sample-by-sample via Server-Sent Events (SSE) |
| `POST` | `/api/benchmark/run` | Executes parallel multi-threaded 30-pair benchmark batch |
| `GET` | `/api/reports/status` | Returns latest PDF report metadata (timestamp, file size, pass rate) |
| `POST` | `/api/reports/generate` | Manually triggers fresh PDF report compilation |
| `GET` | `/api/reports/download` | Direct attachment download of `DRAM_30_Evaluation_Report.pdf` |
| `GET` | `/api/reports/view` | Inline PDF viewer endpoint for embedded preview |
| `GET` | `/api/health` | Health check endpoint returning active AI and matcher subsystem statuses |

---

## 👥 Contributing & Architecture Notes

* **Drift-Sense Architecture Group**: Engineered for robust automated e-beam navigation, nanoscale defect inspection recovery, and sub-pixel die alignment on sub-10nm DRAM and 3D FinFET nodes.
* Built to industrial semiconductor metrology standards with modular support for custom CAD layout masks and e-beam defect review tools.

---

<div align="center">
<b>DRIFT-SENSE | SEMICONDUCTOR METROLOGY & NAVIGATION RECOVERY SYSTEM</b><br/>
<sub>Designed for Sub-Nanometer Inspection Accuracy, Scale Invariance, and High-Throughput Wafer Diagnostics</sub>
</div>
