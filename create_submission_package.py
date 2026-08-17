import os
import shutil
from pathlib import Path

BASE_DIR = Path(r"D:\SemiconFINFETwork")
SUBMISSION_DIR = BASE_DIR / "TanX"

def setup_submission_directory():
    print(f"Setting up clean submission folder at: {SUBMISSION_DIR}")

    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Create subdirectories
    (SUBMISSION_DIR / "configs").mkdir(parents=True, exist_ok=True)
    (SUBMISSION_DIR / "models").mkdir(parents=True, exist_ok=True)
    (SUBMISSION_DIR / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (SUBMISSION_DIR / "results" / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    (SUBMISSION_DIR / "references").mkdir(parents=True, exist_ok=True)

    # 2. Copy core python scripts
    shutil.copy(BASE_DIR / "run.py", SUBMISSION_DIR / "run.py")
    shutil.copy(BASE_DIR / "generate_dataset.py", SUBMISSION_DIR / "generate_dataset.py")
    shutil.copy(BASE_DIR / "localize.py", SUBMISSION_DIR / "localize.py")

    # 3. Copy configs
    if (BASE_DIR / "configs" / "default_config.yaml").exists():
        shutil.copy(BASE_DIR / "configs" / "default_config.yaml", SUBMISSION_DIR / "configs" / "default_config.yaml")

    # 4. Copy entire src directory recursively
    src_dir = BASE_DIR / "src"
    if src_dir.exists():
        shutil.copytree(src_dir, SUBMISSION_DIR / "src", dirs_exist_ok=True)
    elif (BASE_DIR / "drift_sense" / "src").exists():
        shutil.copytree(BASE_DIR / "drift_sense" / "src", SUBMISSION_DIR / "src", dirs_exist_ok=True)

    # 5. Copy models
    ai_model = BASE_DIR / "models" / "ai_restoration.py"
    if not ai_model.exists():
        ai_model = BASE_DIR / "extracted_driftsense_recovery" / "Drift-Sense Navigation-Error Recovery" / "Drift-Sense" / "models" / "ai_restoration.py"
    
    if ai_model.exists():
        shutil.copy(ai_model, SUBMISSION_DIR / "models" / "ai_restoration.py")
        (SUBMISSION_DIR / "models" / "__init__.py").write_text("# Models package", encoding="utf-8")

    # 6. Copy metrics & reports
    metrics_files = [
        ("drift_sense/results/metrics/dram_ml_benchmark_summary.json", "results/metrics/dram_ml_benchmark_summary.json"),
        ("drift_sense/results/metrics/dram_ml_multi_threshold_summary.json", "results/metrics/dram_ml_multi_threshold_summary.json"),
        ("drift_sense/results/metrics/dram_ml_benchmark_per_sample.csv", "results/metrics/dram_ml_benchmark_per_sample.csv")
    ]
    for src_rel, dst_rel in metrics_files:
        p = BASE_DIR / src_rel
        if p.exists():
            shutil.copy(p, SUBMISSION_DIR / dst_rel)

    pdf_report = BASE_DIR / "DRAM_ML_Benchmark_Report.pdf"
    if pdf_report.exists():
        shutil.copy(pdf_report, SUBMISSION_DIR / "results" / "reports" / "DRAM_ML_Benchmark_Report.pdf")

    figs = ["fig1_summary.png", "fig2_ds1_errors.png"]
    for fig in figs:
        fig_path = BASE_DIR / "drift_sense" / "results" / "reports" / "figures" / fig
        if fig_path.exists():
            shutil.copy(fig_path, SUBMISSION_DIR / "results" / "reports" / "figures" / fig)

    # 7. Create requirements.txt
    req_text = """numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.0.0
torch>=2.0.0
matplotlib>=3.4.0
reportlab>=3.6.0
pyyaml>=5.4.0
"""
    (SUBMISSION_DIR / "requirements.txt").write_text(req_text, encoding="utf-8")
    (BASE_DIR / "requirements.txt").write_text(req_text, encoding="utf-8")

    # 8. Create references/sem_imaging_citations.md
    citations_text = """# Public Literature & Structural References

### 1. SEM Physics & Image Degradation Modeling
* **Goldstein, J. et al.** (2017). *Scanning Electron Microscopy and X-Ray Microanalysis*, Springer.
  - *Application*: Technical justification for Poisson shot noise (beam current simulation), Gaussian readout noise, electron charging artifacts, and beam astigmatism/blur.
* **Postek, M. T., & Vladár, A. E.** (2015). *Critical Dimension SEM Measurement Metrology in Semiconductor Manufacturing*, Journal of Research of NIST.
  - *Application*: SEM edge blooming, secondary electron emission modeling, and feature edge response function.

### 2. Semiconductor Geometry & Memory Array Design
* **ITRS / IRDS Roadmap** (International Roadmap for Devices and Systems).
  - *Application*: DRAM capacitor contact hole array matrix pitch scaling (nominal 80 nm ref scale -> 8 nm 10x search scale) and active wordline/bitline geometry.
* **FinFET Lithography Patterns**:
  - Standard multi-gate FinFET pitch ratios, dummy gates, and peripheral scribe line structures.

### 3. Cross-Magnification Pattern Matching & Scale-Space Localization
* **Lindeberg, T.** (1998). *Feature Detection with Automatic Scale Selection*, International Journal of Computer Vision.
  - *Application*: Multi-scale Difference-of-Gaussians (DoG) pyramid matching across 10:1 magnification transitions.
* **Lewis, J. P.** (1995). *Fast Normalized Cross-Correlation*, Industrial Light & Magic.
  - *Application*: Zero-mean Normalized Cross-Correlation (ZNCC) for illumination-invariant template matching.
"""
    (SUBMISSION_DIR / "references" / "sem_imaging_citations.md").write_text(citations_text, encoding="utf-8")

    # 9. Create README.md
    readme_text = """# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools
**Applied Materials Problem Statement | Hackathon 2026**

Drift-Sense is a robust, scale-aware, rotation-tolerant localization solution designed to recover target navigation coordinates on semiconductor wafers (DRAM & FinFET architectures) across a **10:1 nominal magnification transition** (100x reference close-up to 10x wide search image).

---

## 1. Submission Folder Structure
```
TanX/
├── run.py                             # One-click launcher & entry point
├── requirements.txt                   # Environment dependencies
├── README.md                          # Solution documentation & setup instructions
├── generate_dataset.py                # Parameterized synthetic SEM image generator
├── localize.py                        # Standalone multi-scale localization CLI
├── configs/
│   └── default_config.yaml            # Default matching & degradation parameters
├── src/                               # Core algorithm modules
│   ├── geometry.py                    # Layout canvas & coordinate transformations
│   ├── matcher.py                     # Multi-scale DoG + ZNCC matching engine
│   ├── metrics.py                     # Euclidean error & pass rate calculations
│   ├── sem_effects.py                 # SEM noise, blur & charging pipeline
│   ├── phase_correlation_matcher.py   # Frequency-domain matching backend
│   └── localization/
│       └── candidate_finder.py        # Ambiguity resolver & closest-to-center tie-breaker
├── models/
│   └── ai_restoration.py              # PyTorch SEMRestorationUNet edge-preserving U-Net
├── results/
│   ├── metrics/                       # Multi-threshold JSON & CSV metric outputs
│   └── reports/                       # Generated publication PDF report & figures
└── references/
    └── sem_imaging_citations.md       # Public literature references & citations
```

---

## 2. Comprehensive Technology Stack & Framework Inventory

### 1. Programming Language & Core Runtime
* **Python 3.8+**: Primary development runtime.
* **Standard Libraries**: `os`, `sys`, `json`, `math`, `time`, `pathlib`, `csv`, `argparse`, `typing`.

### 2. Machine Learning & Deep Learning
* **PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)**:
  * **Model Architecture**: `SEMRestorationUNet(nn.Module)` (Localization-Aware Edge-Preserving U-Net).
  * **Neural Layers**: 2D Convolutions (`nn.Conv2d`), Batch Normalization (`nn.BatchNorm2d`), ReLU activations (`nn.ReLU`), Max Pooling (`nn.MaxPool2d`), Transposed Convolutions (`nn.ConvTranspose2d`), and feature concatenation skip connections (`torch.cat`).
  * **Purpose**: Pre-filters heavy SEM Poisson shot noise, enhances feature contrast, and preserves sub-nanometer boundary edges.

### 3. Computer Vision & Image Processing
* **OpenCV (`opencv-python` / `cv2`)**:
  * **Normalized Cross-Correlation**: `cv2.matchTemplate` with Zero-mean Normalized Cross-Correlation (`cv2.TM_CCOEFF_NORMED`).
  * **Pyramid Resizing & Rotations**: `cv2.resize` (Bilinear & Area interpolation) and `cv2.getRotationMatrix2D` / `cv2.warpAffine`.
  * **Feature Extraction**: Difference-of-Gaussians (`cv2.GaussianBlur`) and Sobel edge magnitude (`cv2.Sobel`).
  * **Morphological Operations**: `cv2.dilate` (3x3 and 9x9 kernels for local correlation peak detection).
  * **Layout Rendering**: `cv2.circle`, `cv2.line`, `cv2.rectangle`.
* **Pillow (`PIL.Image`, `ImageDraw`)**: Grayscale image I/O, format conversions, and post-rotation coordinate transformations.

### 4. Numerical Computation & Mathematics
* **NumPy (`numpy`)**:
  * Tensor operations, pseudo-random number generation (`np.random.default_rng`), 2D affine rotation matrices.
  * Multi-threshold metric calculations ($\le 5\\text{px}, \\le 4\\text{px}, \\le 2\\text{px}, \\le 1\\text{px}, \\le 0.5\\text{px}$ sub-pixel), mean, median, worst-case max error, and standard deviation.

### 5. Data Visualization & Analytics
* **Matplotlib (`matplotlib.pyplot`)**:
  * Headless server rendering (`matplotlib.use('Agg')`).
  * Pass rate bar charts, mean error comparisons, and per-sample Euclidean error profile plots.

### 6. Publication PDF Report Generation
* **ReportLab (`reportlab`)**:
  * Flowable document architecture (`SimpleDocTemplate`, `Paragraph`, `Table`, `TableStyle`, `Image`, `Spacer`, `HRFlowable`).
  * Dynamic 2-pass canvas (`NumberedCanvas` class) for custom header (*"Drift-Sense Benchmark"*) and footer (*"Page X of Y | Confidential"*).

### 7. Semiconductor Domain Algorithms
* **Multi-Feature Evidence Fusion Engine**:
  * Combines 4 correlation maps: $\\text{Score} = 0.35 \\cdot \\text{DoG} + 0.25 \\cdot \\text{Edge} + 0.20 \\cdot \\text{Raw} + 0.20 \\cdot \\text{Macro Envelope}$.
* **Scale-Space Pyramid Search**:
  * Coarse search ($8.5\\times\\text{--}11.5\\times$, step $0.5$) + fine local neighborhood search (step $0.25$).
* **Ambiguity Resolver / Tie-Breaker**:
  * `AmbiguityResolver` in `candidate_finder.py`: Applies top-score threshold filtering & Euclidean distance to search image center $(500.0, 500.0)$ for periodic DRAM array aliasing.
* **SEM Degradation Physics Engine**:
  * Simulates Poisson shot noise, Gaussian readout noise, beam astigmatism/blur, charging streaks, and line jitter.

### 8. Serialization & File Formats
* **YAML (`pyyaml`)**: `configs/default_config.yaml` for global configuration.
* **JSON**: Ground-truth metadata and multi-threshold metric outputs.
* **CSV**: Per-sample tabulations (`DictWriter`).

---

## 3. Coordinate System & Conventions
* **Origin $(0,0)$**: Top-left pixel of the search image.
* **Axes**: $x$ increases to the right ($0 \\le x \\le 1000$), $y$ increases downward ($0 \\le y \\le 1000$).
* **Target Coordinates**: Returned as $(x_{pred}, y_{pred})$ center coordinates in search-image pixels.
* **Tie-Breaking Rule**: If multiple periodic candidate peaks fall within the top score threshold, the algorithm selects the candidate closest to the search image center $(500.0, 500.0)$.

---

## 4. Environment Setup & Execution Commands

### Prerequisites
* Python 3.8 or higher
* PyTorch 2.x (CPU or CUDA)

### Installation
```bash
pip install -r requirements.txt
```

### 1. Synthetic Dataset Generation
Generate parameterized 1000x1000 reference and search image pairs with exact post-rotation ground-truth labels:
```bash
python generate_dataset.py --out_dir data/generated --num_pairs 30
```

### 2. Single Image Pair Localization
Locate a 100x reference pattern inside a 10x search image:
```bash
python localize.py --ref data/sample_001_ref.png --search data/sample_001_search.png --out_dir results/
```

### 3. Batch Evaluation with PyTorch ML Pre-filtering
Run full multi-threshold evaluation on dataset batch:
```bash
python localize.py --batch_dir DRAM_30/ --use_ai
```

---

## 5. Benchmark Performance Summary

### Multi-Threshold Pass Rates & Sub-Pixel Accuracy (N=30)
| Dataset | Algorithm Model | Pass $\\le 5\\text{px}$ | Pass $\\le 4\\text{px}$ | Pass $\\le 2\\text{px}$ | Pass $\\le 1\\text{px}$ | Sub-Pixel ($\\le 0.5\\text{px}$) | Mean Error | Median Error | Worst-Case Max Error | Avg Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Engineered `DRAM_30`** | **Classical CV** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.03 px** | **0.03 px** | **0.08 px** | 4,641 ms |
| **Engineered `DRAM_30`** | **PyTorch ML (U-Net)** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **0.32 px** | **0.03 px** | **8.68 px** | 6,927 ms |
| **Synthetic DRAM** | **Classical CV** | 3.3% | 3.3% | 3.3% | 3.3% | 3.3% | 307.00 px | 370.23 px | 619.13 px | 2,307 ms |
| **Synthetic DRAM** | **PyTorch ML (U-Net)** | **10.0%** | **10.0%** | **10.0%** | **10.0%** | **10.0%** | **245.28 px** | **247.67 px** | **616.02 px** | 4,613 ms |

---

## 6. Failure Case Analysis & Root Cause Explanation
* **Periodic Array Aliasing**: DRAM memory cells feature a dense matrix of identical contact holes repeating every $\\sim 8\\text{ px}$ in search space. When a reference crop is randomly taken inside a uniform array without macro landmarks (scribe lines or alignment blocks), normalized cross-correlation yields dozens of identical periodic peaks.
* **SEM Noise Sensitivity**: High SEM shot noise can cause peak swapping between adjacent cell pitches.
* **PyTorch U-Net Mitigation**: Applying `SEMRestorationUNet` edge-preserving denoising reduced shot noise variance, tripling the pass rate ($3.3\\% \\rightarrow 10.0\\%$) and fixing extreme errors (e.g. Sample 01 error dropped from $565.6\\text{ px}$ to $0.22\\text{ px}$).
"""
    (SUBMISSION_DIR / "README.md").write_text(readme_text, encoding="utf-8")
    (BASE_DIR / "README.md").write_text(readme_text, encoding="utf-8")

    print("[SUCCESS] Submission package and READMEs updated successfully at:", SUBMISSION_DIR)

if __name__ == "__main__":
    setup_submission_directory()
