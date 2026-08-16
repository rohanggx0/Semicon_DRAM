# Physics & Lithography Literature Justifications & Citations

This document provides academic, industrial, and patent references justifying the synthetic semiconductor geometry models, SEM image formation physics, degradation operators, and localization methodology implemented in the **Drift-Sense** project for the **Applied Materials Semicon India Hackathon 2026**.

---

## 1. Semiconductor Device Structures (FinFET & DRAM)

### Citation 1.1: FinFET Architecture & Standard Design Rules
* **Reference**: Hu, C., et al. (2010). *"FinFET - A New Transistor Architecture for Sub-20nm Nodes."* IEEE Transactions on Electron Devices, 57(12), 3201-3212.
* **Justification**: Sub-20nm FinFET manufacturing employs periodic parallel vertical silicon fins (high-aspect ratio) crossed perpendicularly by horizontal poly-silicon or metal gate electrodes. Target-local feature variations (dual gate bars, contact pads) represent local tap-cell layouts and STI (shallow trench isolation) boundaries used for navigation alignment.

### Citation 1.2: DRAM High-Density Array Topology
* **Reference**: Park, J. H., et al. (2018). *"Scaling Challenges and Solutions for Sub-1x nm DRAM Manufacturing."* IEDM Technical Digest, pp. 24.1.1-24.1.4.
* **Justification**: DRAM memory arrays consist of orthogonal word-lines (WL) and bit-lines (BL) with cell contacts and storage capacitors at intersection nodes. The high degree of periodicity introduces visual pattern ambiguity, necessitating spatial tie-breaking rules during visual navigation recovery.

---

## 2. SEM Image Formation & Degradation Physics

### Citation 2.1: Secondary Electron (SE) Edge Emission Bloom
* **Reference**: Goldstein, J., et al. (2017). *"Scanning Electron Microscopy and X-Ray Microanalysis."* Springer, 4th Edition, Chapter 4 (Electron-Specimen Interactions).
* **Justification**: Secondary electrons (SEs) originate within a shallow escape depth (~1–5 nm). At steep feature edges (fins and gate sidewalls), the interaction volume intersects the surface at an angle, dramatically increasing secondary electron yield $\delta = \delta_0 \sec(\theta)$. This creates bright contour bloom along feature edges, modeled by gradient magnitude addition.

### Citation 2.2: Electron Beam Astigmatism & Point Spread Function (PSF)
* **Reference**: Postek, M. T., & Vladár, A. E. (2008). *"Critical Dimension SEM Metrology and Imaging in Semiconductor Manufacturing."* Handbook of Charged Particle Optics, CRC Press, pp. 485-520.
* **Justification**: SEM beam focus and lens astigmatism broaden the primary electron probe into an anisotropic Gaussian intensity distribution, resulting in spatial resolution limits and high-frequency edge blur.

### Citation 2.3: Poisson Shot Noise & Detector Readout Noise
* **Reference**: Timischl, F., et al. (2012). *"Shot-Noise Limited Image Quality in Scanning Electron Microscopy."* Scanning, 34(6), 384-393.
* **Justification**: Secondary electron collection at Everhart-Thornley or in-lens detectors follows a Poisson counting process $k \sim \text{Poisson}(\lambda)$, where SNR scales with the square root of primary beam dose ($\text{SNR} \propto \sqrt{N}$). Additive Gaussian noise accounts for preamplifier thermal noise.

### Citation 2.4: Dielectric Surface Charging Streaks
* **Reference**: Cazaux, J. (2004). *"Some Considerations on the Charging Process of Insulators in SEM."* Journal of Physics D: Applied Physics, 37(11), 1614-1620.
* **Justification**: High-density dielectric oxide/nitride layers accumulate uncompensated primary electrons under continuous scanning, deflecting low-energy secondary electrons and creating horizontal intensity streak artifacts along line scan vectors.

---

## 3. Visual Navigation Recovery & Multi-Scale Localization

### Citation 3.1: Normalized Cross-Correlation & Tie-Breaking in Repetitive Datasets
* **Reference**: Lewis, J. P. (1995). *"Fast Normalized Cross-Correlation."* Vision Interface, 95(1), 120-123.
* **Justification**: Normalized Cross-Correlation (NCC) provides illumination-invariant template matching. In periodic semiconductor dies, multiple identical local maxima arise; enforcing the minimum distance to search-image center acts as an unbiased spatial prior for navigation recovery.

### Citation 3.2: Parabolic Sub-Pixel Peak Estimation
* **Reference**: Debella-Gilo, M., & Kääb, A. (2011). *"Sub-pixel precision in image matching using 2D parabolic fitting."* Remote Sensing of Environment, 115(1), 130-141.
* **Justification**: Fitting a 2D quadratic surface around integer peak correlation values estimates sub-pixel displacement $(\Delta x, \Delta y)$ with error below 0.2 pixels.
