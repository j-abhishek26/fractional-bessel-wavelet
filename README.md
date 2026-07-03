# Fractional Bessel–Hypergeometric Wavelets

> **Companion code for:**
> *"Fractional Bessel–Hypergeometric Wavelets: Rigorous Construction, Spectral Theory, and Multi-Domain Applications"*
>
> Submitted to **Applied Mathematics and Computation** (Elsevier, Q1)

---

## Authors

- **Linu Tess Antony** — Dept. of ECE, Amal Jyothi College of Engineering, Kerala, India
- **S. N. Kumar** — Dept. of ECE, Amal Jyothi College of Engineering, Kerala, India
- **J. Rajasingh** — Dept. of Mathematics, Kumaraguru College of Technology, Coimbatore, India
- **Abhishek J** — Dept. of AI&DS, Kumaraguru College of Technology, Coimbatore, India

---

## Overview

This repository contains the complete Python implementation used to generate all **10 figures** and **quantitative validation tables** presented in the paper. The code implements the fractional Bessel–Hypergeometric mother wavelet:

$$\Psi_\nu(x) = x^\nu \, e^{-x} \; {}_2F_1\!\left(\nu+1,\; \nu+\tfrac{3}{2};\; 2\nu+2;\; -x\right), \qquad x \ge 0, \quad \nu > -\tfrac{1}{2}$$

and demonstrates its properties through spectral analysis, continuous wavelet transforms, and four application domains.

---

## Repository Structure

```
.
├── bessel_wavelet_all_figures.py   # Main script — generates all figures and tables
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
├── README.md                       # This file
└── figures/                        # Generated figures (after running the script)
    ├── fig1_time_freq.png
    ├── fig2_3d_surface.png
    ├── fig3_scalogram.png
    ├── fig4_admissibility.png
    ├── fig5_localization.png
    ├── fig6_snr_comparison.png
    ├── fig7_fde_convergence.png
    ├── fig8_2d_filter.png
    ├── fig9_seismic.png
    └── fig10_power_quality.png
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.9

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/fractional-bessel-wavelet.git
cd fractional-bessel-wavelet

# Install dependencies
pip install -r requirements.txt
```

### Generate All Figures

```bash
python bessel_wavelet_all_figures.py
```

All 10 figures will be saved as high-resolution PNG files in the current directory. The console will also display quantitative validation tables (Parseval verification and time–frequency spread analysis).

---

## Figures

| Figure | Filename | Description |
|--------|----------|-------------|
| Fig. 1 | `fig1_time_freq.png` | Time-domain wavelet profiles and Fourier spectra |
| Fig. 2 | `fig2_3d_surface.png` | 3-D parameter surface Ψ_ν(x) over (x, ν) |
| Fig. 3 | `fig3_scalogram.png` | CWT scalogram of a noisy chirp signal |
| Fig. 4 | `fig4_admissibility.png` | Admissibility constant vs. fractional order |
| Fig. 5 | `fig5_localization.png` | Time–frequency localisation (Heisenberg product) |
| Fig. 6 | `fig6_snr_comparison.png` | SNR robustness: proposed wavelet vs. Morlet |
| Fig. 7 | `fig7_fde_convergence.png` | FDE collocation convergence rates |
| Fig. 8 | `fig8_2d_filter.png` | 2-D isotropic filter bank (medical imaging) |
| Fig. 9 | `fig9_seismic.png` | Seismic facies decomposition |
| Fig. 10 | `fig10_power_quality.png` | Power-quality disturbance detection |

---

## Key Theoretical Results Validated

| Result | Paper Reference | Numerical Validation |
|--------|----------------|----------------------|
| L² membership and admissibility | Theorem 3.8 | Fig. 4 — C_Ψ finite for all tested ν |
| Closed-form Fourier transform (₃F₂ series) | Theorem 3.5 | Fig. 1(b) — magnitude and phase spectra |
| Heisenberg uncertainty bound | Theorem 3.10 | Fig. 5 — minimum product near ν ≈ 3 |
| Parseval equality | — | Table 2 — relative error < 10⁻¹⁰ |
| Super-algebraic convergence (≈ N⁻²·⁸) | Section 5.3 | Fig. 7 — log-log convergence plot |

---



