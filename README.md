# EllPHi – a fast ellipse-tangency solver for anisotropic persistent homology
<img src="https://github.com/t-uda/ellphi/raw/main/ellphi-logo.png" alt="ellphi-logo" width="256" />

**EllPHi** brings anisotropy to persistent-homology workflows.

Starting from an ordinary 2-D point cloud, it estimates local covariance, inflates **ellipses** instead of balls, and feeds the resulting *tangency distance* into your favourite PH backend (HomCloud, Ripser, and so on). The result: cleaner barcodes, longer lifetimes, and ring structures that survive heavy noise — all without rewriting your topology code.

## Installation

A PyPI release is in progress. Until then, install from GitHub:

```bash
pip install ellphi
```

## Quick start (under construction)

* [`quickstart.ipynb`](notebooks/quickstart.ipynb) – 5-minute tour  
* [`eph-6rings-PH.ipynb`](notebooks/eph-6rings-PH.ipynb) – full pipeline  
* [`eph-6rings-PH-figures.ipynb`](notebooks/eph-6rings-PH-figures.ipynb) – figures presented in ATMCS 2025 poster

> **For ATMCS 2025 attendees**  
> See **[`eph-6rings-PH-figures.ipynb`](notebooks/eph-6rings-PH-figures.ipynb)**  
> which accompanies the conference poster.

