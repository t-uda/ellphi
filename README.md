# EllPHi – a fast ellipse-tangency solver for anisotropic persistent homology
[![CI](https://github.com/t-uda/ellphi/actions/workflows/python-app.yml/badge.svg)](https://github.com/t-uda/ellphi/actions/workflows/python-app.yml)
<img src="https://github.com/t-uda/ellphi/raw/main/ellphi-logo.png" alt="ellphi-logo" width="256" />

**EllPHi** brings anisotropy to persistent-homology workflows.

Starting from an ordinary 2-D point cloud, it estimates local covariance, inflates **ellipses** instead of balls, and feeds the resulting *tangency distance* into your favourite PH backend (HomCloud, Ripser, and so on). The result: cleaner barcodes, longer lifetimes, and ring structures that survive heavy noise — all without rewriting your topology code.

## Installation

Install from PyPI:

```bash
pip install ellphi
```

## Supported Python Versions

Python 3.10 or later.

## Quick start

Install and solve a tangency query in just a few lines:

```bash
pip install ellphi
```

```python
import numpy as np
import ellphi

pcoef = ellphi.coef_from_cov([0.0, 0.0], [[0.2, 0.0], [0.0, 0.1]])[0]
qcoef = ellphi.coef_from_cov([1.0, 0.25], [[0.15, 0.0], [0.0, 0.25]])[0]

result = ellphi.tangency(pcoef, qcoef)
print(f"t = {result.t:.3f}")       # tangency distance
print(f"point = {result.point}")
```

For deeper workflows, see the accompanying notebooks:

* [`quickstart.ipynb`](https://github.com/t-uda/ellphi/notebooks/quickstart.ipynb) – 5-minute tour
* [`eph-6rings-PH.ipynb`](https://github.com/t-uda/ellphi/notebooks/eph-6rings-PH.ipynb) – full pipeline
* [`eph-6rings-PH-figures.ipynb`](https://github.com/t-uda/ellphi/notebooks/eph-6rings-PH-figures.ipynb) – figures presented in ATMCS 2025 poster
* [`ndim-demo-3d.ipynb`](https://github.com/t-uda/ellphi/notebooks/ndim-demo-3d.ipynb) – 3-D ellipsoid cloud + tangency distance walkthrough

> **For ATMCS 2025 attendees**
> See **[`eph-6rings-PH-figures.ipynb`](https://github.com/t-uda/ellphi/notebooks/eph-6rings-PH-figures.ipynb)**
> which accompanies the conference poster.

## Check the installed version

Inside Python:

```python
import ellphi

print(ellphi.version_info())
```

From the shell:

```bash
python -m ellphi --version
```
