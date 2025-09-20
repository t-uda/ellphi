# EllPHi – a fast ellipse-tangency solver for anisotropic persistent homology
<img src="https://github.com/t-uda/ellphi/raw/main/ellphi-logo.png" alt="ellphi-logo" width="256" />

**EllPHi** brings anisotropy to persistent-homology workflows.

Starting from an ordinary 2-D point cloud, it estimates local covariance, inflates **ellipses** instead of balls, and feeds the resulting *tangency distance* into your favourite PH backend (HomCloud, Ripser, and so on). The result: cleaner barcodes, longer lifetimes, and ring structures that survive heavy noise — all without rewriting your topology code.

> **For ATMCS 2025 attendees**  
> See **[`eph-6rings-PH-figures.ipynb`](notebooks/eph-6rings-PH-figures.ipynb)**  
> which accompanies the conference poster.

## Quick start (under construction)

* [`quickstart.ipynb`](notebooks/quickstart.ipynb) – 5-minute tour  
* [`eph-6rings-PH.ipynb`](notebooks/eph-6rings-PH.ipynb) – full pipeline  
* [`eph-6rings-PH-figures.ipynb`](notebooks/eph-6rings-PH-figures.ipynb) – figures presented in ATMCS 2025 poster

## Installation (under construction 🚧)

A PyPI release is in progress. Until then, install from GitHub:

```bash
pip install git+https://github.com/t-uda/ellphi.git
```

### C++ toolchain support

EllPHi ships a prebuilt C++ backend when you install from source. The build
step now relies on Python's compiler configuration, and our GitHub Actions
matrix exercises Linux, macOS, and Windows builds on every push. You will need
one of the following toolchains if you compile locally:

* **Linux** – GCC or Clang with support for `-std=c++17`, `-O3`, `-fPIC`, and
  shared libraries (`-shared`).
* **macOS** – Apple Clang (Xcode 15 or newer) with `-dynamiclib` and
  `-undefined dynamic_lookup`.
* **Windows** – Microsoft Visual C++ 2019+ (MSVC) or compatible toolchains with
  `/std:c++17` and `/LD`.

The CI smoke tests ensure the packaged wheel loads the compiled backend on all
three platforms, so end users installing via `pip` receive a working shared
library by default.
