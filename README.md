# EllPHi – a fast ellipse-tangency solver for anisotropic persistent homology
[![CI](https://github.com/t-uda/ellphi/actions/workflows/python-app.yml/badge.svg)](https://github.com/t-uda/ellphi/actions/workflows/python-app.yml) [![codecov](https://codecov.io/gh/t-uda/ellphi/graph/badge.svg)](https://codecov.io/gh/t-uda/ellphi)
<img src="https://github.com/t-uda/ellphi/raw/main/ellphi-logo.png" alt="ellphi-logo" width="256" />

**EllPHi** brings anisotropy to persistent-homology workflows.

Starting from an ordinary point cloud, it estimates local covariance, inflates **ellipsoids** instead of balls, and feeds the resulting *tangency distance* into your favourite PH backend (HomCloud, Ripser, and so on). The result: cleaner barcodes, longer lifetimes, and ring structures that survive heavy noise — all without rewriting your topology code.

## Features

- **Ellipse Creation**: Easily create ellipses from covariance matrices or directly from point cloud neighborhoods.
- **Tangency Distance**: Compute the tangency distance between pairs of ellipses, a key component for anisotropic persistent homology.
- **High-Performance Backend**: Includes a C++ backend for fast and efficient tangency calculations, with a pure Python fallback for portability.
- **N-Dimensional Support**: Works with n-dimensional ellipsoids, allowing for analysis in higher-dimensional spaces.
- **Visualization**: Comes with helper functions to quickly visualize ellipse clouds using Matplotlib.

## Installation

Install from PyPI:

```bash
pip install ellphi
```

To install with demo dependencies for notebooks and examples:

```bash
pip install ellphi[demo]
```

## Optional Eigen build (from source)

If you build from source (for example, with `poetry install`), you can enable
the C++ linear algebra implementation based on Eigen.

Install Eigen headers:

- Ubuntu: `apt-get install libeigen3-dev`
- macOS (Homebrew): `brew install eigen`

Then build with:

```bash
ELLPHI_USE_EIGEN=1 ELLPHI_EIGEN_INCLUDE=/usr/include/eigen3 poetry install
```

On macOS, use `/opt/homebrew/include/eigen3` (Apple Silicon) or
`/usr/local/include/eigen3` (Intel).

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

## Usage

Here's a complete example of how to create an ellipse cloud from a point cloud and compute the pairwise tangency distances:

```python
import numpy as np
import ellphi

# 1. Generate a sample point cloud
np.random.seed(42)
X = np.random.rand(100, 2)

# 2. Create an ellipse cloud from the point cloud
# This will estimate the local covariance for each point's neighborhood
ellipses = ellphi.ellipse_cloud(X, k=5)

# 3. Compute the pairwise tangency distances
# This will use the C++ backend if available
distances = ellipses.pdist_tangency()

print("Computed tangency distances for", len(distances), "pairs of ellipses.")
```

For deeper workflows, see the accompanying notebooks:

* [`quickstart.ipynb`](https://github.com/t-uda/ellphi/blob/main/notebooks/quickstart.ipynb) – 5-minute tour
* [`eph-6rings-PH.ipynb`](https://github.com/t-uda/ellphi/blob/main/notebooks/eph-6rings-PH.ipynb) – full pipeline
* [`eph-6rings-PH-figures.ipynb`](https://github.com/t-uda/ellphi/blob/main/notebooks/eph-6rings-PH-figures.ipynb) – figures presented in ATMCS 11 poster
* [`ndim-demo-3d.ipynb`](https://github.com/t-uda/ellphi/blob/main/notebooks/ndim-demo-3d.ipynb) – 3-D ellipsoid cloud + tangency distance walkthrough

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

Build info (including the C++ linear algebra choice):

```bash
python -m ellphi --build-info
```

## Citation

Citation metadata for both the paper and the software is kept in
[`CITATION.cff`](CITATION.cff).

### Citing the paper

If you use EllPHi in your research, please cite the JSIAM Letters paper:

> Tomoki Uda, *Fast and accurate computation of ellipse tangency time for
> anisotropic persistent homology*, JSIAM Letters **18** (2026), 13–16.
> DOI: [10.14495/jsiaml.18.13](https://doi.org/10.14495/jsiaml.18.13)
> ([J-STAGE](https://www.jstage.jst.go.jp/article/jsiaml/18/0/18_13/_article/-char/en))

This is also what GitHub's "Cite this repository" widget (in the sidebar of
the repository page) generates, via the `preferred-citation` entry in
`CITATION.cff`.

BibTeX:

```bibtex
@article{Uda2026EllipseTangency,
  author  = {Uda, Tomoki},
  title   = {Fast and accurate computation of ellipse tangency time for
             anisotropic persistent homology},
  journal = {JSIAM Letters},
  volume  = {18},
  year    = {2026},
  pages   = {13--16},
  doi     = {10.14495/jsiaml.18.13},
}
```

### Citing the software

To cite the software itself:

```bibtex
@software{Uda2026EllPHi,
  author  = {Uda, Tomoki},
  title   = {EllPHi},
  version = {0.1.2},
  year    = {2026},
  url     = {https://github.com/t-uda/ellphi},
  license = {MIT},
}
```

## Contributing

Interested in contributing? We welcome pull requests!

Please read `CONTRIBUTING.md` for the branching/versioning flow and
`AGENTS.md` (plus `.github/workflows/python-app.yml`) for the required local
checks before opening a PR.

To get started with development, clone the repository and set up your environment:

```bash
git clone https://github.com/t-uda/ellphi.git
cd ellphi

# Install dependencies, including development tools
poetry install --with dev

# Run the tests to make sure everything is set up correctly
poetry run pytest
```
