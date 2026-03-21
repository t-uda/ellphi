![EllPHi](assets/logo.png){ width=220 style="display:block;margin:0 auto 1rem;" }

# EllPHi

**EllPHi** is a fast ellipse (and ellipsoid) tangency solver for Python, designed for applications in Topological Data Analysis (TDA) and Computational Geometry.

## Features

- **Ellipse cloud construction** — fit local covariance ellipses to a point cloud with `ellipse_cloud`.
- **Fast tangency distances** — compute the pairwise tangency distance matrix with `pdist_tangency`, dispatching automatically to a compiled C++ backend (Eigen) when available.
- **Differentiable solver** — `ellphi.grad` provides per-pair gradients (`tangency_grad`) and a batch pairwise VJP (`pdist_tangency_grad`) for gradient-based optimisation and persistent-homology backprop.
- **n-dimensional support** — geometry and solver work in any dimension ≥ 2.

## Quick install

```bash
pip install ellphi
```

## Five-line example

```python
import numpy as np
from ellphi import ellipse_cloud, pdist_tangency

rng = np.random.default_rng(0)
X = rng.standard_normal((60, 2))         # 60 points in 2D
cloud = ellipse_cloud(X, k=5)            # fit local-covariance ellipses
dists = cloud.pdist_tangency()           # condensed distance matrix (1770,)
print(dists.min(), dists.max())
```

## Documentation

- [Getting Started](guide/quickstart.md) — install and first steps
- [TDA Workflow](guide/tda_workflow.md) — full pipeline from point cloud to persistence diagram
- [Differentiable Tangency](guide/differentiable.md) — gradient-based optimisation with `ellphi.grad`
- [API Reference](reference/index.md) — full module documentation
