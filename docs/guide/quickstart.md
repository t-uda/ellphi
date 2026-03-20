# Getting Started

## Installation

```bash
pip install ellphi
```

The C++ backend (recommended for performance) is compiled during installation when a suitable compiler is available. Check whether it loaded:

```python
import ellphi
print(ellphi.build_info())
```

## Create an ellipse cloud

`ellipse_cloud` fits a local-covariance ellipse around the *k*-nearest neighbours of each point:

```python
import numpy as np
from ellphi import ellipse_cloud

rng = np.random.default_rng(42)
X = rng.standard_normal((80, 2))   # 80 points in ℝ²

cloud = ellipse_cloud(X, k=5)
print(cloud)  # EllipseCloud(coef=array<(N, 6)>, ...)
```

Each ellipse is stored as a packed conic coefficient vector of length
$m = (d+1)(d+2)/2$ (6 for 2D, 10 for 3D).

## Compute pairwise tangency distances

```python
dists = cloud.pdist_tangency()   # condensed array, same layout as scipy.pdist
print(dists.shape)               # (N*(N-1)//2,)
```

Convert to a square matrix when needed:

```python
from scipy.spatial.distance import squareform
D = squareform(dists)
```

## Visualise

```python
import matplotlib.pyplot as plt

ax = cloud.plot(alpha=0.4, edgecolor="steelblue")
ax.scatter(X[:, 0], X[:, 1], s=10, c="k")
ax.set_aspect("equal")
plt.show()
```

## Next steps

- [TDA Workflow](tda_workflow.md) — build a persistence diagram from the tangency distance matrix.
- [Differentiable Tangency](differentiable.md) — gradient-based optimisation with `ellphi.grad`.
- [API Reference](../reference/index.md) — full module documentation.
