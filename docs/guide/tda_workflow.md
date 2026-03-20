# TDA Workflow

This page shows how to go from a raw point cloud to a persistence diagram using elliptic tangency distances.

## Overview

1. Sample a point cloud $X \subset \mathbb{R}^d$.
2. Fit an ellipse cloud with `ellipse_cloud`.
3. Compute the condensed pairwise tangency distance matrix with `pdist_tangency`.
4. Feed the matrix to a persistence homology library (e.g. [HomCloud](https://homcloud.dev/) or [Ripser](https://github.com/scikit-tda/ripser.py)).

## Step-by-step example

```python
import numpy as np
from ellphi import ellipse_cloud

# 1. Point cloud — six interlocking rings
rng = np.random.default_rng(0)
angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
rings = [
    np.column_stack([np.cos(angles) + dx, np.sin(angles) + dy])
    for dx, dy in [(0, 0), (1.5, 0), (3, 0), (0.75, 1.3), (2.25, 1.3), (1.5, 2.6)]
]
X = np.vstack(rings) + rng.normal(scale=0.05, size=(180, 2))

# 2. Ellipse cloud
cloud = ellipse_cloud(X, k=8)

# 3. Pairwise tangency distances
dists = cloud.pdist_tangency()
```

### Persistence diagram with HomCloud

```python
import homcloud.interface as hc

pd = hc.PDList.from_rips_filtration(dists, maxdim=1, metric="precomputed")
pd.histogram(1).plot()          # H₁ barcode
```

### Persistence diagram with Ripser

```python
from scipy.spatial.distance import squareform
from ripser import ripser
from persim import plot_diagrams

D = squareform(dists)
result = ripser(D, metric="precomputed", maxdim=1)
plot_diagrams(result["dgms"])
```

## Notes on distance semantics

The tangency distance $t(E_i, E_j) \in [0, 1]$ measures how far two ellipses must be
uniformly inflated before they first touch.  Values close to 0 mean near-tangency;
values close to 1 indicate that the union of the two ellipses fills the entire pencil.
This is analogous to a Čech filtration radius but adapted to the local geometry of the
data.

## Choosing *k*

Larger *k* produces fatter, more overlapping ellipses and tends to smooth out local
noise.  A good starting range is $k \in [5, 15]$ for 2D data; for higher-dimensional
data increase *k* proportionally.  Use `cloud.rescale(method="median")` to normalise
the scale of the ellipses before computing distances.
