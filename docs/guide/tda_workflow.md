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

For generic (nondegenerate) inputs, the solver finds $\mu^* \in (0, 1)$ such that the
center $x_c$ of the pencil conic $(1-\mu^*)p + \mu^* q$ satisfies $p(x_c) = q(x_c)$.
The tangency distance is

$$t(E_p, E_q) = \sqrt{p(x_c)} = \sqrt{q(x_c)}.$$

Because $x_c$ is the center of the pencil element, $\nabla F_{\mu^*}(x_c) = 0$, which
gives

$$
(1 - \mu^*)\,\nabla p(x_c) + \mu^*\,\nabla q(x_c) = 0.
$$

For $t > 0$, the normals to the level sets $\{p = t^2\}$ and $\{q = t^2\}$ are
anti-parallel at $x_c$: these level sets are the original ellipsoids uniformly scaled
by $t$ from their respective centers, and they are externally tangent at $x_c$.

Equivalently, $t$ is the smallest scale at which the inflated sublevel sets meet:

$$
t = \inf\bigl\{s \ge 0 : E_p(s) \cap E_q(s) \neq \varnothing\bigr\},
\qquad E_p(s) = \{x : p(x) \le s^2\},
$$

or $t^2 = \min_x \max\{p(x),\, q(x)\}$.  This characterisation yields a coarse
trichotomy for the reference ellipsoids $E_p = \{p \le 1\}$,
$E_q = \{q \le 1\}$:

- $0 \le t < 1$: the unit ellipsoids have non-empty intersection (overlap,
  containment, or internal tangency).
- $t = 1$: $E_p$ and $E_q$ are externally tangent.
- $t > 1$: $E_p$ and $E_q$ are disjoint; both must be scaled by $t$ to touch.

Degenerate cases (identical or concentric ellipsoids) yield $t = 0$; the level sets
then collapse to a point rather than forming two tangent ellipsoids.

$t$ is non-negative with no upper bound.

## Choosing *k*

Larger *k* produces fatter, more overlapping ellipses and tends to smooth out local
noise.  A good starting range is $k \in [5, 15]$ for 2D data; for higher-dimensional
data increase *k* proportionally.  Use `cloud.rescale(method="median")` to normalise
the scale of the ellipses before computing distances.
