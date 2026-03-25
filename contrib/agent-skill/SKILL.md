---
name: ellphi-api
description: >
  EllPHi library usage guide for anisotropic persistent homology.
  Use when: (1) writing code that imports ellphi, (2) working with
  anisotropic persistent homology or ellipsoid tangency distances,
  (3) debugging ellphi solver issues or understanding coefficient format.
---

# EllPHi API Skill

## Version check

```bash
python -c "import ellphi; print(ellphi.__version__)"
```

Require >= 0.1.2 for the `grad` module. If older: `pip install -U ellphi`.

## Core concepts

### Ellipsoid coefficient format

Each ellipsoid is a quadratic `Q(x) = x^T A x + 2 b^T x + c`, packed into a flat vector of length `m = (d+1)(d+2)/2` (m=6 for d=2, m=10 for d=3). Packing order: upper-triangular A, then b (length d), then c (scalar).

**Critical: b = -A x_bar** (negative sign). To recover center: `x_bar = -A^{-1} b`. Use `unpack_conic` to extract (A, b, c).

### Tangency distance

The inflation factor `t` at which two ellipsoids become externally tangent. The solver finds pencil parameter `mu` in [0,1] minimizing the discriminant, then `t = sqrt(alpha)`.

## Quick recipe: point cloud to PH

```python
import ellphi

cloud = ellphi.ellipse_cloud(X, k=5)   # X: (N, d) point cloud
dists = ellphi.pdist_tangency(cloud)    # condensed distance matrix (like scipy pdist)

# Feed to ripser, GUDHI, etc.
```

## Quick recipe: gradient optimization

```python
from ellphi.grad import pdist_tangency_grad

dists, vjp = pdist_tangency_grad(cloud.coef)  # cloud.coef: (N, m)
grad_coefs = vjp(upstream_grad)               # upstream: (N*(N-1)//2,) -> (N, m)
```

### End-to-end: centers + covariances → loss

```python
from ellphi.grad import coef_from_cov_grad, pdist_tangency_grad

# Forward + VJP for coef_from_cov
coefs, vjp_coef = coef_from_cov_grad(centers, covs)

# Forward + VJP for tangency distances
dists, vjp_dist = pdist_tangency_grad(coefs)

# Backward: chain the two VJPs
loss = dists.sum()
grad_coefs = vjp_dist(np.ones_like(dists))    # (N, m)
grad_centers, grad_covs = vjp_coef(grad_coefs) # (N,d), (N,d,d)
```

## Solver methods

Pass `method=` to `tangency()`:

| Method | Notes |
|--------|-------|
| `"brentq+newton"` | **(default)** Brentq bracketing + Newton polish |
| `"brentq"` | Pure Brentq. Robust, slower |
| `"bisect"` | Most robust, slowest |
| `"newton"` | Requires `x0`. Fast but can diverge |

## Backends

`backend=` in `tangency()` / `pdist_tangency()`: `"auto"` (default, C++ if available), `"python"`, `"cpp"`.

## Common pitfalls

1. **b sign**: `b = -A x_bar`. Forgetting the negative sign is the #1 bug.
2. **Condensed distances**: `pdist_tangency` returns 1-D like `scipy.spatial.distance.pdist`. Use `squareform` for a full matrix.
3. **Degenerate inputs**: `tangency_grad` raises `ZeroDivisionError` for identical/concentric ellipsoids.
4. **Singular covariance**: `coef_from_cov` returns NaN for singular matrices.

## Envelope theorem (grad module)

See [references/grad_module.md](references/grad_module.md) for how the envelope theorem enables efficient exact gradients.

## API details

For full function signatures and parameters:

```bash
python -c "import ellphi; help(ellphi.tangency)"
python -c "from ellphi.grad import tangency_grad; help(tangency_grad)"
```

- Current exports: `src/ellphi/__init__.py`
- Full reference: the project's MkDocs site (`mkdocs serve` or published docs)
