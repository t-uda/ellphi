# Differentiable Tangency

The `ellphi.grad` module exposes gradient-enabled versions of the tangency solver.
Gradients are computed analytically via the **envelope theorem**: because the tangency
point $\bm{x}_c$ is optimal at the solution, its implicit derivative with respect to the
parameters drops out, and only the explicit dependence of the pencil on the coefficient
vectors $\bm{p}$ and $\bm{q}$ contributes.

## Single-pair gradient

`tangency_grad` returns a [`TangencyGrad`][ellphi.grad.TangencyGrad] dataclass with the
tangency distance $t$ and the gradients $\partial t/\partial \bm{p}$ and
$\partial t/\partial \bm{q}$:

```python
import numpy as np
from ellphi import ellipse_cloud
from ellphi.grad import tangency_grad

rng = np.random.default_rng(0)
X = rng.standard_normal((10, 2))
cloud = ellipse_cloud(X, k=5)

p, q = cloud[0], cloud[1]
g = tangency_grad(p, q)

print(f"t = {g.t:.4f}")
print(f"∂t/∂p shape: {g.dt_dp.shape}")  # (6,) for 2D
```

## Batch pairwise VJP

`pdist_tangency_grad` computes all pairwise distances and returns a **VJP (pullback)**
function.  This is the building block for optimising a loss that depends on the full
distance matrix:

```python
import numpy as np
from ellphi import ellipse_cloud
from ellphi.grad import pdist_tangency_grad

rng = np.random.default_rng(1)
X = rng.standard_normal((20, 2))
cloud = ellipse_cloud(X, k=5)

dists, vjp = pdist_tangency_grad(cloud.coef)

# Suppose the loss is the sum of all tangency distances.
# Its gradient w.r.t. each distance is 1.
grad_coefs = vjp(np.ones_like(dists))   # shape (20, 6)
```

## Gradient-based optimisation

The VJP integrates naturally into any gradient descent loop:

```python
import numpy as np
from ellphi import ellipse_cloud
from ellphi.grad import pdist_tangency_grad

rng = np.random.default_rng(2)
X = rng.standard_normal((15, 2))
cloud = ellipse_cloud(X, k=5)
coefs = cloud.coef.copy()

lr = 1e-3
for step in range(50):
    dists, vjp = pdist_tangency_grad(coefs)
    loss = dists.sum()
    grad = vjp(np.ones_like(dists))    # ∂loss/∂coefs
    coefs -= lr * grad
    if step % 10 == 0:
        print(f"step {step:3d}  loss={loss:.4f}")
```

## Persistent-homology backprop

When combined with modern TDA libraries, `pdist_tangency_grad` enables differentiating
through a persistence diagram — see `notebooks/topology_optimization.ipynb` for a
worked example.

## Degenerate inputs

`tangency_grad` may raise `ZeroDivisionError` for degenerate configurations (identical
or concentric ellipsoids) where $\partial F/\partial \mu$ vanishes at the solution.
Removing duplicate ellipsoids from the inputs reduces one source of degeneracy, but
distinct concentric or nested ellipsoids can trigger the same error.  In optimisation
loops, either catch the exception and skip the offending pair, or perturb the inputs
slightly to escape the degenerate configuration.
