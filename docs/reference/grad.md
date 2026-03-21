# ellphi.grad

Differentiable tangency distances for gradient-based optimisation and persistent-homology backprop.

Gradients are derived via the **envelope theorem**: the derivative through the optimal tangent point $\bm{x}^*$ vanishes, so only the explicit dependence of the pencil on the input coefficients survives.  This makes the gradient cheap — no second solver pass is required.

```python
from ellphi.grad import tangency_grad, pdist_tangency_grad

# Single pair
g = tangency_grad(p, q)
# g.t, g.dt_dp, g.dt_dq

# Batch pairwise with VJP
dists, vjp = pdist_tangency_grad(coefs)
grad_coefs = vjp(upstream_grad)   # ∂loss/∂coefs
```

See the [Differentiable Tangency guide](../guide/differentiable.md) for full examples.

::: ellphi.grad
