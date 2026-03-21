# EllPHi Grad Module — Conceptual Background

## Envelope theorem approach

The tangency distance `t` depends on coefficient vectors `p` and `q` through:

1. The pencil parameter `mu*` (optimal blend), itself a function of (p, q).
2. The tangent point `x*(mu*)`, the pencil ellipsoid center at `mu*`.

By the envelope theorem, at the optimum `mu*`, the derivative of the
tangent point `x*(mu*)` w.r.t. `p` drops out of the gradient formula.
The `d_mu/dp` terms remain (computed via implicit differentiation):

```
dt/dp_i = (1/(2t)) * [(1 - mu*) * base_i + scalar * d_mu/dp_i]
dt/dq_i = (1/(2t)) * [mu* * base_i + scalar * d_mu/dq_i]
```

where `base` is the monomial basis at the tangent point, and
`scalar = base . (q - p)`.

The `d_mu/dp` terms come from implicit differentiation of the optimality
condition `F(mu, p, q) = 0` (handled internally by `solve_mu_gradients`
in `ellphi.differentiable_solver`).

## Key design points

- `tangency_grad` / `pdist_tangency_grad` are the public API (in `ellphi.grad`).
- They handle the full chain: mu gradients -> t gradients, including the
  `1/(2t)` factor and monomial basis evaluation.
- `pdist_tangency_grad` returns a VJP (vector-Jacobian product) pullback that
  accumulates pair contributions into per-ellipsoid gradients.
- Degenerate configurations (identical/concentric ellipsoids) raise
  `ZeroDivisionError` because the implicit function theorem breaks down.

For function signatures: `python -c "from ellphi.grad import tangency_grad; help(tangency_grad)"`
