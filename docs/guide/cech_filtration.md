# Čech filtration

The public `ellphi.cech` API computes the anisotropic Čech filtration value for
a simplex of packed ellipsoid coefficient vectors. For
`f_i(x)` describing the growing ellipsoids
`E_i(t) = {f_i <= t^2}`, `cech(coefs)` returns the Čech filtration time
`t`, where

`t^2 = alpha(sigma) = min_x max_i f_i(x)`

is the least squared scale at which all bodies share a common point.

The result fields are ordered as
`(t, point, mu, support, active_set)`. The gradient result fields are
ordered as `(t, point, mu, dt_dcoef, support, active_set)`.

- `t`: the Čech filtration time, with `t^2 = alpha(sigma)`.
- `point`: the common intersection point `x*` at the critical scale.
- `mu`: the Lagrange multipliers, or dual weights on the simplex.
- `support`: the thresholded weight support
  `{i : mu_i > weight_tol}`.
- `active_set`: the tight-constraint set at the returned point,
  `I = {i : t^2 - f_i(point) <= active_tol * max(1, t^2)}`.

The public `cech` and `cech_grad` signatures intentionally do not expose
matrix stabilization controls such as `regularization`,
`condition_number_limit`, or `max_conditioning_steps`. Stabilized solves are
available only through the private `ellphi._minimax_python` engine for
research use, and the public coefficient-space gradient carries no guarantee
for those stabilized solves.

At the critical scale the intersection is the single point `x*`. The active
ellipsoid boundaries pass through `x*` and satisfy
`sum_i mu_i grad f_i(x*) = 0`, so their normals are positively dependent;
the boundaries are not tangent to each other. Pairwise tangency is only the
`k = 2` special case: for `k = 2` the Čech time equals the tangency time
computed by `tangency()`, and `cech()` agrees with it.

Up to numerical tolerance, `support` is a subset of `active_set`; under
strict complementarity (ND1), they coincide. For three unit balls centred at
`(0, 0)`, `(2, 0)`, and `(0, 2)`, the point is `(1, 1)`, support has
two indices, and `active_set` has three. A solver that does not converge or
produces non-finite output raises `RuntimeError` with its diagnostics.

The default solver uses the away-step Frank-Wolfe (AFW) method of
Lacoste-Julien and Jaggi (2015) on the dual simplex with an adaptive Brent
line search. At each iteration it compares the ordinary FW direction with an
away direction from the least useful positive-weight vertex; a capped away
step removes that vertex exactly. AFW can still require more than the default
iteration budget before identifying a minimal face on rank-deficient
many-body problems. An exhausted AFW phase therefore starts SLSQP from its
current weights. Newton-bearing method variants subsequently polish the
thresholded fallback face. Every accepted result is checked against the
all-index stabilised FW gap.

The private engine's internal field named `active_set` is the weight support
(historical ellcech naming) and is not part of the public API.

If `unpack_conic` returns `(A_i, b_i, c_i)`, the center is
`xbar_i = -A_i^{-1} b_i` and the completed-square constant is
`delta_i = c_i - b_i^T A_i^{-1} b_i`. The public API passes this exactly
computed `delta_i` to the many-body engine, including when it is small, so
general independent constants are supported.

A packed row encodes its constant as
`c_i = xbar_i^T A_i xbar_i + delta_i`. For `|xbar_i|^2 >> 1` or
ill-conditioned `A_i`, the representable `delta_i` carries roundoff of order
`eps * |xbar_i|^2`, with a `cond2(A_i)` factor for ill-conditioned matrices.
Callers needing exact normalization at large translations should center their
data first. Packed input requires `d >= 2`; d=1 packed conics are not
supported and raise `ValueError`.

The many-body engine uses the completed-square offsets above. Thus
`cech_grad(coefs)` returns `dt_dcoef`, the gradient for arbitrary
independent packed-coordinate perturbations, including the constant component.
For the upper-triangular packed basis
`[x_i**2, 2*x_i*x_j (i < j), 2*x_i, 1]`, the formula is
`mu_i * basis(point) / (2*t)`. It assumes a non-degenerate input, at least
two indices in `active_set`, multipliers supported on `active_set`, and an
exact weighted linear solve. Convergence or a small residual alone is not a
derivative certificate, and the gradient is undefined at `t == 0`.
For far-translated inputs, the uncentred packed basis used by this derivative
can be poorly conditioned.
