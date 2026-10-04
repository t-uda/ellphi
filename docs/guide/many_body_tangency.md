# Many-body tangency

The provisional `ellphi.simplex` API extends pairwise tangency to a simplex of
packed ellipsoid coefficient vectors. `tangency_simplex(coefs)` returns the
tangency time `t`, point, simplex multipliers `mu`, multiplier `support`, and
numerically determined `active_set`. Unlike the internal minimax engine, the
public API reports `t`, not the squared value `t**2`.

The public result fields are ordered as `(t, point, mu, support, active_set)`.
The gradient result fields are ordered as
`(t, point, mu, dt_dcoef, support, active_set)`.

`support` is the thresholded weight support
`{i : mu_i > weight_tol}`. `active_set` is the tight-constraint set at the
returned point,
`I = {i : t**2 - f_i(point) <= active_tol * max(1, t**2)}`. Up to numerical
tolerance, `support` is a subset of `active_set`; under strict complementarity
(ND1), they coincide. For three unit balls centred at `(0, 0)`, `(2, 0)`, and
`(0, 2)`, the point is `(1, 1)`, support has two indices, and `active_set`
has three. A solver that does not converge or produces non-finite output
raises `RuntimeError` with its diagnostics.

The private engine's internal field named `active_set` is the weight support
(historical ellcech naming) and is not part of the public API.

Every coefficient row must be a normalized conic. If `unpack_conic` returns
`(A_i, b_i, c_i)`, the required convention is
`b_i = -A_i xbar_i` and `c_i = xbar_i^T A_i xbar_i`, equivalently
`c_i = b_i^T A_i^{-1} b_i` within the maximum of
`norm_tol * max(1, abs(c_i))` and
`64 * eps * cond2(A_i) * max(1, abs(c_i), abs(b_i^T A_i^{-1} b_i))`,
where `eps` is machine epsilon and `norm_tol` defaults to `1e-9`.
`tangency_simplex` and `tangency_simplex_grad` reject an offending row with
`ValueError`; `ellphi.coef_from_cov` constructs coefficients in this form.
Packed input requires `d >= 2`, as for `ellphi.tangency`; d=1 packed conics
are not supported and raise `ValueError`.

The private engine reconstructs each center from `(A_i, b_i)` and ignores
`c_i` during optimization. Thus `tangency_simplex_grad(coefs)` returns
`dt_dcoef`, a coefficient-space covector whose contraction with a packed
direction is the derivative of `t` only for normalization-preserving
perturbations. For the upper-triangular packed basis
`[x_i**2, 2*x_i*x_j (i < j), 2*x_i, 1]`, the formula is
`mu_i * basis(point) / (2*t)`. The constant-term component must be read in
this chain-rule sense, not as a derivative under an arbitrary independent
change to a packed constant. It assumes a non-degenerate input, at least two
indices in `active_set`, multipliers supported on `active_set`, and an exact
weighted linear solve. Convergence or a small residual alone is not a
derivative certificate, and the gradient is undefined at `t == 0`.
For far-translated inputs, the uncentred packed basis used by this derivative
can be poorly conditioned.

Support for general independent constants inside the minimax solve is planned
engine work; it is not part of this wrapper contract.
