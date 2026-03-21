# ellphi.geometry

Utilities for encoding and decoding ellipsoid geometry as packed conic coefficient vectors.

A *d*-dimensional ellipsoid centred at $\bm{x}_0$ with covariance $\Sigma$ is represented by the quadratic form

$$
E = \{ \bm{x} : (\bm{x} - \bm{x}_0)^\top \Sigma^{-1} (\bm{x} - \bm{x}_0) \le 1 \}
$$

The coefficient vector packs the upper-triangular entries of $A = \Sigma^{-1}$, followed by the linear term $\bm{b} = -A\bm{x}_0$ and the scalar $c = \bm{x}_0^\top A \bm{x}_0$, into a flat array of length $m = (d+1)(d+2)/2$.  The internal evaluator applies a factor of 2 to $\bm{b}$, so the effective evaluation is $(\bm{x} - \bm{x}_0)^\top A (\bm{x} - \bm{x}_0)$, which equals 1 on the ellipsoid boundary and 0 at its center.

::: ellphi.geometry
