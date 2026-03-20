# ellphi.geometry

Utilities for encoding and decoding ellipsoid geometry as packed conic coefficient vectors.

A *d*-dimensional ellipsoid centred at $\mathbf{x}_0$ with covariance $\Sigma$ is represented by the quadratic form

$$
E = \{ \mathbf{x} : (\mathbf{x} - \mathbf{x}_0)^\top \Sigma^{-1} (\mathbf{x} - \mathbf{x}_0) \le 1 \}
$$

The coefficient vector packs the upper-triangular entries of $A = \Sigma^{-1}$, followed by the linear term $b = -2A\mathbf{x}_0$ and the scalar $c = \mathbf{x}_0^\top A \mathbf{x}_0 - 1$, into a flat array of length $m = (d+1)(d+2)/2$.

::: ellphi.geometry
