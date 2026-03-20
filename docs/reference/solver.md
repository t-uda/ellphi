# ellphi.solver

The solver module dispatches tangency computations to the best available backend:

- **C++ / Eigen** — compiled extension, used automatically when available (`backend="auto"`).
- **Pure Python** — SciPy-based fallback, always available.

The default method `"brentq+newton"` uses a short bracketing phase followed by Newton refinement, which is significantly faster than pure Brent's method for high-dimensional inputs.

```python
from ellphi import tangency, pdist_tangency, build_info

print(build_info())         # shows cpp_backend_available, cpp_linalg_kind, etc.

result = tangency(p, q)
print(result.t, result.point, result.mu)

dists = pdist_tangency(cloud)   # condensed array
```

::: ellphi.solver
