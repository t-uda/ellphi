# API Reference

EllPHi's public API is organised into the following modules:

| Module | Description |
|--------|-------------|
| [`ellphi.geometry`](geometry.md) | Conic coefficient encoding/decoding and covariance-to-ellipse conversion. |
| [`ellphi.ellcloud`](ellcloud.md) | `EllipseCloud` container and the `ellipse_cloud` factory function. |
| [`ellphi.solver`](solver.md) | Tangency solver — `tangency` (single pair) and `pdist_tangency` (pairwise). |
| [`ellphi.grad`](grad.md) | Differentiable solver — gradients and VJP for gradient-based optimisation. |
| [`ellphi.visualization`](visualization.md) | Matplotlib helpers for visualising ellipse clouds. |

The top-level `ellphi` namespace re-exports the most commonly used symbols:

```python
from ellphi import (
    ellipse_cloud,       # factory: point cloud → EllipseCloud
    EllipseCloud,
    tangency,            # single-pair tangency distance
    pdist_tangency,      # pairwise condensed distance matrix
    TangencyResult,      # NamedTuple: t, point, mu
    build_info,          # solver backend details
)
```
