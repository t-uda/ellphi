# ellphi.visualization

Matplotlib helpers for rendering ellipse clouds.  For most use cases, `EllipseCloud.plot()` is the recommended entry point — it calls `ellipse_patch` internally.

```python
import matplotlib.pyplot as plt
from ellphi.visualization import ellipse_patch

fig, ax = plt.subplots()
patch = ellipse_patch(center, r_major, r_minor, theta, edgecolor="teal", alpha=0.5)
ax.add_patch(patch)
ax.set_aspect("equal")
plt.show()
```

::: ellphi.visualization
