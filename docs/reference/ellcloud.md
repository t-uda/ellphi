# ellphi.ellcloud

The `EllipseCloud` class is the central data structure of EllPHi.  It holds an array of ellipsoid coefficient vectors together with the original means and covariances, and provides convenience methods for computing tangency distances and visualisation.

The `ellipse_cloud` factory function (an alias for `EllipseCloud.from_point_cloud`) is the recommended entry point:

```python
from ellphi import ellipse_cloud
cloud = ellipse_cloud(X, k=5)
```

When centres and covariance matrices are already available, prefer the direct
constructor:

```python
cloud = EllipseCloud.from_cov(centers, covs)
```

::: ellphi.ellcloud
