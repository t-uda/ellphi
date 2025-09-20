"""
ellphi top-level package initialisation
--------------------------------------

Re-exports the most frequently used symbols so users can::

    import ellphi as el
    el.tangency(...)
"""

from importlib.metadata import PackageNotFoundError, version as _version

# geometry
from .geometry import (
    unit_vector,
    axes_from_cov,
    coef_from_axes,
    coef_from_cov,
)

from .ellcloud import ellipse_cloud, EllipseCloud, LocalCov

# solver
from .solver import (
    quad_eval,
    pencil,
    tangency,
    pdist_tangency,
    TangencyResult,
    has_cpp_backend,
)

__all__ = [
    # geometry
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_cov",
    # ellcloud
    "ellipse_cloud",
    "EllipseCloud",
    "LocalCov",
    # solver
    "quad_eval",
    "pencil",
    "tangency",
    "pdist_tangency",
    "TangencyResult",
    "has_cpp_backend",
]

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    __version__ = "0+unknown"
