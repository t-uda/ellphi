"""
ellphi top-level package initialisation
--------------------------------------

Re-exports the most frequently used symbols so users can::

    import ellphi as el
    el.tangency(...)
"""

from importlib.metadata import version as _version

# geometry
from .geometry import (
    unit_vector,
    axes_from_cov,
    coef_from_axes,
    coef_from_array,
    coef_from_cov,
)

# solver
from .solver import (
    quad_eval,
    pencil,
    tangency,
    pairwise_tangency,
    TangencyResult,
)

__all__ = [
    # geometry
    "unit_vector",
    "axes_from_cov",
    "coef_from_axes",
    "coef_from_array",
    "coef_from_cov",
    # solver
    "quad_eval",
    "pencil",
    "tangency",
    "pairwise_tangency",
    "TangencyResult",
]

__version__ = _version(__name__)

