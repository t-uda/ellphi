"""
ellphi top-level package initialisation
--------------------------------------

Re-exports the most frequently used symbols so users can::

    import ellphi as el
    el.tangency(...)
"""

import numpy as np
from numpy.typing import NDArray

from ._version import __version__

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


FloatArray = NDArray[np.float64]


__all__ = [
    "FloatArray",
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
    "__version__",
    "version_info",
]


def version_info() -> str:
    """Return the current :mod:`ellphi` version string."""

    return __version__


def _main() -> None:
    """A minimal CLI for printing the current version."""

    import argparse

    parser = argparse.ArgumentParser(description="ellphi command-line interface")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed ellphi version and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(version_info())
    else:
        parser.print_help()
