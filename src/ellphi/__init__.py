"""A Python library for ellipse-based geometric analysis.

This package provides a comprehensive set of tools for working with ellipses,
including geometric calculations, ellipse cloud generation, and tangency
analysis. It is designed to be both efficient and user-friendly, with a
high-level API that abstracts away the complexities of the underlying
mathematics.

The library is organized into the following subpackages:

- `ellphi.geometry`: Core geometric functions for working with ellipses.
- `ellphi.ellcloud`: Tools for creating and manipulating ellipse clouds.
- `ellphi.solver`: Tangency solvers for finding the point of contact between
    two ellipses.
- `ellphi.visualization`: Helper functions for visualizing ellipses and
    ellipse clouds.
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
