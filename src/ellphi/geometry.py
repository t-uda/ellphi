
from collections import namedtuple
from typing import Union, Sequence

import numpy

__all__ = [
    "Ellipse",
    "ellipse_coeffs",
    "coeff_matrix",
    "_to_array",
]

# ---------------------------------------------------------------------------
# Lightweight parameter container
# ---------------------------------------------------------------------------

Ellipse = namedtuple("Ellipse", ["x", "y", "r1", "r2", "theta"])


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def ellipse_coeffs(e: Ellipse) -> numpy.ndarray:
    """Convert an :class:`Ellipse` to six conic coefficients `[a,b,c,d,e,f]`.

    The coefficients are normalised so that the polynomial equals 1 on the
    ellipse boundary.
    """
    x0, y0, r1, r2, th = e
    c, s = numpy.cos(th), numpy.sin(th)
    a = (s ** 2) / (r2 ** 2) + (c ** 2) / (r1 ** 2)
    b = (-s * c) / (r2 ** 2) + (s * c) / (r1 ** 2)
    c2 = (c ** 2) / (r2 ** 2) + (s ** 2) / (r1 ** 2)
    d = (-x0 * s ** 2 + y0 * s * c) / (r2 ** 2) - (x0 * c ** 2 + y0 * s * c) / (r1 ** 2)
    ecoef = (x0 * s * c - y0 * c ** 2) / (r2 ** 2) - (x0 * s * c + y0 * s ** 2) / (r1 ** 2)
    f = (
        (x0 ** 2 * s ** 2 - 2 * x0 * y0 * s * c + y0 ** 2 * c ** 2) / (r2 ** 2)
        + (x0 ** 2 * c ** 2 + 2 * x0 * y0 * s * c + y0 ** 2 * s ** 2) / (r1 ** 2)
    )
    return numpy.array([a, b, c2, d, ecoef, f], dtype=float)


def _to_array(obj: Union[Ellipse, Sequence[float], numpy.ndarray]) -> numpy.ndarray:
    """Return *obj* as a numpy float64 1‑D array (copy if needed)."""
    if isinstance(obj, Ellipse):
        return ellipse_coeffs(obj)
    return numpy.asarray(obj, dtype=float)


# ---------------------------------------------------------------------------
# Vectorised batch conversion
# ---------------------------------------------------------------------------

def coeff_matrix(params: numpy.ndarray) -> numpy.ndarray:
    """Convert ``(N,5)`` param rows to ``(N,6)`` coefficient rows."""
    x, y, r1, r2, th = params.T
    c, s = numpy.cos(th), numpy.sin(th)
    a = (s ** 2) / r2 ** 2 + (c ** 2) / r1 ** 2
    b = (-s * c) / r2 ** 2 + (s * c) / r1 ** 2
    c2 = (c ** 2) / r2 ** 2 + (s ** 2) / r1 ** 2
    d = (-x * s ** 2 + y * s * c) / r2 ** 2 - (x * c ** 2 + y * s * c) / r1 ** 2
    ecoef = (x * s * c - y * c ** 2) / r2 ** 2 - (x * s * c + y * s ** 2) / r1 ** 2
    f = (
        (x ** 2 * s ** 2 - 2 * x * y * s * c + y ** 2 * c ** 2) / r2 ** 2
        + (x ** 2 * c ** 2 + 2 * x * y * s * c + y ** 2 * s ** 2) / r1 ** 2
    )
    return numpy.stack([a, b, c2, d, ecoef, f], axis=1)

