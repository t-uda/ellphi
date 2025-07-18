
import numpy as np
import pytest

from ellphi import Ellipse, tangency

# -----------------------------------------------------------------------------
# 1. Unit‑circle tangency (simple, deterministic)
# -----------------------------------------------------------------------------

def test_tangent_unit_circles():
    a = Ellipse(0, 0, 1, 1, 0)
    b = Ellipse(2, 0, 1, 1, 0)
    res = tangency(a, b)
    assert not np.isnan(res.mu), "mu should be finite"
    assert res.mu == pytest.approx(0.5)
    assert res.point.tolist() == pytest.approx([1.0, 0.0])
    assert res.t == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# 2. Symmetry check with generic, non‑degenerate ellipses
#    (avoid parameters that lead to singular centre computation)
# -----------------------------------------------------------------------------

def test_symmetry_generic():
    p = Ellipse(0.3, -0.7, 1.2, 0.9, 0.4)
    q = Ellipse(-1.1, 1.4, 0.8, 1.5, 1.0)

    r1 = tangency(p, q)
    r2 = tangency(q, p)

    # Ensure results are finite
    for r in (r1, r2):
        assert not np.isnan(r.t)
        assert not np.isnan(r.mu)
        assert np.all(np.isfinite(r.point))

    # Distances and points should match; mu should complement
    assert r1.t == pytest.approx(r2.t, rel=1e-6)
    assert r1.point.tolist() == pytest.approx(r2.point.tolist(), rel=1e-6)
    assert r1.mu == pytest.approx(1.0 - r2.mu, rel=1e-6)


# -----------------------------------------------------------------------------
# 3. Error handling: Newton method requires x0
# -----------------------------------------------------------------------------

def test_newton_requires_x0():
    p = Ellipse(0, 0, 1, 1, 0)
    q = Ellipse(1, 0, 1, 1, 0)
    with pytest.raises(ValueError):
        tangency(p, q, method="newton")

