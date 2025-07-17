
import numpy as np
import pytest
from ellphi import epencil  # Import the module containing find_intersect

def test_tangent_circles():
    """Two identical unit circles touching at one point (tangency)."""
    # Circle 1 centered at (0,0), Circle 2 centered at (2,0), both radius = 1
    p = epencil.to_abcdef(0, 0, 1, 1, 0)
    q = epencil.to_abcdef(2, 0, 1, 1, 0)
    t, xc, mu = epencil.find_intersect(p, q)
    # The circles touch at (1, 0)
    assert mu == pytest.approx(0.5, rel=1e-7), "Symmetric circles should give mu=0.5"
    assert isinstance(xc, np.ndarray) and xc.shape == (2,), "xc should be a length-2 numpy array"
    assert xc.tolist() == pytest.approx([1.0, 0.0], rel=1e-7), "Tangency point should be (1.0, 0.0)"
    # Distance measure t should equal the radius (1.0) for external tangency
    assert isinstance(t, (float, np.floating))
    assert t == pytest.approx(1.0, rel=1e-7)
    # At the tangency point, the two ellipse equations should evaluate almost equal (difference ~ 0)
    diff = epencil.epoly(*p, *xc) - epencil.epoly(*q, *xc)
    assert abs(diff) < 1e-8, "Ellipses should have equal polynomial value at tangency point"

def test_separated_circles():
    """Two identical circles with a gap (no actual intersection)."""
    # Circle 1 at (0,0), Circle 2 at (2.5, 0) -> centers distance 2.5 (radii 1, so 0.5 gap)
    p = epencil.to_abcdef(0, 0, 1, 1, 0)
    q = epencil.to_abcdef(2.5, 0, 1, 1, 0)
    t, xc, mu = epencil.find_intersect(p, q)
    # Expect the virtual tangency at midpoint (1.25, 0) with mu=0.5
    assert mu == pytest.approx(0.5, rel=1e-7), "Symmetric circles should give mu=0.5"
    assert xc.tolist() == pytest.approx([1.25, 0.0], rel=1e-7), "Intersection point should be midpoint (1.25, 0.0)"
    # t should be half the center distance (2.5/2 = 1.25) in this symmetric case
    expected_t = 0.5 * np.linalg.norm(np.array([2.5, 0]) - np.array([0, 0]))
    assert t == pytest.approx(expected_t, rel=1e-7), "Distance measure t should be half the gap (1.25)"
    # Ellipse polynomial values at the midpoint should be equal
    diff = epencil.epoly(*p, *xc) - epencil.epoly(*q, *xc)
    assert abs(diff) < 1e-8

def test_symmetry_and_consistency():
    """Swapping p and q yields complementary mu and same tangency point."""
    # Define two distinct ellipses via parameters
    p_params = [1, 1, 2, 3, np.pi/8]       # ellipse centered at (1,1)...
    q_params = [-2, -1, 1, 5, 2*np.pi/3]   # another ellipse
    p = epencil.to_abcdef(*p_params)
    q = epencil.to_abcdef(*q_params)
    # Solve using default method
    t1, xc1, mu1 = epencil.find_intersect(p, q)
    # Solve with roles swapped
    t2, xc2, mu2 = epencil.find_intersect(q, p)
    # Tangency distance and point should be identical
    assert t1 == pytest.approx(t2, rel=1e-7)
    assert xc1.tolist() == pytest.approx(xc2.tolist(), rel=1e-7)
    # The mu values should sum to ~1 (complementary) for swapped inputs
    assert mu1 == pytest.approx(1.0 - mu2, rel=1e-7)
    # The solution should make the polynomial values equal at xc1
    diff = epencil.epoly(*p, *xc1) - epencil.epoly(*q, *xc1)
    assert abs(diff) < 1e-8

def test_invalid_method_raises():
    """Invalid solver method or missing x0 for Newton should raise ValueError."""
    p = epencil.to_abcdef(0, 0, 1, 1, 0)
    q = epencil.to_abcdef(1, 0, 1, 1, 0)
    # Unsupported method name
    with pytest.raises(ValueError):
        epencil.find_intersect_mu(p, q, method="undefined_method")
    # Newton method without initial guess
    with pytest.raises(ValueError):
        epencil.find_intersect_mu(p, q, method="newton")

