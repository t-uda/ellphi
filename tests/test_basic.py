
import numpy as np
import pytest
from ellphi import Ellipse, tangency

# --- 1. 基本動作: 二つの単位円が外接 -----------------------------
def test_tangent_unit_circles():
    a = Ellipse(0, 0, 1, 1, 0)
    b = Ellipse(2, 0, 1, 1, 0)
    res = tangency(a, b)
    assert res.mu == pytest.approx(0.5)
    assert res.point.tolist() == pytest.approx([1.0, 0.0])
    assert res.t == pytest.approx(1.0)

# --- 2. 対称性: p⟺q で mu が補数 -------------------------------
def test_symmetry():
    p = Ellipse(1, 2, 0.8, 1.2, 0.3)
    q = Ellipse(-1, 0, 1.1, 0.9, 1.0)
    r1 = tangency(p, q)
    r2 = tangency(q, p)
    assert r1.t == pytest.approx(r2.t)
    assert r1.point.tolist() == pytest.approx(r2.point.tolist())
    assert r1.mu == pytest.approx(1.0 - r2.mu)

# --- 3. エラー処理: Newton 法に x0 必須 --------------------------
def test_newton_requires_x0():
    p = Ellipse(0, 0, 1, 1, 0)
    q = Ellipse(1, 0, 1, 1, 0)
    with pytest.raises(ValueError):
        tangency(p, q, method="newton")

