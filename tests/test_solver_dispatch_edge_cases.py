import numpy as np
import pytest

import ellphi.solver as solver_mod
from ellphi.geometry import coef_from_axes

from .factories import random_cloud


def test_extract_coef_array_requires_2d():
    with pytest.raises(ValueError, match="shape"):
        solver_mod._extract_coef_array(np.zeros(6))


def test_should_use_cpp_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        solver_mod._should_use_cpp("unknown")


def test_should_use_cpp_python_returns_false():
    assert solver_mod._should_use_cpp("python") is False


def test_should_use_cpp_requires_available_backend(monkeypatch):
    monkeypatch.setattr(solver_mod._cpp, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="C\\+\\+ backend requested"):
        solver_mod._should_use_cpp("cpp")


def test_normalize_method_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown method"):
        solver_mod._normalize_method("not-a-method")


def test_tangency_rejects_mismatched_coefficients():
    p = np.zeros(6)
    q = np.zeros(7)
    with pytest.raises(ValueError, match="same length"):
        solver_mod.tangency(p, q, backend="python")


def test_tangency_rejects_unknown_backend():
    p = coef_from_axes([0.0, 0.0], 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="Unknown backend"):
        solver_mod.tangency(p, p, backend="unknown")


def test_pdist_tangency_rejects_unknown_backend(rng):
    cloud = random_cloud(rng, n_ellipses=2)
    with pytest.raises(ValueError, match="Unknown backend"):
        solver_mod.pdist_tangency(cloud, backend="unknown")
