"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from ellphi import has_cpp_backend


_BACKEND_PARAMS = [
    "python",
    pytest.param(
        "cpp",
        marks=pytest.mark.skipif(
            not has_cpp_backend(), reason="C++ backend not available"
        ),
    ),
]


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a dedicated, reproducible random generator per test."""
    return np.random.default_rng(12345)


@pytest.fixture(params=_BACKEND_PARAMS)
def solver_backend(request: pytest.FixtureRequest) -> str:
    """Backends supported by tangency-related tests."""

    return request.param
