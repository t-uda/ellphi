"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a dedicated, reproducible random generator per test."""
    return np.random.default_rng(12345)
