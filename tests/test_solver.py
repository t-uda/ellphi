import numpy as np
import numpy as np
import pytest

from ellphi.solver import pdist_tangency

from .factories import random_cloud


@pytest.fixture
def ellipse_cloud(rng):
    """Return a reproducible ellipse cloud for tangency checks."""
    # 32 ellipses keep the workload representative while staying quick.
    return random_cloud(rng, n_ellipses=32)


def test_pdist_tangency_consistency(ellipse_cloud, solver_backend):
    """Serial and parallel ``pdist_tangency`` implementations agree."""
    serial_result = pdist_tangency(
        ellipse_cloud, parallel=False, backend=solver_backend
    )
    parallel_result = pdist_tangency(
        ellipse_cloud, parallel=True, backend=solver_backend
    )

    np.testing.assert_allclose(
        serial_result,
        parallel_result,
        err_msg="Serial and parallel results are not close enough.",
    )


def test_pdist_tangency_three_dimensional_python_backend(rng):
    cloud = random_cloud(rng, n_ellipses=8, dim=3)
    result = pdist_tangency(cloud, parallel=False, backend="python")
    expected_length = cloud.n * (cloud.n - 1) // 2
    assert result.shape == (expected_length,)
    assert np.all(result >= 0.0)
