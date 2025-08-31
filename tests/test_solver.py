import numpy as np

from ellphi.ellcloud import EllipseCloud
from ellphi.solver import pdist_tangency
from ellphi.geometry import coef_from_cov


def generate_ellipses(n_ellipses, seed=42):
    np.random.seed(seed)
    means = np.random.rand(n_ellipses, 2) * 100
    covs_list = []
    for _ in range(n_ellipses):
        a = np.random.rand() * 5 + 1
        b = np.random.rand() * 5 + 1
        angle = np.random.rand() * np.pi
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        cov = rot @ np.diag([a, b]) @ rot.T
        covs_list.append(cov)
    covs = np.array(covs_list)
    coefs = coef_from_cov(means, covs)
    # Create dummy nbd and k, as they are not used in pdist_tangency
    dummy_nbd = np.array([[] for _ in range(n_ellipses)])
    return EllipseCloud(coef=coefs, mean=means, cov=covs, k=0, nbd=dummy_nbd)


def test_pdist_tangency_consistency():
    """
    Test that serial and parallel pdist_tangency implementations give the same result.
    """
    n_ellipses = 50  # A reasonable number to test parallelism
    ellipses = generate_ellipses(n_ellipses)

    serial_result = pdist_tangency(ellipses, parallel=False)
    parallel_result = pdist_tangency(ellipses, parallel=True)

    np.testing.assert_allclose(
        serial_result,
        parallel_result,
        err_msg="Serial and parallel results are not close enough.",
    )
