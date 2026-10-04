"""Tests for the minimax gradient with respect to centers and metric matrices.

Adapted from uda-lab/ellcech ``tests/test_minimax_grad.py`` at commit
82d13e3e174f4903cdcd6adcee1f469348361c91.

Validation strategy: finite-difference check with relative error < 1e-5.

The gradient formulas:
    d alpha / d xbar_k = 2 mu_k* A_k (xbar_k - x*)
    d alpha / d A_k    = mu_k* outer(xbar_k - x*, xbar_k - x*)
"""

from __future__ import annotations

import numpy as np
import pytest

from ellphi._minimax_grad_python import GradientResult, compute_gradient
from ellphi._minimax_python import solve_minimax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spd(rng: np.random.Generator, d: int, scale: float = 1.0) -> np.ndarray:
    """Return a random d×d SPD matrix."""
    L = rng.standard_normal((d, d))
    return scale * (L @ L.T) + np.eye(d) * 0.5


def _solve_alpha(matrices: np.ndarray, centers: np.ndarray) -> float:
    """Compute alpha via fw+brentq+newton (recommended method)."""
    res = solve_minimax(matrices, centers, method="fw+brentq+newton")
    return res.alpha


def _fd_grad_xbar(matrices, centers, k, h=1e-5):
    """Finite-difference gradient of alpha w.r.t. centers[k], shape (d,)."""
    d = centers.shape[1]
    grad = np.zeros(d)
    for j in range(d):
        c_plus = centers.copy()
        c_plus[k, j] += h
        c_minus = centers.copy()
        c_minus[k, j] -= h
        alpha_plus = _solve_alpha(matrices, c_plus)
        alpha_minus = _solve_alpha(matrices, c_minus)
        grad[j] = (alpha_plus - alpha_minus) / (2 * h)
    return grad


def _fd_grad_A(matrices, centers, k, h=1e-6):
    """Finite-difference gradient of alpha w.r.t. matrices[k], shape (d,d)."""
    d = centers.shape[1]
    grad = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            m_plus = matrices.copy()
            m_minus = matrices.copy()
            # Symmetric perturbation
            m_plus[k, i, j] += h
            m_plus[k, j, i] += h
            m_minus[k, i, j] -= h
            m_minus[k, j, i] -= h
            # Ensure still SPD (perturbation is small)
            alpha_plus = _solve_alpha(m_plus, centers)
            alpha_minus = _solve_alpha(m_minus, centers)
            grad[i, j] = (alpha_plus - alpha_minus) / (4 * h)  # factor 2 from symmetry
    return grad


def _rel_err(analytical, fd):
    """Relative error between analytical and finite-difference gradients."""
    denom = max(np.linalg.norm(fd), 1e-10)
    return float(np.linalg.norm(analytical - fd) / denom)


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------


class TestGradientResult:
    def test_returns_gradient_result(self):
        rng = np.random.default_rng(0)
        d, k = 2, 3
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))
        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)
        assert isinstance(grad, GradientResult)

    def test_output_shapes(self):
        rng = np.random.default_rng(1)
        d, k = 3, 4
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))
        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)
        assert len(grad.d_xbar) == k
        assert len(grad.d_A) == k
        for i in range(k):
            assert grad.d_xbar[i].shape == (d,)
            assert grad.d_A[i].shape == (d, d)

    def test_passive_vertices_zero(self):
        """Inactive vertices (mu_k=0) should have zero gradient."""
        rng = np.random.default_rng(2)
        d = 2
        # k=2 pairwise: both vertices are active. Test k=4 where some may be inactive.
        k = 4
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))
        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)
        for i in range(k):
            if result.weights[i] < 1e-10:
                np.testing.assert_allclose(grad.d_xbar[i], 0.0, atol=1e-14)
                np.testing.assert_allclose(grad.d_A[i], 0.0, atol=1e-14)

    def test_d_A_symmetric(self):
        """d alpha / d A_k = mu_k outer(diff, diff) is symmetric."""
        rng = np.random.default_rng(3)
        d, k = 3, 3
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))
        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)
        for i in range(k):
            np.testing.assert_allclose(grad.d_A[i], grad.d_A[i].T, atol=1e-15)

    def test_alpha_matches_result(self):
        rng = np.random.default_rng(4)
        d, k = 2, 2
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))
        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)
        assert grad.alpha == pytest.approx(result.alpha, rel=1e-12)

    def test_weights_match_result(self):
        rng = np.random.default_rng(5)
        d, k = 2, 2
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))
        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)
        np.testing.assert_allclose(grad.weights, result.weights, atol=1e-15)


# ---------------------------------------------------------------------------
# Finite-difference validation tests
# ---------------------------------------------------------------------------


class TestFiniteDifferenceValidation:
    """Validate analytical gradients against finite differences.

    Target: relative error < 1e-5 for all cases.
    """

    @pytest.mark.parametrize(
        "k,d,seed",
        [
            (2, 2, 10),
            (2, 3, 11),
            (3, 2, 12),
            (3, 3, 13),
            (4, 2, 14),
        ],
    )
    def test_grad_xbar_fd(self, k, d, seed):
        """d alpha / d xbar_k matches finite differences."""
        rng = np.random.default_rng(seed)
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))

        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)

        for i in range(k):
            if result.weights[i] < 1e-8:
                continue  # skip genuinely inactive
            fd = _fd_grad_xbar(matrices, centers, i)
            err = _rel_err(grad.d_xbar[i], fd)
            assert err < 1e-5, (
                f"k={k},d={d},i={i}: d_xbar rel_err={err:.2e} "
                f"analytical={grad.d_xbar[i]}, fd={fd}"
            )

    @pytest.mark.parametrize(
        "k,d,seed",
        [
            (2, 2, 20),
            (2, 3, 21),
            (3, 2, 22),
        ],
    )
    def test_grad_A_fd(self, k, d, seed):
        """d alpha / d A_k matches finite differences."""
        rng = np.random.default_rng(seed)
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))

        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)

        for i in range(k):
            if result.weights[i] < 1e-8:
                continue  # skip inactive
            fd = _fd_grad_A(matrices, centers, i)
            err = _rel_err(grad.d_A[i], fd)
            assert err < 1e-5, f"k={k},d={d},i={i}: d_A rel_err={err:.2e}"

    def test_pairwise_matches_ellphi_direction(self):
        """For k=2, gradient direction should be consistent with ellphi's formula."""
        rng = np.random.default_rng(30)
        d = 3
        matrices = np.stack([_make_spd(rng, d) for _ in range(2)])
        centers = rng.standard_normal((2, d))

        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        grad = compute_gradient(result, centers, matrices)

        # Verify against FD
        for i in range(2):
            fd = _fd_grad_xbar(matrices, centers, i)
            err = _rel_err(grad.d_xbar[i], fd)
            assert err < 1e-5, f"pairwise xbar[{i}] rel_err={err:.2e}"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_shape_mismatch_raises(self):
        rng = np.random.default_rng(40)
        d, k = 2, 3
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))
        result = solve_minimax(matrices, centers, method="fw+brentq+newton")
        # Wrong: pass centers with k+1 rows but same matrices (k rows) → shape mismatch
        bad_centers = rng.standard_normal((k + 1, d))
        bad_matrices = np.stack([_make_spd(rng, d) for _ in range(k + 1)])
        with pytest.raises(ValueError, match="weights length"):
            compute_gradient(result, bad_centers, bad_matrices)

    def test_single_point_zero_gradient(self):
        """Single-vertex simplex: circumcenter = xbar, weights=[1], grad=0."""
        rng = np.random.default_rng(41)
        d = 2
        A = _make_spd(rng, d)[np.newaxis]
        center = rng.standard_normal((1, d))
        result = solve_minimax(A, center, method="fw+brentq+newton")
        grad = compute_gradient(result, center, A)
        # xstar = xbar, so diff = 0, grad = 0
        np.testing.assert_allclose(grad.d_xbar[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(grad.d_A[0], 0.0, atol=1e-14)

    def test_all_methods_give_same_gradient(self):
        """Gradient should be consistent across solver methods."""
        rng = np.random.default_rng(42)
        d, k = 2, 3
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        centers = rng.standard_normal((k, d))

        methods = ["fw+bisect", "fw+brentq", "fw+bisect+newton", "fw+brentq+newton"]
        grads = []
        for method in methods:
            result = solve_minimax(matrices, centers, method=method)
            grad = compute_gradient(result, centers, matrices)
            grads.append(grad)

        # All methods should agree to ~1e-6
        ref = grads[0]
        for g in grads[1:]:
            for i in range(k):
                np.testing.assert_allclose(
                    g.d_xbar[i],
                    ref.d_xbar[i],
                    atol=1e-6,
                    err_msg=f"d_xbar[{i}] mismatch",
                )
                np.testing.assert_allclose(
                    g.d_A[i], ref.d_A[i], atol=1e-6, err_msg=f"d_A[{i}] mismatch"
                )
