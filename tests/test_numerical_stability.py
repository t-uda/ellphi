import numpy as np
import pytest

from ellphi._solver_python import _center
from ellphi.geometry import pack_conic
from ellphi.solver import TangencyResult, tangency

# --- Divergent Case extracted from benchmark (Case index: 0, Dim: 5) ---
p_coef = np.array(
    [
        1.33228455308525628e00,
        2.43876278816755798e00,
        1.54325898022069286e00,
        -1.55423589451699495e00,
        9.16976812360668192e-01,
        4.52959682037715794e00,
        2.84600636540770369e00,
        -2.86805847867001118e00,
        1.68977194660716101e00,
        1.79536060090871930e00,
        -1.80685639163475087e00,
        1.06546533842783386e00,
        1.82455600160327158e00,
        -1.07515390424123614e00,
        6.34352719116559549e-01,
        5.85796702249347874e01,
        1.08660015682746760e02,
        6.82787962832369431e01,
        -6.89348664118233216e01,
        4.06138994116748506e01,
        2.60990938892542999e03,
    ]
)

q_coef = np.array(
    [
        2.99003917973344904e-04,
        -2.45853320479340581e-04,
        -1.08217875554905756e-04,
        -1.41764124064242014e-04,
        -9.06750164522750536e-06,
        5.72378663522622906e-03,
        3.73579973362120517e-03,
        1.15281000826166521e-03,
        -1.84496428561159789e-03,
        2.95139709332456089e-03,
        6.94065665107187861e-04,
        -1.22310973698516148e-03,
        5.31951220358083468e-04,
        -3.30826002336106128e-04,
        9.22379300255368067e-04,
        -3.97154480378464347e-03,
        -4.95103298681364989e-02,
        -2.63837917271066563e-02,
        -1.27182724051025722e-02,
        2.18558990e-02,
        7.79412124457922406e-01,
    ]
)


@pytest.mark.parametrize("method", ["algsig+newton", "brentq+newton"])
def test_hard_case_stability_and_parity(method: str):
    """Verify solver stability and C++/Python parity on a known hard case.

    This test uses a specific 5-dimensional case known to be numerically
    sensitive.

    Strategies:
    - C++ backend: Used as the ground truth. We expect it to solve this
      case cleanly (failsafe=False) due to consistent stable implementation.
    - Python backend: We allow failsafe=True. Floating point differences
      across environments (BLAS/LAPACK) can cause Newton steps to diverge
      slightly. We prioritize getting the correct result (parity) over
      enforcing Newton convergence in this specific edge case.
    """
    # 1. Establish Ground Truth with C++ (strict)
    # If C++ fails here, the test case itself might be too broken,
    # or the C++ implementation regressed.
    cpp_res = tangency(
        p_coef, q_coef, method=method, backend="cpp", failsafe=False, x0=0.5
    )
    assert isinstance(cpp_res, TangencyResult)
    assert np.isfinite(cpp_res.mu)

    # 2. Test Python Backend (relaxed stability)
    py_res = tangency(
        p_coef, q_coef, method=method, backend="python", failsafe=True, x0=0.5
    )
    assert isinstance(py_res, TangencyResult)
    assert np.isfinite(py_res.mu)

    # 3. Verify Parity
    # Results should be identical within reasonable numerical tolerance
    np.testing.assert_allclose(
        py_res.mu, cpp_res.mu, rtol=1e-12, err_msg="mu mismatch between backends"
    )
    np.testing.assert_allclose(
        py_res.point,
        cpp_res.point,
        rtol=1e-9,
        err_msg="contact point mismatch between backends",
    )


def test_center_calculation_indefinite_matrix_parity():
    """Verify `_center` calculation for indefinite matrices (Gaussian fallback).

    Standard Cholesky decomposition fails for indefinite matrices (even if non-singular).
    This test ensures the Python fallback (Gaussian elimination) produces the
    same result as the expected mathematical solution, matching C++ strategy.
    """
    # Indefinite matrix: [[0, 1.5], [1.5, 0]] -> det = -2.25 != 0
    matrix = np.array([[0.0, 1.5], [1.5, 0.0]])
    linear = np.array([1.0, -2.0])
    coef = pack_conic(matrix, linear, 0.25)

    center = _center(coef)

    # Expected solution for 1.5y + x = 0, 1.5x - 2 = 0
    # => x = 4/3, y = -x/1.5 = -4/4.5 = -8/9 ... wait.
    # Ax = -b
    # [[0, 1.5], [1.5, 0]] * [x, y] = [-1, 2]
    # 1.5y = -1 => y = -2/3
    # 1.5x = 2  => x = 4/3
    expected = np.array([4.0 / 3.0, -2.0 / 3.0])

    np.testing.assert_allclose(center, expected)