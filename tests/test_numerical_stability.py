import numpy as np
import pytest

from ellphi.solver import tangency, TangencyResult
from ellphi._solver_python import _center  # For direct testing if needed

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


def test_divergent_algsig_newton_case():
    # Test C++ backend - should succeed
    cpp_result = tangency(
        p_coef, q_coef, method="algsig+newton", backend="cpp", failsafe=False, x0=0.5
    )
    assert isinstance(cpp_result, TangencyResult)
    assert np.isfinite(cpp_result.mu)
    assert 0.0 < cpp_result.mu < 1.0

    # Test Python backend - should now succeed
    python_result = tangency(
        p_coef, q_coef, method="algsig+newton", backend="python", failsafe=False, x0=0.5
    )
    assert isinstance(python_result, TangencyResult)
    assert np.isfinite(python_result.mu)
    assert 0.0 < python_result.mu < 1.0

    # Optionally, assert that the mu values are close if exact match is expected
    # np.testing.assert_allclose(cpp_result.mu, python_result.mu, rtol=1e-9, atol=1e-10)
