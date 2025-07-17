
import numpy
from scipy.optimize import root_scalar

def to_abcdef(x0, y0, r1, r2, theta):
    """Convert ellipse parameters to polynomial coefficients.

    Given an ellipse defined by its center `(x0, y0)`, semi-axis lengths `r1` and `r2`
    (along the major and minor axes), and rotation angle `theta` (radians from the x-axis),
    compute the coefficients of its quadratic equation. The result is an array
    `[a, b, c, d, e, f]` representing the conic:
    **a**·x² + 2·**b**·x·y + **c**·y² + 2·**d**·x + 2·**e**·y + **f** = 0.

    Args:
        x0 (float): X-coordinate of the ellipse center.
        y0 (float): Y-coordinate of the ellipse center.
        r1 (float): Semi-axis length along the ellipse’s major axis.
        r2 (float): Semi-axis length along the ellipse’s minor axis.
        theta (float): Rotation of the ellipse in radians (0 = aligned with x-axis).

    Returns:
        numpy.ndarray: A length-6 array of coefficients `[a, b, c, d, e, f]` for the ellipse’s implicit equation.
    """
    C, S = numpy.cos(theta), numpy.sin(theta)
    a = (S ** 2) / (r2 ** 2) + (C ** 2) / (r1 ** 2)
    b = (-S * C) / (r2 ** 2) + (S * C) / (r1 ** 2)
    c = (C ** 2) / (r2 ** 2) + (S ** 2) / (r1 ** 2)
    d = (-x0 * S ** 2 + y0 * S * C) / (r2 ** 2) - (x0 * C ** 2 + y0 * S * C) / (r1 ** 2)
    e = (x0 * S * C - y0 * C ** 2) / (r2 ** 2) - (x0 * S * C + y0 * S ** 2) / (r1 ** 2)
    f = (x0 ** 2 * S ** 2 - 2 * x0 * y0 * S * C + y0 ** 2 * C ** 2) / (r2 ** 2) \
      + (x0 ** 2 * C ** 2 + 2 * x0 * y0 * S * C + y0 ** 2 * S ** 2) / (r1 ** 2)
    return numpy.array([a, b, c, d, e, f])

def epoly(a, b, c, d, e, f, x, y):
    return qform(a, b, c, x, y) + 2 * d * x + 2 * e * y + f

def qform(a, b, c, x, y):
    return a * x**2 + 2 * b * x * y + c * y**2

def center_point(coeffs):
    # 多項式係数 (abcdef) を受け取って中心点を返す
    a, b, c, d, e, f = coeffs
    mat = numpy.array([[a, b], [b, c]])
    return -numpy.linalg.inv(mat).dot(numpy.array([d, e]))

def center_vectorized(coeffs):
    a, b, c, d, e, f = coeffs
    matinv = numpy.array([[c, -b], [-b, a]]) / (a * c - b**2)
    return numpy.einsum('ijk,jk->ik', -matinv, numpy.array([d, e]))

def pencil(mu1, p1, mu2, p2):
    return mu1 * p1 + mu2 * p2

def target_function(mu, p, q):
    # q_mu を計算
    q_mu = pencil(1 - mu, p, mu, q)
    
    # 楕円の中心点を計算
    xc = center_point(q_mu)
    
    # Q_0 と Q_1 の中心点における評価
    Q0_value = epoly(*p, *xc)
    Q1_value = epoly(*q, *xc)
    
    # Q_0 - Q_1 の差を返す
    return Q0_value - Q1_value

def target_function_prime(mu, p, q):
    q_mu = pencil(1 - mu, p, mu, q)
    diff = p - q
    xc = center_point(q_mu)
    A_xprime = -(numpy.array([diff[:2], diff[1:3]]) @ xc + diff[3:5])
    A_mu_inv = numpy.linalg.inv(numpy.array([q_mu[:2], q_mu[1:3]]))
    return 2 * A_xprime.T @ A_mu_inv @ A_xprime

def find_intersect(p, q, method='brentq+newton', *, bracket=[0, 1], x0=None):
    """Find the tangency point between two ellipses given by their coefficients.

    This function computes the point of intersection or closest approach between two ellipses
    (defined by coefficient arrays `p` and `q`). It uses a root-finding strategy to find a parameter `mu`
    in [0, 1] such that the two ellipses are tangent (their defining equations are equal at some point).
    By default, a combination of Brent’s method and Newton’s method is used for efficiency and accuracy.

    Args:
        p (array-like): Coefficients `[a, b, c, d, e, f]` of the first ellipse’s quadratic equation.
        q (array-like): Coefficients `[a, b, c, d, e, f]` of the second ellipse’s quadratic equation.
        method (str, optional): Root-finding method to use for solving the tangency condition.
            Options include `'brentq+newton'` (default, two-phase Brent then Newton refinement),
            `'bisect'`, `'brentq'`, `'brenth'`, or `'newton'`.
            See `find_intersect_mu` for details.
        bracket (list of float, optional): Two-element list `[mu_min, mu_max]` bracketing the solution (default `[0, 1]`).
        x0 (float, optional): Initial guess for `mu` (required if `method='newton'` alone is used).

    Returns:
        tuple: A tuple `(t, xc, mu)` where:
          - **t** (float) is the tangential distance measure at the intersection point (non-negative).
          - **xc** (numpy.ndarray of shape (2,)): the *(x, y)* coordinates of the tangency point.
          - **mu** (float): the interpolation parameter in [0, 1] at which the tangency occurs (mu=0 corresponds to ellipse `p`, mu=1 to ellipse `q`).

    Raises:
        ValueError: If an invalid `method` is specified, or if `method='newton'` is used without providing an `x0` initial guess.
    """
    mu = find_intersect_mu(p, q, method=method, bracket=bracket, x0=x0)
    q_mu = pencil(1 - mu, p, mu, q)
    xc = center_point(q_mu)
    t = numpy.sqrt(epoly(*q_mu, *xc))
    return (t, xc, mu)


def find_intersect_mu(p, q, method='brentq+newton', *, bracket=[0, 1], x0=None):
    """Solve for the Lagrange multiplier that equalizes two ellipse quadratic polynomials.

    Finds the value `mu` in [0, 1] such that the "pencil" combination of ellipse `p` and ellipse `q`
    has a point where both original ellipses evaluate equally. In other words, it solves for `mu`
    where the two ellipses are tangent or intersecting. This is a lower-level helper for `find_intersect`
    that returns only the parameter `mu`.

    The root-finding method can be specified:
      - `'brentq+newton'`: first use Brent’s bracketing method to approximate the root, then refine with a couple of Newton steps (default).
      - `'bisect'`, `'brentq'`, `'brenth'`: use the corresponding bracketing method from `scipy.optimize` directly.
      - `'newton'`: use Newton method (requires a good initial guess `x0`).

    Args:
        p (array-like): Coefficients of the first ellipse.
        q (array-like): Coefficients of the second ellipse.
        method (str, optional): Root-finding algorithm to use (default `'brentq+newton'`).
        bracket (list of float, optional): Bracket interval [min, max] for methods that require it (default [0, 1]).
        x0 (float, optional): Initial guess for Newton’s method (required if `method='newton'`).

    Returns:
        float: The solution `mu` (between 0 and 1) that makes the ellipses tangent.

    Raises:
        ValueError: If `method` is not one of the supported options, or if `method='newton'` is chosen without providing an `x0`.
    """
    if method == 'brentq+newton':
        # (Brent’s method followed by Newton refinement)
        brent_result = root_scalar(target_function, args=(p, q), bracket=bracket, method='brentq', options={'maxiter': 8})
        mu_approx = brent_result.root
        newton_result = root_scalar(target_function, args=(p, q), x0=mu_approx, method='newton',
                                    fprime=target_function_prime, options={'maxiter': 3})
        return newton_result.root
    if method in ['bisect', 'brentq', 'brenth']:
        return root_scalar(target_function, args=(p, q), bracket=bracket, method=method).root
    elif method == 'newton' and x0 is not None:
        return root_scalar(target_function, args=(p, q), x0=x0, method=method, fprime=target_function_prime).root
    else:
        raise ValueError("Invalid method or missing x0 for Newton's method")

# モジュールとしての体裁のための __name__ チェック
if __name__ == "__main__":
    # テスト用の簡単な実行例
    p_params = numpy.array([1, 1, 2, 3, numpy.pi/8])
    q_params = numpy.array([-2, -1, 1, 5, 2*numpy.pi/3])
    p = to_abcdef(*p_params)
    q = to_abcdef(*q_params)
    t, xc, mu = find_intersect(p, q, method='newton', x0=0.5)
    print(f"{t=}")
    print(f"{mu=}")
    print(f"{xc=}")

