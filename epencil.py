
import numpy
from scipy.optimize import root_scalar

def to_abcdef(x0, y0, r1, r2, theta):
    C, S = numpy.cos(theta), numpy.sin(theta)
    a = (S ** 2) / (r2 ** 2) + (C ** 2) / (r1 ** 2)
    b = (-S * C) / (r2 ** 2) + (S * C) / (r1 ** 2)
    c = (C ** 2) / (r2 ** 2) + (S ** 2) / (r1 ** 2)
    d = (-x0 * S ** 2 + y0 * S * C) / (r2 ** 2) - (x0 * C ** 2 + y0 * S * C) / (r1 ** 2)
    e = (x0 * S * C - y0 * C ** 2) / (r2 ** 2) - (x0 * S * C + y0 * S ** 2) / (r1 ** 2)
    f = (x0 ** 2 * S ** 2 - 2 * x0 * y0 * S * C + y0 ** 2 * C ** 2) / (r2 ** 2) + (x0 ** 2 * C ** 2 + 2 * x0 * y0 * S * C + y0 ** 2 * S ** 2) / (r1 ** 2)
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
    mu = find_intersect_mu(p, q, method=method, bracket=bracket, x0=x0)
    q_mu = pencil(1 - mu, p, mu, q)
    xc = center_point(q_mu)
    t = numpy.sqrt(epoly(*q_mu, *xc))
    return (t, xc, mu)

def find_intersect_mu(p, q, method='brentq+newton', *, bracket=[0, 1], x0=None):
    if method == 'brentq+newton':
        # Step 1: Use Brent's method to quickly converge to a rough solution
        brent_result = root_scalar(target_function, args=(p, q), bracket=bracket, method='brentq', options={'maxiter': 8})
        mu_approx = brent_result.root
        
        # Step 2: Use Newton's method for fine-tuning with 2 iterations
        newton_result = root_scalar(target_function, args=(p, q), x0=mu_approx, method='newton', fprime=target_function_prime, options={'maxiter': 3})
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

