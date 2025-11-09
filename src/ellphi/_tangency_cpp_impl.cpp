#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double XTOL = std::numeric_limits<double>::epsilon();

[[noreturn]] void raise(const std::string& message) {
    throw std::runtime_error(message);
}

std::size_t infer_dim_from_coef_length(std::size_t length) {
    if (length < 6) {
        raise("Coefficient vector too short to represent a conic");
    }
    std::size_t disc = 1 + 8 * length;
    std::size_t sqrt_disc = static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<long double>(disc)))
    );
    if (sqrt_disc * sqrt_disc != disc) {
        raise("Coefficient length does not correspond to a symmetric quadratic form");
    }
    if (sqrt_disc < 3 || ((sqrt_disc - 3) % 2 != 0)) {
        raise("Coefficient length does not correspond to a valid dimension");
    }
    std::size_t n = (sqrt_disc - 3) / 2;
    if (n < 2 || ((n + 1) * (n + 2)) / 2 != length) {
        raise("Coefficient length does not correspond to a valid dimension");
    }
    return n;
}

struct Conic {
    std::size_t dim;
    std::vector<double> quad;
    std::vector<double> linear;
    double constant;
};

Conic decode_conic(const std::vector<double>& coef) {
    std::size_t length = coef.size();
    std::size_t dim = infer_dim_from_coef_length(length);
    std::size_t n_quad = dim * (dim + 1) / 2;

    Conic result;
    result.dim = dim;
    result.quad.assign(dim * dim, 0.0);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i; j < dim; ++j) {
            double value = coef[idx++];
            result.quad[i * dim + j] = value;
            result.quad[j * dim + i] = value;
        }
    }

    result.linear.assign(dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        result.linear[i] = coef[n_quad + i];
    }
    result.constant = coef[n_quad + dim];
    return result;
}

std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        raise("Coefficient vectors must have the same length");
    }
    std::vector<double> diff(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff[i] = a[i] - b[i];
    }
    return diff;
}

std::vector<double> pencil(const std::vector<double>& p, const std::vector<double>& q, double mu) {
    if (p.size() != q.size()) {
        raise("Coefficient vectors must have the same length");
    }
    std::vector<double> result(p.size());
    double alpha = 1.0 - mu;
    for (std::size_t i = 0; i < p.size(); ++i) {
        result[i] = alpha * p[i] + mu * q[i];
    }
    return result;
}

std::vector<double> as_vector(const double* data, std::size_t length) {
    if (data == nullptr) {
        raise("Null pointer passed for coefficient data");
    }
    return std::vector<double>(data, data + length);
}

struct CholeskyFactor {
    std::size_t dim;
    std::vector<double> lower;
};

bool cholesky_inplace(std::vector<double>& matrix, std::size_t dim) {
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = matrix[i * dim + j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= matrix[i * dim + k] * matrix[j * dim + k];
            }
            if (i == j) {
                if (sum <= 0.0) {
                    return false;
                }
                matrix[i * dim + j] = std::sqrt(sum);
            } else {
                matrix[i * dim + j] = sum / matrix[j * dim + j];
            }
        }
        for (std::size_t j = i + 1; j < dim; ++j) {
            matrix[i * dim + j] = 0.0;
        }
    }
    return true;
}

CholeskyFactor factorize(const Conic& conic) {
    CholeskyFactor factor;
    factor.dim = conic.dim;
    factor.lower = conic.quad;
    if (!cholesky_inplace(factor.lower, factor.dim)) {
        raise("Degenerate conic (determinant zero)");
    }
    return factor;
}

std::vector<double> solve_with_factor(const CholeskyFactor& factor, const std::vector<double>& rhs) {
    if (rhs.size() != factor.dim) {
        raise("Right-hand side dimension mismatch");
    }
    std::vector<double> y(factor.dim, 0.0);
    for (std::size_t i = 0; i < factor.dim; ++i) {
        double sum = rhs[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= factor.lower[i * factor.dim + k] * y[k];
        }
        double diag = factor.lower[i * factor.dim + i];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        y[i] = sum / diag;
    }

    std::vector<double> x(factor.dim, 0.0);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(factor.dim) - 1; i >= 0; --i) {
        double sum = y[static_cast<std::size_t>(i)];
        for (std::size_t k = static_cast<std::size_t>(i) + 1; k < factor.dim; ++k) {
            sum -= factor.lower[k * factor.dim + static_cast<std::size_t>(i)] * x[k];
        }
        double diag = factor.lower[static_cast<std::size_t>(i) * factor.dim + static_cast<std::size_t>(i)];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        x[static_cast<std::size_t>(i)] = sum / diag;
    }
    return x;
}

std::vector<double> center(const Conic& conic, const CholeskyFactor& factor) {
    std::vector<double> rhs(conic.dim, 0.0);
    for (std::size_t i = 0; i < conic.dim; ++i) {
        rhs[i] = -conic.linear[i];
    }
    return solve_with_factor(factor, rhs);
}

double quad_eval(const Conic& conic, const std::vector<double>& point) {
    if (point.size() != conic.dim) {
        raise("Point dimensionality does not match conic coefficients");
    }
    double quad = 0.0;
    for (std::size_t i = 0; i < conic.dim; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < conic.dim; ++j) {
            row_sum += conic.quad[i * conic.dim + j] * point[j];
        }
        quad += point[i] * row_sum;
    }
    double linear = 0.0;
    for (std::size_t i = 0; i < conic.dim; ++i) {
        linear += conic.linear[i] * point[i];
    }
    return quad + 2.0 * linear + conic.constant;
}

double quadratic_form_via_factor(const CholeskyFactor& factor, const std::vector<double>& residual) {
    std::vector<double> solution = solve_with_factor(factor, residual);
    double value = 0.0;
    for (std::size_t i = 0; i < factor.dim; ++i) {
        value += residual[i] * solution[i];
    }
    return value;
}

struct PencilGeometry {
    Conic conic;
    CholeskyFactor factor;
    std::vector<double> center;
};

PencilGeometry build_pencil_geometry(
    double mu,
    const std::vector<double>& pcoef,
    const std::vector<double>& qcoef
) {
    PencilGeometry geom;
    geom.conic = decode_conic(pencil(pcoef, qcoef, mu));
    geom.factor = factorize(geom.conic);
    geom.center = center(geom.conic, geom.factor);
    return geom;
}

double target(
    double mu,
    const std::vector<double>& pcoef,
    const Conic& p,
    const std::vector<double>& qcoef,
    const Conic& q
) {
    PencilGeometry geom = build_pencil_geometry(mu, pcoef, qcoef);
    double p_value = quad_eval(p, geom.center);
    double q_value = quad_eval(q, geom.center);
    return p_value - q_value;
}

double target_prime(
    double mu,
    const std::vector<double>& pcoef,
    const std::vector<double>& qcoef,
    const Conic& diff
) {
    PencilGeometry geom = build_pencil_geometry(mu, pcoef, qcoef);
    if (geom.conic.dim != diff.dim) {
        raise("Dimension mismatch while computing derivative");
    }

    std::vector<double> residual(diff.dim, 0.0);
    for (std::size_t i = 0; i < diff.dim; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < diff.dim; ++j) {
            sum += diff.quad[i * diff.dim + j] * geom.center[j];
        }
        residual[i] = -(sum + diff.linear[i]);
    }

    double value = quadratic_form_via_factor(geom.factor, residual);
    return 2.0 * value;
}

double bisect(
    const std::function<double(double)>& f,
    double a,
    double b,
    double fa,
    double fb,
    int maxiter
) {
    if (fa == 0.0) {
        return a;
    }
    if (fb == 0.0) {
        return b;
    }
    if (fa * fb > 0.0) {
        raise("Bisection interval does not bracket a root");
    }

    double left = a;
    double right = b;
    double f_left = fa;
    double mid = left;
    for (int iter = 0; iter < maxiter; ++iter) {
        mid = 0.5 * (left + right);
        double f_mid = f(mid);
        if (f_mid == 0.0 || 0.5 * std::abs(right - left) < EPS) {
            return mid;
        }
        if (f_left * f_mid < 0.0) {
            right = mid;
        } else {
            left = mid;
            f_left = f_mid;
        }
    }
    return mid;
}

double brent(
    const std::function<double(double)>& f,
    double a,
    double b,
    double fa,
    double fb,
    int maxiter
) {
    if (fa == 0.0) {
        return a;
    }
    if (fb == 0.0) {
        return b;
    }
    if (fa * fb > 0.0) {
        raise("Brent interval does not bracket a root");
    }

    double c = a;
    double fc = fa;
    double d = b - a;
    double e = d;

    for (int iter = 0; iter < maxiter; ++iter) {
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }

        if (std::abs(fc) < std::abs(fb)) {
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = a;
            fc = fa;
        }

        const double tol = 2.0 * EPS * std::abs(b) + 0.5 * XTOL;
        const double m = 0.5 * (c - b);

        if (std::abs(m) <= tol || fb == 0.0) {
            return b;
        }

        if (std::abs(e) >= tol && std::abs(fa) > std::abs(fb)) {
            double s = fb / fa;
            double p;
            double q;

            if (a == c) {
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                const double q_tmp = fa / fc;
                const double r = fb / fc;
                p = s * (2.0 * m * q_tmp * (q_tmp - r) - (b - a) * (r - 1.0));
                q = (q_tmp - 1.0) * (r - 1.0) * (s - 1.0);
            }

            if (p > 0.0) {
                q = -q;
            } else {
                p = -p;
            }

            if (q != 0.0 &&
                2.0 * p < std::min(3.0 * m * q - std::abs(tol * q), std::abs(e * q))) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }

        a = b;
        fa = fb;
        if (std::abs(d) > tol) {
            b += d;
        } else {
            b += (m > 0.0 ? tol : -tol);
        }
        fb = f(b);
        if (fb == 0.0) {
            return b;
        }
    }

    double residual = f(b);
    if (std::abs(residual) > 8.0 * EPS * std::abs(b)) {
        raise("Brent method failed to converge");
    }
    return b;
}

double newton(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double x0,
    int maxiter
) {
    double x = x0;
    for (int iter = 0; iter < maxiter; ++iter) {
        double fx = f(x);
        double dfx = df(x);
        if (dfx == 0.0) {
            raise("Derivative is zero during Newton iteration");
        }
        double step = fx / dfx;
        double next = x - step;
        if (std::abs(step) <= 8.0 * EPS * std::abs(next)) {
            return next;
        }
        x = next;
    }
    return x;
}

double solve_mu(
    const std::vector<double>& pcoef,
    const std::vector<double>& qcoef,
    const Conic& p,
    const Conic& q,
    const Conic& diff,
    const std::string& method,
    const std::pair<double, double>& bracket,
    bool has_x0,
    double x0
) {
    auto target_fn = [&](double mu) { return target(mu, pcoef, p, qcoef, q); };
    auto target_prime_fn = [&](double mu) { return target_prime(mu, pcoef, qcoef, diff); };

    const double a = bracket.first;
    const double b = bracket.second;
    const double fa = target_fn(a);
    const double fb = target_fn(b);

    auto bisect_refined = [&]() { return bisect(target_fn, a, b, fa, fb, 128); };
    auto brent_refined = [&]() { return brent(target_fn, a, b, fa, fb, 256); };

    if (method == "brentq+newton") {
        double mu0 = brent(target_fn, a, b, fa, fb, 64);
        try {
            return newton(target_fn, target_prime_fn, mu0, 3);
        } catch (const std::runtime_error& ex) {
            if (std::string(ex.what()) == "Derivative is zero during Newton iteration") {
                return mu0;
            }
            throw;
        }
    }
    if (method == "bisect") {
        return bisect_refined();
    }
    if (method == "brentq" || method == "brenth") {
        return brent_refined();
    }
    if (method == "newton") {
        if (!has_x0) {
            raise("x0 must be provided for Newton method");
        }
        return newton(target_fn, target_prime_fn, x0, 50);
    }
    raise("Unknown method");
}

void copy_error(char* buffer, std::size_t size, const std::string& message) {
    if (buffer == nullptr || size == 0) {
        return;
    }
    std::size_t copy_len = std::min<std::size_t>(message.size(), size - 1);
    std::memcpy(buffer, message.c_str(), copy_len);
    buffer[copy_len] = '\0';
}

}  // namespace

#if defined(_WIN32) || defined(__CYGWIN__)
#define ELLPHI_EXPORT __declspec(dllexport)
#else
#define ELLPHI_EXPORT
#endif

ELLPHI_EXPORT extern "C" int tangency_solve(
    const double* pcoef,
    const double* qcoef,
    std::size_t coef_length,
    const char* method,
    const double* bracket,
    int has_x0,
    double x0,
    double* out_t,
    double* out_point,
    std::size_t point_length,
    double* out_mu,
    char* err_buffer,
    std::size_t err_buffer_len
) {
    try {
        std::vector<double> p = as_vector(pcoef, coef_length);
        std::vector<double> q = as_vector(qcoef, coef_length);
        if (p.size() != q.size()) {
            raise("Coefficient vectors must have the same length");
        }
        std::size_t dim = infer_dim_from_coef_length(coef_length);
        if (point_length < dim) {
            raise("Output point buffer too small");
        }

        Conic p_conic = decode_conic(p);
        Conic q_conic = decode_conic(q);
        Conic diff_conic = decode_conic(subtract(p, q));

        std::pair<double, double> bracket_pair{bracket[0], bracket[1]};
        double mu = solve_mu(p, q, p_conic, q_conic, diff_conic, std::string(method), bracket_pair, has_x0 != 0, x0);

        PencilGeometry geom = build_pencil_geometry(mu, p, q);
        double t = std::sqrt(quad_eval(geom.conic, geom.center));

        out_t[0] = t;
        for (std::size_t i = 0; i < dim; ++i) {
            out_point[i] = geom.center[i];
        }
        out_mu[0] = mu;
        return 0;
    } catch (const std::exception& ex) {
        copy_error(err_buffer, err_buffer_len, ex.what());
        return 1;
    } catch (...) {
        copy_error(err_buffer, err_buffer_len, "Unknown error");
        return 1;
    }
}

ELLPHI_EXPORT extern "C" int pdist_tangency(
    const double* coef,
    std::size_t m,
    std::size_t coef_length,
    double* out,
    char* err_buffer,
    std::size_t err_buffer_len
) {
    try {
        if (coef_length == 0) {
            return 0;
        }
        std::size_t dim = infer_dim_from_coef_length(coef_length);
        (void)dim;  // unused but validates coefficients.

        std::vector<std::vector<double>> coefs;
        coefs.reserve(m);
        std::vector<Conic> conics;
        conics.reserve(m);
        for (std::size_t i = 0; i < m; ++i) {
            coefs.emplace_back(coef + i * coef_length, coef + (i + 1) * coef_length);
            conics.push_back(decode_conic(coefs.back()));
        }

        std::size_t idx = 0;
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = i + 1; j < m; ++j) {
                Conic diff_conic = decode_conic(subtract(coefs[i], coefs[j]));
                double mu = solve_mu(
                    coefs[i],
                    coefs[j],
                    conics[i],
                    conics[j],
                    diff_conic,
                    "brentq+newton",
                    {0.0, 1.0},
                    false,
                    0.0
                );
                PencilGeometry geom = build_pencil_geometry(mu, coefs[i], coefs[j]);
                double t = std::sqrt(quad_eval(geom.conic, geom.center));
                out[idx++] = t;
            }
        }
        return 0;
    } catch (const std::exception& ex) {
        copy_error(err_buffer, err_buffer_len, ex.what());
        return 1;
    } catch (...) {
        copy_error(err_buffer, err_buffer_len, "Unknown error");
        return 1;
    }
}
