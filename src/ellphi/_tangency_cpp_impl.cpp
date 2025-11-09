#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
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

int infer_dim_from_coef_length(std::size_t length) {
    if (length < 6) {
        raise("Coefficient vector too short to represent a conic");
    }
    unsigned long long disc = 8ULL * static_cast<unsigned long long>(length) + 1ULL;
    unsigned long long sqrt_disc = static_cast<unsigned long long>(
        std::llround(std::sqrt(static_cast<long double>(disc)))
    );
    if (sqrt_disc * sqrt_disc != disc) {
        raise("Coefficient length does not correspond to a symmetric quadratic form");
    }
    if (sqrt_disc < 3 || (sqrt_disc - 3ULL) % 2ULL != 0ULL) {
        raise("Coefficient length does not correspond to a valid dimension");
    }
    long long n = static_cast<long long>(sqrt_disc - 3ULL) / 2LL;
    if (n < 2) {
        raise("Coefficient length does not correspond to a valid dimension");
    }
    std::size_t expected =
        static_cast<std::size_t>(((n + 1) * (n + 2)) / 2);
    if (expected != length) {
        raise("Coefficient length does not correspond to a valid dimension");
    }
    return static_cast<int>(n);
}

struct DecodedConic {
    int dim;
    std::vector<double> quad;
    std::vector<double> linear;
    double constant;
};

DecodedConic decode_conic(const std::vector<double>& coef) {
    DecodedConic decoded{};
    decoded.dim = infer_dim_from_coef_length(coef.size());
    const int dim = decoded.dim;
    const std::size_t quad_entries = static_cast<std::size_t>(dim * (dim + 1) / 2);
    decoded.quad.assign(static_cast<std::size_t>(dim * dim), 0.0);
    decoded.linear.assign(dim, 0.0);

    std::size_t idx = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = i; j < dim; ++j) {
            double value = coef[idx++];
            decoded.quad[static_cast<std::size_t>(i * dim + j)] = value;
            decoded.quad[static_cast<std::size_t>(j * dim + i)] = value;
        }
    }

    for (int i = 0; i < dim; ++i) {
        decoded.linear[static_cast<std::size_t>(i)] = coef[idx++];
    }
    if (idx >= coef.size()) {
        raise("Coefficient vector too short to contain constant term");
    }
    decoded.constant = coef[idx];

    const std::size_t expected_length = quad_entries + static_cast<std::size_t>(dim) + 1U;
    if (coef.size() != expected_length) {
        raise("Coefficient length mismatch during decoding");
    }
    return decoded;
}

std::vector<double> pencil(
    const std::vector<double>& p,
    const std::vector<double>& q,
    double mu
) {
    if (p.size() != q.size()) {
        raise("Coefficient vectors must have the same length");
    }
    std::vector<double> result(p.size(), 0.0);
    const double alpha = 1.0 - mu;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = alpha * p[i] + mu * q[i];
    }
    return result;
}

struct SolverContext {
    const std::vector<double>& pcoef;
    const std::vector<double>& qcoef;
    DecodedConic p_dec;
    DecodedConic q_dec;
    std::vector<double> diff_coef;
    DecodedConic diff_dec;
};

SolverContext build_solver_context(
    const std::vector<double>& pcoef,
    const std::vector<double>& qcoef
) {
    if (pcoef.size() != qcoef.size()) {
        raise("Coefficient vectors must have the same length");
    }
    SolverContext ctx{
        pcoef,
        qcoef,
        decode_conic(pcoef),
        decode_conic(qcoef),
        {},
        {}
    };
    ctx.diff_coef.resize(pcoef.size());
    for (std::size_t i = 0; i < pcoef.size(); ++i) {
        ctx.diff_coef[i] = pcoef[i] - qcoef[i];
    }
    ctx.diff_dec = decode_conic(ctx.diff_coef);
    return ctx;
}

std::vector<double> cholesky_factor(const std::vector<double>& matrix, int dim) {
    std::vector<double> chol(static_cast<std::size_t>(dim * dim), 0.0);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += chol[static_cast<std::size_t>(i * dim + k)] *
                    chol[static_cast<std::size_t>(j * dim + k)];
            }
            if (i == j) {
                double value = matrix[static_cast<std::size_t>(i * dim + i)] - sum;
                if (value <= 0.0) {
                    raise("Degenerate conic (determinant zero)");
                }
                chol[static_cast<std::size_t>(i * dim + j)] = std::sqrt(value);
            } else {
                double diag = chol[static_cast<std::size_t>(j * dim + j)];
                if (diag == 0.0) {
                    raise("Degenerate conic (determinant zero)");
                }
                chol[static_cast<std::size_t>(i * dim + j)] =
                    (matrix[static_cast<std::size_t>(i * dim + j)] - sum) / diag;
            }
        }
    }
    return chol;
}

std::vector<double> solve_with_cholesky(
    const std::vector<double>& chol,
    const std::vector<double>& rhs,
    int dim
) {
    std::vector<double> y(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
            sum += chol[static_cast<std::size_t>(i * dim + k)] * y[static_cast<std::size_t>(k)];
        }
        double diag = chol[static_cast<std::size_t>(i * dim + i)];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        y[static_cast<std::size_t>(i)] = (rhs[static_cast<std::size_t>(i)] - sum) / diag;
    }

    std::vector<double> x(dim, 0.0);
    for (int i = dim - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int k = i + 1; k < dim; ++k) {
            sum += chol[static_cast<std::size_t>(k * dim + i)] * x[static_cast<std::size_t>(k)];
        }
        double diag = chol[static_cast<std::size_t>(i * dim + i)];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        x[static_cast<std::size_t>(i)] = (y[static_cast<std::size_t>(i)] - sum) / diag;
    }
    return x;
}

std::vector<double> matvec(
    const std::vector<double>& matrix,
    const std::vector<double>& vec,
    int dim
) {
    std::vector<double> result(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += matrix[static_cast<std::size_t>(i * dim + j)] * vec[static_cast<std::size_t>(j)];
        }
        result[static_cast<std::size_t>(i)] = sum;
    }
    return result;
}

double quad_eval(const DecodedConic& conic, const std::vector<double>& point) {
    if (static_cast<int>(point.size()) != conic.dim) {
        raise("Point dimensionality does not match conic coefficients");
    }
    double value = 0.0;
    for (int i = 0; i < conic.dim; ++i) {
        double row_dot = 0.0;
        for (int j = 0; j < conic.dim; ++j) {
            row_dot += conic.quad[static_cast<std::size_t>(i * conic.dim + j)] *
                point[static_cast<std::size_t>(j)];
        }
        value += point[static_cast<std::size_t>(i)] * row_dot;
    }

    double linear_term = 0.0;
    for (int i = 0; i < conic.dim; ++i) {
        linear_term += conic.linear[static_cast<std::size_t>(i)] *
            point[static_cast<std::size_t>(i)];
    }
    return value + 2.0 * linear_term + conic.constant;
}

struct PencilGeometry {
    DecodedConic conic;
    std::vector<double> chol;
    std::vector<double> center;
};

PencilGeometry build_pencil_geometry(double mu, const SolverContext& ctx) {
    PencilGeometry geom;
    geom.conic = decode_conic(pencil(ctx.pcoef, ctx.qcoef, mu));
    geom.chol = cholesky_factor(geom.conic.quad, geom.conic.dim);
    std::vector<double> rhs = geom.conic.linear;
    for (double& value : rhs) {
        value = -value;
    }
    geom.center = solve_with_cholesky(geom.chol, rhs, geom.conic.dim);
    return geom;
}

double target(double mu, const SolverContext& ctx) {
    PencilGeometry geom = build_pencil_geometry(mu, ctx);
    double p_value = quad_eval(ctx.p_dec, geom.center);
    double q_value = quad_eval(ctx.q_dec, geom.center);
    return p_value - q_value;
}

double target_prime(double mu, const SolverContext& ctx) {
    PencilGeometry geom = build_pencil_geometry(mu, ctx);
    const int dim = geom.conic.dim;
    if (ctx.diff_dec.dim != dim) {
        raise("Dimension mismatch while computing derivative");
    }
    std::vector<double> mat_center = matvec(ctx.diff_dec.quad, geom.center, dim);
    std::vector<double> residual(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        residual[static_cast<std::size_t>(i)] =
            -(mat_center[static_cast<std::size_t>(i)] + ctx.diff_dec.linear[static_cast<std::size_t>(i)]);
    }
    std::vector<double> solved = solve_with_cholesky(geom.chol, residual, dim);
    double dot = std::inner_product(residual.begin(), residual.end(), solved.begin(), 0.0);
    return 2.0 * dot;
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
    const std::string& method,
    const std::pair<double, double>& bracket,
    bool has_x0,
    double x0
) {
    SolverContext ctx = build_solver_context(pcoef, qcoef);
    auto target_fn = [&](double mu) { return target(mu, ctx); };
    auto target_prime_fn = [&](double mu) { return target_prime(mu, ctx); };

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
    double* out_mu,
    char* err_buffer,
    std::size_t err_buffer_len
) {
    try {
        std::vector<double> p(pcoef, pcoef + coef_length);
        std::vector<double> q(qcoef, qcoef + coef_length);
        std::pair<double, double> bracket_pair{bracket[0], bracket[1]};
        double mu = solve_mu(p, q, std::string(method), bracket_pair, has_x0 != 0, x0);
        SolverContext ctx = build_solver_context(p, q);
        PencilGeometry geom = build_pencil_geometry(mu, ctx);
        double t = std::sqrt(quad_eval(geom.conic, geom.center));

        out_t[0] = t;
        std::copy(geom.center.begin(), geom.center.end(), out_point);
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
        int dim = infer_dim_from_coef_length(coef_length);
        (void)dim;
        std::vector<std::vector<double>> conics(
            m,
            std::vector<double>(coef_length, 0.0)
        );
        for (std::size_t i = 0; i < m; ++i) {
            const double* start = coef + i * coef_length;
            std::copy(start, start + coef_length, conics[i].begin());
        }

        std::size_t idx = 0;
        for (std::size_t i = 0; i < m; ++i) {
            const std::vector<double>& p = conics[i];
            for (std::size_t j = i + 1; j < m; ++j) {
                const std::vector<double>& q = conics[j];
                double mu = solve_mu(p, q, "brentq+newton", {0.0, 1.0}, false, 0.0);
                SolverContext ctx = build_solver_context(p, q);
                PencilGeometry geom = build_pencil_geometry(mu, ctx);
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
