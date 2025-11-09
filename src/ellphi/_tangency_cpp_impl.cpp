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

int infer_dim_from_coef_length(std::size_t length) {
    if (length < 6) {
        raise("Coefficient vector too short to represent a conic");
    }
    const std::size_t disc = 1 + 8 * length;
    const long double disc_ld = static_cast<long double>(disc);
    const std::size_t sqrt_disc = static_cast<std::size_t>(std::llround(std::sqrt(disc_ld)));
    if (sqrt_disc * sqrt_disc != disc) {
        raise("Coefficient length does not correspond to a symmetric quadratic form");
    }
    const long long numerator = static_cast<long long>(sqrt_disc) - 3;
    if (numerator < 0 || numerator % 2 != 0) {
        raise("Coefficient length does not correspond to a valid dimension");
    }
    const long long dim_ll = numerator / 2;
    const std::size_t expected_length = (static_cast<std::size_t>(dim_ll + 1) *
                                         static_cast<std::size_t>(dim_ll + 2)) /
                                        2;
    if (expected_length != length || dim_ll < 2) {
        raise("Coefficient length does not correspond to a valid dimension");
    }
    return static_cast<int>(dim_ll);
}

struct Quadric {
    int dim{};
    std::size_t quad_entries{};
    std::vector<double> coef;

    Quadric() = default;

    Quadric(int dimension, std::vector<double> values)
        : dim(dimension),
          quad_entries(static_cast<std::size_t>(dimension) *
                       static_cast<std::size_t>(dimension + 1) / 2),
          coef(std::move(values)) {
        const std::size_t expected = quad_entries + static_cast<std::size_t>(dim) + 1;
        if (coef.size() != expected) {
            raise("Coefficient vector length does not match inferred dimension");
        }
    }

    static Quadric from_data(const double* data, std::size_t length) {
        const int dimension = infer_dim_from_coef_length(length);
        std::vector<double> values(length);
        for (std::size_t i = 0; i < length; ++i) {
            values[i] = data[i];
        }
        return Quadric(dimension, std::move(values));
    }
};

struct CholeskyFactor {
    int dim{};
    std::vector<double> lower;
};

struct PencilGeometry {
    int dim{};
    CholeskyFactor chol;
    std::vector<double> center;
};

Quadric pencil(const Quadric& p, const Quadric& q, double mu) {
    if (p.dim != q.dim || p.coef.size() != q.coef.size()) {
        raise("Coefficient vectors must share the same dimension");
    }
    std::vector<double> result(p.coef.size());
    const double alpha = 1.0 - mu;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = alpha * p.coef[i] + mu * q.coef[i];
    }
    return Quadric(p.dim, std::move(result));
}

std::vector<double> quad_matrix(const Quadric& coef) {
    std::vector<double> matrix(static_cast<std::size_t>(coef.dim) *
                               static_cast<std::size_t>(coef.dim),
                               0.0);
    std::size_t idx = 0;
    for (int i = 0; i < coef.dim; ++i) {
        for (int j = i; j < coef.dim; ++j) {
            const double value = coef.coef[idx++];
            matrix[static_cast<std::size_t>(i) * coef.dim + j] = value;
            matrix[static_cast<std::size_t>(j) * coef.dim + i] = value;
        }
    }
    return matrix;
}

std::vector<double> linear_vector(const Quadric& coef) {
    std::vector<double> linear(static_cast<std::size_t>(coef.dim));
    std::size_t offset = coef.quad_entries;
    for (int i = 0; i < coef.dim; ++i) {
        linear[static_cast<std::size_t>(i)] = coef.coef[offset + static_cast<std::size_t>(i)];
    }
    return linear;
}

CholeskyFactor cholesky_factor(const std::vector<double>& matrix, int dim) {
    CholeskyFactor factor;
    factor.dim = dim;
    factor.lower.assign(static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim), 0.0);

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = matrix[static_cast<std::size_t>(i) * dim + j];
            for (int k = 0; k < j; ++k) {
                sum -= factor.lower[static_cast<std::size_t>(i) * dim + k] *
                       factor.lower[static_cast<std::size_t>(j) * dim + k];
            }
            if (i == j) {
                if (sum <= 0.0) {
                    raise("Degenerate conic (determinant zero)");
                }
                factor.lower[static_cast<std::size_t>(i) * dim + j] = std::sqrt(sum);
            } else {
                const double diag = factor.lower[static_cast<std::size_t>(j) * dim + j];
                if (diag == 0.0) {
                    raise("Degenerate conic (determinant zero)");
                }
                factor.lower[static_cast<std::size_t>(i) * dim + j] = sum / diag;
            }
        }
    }

    return factor;
}

std::vector<double> cholesky_solve(const CholeskyFactor& factor, const std::vector<double>& rhs) {
    if (rhs.size() != static_cast<std::size_t>(factor.dim)) {
        raise("Right-hand side length does not match matrix dimension");
    }

    const int dim = factor.dim;
    std::vector<double> y(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        double sum = rhs[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k) {
            sum -= factor.lower[static_cast<std::size_t>(i) * dim + k] * y[static_cast<std::size_t>(k)];
        }
        const double diag = factor.lower[static_cast<std::size_t>(i) * dim + i];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        y[static_cast<std::size_t>(i)] = sum / diag;
    }

    std::vector<double> x(static_cast<std::size_t>(dim));
    for (int i = dim - 1; i >= 0; --i) {
        double sum = y[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < dim; ++k) {
            sum -= factor.lower[static_cast<std::size_t>(k) * dim + i] *
                   x[static_cast<std::size_t>(k)];
        }
        const double diag = factor.lower[static_cast<std::size_t>(i) * dim + i];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        x[static_cast<std::size_t>(i)] = sum / diag;
    }

    return x;
}

double quadratic_form_with_inverse(const CholeskyFactor& factor, const std::vector<double>& vector) {
    std::vector<double> solution = cholesky_solve(factor, vector);
    double dot = 0.0;
    for (int i = 0; i < factor.dim; ++i) {
        dot += vector[static_cast<std::size_t>(i)] * solution[static_cast<std::size_t>(i)];
    }
    return dot;
}

PencilGeometry build_geometry(const Quadric& coef) {
    PencilGeometry geometry;
    geometry.dim = coef.dim;

    std::vector<double> matrix = quad_matrix(coef);
    std::vector<double> linear = linear_vector(coef);

    geometry.chol = cholesky_factor(matrix, coef.dim);

    std::vector<double> rhs(static_cast<std::size_t>(coef.dim));
    for (int i = 0; i < coef.dim; ++i) {
        rhs[static_cast<std::size_t>(i)] = -linear[static_cast<std::size_t>(i)];
    }
    geometry.center = cholesky_solve(geometry.chol, rhs);

    return geometry;
}

std::vector<double> matrix_vector_product(const std::vector<double>& matrix, const std::vector<double>& vector, int dim) {
    if (vector.size() != static_cast<std::size_t>(dim)) {
        raise("Vector length does not match matrix dimension");
    }
    std::vector<double> result(static_cast<std::size_t>(dim), 0.0);
    for (int i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += matrix[static_cast<std::size_t>(i) * dim + j] *
                   vector[static_cast<std::size_t>(j)];
        }
        result[static_cast<std::size_t>(i)] = sum;
    }
    return result;
}

void difference_matrix_linear(
    const Quadric& p,
    const Quadric& q,
    std::vector<double>& matrix,
    std::vector<double>& linear
) {
    if (p.dim != q.dim || p.coef.size() != q.coef.size()) {
        raise("Coefficient vectors must share the same dimension");
    }
    const int dim = p.dim;
    matrix.assign(static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim), 0.0);
    linear.assign(static_cast<std::size_t>(dim), 0.0);

    std::size_t idx = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = i; j < dim; ++j) {
            const double value = p.coef[idx] - q.coef[idx];
            matrix[static_cast<std::size_t>(i) * dim + j] = value;
            matrix[static_cast<std::size_t>(j) * dim + i] = value;
            ++idx;
        }
    }
    for (int i = 0; i < dim; ++i) {
        linear[static_cast<std::size_t>(i)] =
            p.coef[idx] - q.coef[idx];
        ++idx;
    }
}

double quad_eval(const Quadric& coef, const std::vector<double>& point) {
    if (point.size() != static_cast<std::size_t>(coef.dim)) {
        raise("Point dimensionality does not match conic coefficients");
    }
    double result = 0.0;
    std::size_t idx = 0;
    for (int i = 0; i < coef.dim; ++i) {
        const double xi = point[static_cast<std::size_t>(i)];
        for (int j = i; j < coef.dim; ++j) {
            const double xj = point[static_cast<std::size_t>(j)];
            const double value = coef.coef[idx++];
            if (i == j) {
                result += value * xi * xj;
            } else {
                result += 2.0 * value * xi * xj;
            }
        }
    }
    for (int i = 0; i < coef.dim; ++i) {
        const double bi = coef.coef[idx++];
        result += 2.0 * bi * point[static_cast<std::size_t>(i)];
    }
    result += coef.coef[idx];
    return result;
}

double target(double mu, const Quadric& p, const Quadric& q) {
    Quadric coef = pencil(p, q, mu);
    PencilGeometry geometry = build_geometry(coef);
    const double value_p = quad_eval(p, geometry.center);
    const double value_q = quad_eval(q, geometry.center);
    return value_p - value_q;
}

double target_prime(double mu, const Quadric& p, const Quadric& q) {
    Quadric coef = pencil(p, q, mu);
    PencilGeometry geometry = build_geometry(coef);

    std::vector<double> diff_matrix;
    std::vector<double> diff_linear;
    difference_matrix_linear(p, q, diff_matrix, diff_linear);

    std::vector<double> matvec = matrix_vector_product(diff_matrix, geometry.center, geometry.dim);
    std::vector<double> residual(static_cast<std::size_t>(geometry.dim));
    for (int i = 0; i < geometry.dim; ++i) {
        residual[static_cast<std::size_t>(i)] =
            -(matvec[static_cast<std::size_t>(i)] + diff_linear[static_cast<std::size_t>(i)]);
    }

    const double quad = quadratic_form_with_inverse(geometry.chol, residual);
    return 2.0 * quad;
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
    const Quadric& p,
    const Quadric& q,
    const std::string& method,
    const std::pair<double, double>& bracket,
    bool has_x0,
    double x0
) {
    auto target_fn = [&](double mu) { return target(mu, p, q); };
    auto target_prime_fn = [&](double mu) { return target_prime(mu, p, q); };

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
        Quadric p = Quadric::from_data(pcoef, coef_length);
        Quadric q = Quadric::from_data(qcoef, coef_length);
        std::pair<double, double> bracket_pair{bracket[0], bracket[1]};
        double mu = solve_mu(p, q, std::string(method), bracket_pair, has_x0 != 0, x0);
        Quadric coef = pencil(p, q, mu);
        PencilGeometry geometry = build_geometry(coef);
        double value = quad_eval(coef, geometry.center);
        if (value < 0.0) {
            value = 0.0;
        }
        double t = std::sqrt(value);

        out_t[0] = t;
        for (int i = 0; i < geometry.dim; ++i) {
            out_point[static_cast<std::size_t>(i)] = geometry.center[static_cast<std::size_t>(i)];
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
        std::vector<Quadric> conics;
        conics.reserve(m);
        for (std::size_t i = 0; i < m; ++i) {
            conics.emplace_back(Quadric::from_data(coef + i * coef_length, coef_length));
        }

        std::size_t idx = 0;
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = i + 1; j < m; ++j) {
                double mu = solve_mu(conics[i], conics[j], "brentq+newton", {0.0, 1.0}, false, 0.0);
                Quadric mix = pencil(conics[i], conics[j], mu);
                PencilGeometry geometry = build_geometry(mix);
                double value = quad_eval(mix, geometry.center);
                if (value < 0.0) {
                    value = 0.0;
                }
                out[idx++] = std::sqrt(value);
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
