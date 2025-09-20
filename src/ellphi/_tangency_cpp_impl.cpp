#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using QuadCoef = std::array<double, 6>;
using Point = std::array<double, 2>;

constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double XTOL = std::numeric_limits<double>::epsilon();

[[noreturn]] void raise(const std::string& message) {
    throw std::runtime_error(message);
}

QuadCoef pencil(const QuadCoef& p, const QuadCoef& q, double mu) {
    QuadCoef result{};
    const double alpha = 1.0 - mu;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = alpha * p[i] + mu * q[i];
    }
    return result;
}

double quad_eval(const QuadCoef& coef, const Point& center) {
    const double a = coef[0];
    const double b = coef[1];
    const double c = coef[2];
    const double d = coef[3];
    const double e = coef[4];
    const double f = coef[5];
    const double x = center[0];
    const double y = center[1];
    return a * x * x + 2.0 * b * x * y + c * y * y + 2.0 * d * x + 2.0 * e * y + f;
}

Point center(const QuadCoef& coef) {
    const double a = coef[0];
    const double b = coef[1];
    const double c = coef[2];
    const double d = coef[3];
    const double e = coef[4];

    const double det = a * c - b * b;
    if (det == 0.0) {
        raise("Degenerate conic (determinant zero)");
    }

    const double x = (b * e - c * d) / det;
    const double y = (b * d - a * e) / det;
    return Point{x, y};
}

double target(double mu, const QuadCoef& p, const QuadCoef& q) {
    QuadCoef coef = pencil(p, q, mu);
    Point xc = center(coef);
    return quad_eval(p, xc) - quad_eval(q, xc);
}

double target_prime(double mu, const QuadCoef& p, const QuadCoef& q) {
    QuadCoef coef = pencil(p, q, mu);
    const double a = coef[0];
    const double b = coef[1];
    const double c = coef[2];
    const double d = coef[3];
    const double e = coef[4];

    QuadCoef diff{};
    for (std::size_t i = 0; i < diff.size(); ++i) {
        diff[i] = p[i] - q[i];
    }

    const double det = a * c - b * b;
    if (det == 0.0) {
        raise("Degenerate conic (determinant zero)");
    }

    const double xc0 = (b * e - c * d) / det;
    const double xc1 = (b * d - a * e) / det;

    const double diff00 = diff[0];
    const double diff01 = diff[1];
    const double diff11 = diff[2];
    const double diff0 = diff[3];
    const double diff1 = diff[4];

    const double Axprime0 = -(diff00 * xc0 + diff01 * xc1 + diff0);
    const double Axprime1 = -(diff01 * xc0 + diff11 * xc1 + diff1);

    const double v0 = Axprime0;
    const double v1 = Axprime1;
    const double numerator = c * v0 * v0 - 2.0 * b * v0 * v1 + a * v1 * v1;
    return 2.0 * numerator / det;
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
    const QuadCoef& p,
    const QuadCoef& q,
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

QuadCoef as_coef(const double* data) {
    QuadCoef coef{};
    for (std::size_t i = 0; i < coef.size(); ++i) {
        coef[i] = data[i];
    }
    return coef;
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
        QuadCoef p = as_coef(pcoef);
        QuadCoef q = as_coef(qcoef);
        std::pair<double, double> bracket_pair{bracket[0], bracket[1]};
        double mu = solve_mu(p, q, std::string(method), bracket_pair, has_x0 != 0, x0);
        QuadCoef coef = pencil(p, q, mu);
        Point pt = center(coef);
        double t = std::sqrt(quad_eval(coef, pt));

        out_t[0] = t;
        out_point[0] = pt[0];
        out_point[1] = pt[1];
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
    double* out,
    char* err_buffer,
    std::size_t err_buffer_len
) {
    try {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < m; ++i) {
            QuadCoef p = as_coef(coef + i * 6);
            for (std::size_t j = i + 1; j < m; ++j) {
                QuadCoef q = as_coef(coef + j * 6);
                double mu = solve_mu(p, q, "brentq+newton", {0.0, 1.0}, false, 0.0);
                QuadCoef mix = pencil(p, q, mu);
                Point pt = center(mix);
                double t = std::sqrt(quad_eval(mix, pt));
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
