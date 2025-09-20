#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace {

using Coef = std::array<double, 6>;
using Point = std::array<double, 2>;

constexpr int SUCCESS = 0;
constexpr int ERROR_DEGENERATE = 1;
constexpr int ERROR_BRACKET = 2;
constexpr int ERROR_METHOD = 3;
constexpr int ERROR_MISSING_X0 = 4;
constexpr int ERROR_ZERO_DERIVATIVE = 5;
constexpr int ERROR_INVALID_BRACKET = 6;

inline void load_coef(const double* src, Coef& dst) {
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = src[i];
    }
}

inline double quad_eval(const Coef& coef, const Point& center) {
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

inline Coef pencil(const Coef& p, const Coef& q, double mu) {
    Coef coef{};
    const double one_minus_mu = 1.0 - mu;
    for (std::size_t i = 0; i < coef.size(); ++i) {
        coef[i] = one_minus_mu * p[i] + mu * q[i];
    }
    return coef;
}

inline int center_from_coef(const Coef& coef, Point& point) {
    const double a = coef[0];
    const double b = coef[1];
    const double c = coef[2];
    const double d = coef[3];
    const double e = coef[4];

    const double det = a * c - b * b;
    if (det == 0.0) {
        return ERROR_DEGENERATE;
    }

    point[0] = (b * e - c * d) / det;
    point[1] = (b * d - a * e) / det;
    return SUCCESS;
}

inline int target(double mu, const Coef& p, const Coef& q, double& value) {
    const Coef coef = pencil(p, q, mu);
    Point center{};
    const int status = center_from_coef(coef, center);
    if (status != SUCCESS) {
        return status;
    }
    value = quad_eval(p, center) - quad_eval(q, center);
    return SUCCESS;
}

inline int target_prime(double mu, const Coef& p, const Coef& q, double& value) {
    const Coef coef = pencil(p, q, mu);
    const double a = coef[0];
    const double b = coef[1];
    const double c = coef[2];
    const double d = coef[3];
    const double e = coef[4];

    const double det = a * c - b * b;
    if (det == 0.0) {
        return ERROR_DEGENERATE;
    }

    Point center{};
    center[0] = (b * e - c * d) / det;
    center[1] = (b * d - a * e) / det;

    const double diff0 = p[0] - q[0];
    const double diff1 = p[1] - q[1];
    const double diff2 = p[2] - q[2];
    const double diff3 = p[3] - q[3];
    const double diff4 = p[4] - q[4];

    const double v0 = -(diff0 * center[0] + diff1 * center[1] + diff3);
    const double v1 = -(diff1 * center[0] + diff2 * center[1] + diff4);

    const double numerator = c * v0 * v0 - 2.0 * b * v0 * v1 + a * v1 * v1;
    value = 2.0 * numerator / det;
    return SUCCESS;
}

inline int bisect(const Coef& p, const Coef& q, double a, double b, int maxiter, double tol, double& root) {
    double fa = 0.0;
    double fb = 0.0;
    int status = target(a, p, q, fa);
    if (status != SUCCESS) {
        return status;
    }
    status = target(b, p, q, fb);
    if (status != SUCCESS) {
        return status;
    }
    if (fa == 0.0) {
        root = a;
        return SUCCESS;
    }
    if (fb == 0.0) {
        root = b;
        return SUCCESS;
    }
    if (fa * fb > 0.0) {
        return ERROR_BRACKET;
    }

    double left = a;
    double right = b;
    double mid = 0.5 * (left + right);
    for (int iter = 0; iter < maxiter; ++iter) {
        mid = 0.5 * (left + right);
        double fm = 0.0;
        status = target(mid, p, q, fm);
        if (status != SUCCESS) {
            return status;
        }
        if (std::abs(fm) < tol || 0.5 * (right - left) < tol) {
            break;
        }
        if (fa * fm < 0.0) {
            right = mid;
            fb = fm;
        } else {
            left = mid;
            fa = fm;
        }
    }
    root = mid;
    return SUCCESS;
}

inline int brent(const Coef& p, const Coef& q, double a, double b, int maxiter, double tol, double& root) {
    double fa = 0.0;
    double fb = 0.0;
    int status = target(a, p, q, fa);
    if (status != SUCCESS) {
        return status;
    }
    status = target(b, p, q, fb);
    if (status != SUCCESS) {
        return status;
    }
    if (fa == 0.0) {
        root = a;
        return SUCCESS;
    }
    if (fb == 0.0) {
        root = b;
        return SUCCESS;
    }
    if (fa * fb > 0.0) {
        return ERROR_BRACKET;
    }

    double left = a;
    double right = b;
    double c = a;
    double fc = fa;
    double s = right;
    double d = 0.0;
    bool mflag = true;

    for (int iter = 0; iter < maxiter; ++iter) {
        if (fa != fc && fb != fc) {
            s = (left * fb * fc) / ((fa - fb) * (fa - fc)) +
                (right * fa * fc) / ((fb - fa) * (fb - fc)) +
                (c * fa * fb) / ((fc - fa) * (fc - fb));
        } else {
            s = right - fb * (right - left) / (fb - fa);
        }

        const double condition1 = (s < (3.0 * left + right) * 0.25) || (s > right);
        const double condition2 = mflag && std::abs(s - right) >= std::abs(right - c) * 0.5;
        const double condition3 = !mflag && std::abs(s - right) >= std::abs(c - d) * 0.5;
        const double condition4 = mflag && std::abs(right - c) < tol;
        const double condition5 = !mflag && std::abs(c - d) < tol;

        if (condition1 || condition2 || condition3 || condition4 || condition5) {
            s = 0.5 * (left + right);
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = 0.0;
        status = target(s, p, q, fs);
        if (status != SUCCESS) {
            return status;
        }

        d = c;
        c = right;
        fc = fb;

        if (fa * fs < 0.0) {
            right = s;
            fb = fs;
        } else {
            left = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(left, right);
            std::swap(fa, fb);
        }

        if (std::abs(right - left) < tol) {
            break;
        }
    }

    root = right;
    return SUCCESS;
}

inline int newton(const Coef& p, const Coef& q, double x0, int maxiter, double tol, double& root) {
    double x = x0;
    for (int iter = 0; iter < maxiter; ++iter) {
        double fx = 0.0;
        double dfx = 0.0;
        int status = target(x, p, q, fx);
        if (status != SUCCESS) {
            return status;
        }
        status = target_prime(x, p, q, dfx);
        if (status != SUCCESS) {
            return status;
        }
        if (dfx == 0.0) {
            return ERROR_ZERO_DERIVATIVE;
        }
        const double step = fx / dfx;
        x -= step;
        if (std::abs(step) < tol) {
            break;
        }
    }
    root = x;
    return SUCCESS;
}

struct TangencyState {
    double t;
    Point point;
    double mu;
};

inline int tangency_impl(
    const Coef& p,
    const Coef& q,
    const std::string& method,
    double a,
    double b,
    bool has_x0,
    double x0,
    TangencyState& out
) {
    if (!(a < b)) {
        return ERROR_INVALID_BRACKET;
    }

    constexpr double tol = 1e-12;
    constexpr int brent_iter = 64;
    constexpr int bisect_iter = 100;
    constexpr int newton_iter = 3;

    double mu = 0.0;

    if (method == "brentq+newton") {
        double mu0 = 0.0;
        int status = brent(p, q, a, b, 8, tol, mu0);
        if (status != SUCCESS) {
            return status;
        }
        status = newton(p, q, mu0, newton_iter, tol, mu);
        if (status == ERROR_ZERO_DERIVATIVE) {
            mu = mu0;
        } else if (status != SUCCESS) {
            return status;
        }
    } else if (method == "brentq" || method == "brenth") {
        int status = brent(p, q, a, b, brent_iter, tol, mu);
        if (status != SUCCESS) {
            return status;
        }
    } else if (method == "bisect") {
        int status = bisect(p, q, a, b, bisect_iter, tol, mu);
        if (status != SUCCESS) {
            return status;
        }
    } else if (method == "newton") {
        if (!has_x0) {
            return ERROR_MISSING_X0;
        }
        int status = newton(p, q, x0, brent_iter, tol, mu);
        if (status != SUCCESS) {
            return status;
        }
    } else {
        return ERROR_METHOD;
    }

    const Coef coef = pencil(p, q, mu);
    Point center{};
    int status = center_from_coef(coef, center);
    if (status != SUCCESS) {
        return status;
    }

    const double value = quad_eval(coef, center);
    out.t = std::sqrt(value < 0.0 ? 0.0 : value);
    out.point = center;
    out.mu = mu;
    return SUCCESS;
}

}  // namespace

extern "C" {

struct TangencyResult {
    double t;
    double point_x;
    double point_y;
    double mu;
};

int tangency_solver(
    const double* pcoef,
    const double* qcoef,
    const char* method,
    const double* bracket,
    double x0,
    int has_x0,
    TangencyResult* out
) {
    if (pcoef == nullptr || qcoef == nullptr || bracket == nullptr || out == nullptr) {
        return ERROR_METHOD;
    }

    Coef p{};
    Coef q{};
    load_coef(pcoef, p);
    load_coef(qcoef, q);

    const std::string method_str = method != nullptr ? std::string(method) : std::string("brentq+newton");
    const double a = bracket[0];
    const double b = bracket[1];

    TangencyState state{};
    const int status = tangency_impl(p, q, method_str, a, b, has_x0 != 0, x0, state);
    if (status != SUCCESS) {
        return status;
    }

    out->t = state.t;
    out->point_x = state.point[0];
    out->point_y = state.point[1];
    out->mu = state.mu;
    return SUCCESS;
}

int pdist_tangency_solver(
    const double* coefficients,
    std::int64_t m,
    const char* method,
    const double* bracket,
    double* out
) {
    if (coefficients == nullptr || bracket == nullptr || out == nullptr) {
        return ERROR_METHOD;
    }
    if (m <= 1) {
        return SUCCESS;
    }

    const std::string method_str = method != nullptr ? std::string(method) : std::string("brentq+newton");
    const double a = bracket[0];
    const double b = bracket[1];

    for (std::int64_t i = 0; i < m; ++i) {
        Coef p{};
        load_coef(coefficients + i * 6, p);
        for (std::int64_t j = i + 1; j < m; ++j) {
            Coef q{};
            load_coef(coefficients + j * 6, q);
            TangencyState state{};
            const int status = tangency_impl(p, q, method_str, a, b, false, 0.0, state);
            if (status != SUCCESS) {
                return status;
            }
            const std::int64_t idx = m * i + j - ((i + 2) * (i + 1)) / 2;
            out[idx] = state.t;
        }
    }

    return SUCCESS;
}

}  // extern "C"
