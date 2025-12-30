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

#if defined(ELLPHI_USE_LAPACK)
extern "C" {
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
void dpotrs_(
    const char* uplo,
    const int* n,
    const int* nrhs,
    const double* a,
    const int* lda,
    double* b,
    const int* ldb,
    int* info
);
}
#endif

#ifndef TANGENCY_VERSION
#define TANGENCY_VERSION "0+unknown"
#endif

extern "C" const char* tangency_backend_version() { return TANGENCY_VERSION; }

namespace {

constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double BRENT_XTOL = std::numeric_limits<double>::epsilon();
constexpr double NEWTON_RTOL = 4.0 * EPS;
constexpr double NEWTON_XTOL = 1e-8;
constexpr int HYBRID_BRACKET_MAXITER = 28;
constexpr int HYBRID_NEWTON_MAXITER = 3;
constexpr int HYBRID_BRACKET_MAXITER_FAILSAFE = 64;
constexpr int NEWTON_ONLY_MAXITER = 50;

std::pair<int, int> default_hybrid_iterations(int dim) {
    // The dimension argument `dim` is preserved for API compatibility, even
    // though the defaults are now dimension-independent.
    return {HYBRID_BRACKET_MAXITER, HYBRID_NEWTON_MAXITER};
}

[[noreturn]] void raise(const std::string& message) {
    throw std::runtime_error(message);
}

bool is_degenerate_error(const std::runtime_error& ex) {
    return std::strcmp(ex.what(), "Degenerate conic (determinant zero)") == 0;
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

// Forward declarations for optimization functions
void decode_conic_into(const std::vector<double>& coef, DecodedConic& out);
void pencil_into(const std::vector<double>& p, const std::vector<double>& q, double mu, std::vector<double>& out);
void cholesky_factor_into(const std::vector<double>& matrix, int dim, std::vector<double>& out);
void solve_with_cholesky_into(const std::vector<double>& chol, const std::vector<double>& rhs, int dim, std::vector<double>& out);
void cholesky_factor_into_internal(const std::vector<double>& matrix, int dim, std::vector<double>& out);
void solve_with_cholesky_into_internal(const std::vector<double>& chol, const std::vector<double>& rhs, int dim, std::vector<double>& out);
#if defined(ELLPHI_USE_LAPACK)
void cholesky_factor_into_lapack(const std::vector<double>& matrix, int dim, std::vector<double>& out);
void solve_with_cholesky_into_lapack(const std::vector<double>& chol, const std::vector<double>& rhs, int dim, std::vector<double>& out);
#endif
void gaussian_elimination_into(std::vector<double> matrix, std::vector<double> rhs, int dim, std::vector<double>& out);
void matvec_into(const std::vector<double>& matrix, const std::vector<double>& vec, int dim, std::vector<double>& out);
void center_with_fallback_into(const DecodedConic& conic, std::vector<double>& out_center, std::vector<double>& work_rhs, std::vector<double>& work_chol);

DecodedConic decode_conic(const std::vector<double>& coef) {
    DecodedConic decoded{};
    decode_conic_into(coef, decoded);
    return decoded;
}

void decode_conic_into(const std::vector<double>& coef, DecodedConic& out) {
    out.dim = infer_dim_from_coef_length(coef.size());
    const int dim = out.dim;
    const std::size_t quad_size = static_cast<std::size_t>(dim * dim);
    if (out.quad.size() != quad_size) {
        out.quad.resize(quad_size);
    }
    if (out.linear.size() != static_cast<std::size_t>(dim)) {
        out.linear.resize(dim);
    }

    std::fill(out.quad.begin(), out.quad.end(), 0.0);

    std::size_t idx = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = i; j < dim; ++j) {
            double value = coef[idx++];
            out.quad[static_cast<std::size_t>(i * dim + j)] = value;
            out.quad[static_cast<std::size_t>(j * dim + i)] = value;
        }
    }

    for (int i = 0; i < dim; ++i) {
        out.linear[static_cast<std::size_t>(i)] = coef[idx++];
    }
    if (idx >= coef.size()) {
        raise("Coefficient vector too short to contain constant term");
    }
    out.constant = coef[idx];

    const std::size_t expected_length = idx + 1;
    if (coef.size() != expected_length) {
        raise("Coefficient length mismatch during decoding");
    }
}
std::vector<double> pencil(
    const std::vector<double>& p,
    const std::vector<double>& q,
    double mu
) {
    std::vector<double> result(p.size());
    pencil_into(p, q, mu, result);
    return result;
}

void pencil_into(
    const std::vector<double>& p,
    const std::vector<double>& q,
    double mu,
    std::vector<double>& out
) {
    if (p.size() != q.size()) {
        raise("Coefficient vectors must have the same length");
    }
    if (out.size() != p.size()) {
        out.resize(p.size());
    }
    const double alpha = 1.0 - mu;
    const std::size_t n = p.size();
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = alpha * p[i] + mu * q[i];
    }
}

struct SolverContext {
    const std::vector<double>& pcoef;
    const std::vector<double>& qcoef;
    DecodedConic p_dec;
    DecodedConic q_dec;
    std::vector<double> diff_coef;
    DecodedConic diff_dec;
    
    // Working buffers to avoid reallocation
    std::vector<double> work_coef;
    DecodedConic work_conic;
    std::vector<double> work_chol;
    std::vector<double> work_rhs;
    std::vector<double> work_center;
    std::vector<double> work_residual;
    std::vector<double> work_solved;
    double cache_mu = std::numeric_limits<double>::quiet_NaN();
    bool cache_valid = false;
    bool cache_chol_ok = false;
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
        {},
        {}, {}, {}, {}, {}, {}, {}
    };
    ctx.diff_coef.resize(pcoef.size());
    for (std::size_t i = 0; i < pcoef.size(); ++i) {
        ctx.diff_coef[i] = pcoef[i] - qcoef[i];
    }
    ctx.diff_dec = decode_conic(ctx.diff_coef);
    
    // Initialize working buffers
    int dim = ctx.p_dec.dim;
    ctx.work_coef.resize(pcoef.size());
    // work_conic will be resized in decode_conic_into
    ctx.work_chol.resize(static_cast<std::size_t>(dim * dim));
    ctx.work_rhs.resize(dim);
    ctx.work_center.resize(dim);
    ctx.work_residual.resize(dim);
    ctx.work_solved.resize(dim);
    
    return ctx;
}



void cholesky_factor_into_internal(const std::vector<double>& matrix, int dim, std::vector<double>& out) {
    const std::size_t size = static_cast<std::size_t>(dim * dim);
    if (out.size() != size) {
        out.resize(size);
    }
    std::fill(out.begin(), out.end(), 0.0);
    
    // Pointer access for optimization
    const double* mat_ptr = matrix.data();
    double* out_ptr = out.data();

    for (int i = 0; i < dim; ++i) {
        double* row_i = out_ptr + i * dim;
        const double* mat_row_i = mat_ptr + i * dim;

        for (int j = 0; j <= i; ++j) {
            double* row_j = out_ptr + j * dim;
            double sum = 0.0;
            
            // Vectorizable loop
            for (int k = 0; k < j; ++k) {
                sum += row_i[k] * row_j[k];
            }

            if (i == j) {
                double value = mat_row_i[i] - sum;
                if (value <= 0.0) {
                    raise("Degenerate conic (determinant zero)");
                }
                row_i[j] = std::sqrt(value);
            } else {
                double diag = row_j[j];
                if (diag == 0.0) {
                    raise("Degenerate conic (determinant zero)");
                }
                row_i[j] = (mat_row_i[j] - sum) / diag;
            }
        }
    }
}



void solve_with_cholesky_into_internal(
    const std::vector<double>& chol,
    const std::vector<double>& rhs,
    int dim,
    std::vector<double>& out
) {
    if (out.size() != static_cast<std::size_t>(dim)) {
        out.resize(dim);
    }
    // Forward substitution (Ly = b)
    // Reuse out as y
    for (int i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
            sum += chol[static_cast<std::size_t>(i * dim + k)] * out[static_cast<std::size_t>(k)];
        }
        double diag = chol[static_cast<std::size_t>(i * dim + i)];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        out[static_cast<std::size_t>(i)] = (rhs[static_cast<std::size_t>(i)] - sum) / diag;
    }

    // Backward substitution (L^T x = y)
    // Reuse out as x (in-place)
    for (int i = dim - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int k = i + 1; k < dim; ++k) {
            sum += chol[static_cast<std::size_t>(k * dim + i)] * out[static_cast<std::size_t>(k)];
        }
        double diag = chol[static_cast<std::size_t>(i * dim + i)];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        out[static_cast<std::size_t>(i)] = (out[static_cast<std::size_t>(i)] - sum) / diag;
    }
}

#if defined(ELLPHI_USE_LAPACK)
void cholesky_factor_into_lapack(const std::vector<double>& matrix, int dim, std::vector<double>& out) {
    const std::size_t size = static_cast<std::size_t>(dim * dim);
    if (out.size() != size) {
        out.resize(size);
    }
    std::copy(matrix.begin(), matrix.end(), out.begin());
    int n = dim;
    int lda = dim;
    int info = 0;
    char uplo = 'L';
    dpotrf_(&uplo, &n, out.data(), &lda, &info);
    if (info > 0) {
        raise("Degenerate conic (determinant zero)");
    }
    if (info < 0) {
        raise("LAPACK dpotrf argument error");
    }
}

void solve_with_cholesky_into_lapack(
    const std::vector<double>& chol,
    const std::vector<double>& rhs,
    int dim,
    std::vector<double>& out
) {
    if (out.size() != static_cast<std::size_t>(dim)) {
        out.resize(dim);
    }
    std::copy(rhs.begin(), rhs.end(), out.begin());
    int n = dim;
    int nrhs = 1;
    int lda = dim;
    int ldb = dim;
    int info = 0;
    char uplo = 'L';
    dpotrs_(&uplo, &n, &nrhs, chol.data(), &lda, out.data(), &ldb, &info);
    if (info != 0) {
        raise("LAPACK dpotrs argument error");
    }
}
#endif

void cholesky_factor_into(const std::vector<double>& matrix, int dim, std::vector<double>& out) {
#if defined(ELLPHI_USE_LAPACK)
    cholesky_factor_into_lapack(matrix, dim, out);
#else
    cholesky_factor_into_internal(matrix, dim, out);
#endif
}

void solve_with_cholesky_into(
    const std::vector<double>& chol,
    const std::vector<double>& rhs,
    int dim,
    std::vector<double>& out
) {
#if defined(ELLPHI_USE_LAPACK)
    solve_with_cholesky_into_lapack(chol, rhs, dim, out);
#else
    solve_with_cholesky_into_internal(chol, rhs, dim, out);
#endif
}



void gaussian_elimination_into(
    std::vector<double> matrix, // Pass by value to copy
    std::vector<double> rhs,    // Pass by value to copy
    int dim,
    std::vector<double>& out
) {
    if (out.size() != static_cast<std::size_t>(dim)) {
        out.resize(dim);
    }
    for (int k = 0; k < dim; ++k) {
        int pivot = k;
        double pivot_value = std::abs(matrix[static_cast<std::size_t>(k * dim + k)]);
        for (int i = k + 1; i < dim; ++i) {
            double value = std::abs(matrix[static_cast<std::size_t>(i * dim + k)]);
            if (value > pivot_value) {
                pivot = i;
                pivot_value = value;
            }
        }
        if (pivot_value == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        if (pivot != k) {
            for (int j = k; j < dim; ++j) {
                std::swap(
                    matrix[static_cast<std::size_t>(k * dim + j)],
                    matrix[static_cast<std::size_t>(pivot * dim + j)]
                );
            }
            std::swap(rhs[static_cast<std::size_t>(k)], rhs[static_cast<std::size_t>(pivot)]);
        }
        double diag = matrix[static_cast<std::size_t>(k * dim + k)];
        for (int i = k + 1; i < dim; ++i) {
            double factor = matrix[static_cast<std::size_t>(i * dim + k)] / diag;
            rhs[static_cast<std::size_t>(i)] -= factor * rhs[static_cast<std::size_t>(k)];
            for (int j = k; j < dim; ++j) {
                matrix[static_cast<std::size_t>(i * dim + j)] -=
                    factor * matrix[static_cast<std::size_t>(k * dim + j)];
            }
        }
    }

    for (int i = dim - 1; i >= 0; --i) {
        double sum = rhs[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < dim; ++j) {
            sum -= matrix[static_cast<std::size_t>(i * dim + j)] * out[static_cast<std::size_t>(j)];
        }
        double diag = matrix[static_cast<std::size_t>(i * dim + i)];
        if (diag == 0.0) {
            raise("Degenerate conic (determinant zero)");
        }
        out[static_cast<std::size_t>(i)] = sum / diag;
    }
}



void matvec_into(
    const std::vector<double>& matrix,
    const std::vector<double>& vec,
    int dim,
    std::vector<double>& out
) {
    if (out.size() != static_cast<std::size_t>(dim)) {
        out.resize(dim);
    }
    for (int i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += matrix[static_cast<std::size_t>(i * dim + j)] * vec[static_cast<std::size_t>(j)];
        }
        out[static_cast<std::size_t>(i)] = sum;
    }
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

std::vector<double> center_with_fallback(const DecodedConic& conic) {
    std::vector<double> center(conic.dim);
    std::vector<double> work_rhs(conic.dim);
    std::vector<double> work_chol(static_cast<std::size_t>(conic.dim * conic.dim));
    center_with_fallback_into(conic, center, work_rhs, work_chol);
    return center;
}

void center_with_fallback_into(
    const DecodedConic& conic,
    std::vector<double>& out_center,
    std::vector<double>& work_rhs,
    std::vector<double>& work_chol
) {
    const int dim = conic.dim;
    if (work_rhs.size() != static_cast<std::size_t>(dim)) {
        work_rhs.resize(dim);
    }
    // Copy linear to work_rhs and negate
    for(int i=0; i<dim; ++i) {
        work_rhs[i] = -conic.linear[i];
    }
    
    try {
        cholesky_factor_into(conic.quad, dim, work_chol);
        solve_with_cholesky_into(work_chol, work_rhs, dim, out_center);
    } catch (const std::runtime_error& ex) {
        if (!is_degenerate_error(ex)) {
            throw;
        }
        // Fallback: gaussian_elimination copies inputs, so we can pass quad and work_rhs
        gaussian_elimination_into(conic.quad, work_rhs, dim, out_center);
    }
}

void compute_geometry_for_mu(double mu, SolverContext& ctx) {
    if (ctx.cache_valid && mu == ctx.cache_mu) {
        return;
    }
    ctx.cache_valid = false;
    ctx.cache_chol_ok = false;

    pencil_into(ctx.pcoef, ctx.qcoef, mu, ctx.work_coef);
    decode_conic_into(ctx.work_coef, ctx.work_conic);

    const int dim = ctx.work_conic.dim;
    if (ctx.work_rhs.size() != static_cast<std::size_t>(dim)) {
        ctx.work_rhs.resize(dim);
    }
    for (int i = 0; i < dim; ++i) {
        ctx.work_rhs[i] = -ctx.work_conic.linear[i];
    }

    try {
        cholesky_factor_into(ctx.work_conic.quad, dim, ctx.work_chol);
        solve_with_cholesky_into(ctx.work_chol, ctx.work_rhs, dim, ctx.work_center);
        ctx.cache_chol_ok = true;
    } catch (const std::runtime_error& ex) {
        if (!is_degenerate_error(ex)) {
            throw;
        }
        gaussian_elimination_into(ctx.work_conic.quad, ctx.work_rhs, dim, ctx.work_center);
    }

    ctx.cache_mu = mu;
    ctx.cache_valid = true;
}

double target(double mu, SolverContext& ctx) {
    compute_geometry_for_mu(mu, ctx);

    double p_value = quad_eval(ctx.p_dec, ctx.work_center);
    double q_value = quad_eval(ctx.q_dec, ctx.work_center);
    return p_value - q_value;
}

double target_prime(double mu, SolverContext& ctx) {
    try {
        compute_geometry_for_mu(mu, ctx);
        if (!ctx.cache_chol_ok) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const int dim = ctx.work_conic.dim;
        
        // Now compute derivative
        if (ctx.diff_dec.dim != dim) {
            raise("Dimension mismatch while computing derivative");
        }
        
        // matvec_into(ctx.diff_dec.quad, geom.center, dim, work_residual)
        matvec_into(ctx.diff_dec.quad, ctx.work_center, dim, ctx.work_residual);
        
        // residual = -(mat_center + diff_linear)
        for (int i = 0; i < dim; ++i) {
            ctx.work_residual[i] = -(ctx.work_residual[i] + ctx.diff_dec.linear[i]);
        }
        
        // solved = solve_with_cholesky(chol, residual)
        // use work_solved
        solve_with_cholesky_into(ctx.work_chol, ctx.work_residual, dim, ctx.work_solved);
        
        double dot = std::inner_product(ctx.work_residual.begin(), ctx.work_residual.end(), ctx.work_solved.begin(), 0.0);
        return 2.0 * dot;
    } catch (const std::runtime_error& ex) {
        if (is_degenerate_error(ex)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        throw;
    }
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

double brentq_impl(
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

        const double tol = 2.0 * EPS * std::abs(b) + 0.5 * BRENT_XTOL;
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

    //// DO NOT RAISE HERE. THIS IS A PART OF THE NUMERICAL COMPUTATION CODE.
    //// Even if the iteration would not converge, it MUST NOT be an error.
    // double residual = f(b);
    // if (std::abs(residual) > 8.0 * EPS * std::abs(b)) {
    //     raise("Brent method failed to converge");
    // }
    return b;
}
// Helper function for 3-point rational interpolation (Bus & Dekker, 1975, Algorithm M)
// Points are (x1, f1), (x2, f2), (x3, f3)
// Returns the interpolated value, or NaN if interpolation is not possible/stable.
double rational_interp(double x1, double f1, double x2, double f2, double x3, double f3) {
    // Ensure distinct function values for interpolation
    if (f1 == f2 || f1 == f3 || f2 == f3) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // First divided differences (f[x_i, x_j] = (f(x_i) - f(x_j)) / (x_i - x_j))
    // This helper function uses (x1, f1), (x2, f2), (x3, f3) for interpolation.
    // In the context of the calling brenth_impl, these map to:
    // x1 -> b, f1 -> fb (current best point)
    // x2 -> a, f2 -> fa (point from opposite side of root)
    // x3 -> c, f3 -> fc (previous 'a' or old 'b' if points were swapped)

    double f_x1_x3_div_diff;
    if (x1 == x3) f_x1_x3_div_diff = std::numeric_limits<double>::infinity();
    else f_x1_x3_div_diff = (f1 - f3) / (x1 - x3); // f[b, d]

    double f_x2_x3_div_diff;
    if (x2 == x3) f_x2_x3_div_diff = std::numeric_limits<double>::infinity();
    else f_x2_x3_div_diff = (f2 - f3) / (x2 - x3); // f[a, d]

    // Alpha and Beta as per Bus & Dekker (3.1.5)
    double alpha = f_x1_x3_div_diff * f2; // alpha = f[b, d] * f(a)
    double beta = f_x2_x3_div_diff * f1;  // beta = f[a, d] * f(b)

    if (beta != alpha) {
        if (beta - alpha == 0.0) return std::numeric_limits<double>::quiet_NaN(); // Avoid division by zero
        return x1 - beta * (x1 - x2) / (beta - alpha);
    } else if (beta != 0.0) { // beta == alpha != 0.0 implies next point is x2
        return x2;
    } else { // beta == alpha == 0.0 => problematic, don't use interpolation
        return std::numeric_limits<double>::quiet_NaN();
    }
}

/*
 * Implementation of Brent's method with 3-point rational interpolation (hyperbolic extrapolation),
 * as described in "Algorithm M" by Bus & Dekker (1975).
 */
double brenth_impl(
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
    double d = b - a; // 'd' in brentq is the step size.
    double e = d;     // 'e' in brentq is previous step size.

    for (int iter = 0; iter < maxiter; ++iter) {
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            // Re-establish bracketing and reset interpolation history if bracket is lost.
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }

        if (std::abs(fc) < std::abs(fb)) {
            // Ensure 'b' is always the current best estimate (|f(b)| smallest).
            // Update the points for interpolation: c becomes old a, a becomes old b, b becomes old c.
            a = b;
            fa = fb;

            b = c;
            fb = fc;

            c = a;
            fc = fa;
        }

        const double tol = 2.0 * EPS * std::abs(b) + 0.5 * BRENT_XTOL;
        const double m = 0.5 * (c - b); // Half of the bracketing interval [b, c]

        if (std::abs(m) <= tol || fb == 0.0) {
            return b; // Converged
        }

        // --- Attempt Rational Interpolation ---
        // Uses three points: (b, fb), (a, fa), (c, fc)
        // Mapped to rational_interp(x1, f1, x2, f2, x3, f3) -> rational_interp(b, fb, a, fa, c, fc)
        double s_interp = std::numeric_limits<double>::quiet_NaN();

        // Only attempt rational interpolation if we have three distinct points.
        // And if a, c are not too close to b for stability.
        if (std::abs(b - a) > EPS && std::abs(b - c) > EPS && std::abs(a - c) > EPS) {
            s_interp = rational_interp(b, fb, a, fa, c, fc);
        }

        // Determine the next step based on interpolation and Brent's safeguards.
        double next_step_len;

        // Apply Brent's safeguard logic to the interpolated step 's_interp'.
        // This closely mimics the conditions in brentq_impl.
        // If s_interp is valid and conditions for superlinear step are met, use it.
        // Otherwise, fall back to bisection.
        if (!std::isnan(s_interp) && std::abs(e) >= tol && std::abs(fa) > std::abs(fb)) {
            double p = s_interp - b; // Proposed step from current 'b' to 's_interp'
            double q = 1.0; // Denominator for comparison in safeguard logic.

            // This safeguard checks if the interpolated step 'p' is not too large
            // and is in a reasonable direction, and ensures sufficient reduction.
            if (p * q > 0.0) { // If 'p' is in the expected direction from 'b'
                if (2.0 * std::abs(p) < std::min(3.0 * std::abs(m) * q - std::abs(tol * q), std::abs(e * q))) {
                    next_step_len = p / q;
                    e = d; // Store previous effective step size
                    d = next_step_len; // Store current effective step size
                } else {
                    next_step_len = m; // Fallback to bisection
                    e = m;
                    d = m;
                }
            } else { // If proposed 'p' is not in expected direction or zero
                next_step_len = m; // Fallback to bisection
                e = m;
                d = m;
            }
        } else {
            // If rational interpolation failed, or conditions for using it are not met, use bisection.
            next_step_len = m;
            e = m;
            d = m;
        }

        // --- Take the step and update points ---
        // Update 'd' (previous step) and 'e' (second previous step) for the next iteration.
        d = e; 
        e = next_step_len;

        a = b;
        fa = fb;

        if (std::abs(next_step_len) > tol) {
            b += next_step_len;
        } else {
            b += (m > 0.0 ? tol : -tol); // Move by tolerance if step is too small
        }
        fb = f(b);

        if (fb == 0.0) {
            return b;
        }
    }

    //// DO NOT RAISE HERE. THIS IS A PART OF THE NUMERICAL COMPUTATION CODE.
    //// Even if the iteration would not converge, it MUST NOT be an error.
    // double residual = f(b);
    // if (std::abs(residual) > 8.0 * EPS * std::abs(b)) {
    //     raise("Brenth method failed to converge");
    // }
    return b;
}

struct NewtonResult {
    double root;
    bool converged;
};

NewtonResult newton(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double x0,
    int maxiter
) {
    double x = x0;
    for (int iter = 0; iter < maxiter; ++iter) {
        double fx = f(x);
        double dfx = df(x);
        if (!std::isfinite(fx) || !std::isfinite(dfx)) {
            break;
        }
        if (std::abs(dfx) < EPS) {
            break;
        }
        double step = fx / dfx;
        double next = x - step;
        if (std::abs(step) <= NEWTON_RTOL * std::abs(next) + NEWTON_XTOL) {
            return {next, true};
        }
        x = next;
    }
    return {x, false};
}

// --- Algsig+Newton Helpers ---

double x_from_u(double u) {
    return 0.5 * (1.0 + u / std::sqrt(1.0 + u * u));
}

double u_from_x(double x) {
    if (x <= 0.0 || x >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double x_safe = std::max(1e-15, std::min(1.0 - 1e-15, x));
    return (2.0 * x_safe - 1.0) / (2.0 * std::sqrt(x_safe * (1.0 - x_safe)));
}

double x_prime_from_u(double u) {
    return 0.5 * std::pow(1.0 + u * u, -1.5);
}

NewtonResult algsig_newton(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double x0,
    int maxiter
) {
    double u = u_from_x(x0);
    if (!std::isfinite(u)) {
        u = 0.0; // Default start if x0 is invalid
    }

    for (int i = 0; i < maxiter; ++i) {
        double x = x_from_u(u);
        double f_val = f(x);
        double f_prime_val = df(x);
        double x_prime_val = x_prime_from_u(u);
        double F_prime_u = f_prime_val * x_prime_val;

        if (!std::isfinite(f_val) || !std::isfinite(F_prime_u) || F_prime_u == 0.0) {
            return {x_from_u(u), false};
        }

        // Backtracking Line Search to find a step that decreases the function value
        double delta_u = -f_val / F_prime_u;
        double alpha = 1.0;
        double u_next = u;
        bool step_accepted = false;

        for (int j = 0; j < 10; ++j) { // Max 10 backtracking steps
            double u_candidate = u + alpha * delta_u;
            if (!std::isfinite(u_candidate)) {
                alpha *= 0.5;
                continue;
            }
            double x_candidate = x_from_u(u_candidate);
            double f_candidate = f(x_candidate);

            if (std::isfinite(f_candidate) && std::abs(f_candidate) < std::abs(f_val)) {
                u_next = u_candidate;
                step_accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!step_accepted) {
            return {x_from_u(u), false}; // Backtracking failed
        }
        
        // Convergence criterion on the step size in the transformed space
        if (std::abs(u_next - u) <= NEWTON_XTOL + NEWTON_RTOL * std::abs(u_next)) {
            return {x_from_u(u_next), true};
        }

        u = u_next;
    }

    return {x_from_u(u), false};
}

double solve_mu(
    const std::vector<double>& pcoef,
    const std::vector<double>& qcoef,
    const std::string& method,
    const std::pair<double, double>& bracket,
    bool has_x0,
    double x0,
    int hybrid_bracket_maxiter,
    int hybrid_newton_maxiter,
    bool failsafe
) {
    SolverContext ctx = build_solver_context(pcoef, qcoef);
    auto target_fn = [&](double mu) { return target(mu, ctx); };
    auto target_prime_fn = [&](double mu) { return target_prime(mu, ctx); };

    const double a = bracket.first;
    const double b = bracket.second;
    const double fa = target_fn(a);
    const double fb = target_fn(b);

    auto bisect_refined = [&]() { return bisect(target_fn, a, b, fa, fb, 128); };
    auto brentq_refined = [&]() { return brentq_impl(target_fn, a, b, fa, fb, 256); };
    auto brenth_refined = [&]() { return brenth_impl(target_fn, a, b, fa, fb, 256); };
    auto failsafe_refined = [&]() { return brentq_impl(target_fn, a, b, fa, fb, HYBRID_BRACKET_MAXITER_FAILSAFE); };

    if (method == "brentq+newton") {
        if (hybrid_bracket_maxiter <= 0 || hybrid_newton_maxiter <= 0) {
            raise("Hybrid iteration counts must be positive");
        }
        double mu0 = brentq_impl(target_fn, a, b, fa, fb, hybrid_bracket_maxiter);

        NewtonResult res = newton(target_fn, target_prime_fn, mu0, hybrid_newton_maxiter);

        if (res.converged) {
            return res.root;
        } else if (failsafe) {
            return failsafe_refined();
        } else {
            return mu0;
        }
    }
    if (method == "algsig+newton") {
        double mu_guess = has_x0 ? x0 : 0.5;
        NewtonResult res = algsig_newton(target_fn, target_prime_fn, mu_guess, NEWTON_ONLY_MAXITER);
        if (res.converged) {
            return res.root;
        }
        if (failsafe) {
            return failsafe_refined();
        }
        return res.root;
    }
    if (method == "bisect") {
        return bisect_refined();
    }
    if (method == "brentq") {
        return brentq_refined();
    }
    if (method == "brenth") {
        return brenth_refined();
    }
    if (method == "newton") {
        if (!has_x0) {
            raise("x0 must be provided for Newton method");
        }
        NewtonResult res = newton(
            target_fn,
            target_prime_fn,
            x0,
            NEWTON_ONLY_MAXITER
        );

        if (failsafe && !res.converged) {
            return failsafe_refined();
        }
        return res.root;
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
    int hybrid_bracket_maxiter,
    int hybrid_newton_maxiter,
    int failsafe,
    double* out_t,
    double* out_point,
    std::size_t point_length,
    double* out_mu,
    char* err_buffer,
    std::size_t err_buffer_len
) {
    try {
        std::vector<double> p(pcoef, pcoef + coef_length);
        std::vector<double> q(qcoef, qcoef + coef_length);
        std::pair<double, double> bracket_pair{bracket[0], bracket[1]};
        double mu = solve_mu(
            p,
            q,
            std::string(method),
            bracket_pair,
            has_x0 != 0,
            x0,
            hybrid_bracket_maxiter,
            hybrid_newton_maxiter,
            failsafe != 0
        );
        DecodedConic conic = decode_conic(pencil(p, q, mu));
        std::vector<double> center = center_with_fallback(conic);
        const std::size_t dim = static_cast<std::size_t>(center.size());
        if (point_length < dim) {
            raise("Output point buffer too small");
        }
        double value = quad_eval(conic, center);
        if (value < 0.0) {
            value = 0.0;
        }
        double t = std::sqrt(value);

        out_t[0] = t;
        std::copy(center.begin(), center.end(), out_point);
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
                auto hybrid_iters = default_hybrid_iterations(dim);
        double mu = solve_mu(
            p,
            q,
            "brentq+newton",
            {0.0, 1.0},
            false,
            0.0,
            hybrid_iters.first,
            hybrid_iters.second,
            true // failsafe enabled by default for pdist
        );
        DecodedConic conic = decode_conic(pencil(p, q, mu));
        std::vector<double> center = center_with_fallback(conic);
        double value = quad_eval(conic, center);
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
