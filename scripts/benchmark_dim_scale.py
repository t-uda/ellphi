import time
import numpy as np
import ellphi
import sys


def generate_random_ellipse(dim):
    # Random center
    center = np.random.rand(dim) * 10.0

    # Random positive definite matrix
    X = np.random.rand(dim, dim)
    cov = X @ X.T + np.eye(dim) * 0.1

    # Convert to coefficients
    # coef_from_cov returns (n, m) array
    # We pass a list of centers and list of covs (length 1)
    coefs = ellphi.coef_from_cov([center], [cov])
    return coefs[0]


def benchmark():
    dims = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    n_trials = 20  # Number of different ellipse pairs to test per dimension
    n_repeats = 20  # Number of repeats for timing per pair

    print("dim,backend,avg_time_ms")

    for dim in dims:
        # Pre-generate ellipse pairs to avoid measuring generation time
        pairs = []
        for _ in range(n_trials):
            p = generate_random_ellipse(dim)
            q = generate_random_ellipse(dim)
            pairs.append((p, q))

        # Benchmark Python backend
        py_times = []
        for p, q in pairs:
            start = time.perf_counter()
            for _ in range(n_repeats):
                ellphi.tangency(p, q, backend="python")
            end = time.perf_counter()
            py_times.append((end - start) / n_repeats)
        avg_py = np.mean(py_times) * 1000.0
        print(f"{dim},python,{avg_py:.4f}")

        # Benchmark C++ backend
        if ellphi.has_cpp_backend():
            cpp_times = []
            for p, q in pairs:
                start = time.perf_counter()
                for _ in range(n_repeats):
                    ellphi.tangency(p, q, backend="cpp")
                end = time.perf_counter()
                cpp_times.append((end - start) / n_repeats)
            avg_cpp = np.mean(cpp_times) * 1000.0
            print(f"{dim},cpp,{avg_cpp:.4f}")
        else:
            print(f"{dim},cpp,NaN")

        sys.stdout.flush()


if __name__ == "__main__":
    benchmark()
