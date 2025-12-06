#!/usr/bin/env python3
"""Prototype: algebraic sigmoid transform for Newton's method."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

from ellphi._solver_python import _target, _target_prime
from ellphi.geometry import coef_from_axes


# --- Algebraic Sigmoid Transform ---


def x_from_u(u: float | np.ndarray) -> float | np.ndarray:
    """x(u) = 0.5 * (1 + u / sqrt(1 + u^2)), maps R -> (0, 1)"""
    # Ensure x stays strictly within (0, 1) to avoid numerical issues
    val = 0.5 * (1.0 + u / np.sqrt(1.0 + u**2))
    epsilon = 1e-10  # Small epsilon to prevent reaching exactly 0 or 1
    return np.clip(val, epsilon, 1.0 - epsilon)


def u_from_x(x: float | np.ndarray) -> float | np.ndarray:
    """Inverse: u = (2x - 1) / (2 * sqrt(x(1-x)))"""
    if np.any(x <= 0) or np.any(x >= 1):
        return np.nan

    # Clip to avoid domain errors at the boundaries
    x_safe = np.clip(x, 1e-15, 1.0 - 1e-15)

    # Numerically stable calculation for u
    # For x near 0.5, (2x-1) is fine.
    # For x near 0 or 1, sqrt(x-x^2) is also fine.
    # Let's use a direct formula.
    return (2.0 * x_safe - 1.0) / (2.0 * np.sqrt(x_safe * (1.0 - x_safe)))


def x_prime_from_u(u: float | np.ndarray) -> float | np.ndarray:
    """x'(u) = dx/du = 0.5 * (1 + u^2)^(-1.5)"""
    return 0.5 * (1.0 + u**2) ** (-1.5)


# --- Test case ---


def build_case(extreme: bool) -> tuple[np.ndarray, np.ndarray]:
    if extreme:
        p = coef_from_axes(np.array([0.0, 0.0]), r0=0.09, r1=40.0, theta=0.4)
        q = coef_from_axes(np.array([1.0, -0.5]), r0=0.01, r1=20.0, theta=1.3)
    else:
        p = coef_from_axes(np.array([0.0, 0.0]), r0=1.0, r1=2.0, theta=0.0)
        q = coef_from_axes(np.array([1.0, 0.5]), r0=1.5, r1=0.8, theta=0.3)
    return p, q


def main(output: Path, extreme: bool) -> int:
    p, q = build_case(extreme)

    def compute_F_and_deriv(u: float) -> tuple[float, float]:
        """Compute F(u) and dF/du."""
        x = x_from_u(u)
        if not (0 < x < 1):
            return np.nan, np.nan

        f_x = _target(x, p, q)
        f_prime_x = _target_prime(x, p, q)
        x_prime_u = x_prime_from_u(u)

        # F(u) = f(x(u))
        F_u = f_x
        # F'(u) = f'(x) * x'(u)
        F_prime_u = f_prime_x * x_prime_u

        return F_u, F_prime_u

    # Find reference root in the original domain [0, 1]
    brent_res = root_scalar(lambda mu: _target(mu, p, q), bracket=(0.0, 1.0))
    x_root = float(brent_res.root)
    u_root = u_from_x(x_root)
    print(f"Brent root: x = {x_root:.6g}, u = {u_root:.6g}")

    # Newton iteration starting at u=0 (corresponds to x=0.5)
    u_path = []
    F_path = []
    u = 0.0

    print(f"Starting Newton at u = {u}, x = {x_from_u(u):.6g}")
    for i in range(25):
        F, dF = compute_F_and_deriv(u)
        u_path.append(u)
        F_path.append(F)

        x = x_from_u(u)
        print(f"Iter {i}: u={u:.4g}, x={x:.4g}, F(u)={F:.4g}, dF/du={dF:.4g}")

        if np.isnan(F) or np.isnan(dF):
            print(f"Iter {i}: NaN")
            break
        if abs(F) < 1e-12:
            print(f"Iter {i}: Converged")
            break

        if not np.isfinite(dF) or dF == 0:
            print(f"Iter {i}: Zero or NaN derivative, stopping.")
            break

        # --- Backtracking Line Search ---
        delta_u = -F / dF
        alpha = 1.0
        u_next = u  # Fallback
        valid_step_found = False

        for j in range(10):  # Max 10 backtracking steps
            u_candidate = u + alpha * delta_u

            if not np.isfinite(u_candidate):
                alpha *= 0.5
                continue

            x_candidate = x_from_u(u_candidate)
            F_candidate = _target(x_candidate, p, q)

            # Armijo condition: check for sufficient decrease
            if np.isfinite(F_candidate) and abs(F_candidate) < abs(F):
                u_next = u_candidate
                valid_step_found = True
                print(
                    f"Iter {i}: Backtrack(j={j}) alpha={alpha:.3g}, step={alpha * delta_u:.4g}"
                )
                break

            alpha *= 0.5

        if not valid_step_found:
            print(f"Iter {i}: Backtracking failed, stopping.")
            break

        u = u_next

        # Safety clamp for u to prevent extreme values due to floating point limits
        u = np.clip(u, -1e7, 1e7)
        # --- End of Backtracking ---

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Original Target f(x)
    x_grid = np.linspace(1e-6, 1.0 - 1e-6, 500)
    f_vals_orig = [_target(x, p, q) for x in x_grid]
    axes[0].plot(x_grid, f_vals_orig)
    axes[0].axhline(0, color="gray", linestyle="-", alpha=0.3)
    axes[0].axvline(
        x_root, color="k", linestyle="--", alpha=0.6, label=f"Root x={x_root:.4f}"
    )
    axes[0].set_title("Original Target f(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].legend()

    # Plot 2: Transformed Target F(u) = f(x(u))
    u_grid = np.linspace(-8, 8, 500)
    x_from_u_grid = x_from_u(u_grid)
    F_vals_transformed = [_target(x, p, q) for x in x_from_u_grid]
    axes[1].plot(u_grid, F_vals_transformed)
    axes[1].axhline(0, color="gray", linestyle="-", alpha=0.3)
    axes[1].axvline(
        u_root, color="k", linestyle="--", alpha=0.6, label=f"Root u={u_root:.4f}"
    )
    axes[1].set_title("Transformed Target F(u) = f(x(u))")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("F(u)")
    axes[1].legend()

    # Plot 3: F(u) with Newton path
    axes[2].plot(u_grid, F_vals_transformed, alpha=0.5, label="F(u)")
    axes[2].plot(u_path, F_path, "o-", color="tab:red", label="Newton Path")
    if u_path:
        axes[2].plot(
            u_path[0],
            F_path[0],
            "D",
            color="tab:green",
            markersize=10,
            label="Start (u=0)",
        )
        axes[2].plot(
            u_path[-1], F_path[-1], "X", color="tab:purple", markersize=10, label="End"
        )
    axes[2].axhline(0, color="gray", linestyle="-", alpha=0.3)
    axes[2].axvline(u_root, color="k", linestyle="--", alpha=0.6)
    axes[2].plot(u_root, 0, "s", color="black", label="Brent root")
    axes[2].set_title("Newton Method in u-space")
    axes[2].set_xlabel("u")
    axes[2].legend()

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot algebraic sigmoid + Newton strategy."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/benchmarks/algebraic_sigmoid_newton.png"),
    )
    parser.add_argument(
        "--extreme", action="store_true", help="Use an extreme anisotropic test case."
    )
    args = parser.parse_args()
    raise SystemExit(main(args.output, args.extreme))
