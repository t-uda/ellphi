#!/usr/bin/env python
"""Utility script for comparing tangency strategies across backends."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence, Tuple, cast, get_args

import numpy as np

from ellphi.geometry import coef_from_cov
from ellphi.solver import MethodName, has_cpp_backend, quad_eval, tangency

BACKEND_CHOICES: tuple[str, ...] = ("python", "cpp", "auto")
METHOD_CHOICES: tuple[MethodName, ...] = cast(
    Tuple[MethodName, ...], get_args(MethodName)
)
BASELINE_EPS = 1e-12


@dataclass
class MetricsAggregate:
    """Running metrics for a single (strategy, scenario) pairing."""

    count: int = 0
    failures: int = 0
    runtime_sum: float = 0.0
    runtime_samples: list[float] = field(default_factory=list)
    rel_err_sum: float = 0.0
    rel_err_max: float = 0.0
    residual_sum: float = 0.0
    residual_max: float = 0.0

    def update_success(self, runtime: float, rel_err: float, residual: float) -> None:
        self.count += 1
        self.runtime_sum += runtime
        self.runtime_samples.append(runtime)
        self.rel_err_sum += rel_err
        self.rel_err_max = max(self.rel_err_max, rel_err)
        self.residual_sum += residual
        self.residual_max = max(self.residual_max, residual)

    def update_failure(self) -> None:
        self.failures += 1

    def summary(self) -> dict[str, float | int]:
        if self.count:
            mean_runtime = self.runtime_sum / self.count
            median_runtime = statistics.median(self.runtime_samples)
            mean_rel_err = self.rel_err_sum / self.count
            mean_residual = self.residual_sum / self.count
        else:
            mean_runtime = float("nan")
            median_runtime = float("nan")
            mean_rel_err = float("nan")
            mean_residual = float("nan")
        return {
            "cases": self.count,
            "failures": self.failures,
            "mean_runtime_s": mean_runtime,
            "median_runtime_s": median_runtime,
            "mean_relative_error": mean_rel_err,
            "max_relative_error": self.rel_err_max,
            "mean_objective_residual": mean_residual,
            "max_objective_residual": self.residual_max,
        }


@dataclass
class StrategyStats:
    """Collect metrics for a solver strategy."""

    label: str
    overall: MetricsAggregate = field(default_factory=MetricsAggregate)
    per_scenario: dict[str, MetricsAggregate] = field(default_factory=dict)

    def update_success(
        self, scenario: str, runtime: float, rel_err: float, residual: float
    ) -> None:
        self.overall.update_success(runtime, rel_err, residual)
        self.per_scenario.setdefault(scenario, MetricsAggregate()).update_success(
            runtime, rel_err, residual
        )

    def update_failure(self, scenario: str) -> None:
        self.overall.update_failure()
        self.per_scenario.setdefault(scenario, MetricsAggregate()).update_failure()

    def serialize(self) -> dict[str, object]:
        payload = {"strategy": self.label, "overall": self.overall.summary()}
        payload["scenarios"] = {
            scenario: metrics.summary()
            for scenario, metrics in self.per_scenario.items()
        }
        return payload


def rotation_matrix(angle: float) -> np.ndarray:
    """Return a 2D rotation matrix."""

    cos_val, sin_val = math.cos(angle), math.sin(angle)
    return np.array([[cos_val, -sin_val], [sin_val, cos_val]], dtype=float)


def orthogonal_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Return a random orthogonal matrix using QR decomposition."""

    mat = rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(mat)
    return q


def random_covariance(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Sample a symmetric positive-definite covariance."""

    if dim == 2:
        axes = rng.uniform(1.0, 6.0, size=2)
        rot = rotation_matrix(rng.uniform(0.0, math.pi))
        return rot @ np.diag(axes) @ rot.T
    mat = rng.standard_normal((dim, dim))
    cov = mat @ mat.T
    return cov + dim * np.eye(dim, dtype=float)


def biased_covariance(
    rng: np.random.Generator, dim: int, *, max_aspect_ratio: float
) -> np.ndarray:
    """Return an SPD covariance matrix with controlled anisotropy."""

    aspect = max_aspect_ratio
    scales = np.geomspace(1.0, aspect, num=dim)
    rng.shuffle(scales)
    diag = np.diag(scales)
    if dim == 2:
        rot = rotation_matrix(rng.uniform(0.0, math.pi))
        return rot @ diag @ rot.T
    ortho = orthogonal_matrix(rng, dim)
    return ortho @ diag @ ortho.T


def random_coefficients(
    rng: np.random.Generator, *, dim: int, bias: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two conic coefficient vectors."""

    means = rng.uniform(-50.0, 50.0, size=(2, dim))
    if bias is None:
        covs = np.stack([random_covariance(rng, dim) for _ in range(2)])
    else:
        covs = np.stack(
            [
                biased_covariance(rng, dim, max_aspect_ratio=bias)
                for _ in range(2)
            ]
        )
    coefs = coef_from_cov(means, covs)
    return coefs[0], coefs[1]


def tangency_residual(pcoef: np.ndarray, qcoef: np.ndarray, point: np.ndarray) -> float:
    """Return an objective-space residual for the tangency point."""

    p_val = quad_eval(pcoef, point)
    q_val = quad_eval(qcoef, point)
    denom = abs(p_val) + abs(q_val) + BASELINE_EPS
    return abs(p_val - q_val) / denom


def relative_error(reference: float, candidate: float) -> float:
    """Compute the relative error between two scalars."""

    scale = max(abs(reference), BASELINE_EPS)
    return abs(candidate - reference) / scale


def generate_cases(
    rng: np.random.Generator,
    dims: Sequence[int],
    n_random: int,
    n_biased: int,
    bias_ratio: float,
) -> Iterable[tuple[str, int, np.ndarray, np.ndarray]]:
    """Yield random and biased tangency cases."""

    for dim in dims:
        for _ in range(n_random):
            yield (f"random-d{dim}", dim, *random_coefficients(rng, dim=dim, bias=None))
        for _ in range(n_biased):
            yield (
                f"biased-d{dim}",
                dim,
                *random_coefficients(rng, dim=dim, bias=bias_ratio),
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare tangency solver strategies across Python and C++ backends "
            "by sweeping random and biased ellipsoid pairs."
        )
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Dimensions",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Random samples per dimension",
    )
    parser.add_argument(
        "--biased-samples",
        type=int,
        default=25,
        help="Biased samples per dimension",
    )
    parser.add_argument(
        "--biased-aspect",
        type=float,
        default=250.0,
        help="Maximum aspect ratio for biased ellipsoids",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_CHOICES),
        choices=list(METHOD_CHOICES),
        help="Solver methods to test",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKEND_CHOICES),
        choices=list(BACKEND_CHOICES),
        help="Backends to test",
    )
    parser.add_argument(
        "--baseline-backend",
        default="python",
        choices=list(BACKEND_CHOICES),
        help="Backend used for the reference solution",
    )
    parser.add_argument(
        "--baseline-method",
        default="brentq+newton",
        choices=list(METHOD_CHOICES),
        help="Method used for the reference solution",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for dumping the aggregated metrics as JSON",
    )
    return parser.parse_args(argv)


def ensure_cpp_availability(backends: Sequence[str]) -> list[str]:
    """Filter unavailable backends and warn if needed."""

    filtered: list[str] = []
    cpp_available = has_cpp_backend()
    for backend in backends:
        if backend == "cpp" and not cpp_available:
            print("[warning] Skipping C++ backend: not available", file=sys.stderr)
            continue
        filtered.append(backend)
    return filtered


def evaluate_strategies(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    backends = ensure_cpp_availability(args.backends)
    strategies: dict[tuple[str, str], StrategyStats] = {}
    for backend in backends:
        for method in args.methods:
            label = f"{backend}:{method}"
            strategies[(backend, method)] = StrategyStats(label)

    cases = list(
        generate_cases(
            rng,
            dims=args.dims,
            n_random=args.samples,
            n_biased=args.biased_samples,
            bias_ratio=args.biased_aspect,
        )
    )

    baseline_key = (args.baseline_backend, cast(MethodName, args.baseline_method))
    if baseline_key not in strategies:
        strategies[baseline_key] = StrategyStats(
            f"{args.baseline_backend}:{args.baseline_method}"
        )

    total_cases = len(cases)
    baseline_failures = 0

    for scenario, _dim, pcoef, qcoef in cases:
        try:
            baseline_result = tangency(
                pcoef,
                qcoef,
                backend=args.baseline_backend,
                method=cast(MethodName, args.baseline_method),
            )
        except Exception as exc:  # noqa: BLE001 - logging
            baseline_failures += 1
            print(
                f"[warning] Baseline failed for {scenario}: {exc}",
                file=sys.stderr,
            )
            continue

        reference_t = float(baseline_result.t)
        for (backend, method), stats in strategies.items():
            start = perf_counter()
            try:
                result = tangency(pcoef, qcoef, backend=backend, method=method)
            except Exception as exc:  # noqa: BLE001 - comparison logging
                stats.update_failure(scenario)
                print(
                    f"[warning] Strategy {stats.label} failed for {scenario}: {exc}",
                    file=sys.stderr,
                )
                continue
            runtime = perf_counter() - start
            rel_err = relative_error(reference_t, float(result.t))
            residual = tangency_residual(pcoef, qcoef, result.point)
            stats.update_success(scenario, runtime, rel_err, residual)

    print("=== Strategy Benchmark Summary ===")
    print(f"Total cases: {total_cases} | baseline failures: {baseline_failures}")
    for stats in strategies.values():
        print(f"\nStrategy: {stats.label}")
        overall = stats.overall.summary()
        print(
            "  successes: {cases} | failures: {failures}".format(**overall)
        )
        print(
            (
                "  mean runtime: {mean_runtime_s:.6f}s | median runtime:"
                " {median_runtime_s:.6f}s"
            ).format(**overall)
        )
        print(
            (
                "  rel err (mean/max): {mean_relative_error:.3e} /"
                " {max_relative_error:.3e}"
            ).format(**overall)
        )
        print(
            (
                "  residual (mean/max): {mean_objective_residual:.3e} /"
                " {max_objective_residual:.3e}"
            ).format(**overall)
        )
        for scenario, metrics in sorted(stats.per_scenario.items()):
            scenario_summary = metrics.summary()
            print(
                (
                    "    - {scenario}: {cases} ok | {failures} fail | mean"
                    " runtime {mean_runtime_s:.6f}s | rel err"
                    " {mean_relative_error:.3e}"
                ).format(scenario=scenario, **scenario_summary)
            )

    serialized = [stats.serialize() for stats in strategies.values()]
    if args.output_json:
        args.output_json.write_text(json.dumps(serialized, indent=2))
        print(f"\nWrote JSON report to {args.output_json}")

    return serialized


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate_strategies(args)


if __name__ == "__main__":
    main()
