#!/usr/bin/env python
"""Utility script for comparing tangency strategies across backends and methods."""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy
import pandas
from scipy.optimize import root_scalar

from ellphi import _solver_python as py_backend
from ellphi import geometry, solver


@dataclass(frozen=True)
class ScenarioConfig:
    """Test-case configuration capturing the target anisotropy range."""

    name: str
    ratio_range: tuple[float, float]
    center_scale: float = 0.5


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for a backend/method pair (optionally with overrides)."""

    label: str
    backend: str
    method: str
    bracket: tuple[float, float] = (0.0, 1.0)
    x0: float | None = None
    hybrid_brent_maxiter: int | None = None
    hybrid_newton_maxiter: int | None = None

    def requires_custom_python_hybrid(self) -> bool:
        return (
            self.backend == "python"
            and self.method == "brentq+newton"
            and (
                self.hybrid_brent_maxiter is not None
                or self.hybrid_newton_maxiter is not None
            )
        )


@dataclass
class TangencyCase:
    """Generated ellipsoid pair to evaluate."""

    case_id: int
    dim: int
    scenario: str
    pcoef: numpy.ndarray
    qcoef: numpy.ndarray
    p_ratio: float
    q_ratio: float


DEFAULT_SCENARIOS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig("balanced", (1.0, 5.0)),
    ScenarioConfig("biased", (50.0, 500.0), center_scale=1.5),
)

DEFAULT_STRATEGIES: tuple[StrategyConfig, ...] = (
    StrategyConfig("python:hybrid-8+3", "python", "brentq+newton"),
    StrategyConfig(
        "python:hybrid-6+2",
        "python",
        "brentq+newton",
        hybrid_brent_maxiter=6,
        hybrid_newton_maxiter=2,
    ),
    StrategyConfig(
        "python:hybrid-4+2",
        "python",
        "brentq+newton",
        hybrid_brent_maxiter=4,
        hybrid_newton_maxiter=2,
    ),
    StrategyConfig("python:brentq", "python", "brentq"),
    StrategyConfig("python:brenth", "python", "brenth"),
    StrategyConfig("python:bisect", "python", "bisect"),
    StrategyConfig("python:newton-x0=0.5", "python", "newton", x0=0.5),
    StrategyConfig("cpp:hybrid-8+3", "cpp", "brentq+newton"),
    StrategyConfig("cpp:brentq", "cpp", "brentq"),
)

REFERENCE_STRATEGY = StrategyConfig("reference", "python", "brentq+newton")

_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare backend/method combinations across random ellipsoid pairs. "
            "The script records runtime, relative error (vs. the reference "
            "python hybrid solver), divergence counts, and anisotropy sensitivity."
        )
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=(2, 3, 5),
        help="Dimensions to evaluate.",
    )
    parser.add_argument(
        "--cases-per-dim",
        type=int,
        default=25,
        help="Number of random ellipsoid pairs to generate per dimension and scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the NumPy random generator.",
    )
    parser.add_argument(
        "--case-output",
        type=str,
        default=None,
        help="Optional path to write per-case measurements as CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional path to write the aggregated summary as CSV.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        nargs="+",
        choices=[scenario.name for scenario in DEFAULT_SCENARIOS],
        default=[scenario.name for scenario in DEFAULT_SCENARIOS],
        help="Subset of scenarios to run (default: all).",
    )
    return parser.parse_args()


def _random_covariance(
    dim: int, ratio_range: tuple[float, float], rng: numpy.random.Generator
) -> numpy.ndarray:
    target_ratio = rng.uniform(*ratio_range)
    log_ratio = math.log(target_ratio)
    spectrum = numpy.linspace(-0.5, 0.5, dim)
    eigenvalues = numpy.exp(spectrum * 2.0 * log_ratio)
    eigenvalues *= rng.uniform(0.3, 1.7)
    base = rng.uniform(0.5, 1.5, size=dim)
    eigenvalues *= base / base.min()
    Q, _ = numpy.linalg.qr(rng.normal(size=(dim, dim)))
    cov = Q @ numpy.diag(eigenvalues) @ Q.T
    return cov


def _anisotropy_ratio(cov: numpy.ndarray) -> float:
    eigvals = numpy.linalg.eigvalsh(cov)
    return float(numpy.max(eigvals) / numpy.min(eigvals))


def _build_ellipsoid(
    dim: int, scenario: ScenarioConfig, rng: numpy.random.Generator
) -> tuple[numpy.ndarray, float]:
    cov = _random_covariance(dim, scenario.ratio_range, rng)
    center = rng.normal(scale=scenario.center_scale, size=dim)
    coef = geometry.coef_from_cov(center, cov)[0]
    return coef, _anisotropy_ratio(cov)


def build_cases(
    dims: Sequence[int],
    cases_per_dim: int,
    scenarios: Iterable[ScenarioConfig],
    rng: numpy.random.Generator,
) -> list[TangencyCase]:
    cases: list[TangencyCase] = []
    case_id = 0
    for dim in dims:
        for scenario in scenarios:
            for _ in range(cases_per_dim):
                pcoef, p_ratio = _build_ellipsoid(dim, scenario, rng)
                qcoef, q_ratio = _build_ellipsoid(dim, scenario, rng)
                cases.append(
                    TangencyCase(
                        case_id=case_id,
                        dim=dim,
                        scenario=scenario.name,
                        pcoef=pcoef,
                        qcoef=qcoef,
                        p_ratio=p_ratio,
                        q_ratio=q_ratio,
                    )
                )
                case_id += 1
    return cases


def _assemble_tangency_result(
    pcoef: numpy.ndarray, qcoef: numpy.ndarray, mu: float
) -> py_backend.TangencyResult:
    coef = py_backend.pencil(pcoef, qcoef, mu)
    point = py_backend._center(coef)
    value = float(numpy.sqrt(py_backend.quad_eval(coef, point)))
    return py_backend.TangencyResult(value, numpy.asarray(point), mu)


def _solve_python_hybrid(
    pcoef: numpy.ndarray,
    qcoef: numpy.ndarray,
    strategy: StrategyConfig,
) -> py_backend.TangencyResult:
    if strategy.hybrid_brent_maxiter is None or strategy.hybrid_newton_maxiter is None:
        raise ValueError("Hybrid override requested without both stage iterations.")
    curry_f = lambda mu: py_backend._target(mu, pcoef, qcoef)  # noqa: E731
    curry_df = lambda mu: py_backend._target_prime(mu, pcoef, qcoef)  # noqa: E731
    stage1 = root_scalar(
        curry_f,
        method="brentq",
        bracket=strategy.bracket,
        maxiter=strategy.hybrid_brent_maxiter,
    )
    stage2 = root_scalar(
        curry_f,
        method="newton",
        x0=stage1.root,
        fprime=curry_df,
        maxiter=strategy.hybrid_newton_maxiter,
    )
    return _assemble_tangency_result(pcoef, qcoef, float(stage2.root))


def run_strategy(
    case: TangencyCase, strategy: StrategyConfig
) -> py_backend.TangencyResult:
    pcoef = numpy.asarray(case.pcoef, dtype=float)
    qcoef = numpy.asarray(case.qcoef, dtype=float)
    if strategy.requires_custom_python_hybrid():
        return _solve_python_hybrid(pcoef, qcoef, strategy)
    kwargs = dict(
        method=strategy.method, bracket=strategy.bracket, backend=strategy.backend
    )
    if strategy.x0 is not None:
        kwargs["x0"] = strategy.x0
    return solver.tangency(pcoef, qcoef, **kwargs)


def compute_reference(case: TangencyCase) -> py_backend.TangencyResult:
    return solver.tangency(
        case.pcoef,
        case.qcoef,
        method=REFERENCE_STRATEGY.method,
        bracket=REFERENCE_STRATEGY.bracket,
        backend=REFERENCE_STRATEGY.backend,
    )


def evaluate_case(
    case: TangencyCase,
    strategy: StrategyConfig,
    reference: py_backend.TangencyResult,
) -> dict:
    start = time.perf_counter()
    try:
        result = run_strategy(case, strategy)
        runtime = (time.perf_counter() - start) * 1000.0
        rel_error = abs(result.t - reference.t) / max(abs(reference.t), _EPS)
        success = True
        message = ""
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        runtime = numpy.nan
        rel_error = numpy.nan
        success = False
        message = str(exc)
    return {
        "case_id": case.case_id,
        "dim": case.dim,
        "scenario": case.scenario,
        "strategy": strategy.label,
        "backend": strategy.backend,
        "method": strategy.method,
        "runtime_ms": runtime,
        "rel_error": rel_error,
        "success": success,
        "divergence_reason": message,
        "p_ratio": case.p_ratio,
        "q_ratio": case.q_ratio,
        "max_ratio": max(case.p_ratio, case.q_ratio),
    }


def summarize(results: pandas.DataFrame) -> pandas.DataFrame:
    grouped = results.groupby(
        ["strategy", "backend", "method", "dim", "scenario"],
        dropna=False,
    )
    summary = grouped.agg(
        cases=("success", "size"),
        success_rate=("success", "mean"),
        divergences=("success", lambda values: int((~values).sum())),
        mean_rel_error=("rel_error", "mean"),
        median_rel_error=("rel_error", "median"),
        max_rel_error=("rel_error", "max"),
        mean_runtime_ms=("runtime_ms", "mean"),
        median_runtime_ms=("runtime_ms", "median"),
        mean_max_ratio=("max_ratio", "mean"),
    )
    return summary.reset_index().sort_values(
        ["strategy", "dim", "scenario"],
        ignore_index=True,
    )


def main() -> None:
    args = parse_args()
    rng = numpy.random.default_rng(args.seed)
    scenario_lookup = {scenario.name: scenario for scenario in DEFAULT_SCENARIOS}
    selected_scenarios = [scenario_lookup[name] for name in args.scenario]

    cases = build_cases(args.dims, args.cases_per_dim, selected_scenarios, rng)
    print(
        f"Generated {len(cases)} cases across {len(args.dims)} dimensions "
        f"and {len(selected_scenarios)} scenarios."
    )

    references: dict[int, py_backend.TangencyResult] = {}
    valid_cases: list[TangencyCase] = []
    for case in cases:
        try:
            references[case.case_id] = compute_reference(case)
            valid_cases.append(case)
        except Exception as exc:  # pragma: no cover - diagnostic path
            case_info = (
                f"[skip] case {case.case_id} (dim={case.dim}, "
                f"scenario={case.scenario})"
            )
            print(f"{case_info} failed reference computation: {exc}")
    if not valid_cases:
        raise RuntimeError("No valid cases generated. Adjust the configuration.")

    strategies = [
        s for s in DEFAULT_STRATEGIES if s.backend != "cpp" or solver.has_cpp_backend()
    ]
    if not any(s.backend == "cpp" for s in strategies):
        print("[info] C++ backend unavailable; skipping related strategies.")

    rows = []
    for case in valid_cases:
        reference = references[case.case_id]
        for strategy in strategies:
            rows.append(evaluate_case(case, strategy, reference))

    df = pandas.DataFrame(rows)
    summary = summarize(df)

    print("\n=== Aggregated summary by dimension & scenario ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    overall = summary.groupby(["strategy", "backend", "method"], dropna=False).agg(
        total_cases=("cases", "sum"),
        mean_success_rate=("success_rate", "mean"),
        mean_runtime_ms=("mean_runtime_ms", "mean"),
        mean_rel_error=("mean_rel_error", "mean"),
    )
    print("\n=== Overall summary (averaged across dimensions/scenarios) ===")
    print(
        overall.reset_index().to_string(index=False, float_format=lambda v: f"{v:0.4f}")
    )

    if args.case_output:
        df.to_csv(args.case_output, index=False)
        print(f"Per-case results saved to {args.case_output}")
    if args.summary_output:
        summary.to_csv(args.summary_output, index=False)
        print(f"Summary saved to {args.summary_output}")


if __name__ == "__main__":
    main()
