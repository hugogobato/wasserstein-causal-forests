#!/usr/bin/env python3
"""Numerical validation of the G3 quadrature oracle truth.

Run from the repository root:

    python research/checks/g3_dgp_truth_accuracy.py

`wasserstein_causal_forests.g3.dgps` computes every conditional target by
Gauss-Hermite quadrature over the outer latent pair rather than by Monte Carlo,
so the truth entering an RMSE denominator carries deterministic quadrature
error instead of sampling noise. That is only worth doing if the quadrature
error is negligible. This script measures it: for every regime, arm, and target
it compares the quadrature value against a large Monte Carlo average over the
same outer law.

The comparison is calibrated rather than absolute. The Monte Carlo average is
itself noisy, so a discrepancy is treated as a failure only when it exceeds
both a fixed floor and a multiple of the Monte Carlo's own standard error. A
quadrature rule that agrees to within Monte Carlo noise is as accurate as this
check can certify, and the printed ratio makes that visible.

Exits nonzero when any target fails, and prints a JSON certificate otherwise.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.dgps import DGP_IDS, build_dgp  # noqa: E402
from wasserstein_causal_forests.pta_bcf.targets import (  # noqa: E402
    GRID_FUNCTIONALS,
)

N_POINTS = 100
N_DRAWS = 50_000
N_GRID = 49
#: Floor below which a discrepancy is negligible for tournament purposes.
ABSOLUTE_FLOOR = 5e-3
#: A discrepancy is also admitted when it sits within this many Monte Carlo
#: standard errors, since the reference average is the noisy quantity there.
SE_MULTIPLE = 5.0
TAIL_THRESHOLD = 1.5
SEED = 20260731


class RunningMoments:
    """Per-coordinate mean and standard error of the Monte Carlo average."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.total = np.zeros(shape)
        self.total_squared = np.zeros(shape)
        self.count = 0

    def update(self, value: np.ndarray) -> None:
        self.total += value
        self.total_squared += value * value
        self.count += 1

    @property
    def mean(self) -> np.ndarray:
        return self.total / self.count

    @property
    def standard_error(self) -> np.ndarray:
        variance = np.maximum(self.total_squared / self.count - self.mean**2, 0.0)
        return np.sqrt(variance / self.count)


def monte_carlo_targets(dgp, X: np.ndarray, arm: int, rng: np.random.Generator):
    """Average every target over N_DRAWS independent outer draws per row."""

    weights = dgp.grid.weights
    reference = dgp.grid.reference_quantiles()
    upper = dgp.grid.n_grid - 1
    n_rows = X.shape[0]

    accumulators = {
        "mean_quantiles": RunningMoments((n_rows, dgp.grid.n_grid)),
        "reference_distance": RunningMoments((n_rows,)),
        "tail_probability": RunningMoments((n_rows,)),
    }
    for name in GRID_FUNCTIONALS:
        accumulators[name] = RunningMoments((n_rows,))

    outer = dgp.spec.outer(arm)
    for _ in range(N_DRAWS):
        xi, eta = outer.sample(rng, n_rows)
        grid = dgp._grid_at_latent(X, arm, xi, eta)
        accumulators["mean_quantiles"].update(grid)
        difference = grid - reference
        accumulators["reference_distance"].update(
            np.sqrt(np.sum(weights * difference * difference, axis=-1))
        )
        accumulators["tail_probability"].update(
            (grid[:, upper] > TAIL_THRESHOLD).astype(float)
        )
        for name, function in GRID_FUNCTIONALS.items():
            accumulators[name].update(function(grid, weights))
    return accumulators


def quadrature_targets(dgp, X: np.ndarray, arm: int) -> dict[str, np.ndarray]:
    upper = dgp.grid.n_grid - 1
    targets = {
        "mean_quantiles": dgp.mean_quantiles(X, arm),
        "reference_distance": dgp.reference_distance(X, arm),
        "tail_probability": dgp.tail_probability(
            X, arm, level_index=upper, threshold=TAIL_THRESHOLD
        ),
    }
    for name in GRID_FUNCTIONALS:
        targets[name] = dgp.functional(X, arm, name)
    return targets


def main() -> int:
    started = time.time()
    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for dgp_id in DGP_IDS:
        dgp = build_dgp(dgp_id, N_GRID)
        X = dgp.sample(N_POINTS, seed=SEED).X
        for arm in (0, 1):
            rng = np.random.default_rng(SEED + 1000 * arm + int(dgp_id[1:]))
            accumulators = monte_carlo_targets(dgp, X, arm, rng)
            exact = quadrature_targets(dgp, X, arm)
            for name, quadrature in exact.items():
                accumulator = accumulators[name]
                error = float(np.max(np.abs(quadrature - accumulator.mean)))
                standard_error = float(np.max(accumulator.standard_error))
                budget = max(ABSOLUTE_FLOOR, SE_MULTIPLE * standard_error)
                passed = np.isfinite(error) and error <= budget
                rows.append(
                    {
                        "dgp": dgp_id,
                        "arm": arm,
                        "target": name,
                        "max_abs_error": error,
                        "max_monte_carlo_se": standard_error,
                        "budget": budget,
                        "status": "PASS" if passed else "FAIL",
                    }
                )
                if not passed:
                    failures.append(
                        f"{dgp_id} arm {arm} {name}: max abs error {error:.6f} "
                        f"exceeds budget {budget:.6f}"
                    )
                print(
                    f"{dgp_id} arm {arm} {name:<22} "
                    f"max|quad-mc|={error:.5f}  mc_se={standard_error:.5f}  "
                    f"{'ok' if passed else 'FAIL'}",
                    flush=True,
                )

    certificate = {
        "check": "g3_dgp_truth_accuracy",
        "n_points": N_POINTS,
        "n_draws": N_DRAWS,
        "n_grid": N_GRID,
        "absolute_floor": ABSOLUTE_FLOOR,
        "se_multiple": SE_MULTIPLE,
        "seed": SEED,
        "worst_error": max(float(row["max_abs_error"]) for row in rows),
        "n_comparisons": len(rows),
        "elapsed_seconds": round(time.time() - started, 1),
        "status": "PASS" if not failures else "FAIL",
    }
    if failures:
        certificate["failures"] = failures
    print(json.dumps(certificate, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
