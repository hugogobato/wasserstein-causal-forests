#!/usr/bin/env python3
"""Design checks for the symmetric, nonlinear-assignment DGP family.

No estimator is fit here. For every non-null SYM regime the script draws one
large deterministic Monte Carlo sample and reports the assignment diagnostics
that the sensitivity report requires before any estimator score is read:
propensity range and mean, treated fraction, raw-covariate standardized mean
differences, the same differences for nonlinear prognostic features, per-arm
pointwise grid skewness, and arm sample counts. It also measures the
propensity-marginal agreement of the SYM-ALIGN / SYM-IRREL pair, whose indices
are functions of the independent coordinates x1 and x6.

Run from the repository root:

    python3 research/checks/wcf_sensitivity_design_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.dgps import (  # noqa: E402
    DGPSample,
    DistributionalDGP,
    build_dgp,
)
from wasserstein_causal_forests.g3.sensitivity_dgps import (  # noqa: E402
    SENSITIVITY_ALIGNMENT_DGPS,
    SENSITIVITY_BASE_DGPS,
    SENSITIVITY_DGPS,
    outcome_log_scale,
    outcome_mu,
    outcome_tau,
    register_sensitivity_dgps,
)

SEED = 20260911
N_ROWS = 200_000
N_ROWS_MONOTONE = 2000
N_GRID = 25
MAX_GRID_SKEWNESS = 1e-8
ALIGNMENT_MAX_ABS_DIFF = 0.01
CLIP_LOW = 0.05
CLIP_HIGH = 0.95
CONSTANT_PROPENSITY = {
    "SYM-RANDOM": 0.5,
    "SYM-RANDOM03": 0.3,
    "SYM-RANDOM-NULL": 0.5,
    "SYM-RANDOM03-NULL": 0.3,
}
OUTPUT = ROOT / "results" / "wcf_sensitivity" / "design_checks.json"


def _smd(values: NDArray[np.float64], treatment: NDArray[np.int64]) -> float:
    """Standardized mean difference (treated minus control) of one feature."""

    treated = values[treatment == 1]
    control = values[treatment == 0]
    pooled = np.sqrt(0.5 * (treated.var() + control.var()))
    return float((treated.mean() - control.mean()) / pooled)


def _pointwise_skewness(
    quantiles: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-row weighted third central moment over the grid coordinates."""

    mean = (quantiles * weights).sum(axis=1, keepdims=True)
    centred = quantiles - mean
    variance = ((centred**2) * weights).sum(axis=1)
    third = ((centred**3) * weights).sum(axis=1)
    return third / variance**1.5


def _nonlinear_features(X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    return {
        "sin_pi_x1": np.sin(np.pi * X[:, 0]),
        "x2_x3": X[:, 1] * X[:, 2],
        "x4_sq": X[:, 3] ** 2,
        "cos_pi_x4": np.cos(np.pi * X[:, 3]),
        "mu": outcome_mu(X),
        "tau": outcome_tau(X),
        "s": outcome_log_scale(X),
    }


def _check_propensity_bounds(
    dgp_id: str, propensity: NDArray[np.float64]
) -> None:
    expected = CONSTANT_PROPENSITY.get(dgp_id)
    if expected is not None:
        assert np.array_equal(
            propensity, np.full(propensity.shape, expected)
        ), f"{dgp_id}: constant propensity is not exactly {expected}"
        return
    assert np.all(propensity >= CLIP_LOW), f"{dgp_id}: propensity below {CLIP_LOW}"
    assert np.all(propensity <= CLIP_HIGH), f"{dgp_id}: propensity above {CLIP_HIGH}"


def _regime_report(dgp: DistributionalDGP, sample: DGPSample) -> dict[str, object]:
    X = sample.X
    treatment = sample.treatment
    weights = dgp.grid.weights

    assert np.all(
        np.diff(sample.quantiles, axis=1) > 0.0
    ), f"{sample.dgp_id}: sampled quantile vectors are not strictly increasing"
    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))
    assert n_treated >= 2 and n_control >= 2, f"{sample.dgp_id}: an arm is empty"

    raw_smd = {f"x{j + 1}": _smd(X[:, j], treatment) for j in range(X.shape[1])}
    feature_smd = {
        name: _smd(values, treatment)
        for name, values in _nonlinear_features(X).items()
    }
    all_smd = {**raw_smd, **feature_smd}
    largest = max(all_smd, key=lambda name: abs(all_smd[name]))

    skewness: dict[str, dict[str, float]] = {}
    for arm in (0, 1):
        quantiles = sample.quantiles[treatment == arm]
        pointwise = _pointwise_skewness(quantiles, weights)
        max_abs = float(np.max(np.abs(pointwise)))
        assert (
            max_abs < MAX_GRID_SKEWNESS
        ), f"{sample.dgp_id} arm {arm}: grid skewness {max_abs} exceeds bound"
        skewness[str(arm)] = {"mean": float(pointwise.mean()), "max_abs": max_abs}

    return {
        "description": dgp.spec.description,
        "n_features": dgp.spec.n_features,
        "propensity_mean": float(sample.propensity.mean()),
        "propensity_min": float(sample.propensity.min()),
        "propensity_max": float(sample.propensity.max()),
        "treated_fraction": float(treatment.mean()),
        "arm_counts": {"0": n_control, "1": n_treated},
        "smd_raw_covariates": raw_smd,
        "smd_nonlinear_features": feature_smd,
        "largest_abs_smd": {"feature": largest, "value": all_smd[largest]},
        "grid_pointwise_skewness": skewness,
    }


def run_checks() -> dict[str, object]:
    register_sensitivity_dgps()

    regimes: dict[str, dict[str, object]] = {}
    alignment_propensities: dict[str, NDArray[np.float64]] = {}
    for dgp_id in SENSITIVITY_BASE_DGPS:
        dgp = build_dgp(dgp_id, N_GRID)
        sample = dgp.sample(N_ROWS, seed=SEED)
        _check_propensity_bounds(dgp_id, sample.propensity)
        regimes[dgp_id] = _regime_report(dgp, sample)
        if dgp_id in SENSITIVITY_ALIGNMENT_DGPS:
            alignment_propensities[dgp_id] = sample.propensity.copy()

    for dgp_id in SENSITIVITY_DGPS:
        dgp = build_dgp(dgp_id, N_GRID)
        sample = dgp.sample(N_ROWS_MONOTONE, seed=SEED)
        _check_propensity_bounds(dgp_id, sample.propensity)
        assert np.all(
            np.diff(sample.quantiles, axis=1) > 0.0
        ), f"{dgp_id}: sampled quantile vectors are not strictly increasing"
        assert sample.treatment.sum() >= 2 and (
            1 - sample.treatment
        ).sum() >= 2, f"{dgp_id}: an arm has fewer than two rows"

    aligned = np.sort(alignment_propensities["SYM-ALIGN"])
    irrelevant = np.sort(alignment_propensities["SYM-IRREL"])
    alignment_max_abs_diff = float(np.max(np.abs(aligned - irrelevant)))
    assert (
        alignment_max_abs_diff < ALIGNMENT_MAX_ABS_DIFF
    ), "SYM-ALIGN and SYM-IRREL propensity marginals disagree"

    return {
        "seed": SEED,
        "n_rows": N_ROWS,
        "n_rows_monotone": N_ROWS_MONOTONE,
        "n_grid": N_GRID,
        "regimes": regimes,
        "alignment_marginal_max_abs_diff": alignment_max_abs_diff,
        "checks_passed": True,
    }


def _print_summary(payload: dict[str, object]) -> None:
    regimes = payload["regimes"]
    assert isinstance(regimes, dict)

    print("SYM sensitivity design checks")
    print(
        f"seed={payload['seed']}  n_rows={payload['n_rows']}  "
        f"n_rows_monotone={payload['n_rows_monotone']}  n_grid={payload['n_grid']}"
    )
    print()
    print("Priority diagnostics (read before estimator scores):")
    for dgp_id in ("SYM-NL", "SYM-MU"):
        report = regimes[dgp_id]
        largest = report["largest_abs_smd"]
        print(
            f"  {dgp_id:11s} propensity [{report['propensity_min']:.3f}, "
            f"{report['propensity_max']:.3f}]  "
            f"treated {report['treated_fraction']:.3f}  "
            f"largest |SMD| {abs(largest['value']):.3f} ({largest['feature']})"
        )
    print()
    print("All non-null regimes:")
    for dgp_id, report in regimes.items():
        largest = report["largest_abs_smd"]
        print(
            f"  {dgp_id:11s} propensity [{report['propensity_min']:.3f}, "
            f"{report['propensity_max']:.3f}]  mean {report['propensity_mean']:.3f}  "
            f"treated {report['treated_fraction']:.3f}  "
            f"largest |SMD| {abs(largest['value']):.3f} ({largest['feature']})"
        )
    print()
    print(
        "SYM-ALIGN vs SYM-IRREL propensity marginal max abs diff: "
        f"{payload['alignment_marginal_max_abs_diff']:.6f}"
    )


def main() -> int:
    payload = run_checks()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(payload)
    print(f"checks_passed={payload['checks_passed']}  wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
