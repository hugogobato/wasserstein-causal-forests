"""Illustrative diagnostics for the three WP3 coordinate-scaling rules.

The frozen ``robust_sd`` rule is justified by its Gaussian-consistent robust
scale and sample-size-stable population limit.  This one-seed diagnostic is
not treated as a data-driven model-selection experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from wp3_odcf import CoordinateScaler, trapezoidal_grid_weights  # noqa: E402


def calibration_data(seed: int = 20260727, n: int = 240, K: int = 49, J: int = 3):
    rng = np.random.default_rng(seed)
    curve = rng.normal(size=(n, K)) * np.linspace(0.3, 1.0, K)[None, :]
    functionals = np.c_[
        rng.lognormal(size=n),
        0.4 * rng.normal(size=n),
        0.1 * rng.normal(size=n),
    ][:, :J]
    scores = np.c_[curve, functionals]
    treatment = rng.binomial(1, 0.5, size=n)
    propensity = np.full(n, 0.5)
    return scores, treatment, propensity


def compare_scaling_rules() -> dict[str, dict[str, float]]:
    scores, treatment, propensity = calibration_data()
    K = 49
    weights = trapezoidal_grid_weights(K)
    duplicate_scores = np.c_[np.repeat(scores[:, :K], 2, axis=1), scores[:, K:]]
    duplicate_weights = np.repeat(weights / 2.0, 2)
    results = {}
    for rule in ("robust_sd", "mad", "null_score_se"):
        scaler = CoordinateScaler.fit(
            scores, K, weights, rule, treatment=treatment, propensity=propensity
        )
        scaled = scaler.transform(scores)
        contributions = np.r_[
            np.dot(weights, np.var(scaled[:, :K], axis=0)),
            np.var(scaled[:, K:], axis=0),
        ]
        half = CoordinateScaler.fit(
            scores[: len(scores) // 2],
            K,
            weights,
            rule,
            treatment=treatment[: len(scores) // 2],
            propensity=propensity[: len(scores) // 2],
        )
        duplicate = CoordinateScaler.fit(
            duplicate_scores,
            2 * K,
            duplicate_weights,
            rule,
            treatment=treatment,
            propensity=propensity,
        )
        scalar_scale = scaler.scales[K:]
        half_instability = np.linalg.norm(scalar_scale - half.scales[K:]) / max(
            np.linalg.norm(scalar_scale), 1e-12
        )
        duplicate_instability = np.linalg.norm(
            scalar_scale - duplicate.scales[2 * K :]
        ) / max(np.linalg.norm(scalar_scale), 1e-12)
        balance_cv = float(np.std(contributions) / max(np.mean(contributions), 1e-12))
        results[rule] = {
            "balance_cv": balance_cv,
            "half_sample_instability": float(half_instability),
            "duplicate_grid_instability": float(duplicate_instability),
            # Sample-size stability is weighted twice because the scale must
            # not silently change when n changes between training folds.
            "selection_score": float(
                balance_cv + 2.0 * half_instability + duplicate_instability
            ),
        }
    return results


def main():
    results = compare_scaling_rules()
    diagnostic_winner = min(results, key=lambda rule: results[rule]["selection_score"])
    print("WP3 illustrative scaling diagnostics")
    for rule, metrics in results.items():
        print(rule, *(f"{key}={value:.6g}" for key, value in metrics.items()))
    print("diagnostic_lowest_score=", diagnostic_winner)
    print("frozen_rule= robust_sd")


if __name__ == "__main__":
    main()
