#!/usr/bin/env python3
"""Cheap design check for symmetric, nonlinear stress regimes.

This probe does not fit an estimator and is not a decisive simulation result.
It verifies that a proposed symmetric-outcome/nonlinear-assignment DGP has
nontrivial treatment imbalance, positive overlap, and the intended absence of
inner right skewness before it is added to a frozen manifest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.dgps import (  # noqa: E402
    DGPSpec,
    DistributionalDGP,
    GridSpec,
    OuterLaw,
)


def _clip_logit(index: np.ndarray) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.05, 0.95)


def build_symmetric_nonlinear() -> DistributionalDGP:
    """Symmetric conditional law, nonlinear prognostic surface and propensity."""

    def location(x: np.ndarray, arm: int) -> np.ndarray:
        baseline = (
            0.70 * np.sin(np.pi * x[:, 0])
            + 0.35 * x[:, 1] * x[:, 2]
            + 0.20 * np.cos(np.pi * x[:, 3])
        )
        effect = 0.25 + 0.15 * np.sin(np.pi * x[:, 1])
        return baseline + arm * effect

    def log_scale(x: np.ndarray, arm: int) -> np.ndarray:
        return 0.12 + 0.08 * x[:, 4] ** 2 - 0.05 * x[:, 2]

    def symmetric_shape(x: np.ndarray, arm: int) -> np.ndarray:
        return np.zeros(x.shape[0])

    def nonlinear_propensity(x: np.ndarray) -> np.ndarray:
        index = (
            1.60 * np.sin(np.pi * x[:, 0])
            + 0.80 * x[:, 1] * x[:, 2]
            - 0.70 * x[:, 3] ** 2
            + 0.35 * x[:, 4]
        )
        return _clip_logit(index)

    spec = DGPSpec(
        dgp_id="SYM-NL-PROP",
        description="symmetric inner law with nonlinear outcome and assignment surfaces",
        location=location,
        log_scale=log_scale,
        shape=symmetric_shape,
        # No outer scale shock: the full conditional law remains symmetric.
        outer=lambda arm: OuterLaw(location_sd=0.25, log_scale_sd=0.0),
        propensity=nonlinear_propensity,
        n_features=6,
    )
    return DistributionalDGP(spec, GridSpec(25))


def summarize(dgp: DistributionalDGP, *, n_rows: int = 100_000, seed: int = 20260909) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_rows, dgp.spec.n_features))
    propensity = dgp.spec.propensity(x)
    if not np.all((propensity >= 0.0) & (propensity <= 1.0)):
        raise AssertionError("propensity left [0, 1]")
    treatment = rng.binomial(1, propensity)
    result: dict[str, object] = {
        "dgp": dgp.spec.dgp_id,
        "n_rows": n_rows,
        "propensity_mean": float(propensity.mean()),
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "treated_fraction": float(treatment.mean()),
        "standardized_mean_differences": {},
        "grid_skewness": {},
    }
    smd: dict[str, float] = {}
    skewness: dict[str, float] = {}
    for j in range(x.shape[1]):
        x1 = x[treatment == 1, j]
        x0 = x[treatment == 0, j]
        pooled = np.sqrt(0.5 * (x1.var() + x0.var()))
        smd[f"x{j + 1}"] = float((x1.mean() - x0.mean()) / pooled)
    for arm in (0, 1):
        q = dgp._grid_at_latent(
            x, arm, np.zeros(n_rows, dtype=float), np.zeros(n_rows, dtype=float)
        )
        if not np.all(np.diff(q, axis=1) > 0.0):
            raise AssertionError("generated quantile vectors are not monotone")
        weights = dgp.grid.weights
        mean = (q * weights).sum(axis=1)
        sd = np.sqrt(((q - mean[:, None]) ** 2 * weights).sum(axis=1))
        pointwise = ((q - mean[:, None]) ** 3 * weights).sum(axis=1) / (sd**3)
        if np.max(np.abs(pointwise)) > 1e-10:
            raise AssertionError("strict symmetric probe has nonzero grid skewness")
        skewness[str(arm)] = float(pointwise.mean())
    result["standardized_mean_differences"] = smd
    result["grid_skewness"] = skewness
    return result


def main() -> int:
    output = ROOT / "results" / "dgp_stress_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize(build_symmetric_nonlinear())
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
