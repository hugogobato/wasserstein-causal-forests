"""Oracle distribution-valued DGPs used by the PTA smoke and crossover cells.

Each unit carries a monotone grid vector

    q_k = loc_a(x) + sigma_outer * e + scale_a(x) * z_k,

with z_k the standard normal quantile at the declared grid point. The regimes
differ only in which covariates moderate the location and scale effects:

* `null` (D2)     zero treatment effect on every target coordinate;
* `shared` (D4)   one moderator drives every coordinate of U;
* `separate` (D3) location and scale effects have different moderators, so the
  coordinates of U disagree about which covariate matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from .targets import TargetManifest, uniform_grid_manifest

REGIMES = ("null", "shared", "separate")

DEFAULT_FUNCTIONALS = ("grid_mean", "grid_sd")


@dataclass(frozen=True)
class DGPConfiguration:
    n_features: int = 5
    sigma_outer: float = 0.25
    baseline_scale: float = 0.0
    confounding: float = 0.8


def reference_quantiles(n_grid: int) -> NDArray[np.float64]:
    """Frozen external reference law nu_star: the standard normal."""

    points = (np.arange(n_grid, dtype=float) + 0.5) / n_grid
    return norm.ppf(points)


def pta_manifest(
    n_grid: int = 5,
    *,
    functionals: tuple[str, ...] = DEFAULT_FUNCTIONALS,
    with_reference: bool = True,
) -> TargetManifest:
    """Manifest with D = K + J + 1 coordinates on the midpoint grid."""

    return uniform_grid_manifest(
        n_grid,
        functionals=functionals,
        reference_quantiles=reference_quantiles(n_grid) if with_reference else None,
    )


def _location_effect(X: NDArray[np.float64], regime: str) -> NDArray[np.float64]:
    if regime == "null":
        return np.zeros(X.shape[0])
    if regime == "shared":
        return 0.9 * X[:, 0]
    return 0.9 * X[:, 1]


def _log_scale_effect(X: NDArray[np.float64], regime: str) -> NDArray[np.float64]:
    if regime == "null":
        return np.zeros(X.shape[0])
    if regime == "shared":
        return 0.30 * X[:, 0]
    return -0.30 * X[:, 2]


def _prognostic(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.7 * np.sin(np.pi * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]


def _log_scale_prognostic(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.20 * X[:, 3]


def propensity(
    X: NDArray[np.float64], configuration: DGPConfiguration = DGPConfiguration()
) -> NDArray[np.float64]:
    """Confounded assignment probability with comfortable overlap."""

    linear = configuration.confounding * (X[:, 0] + 0.5 * X[:, 1])
    return np.clip(1.0 / (1.0 + np.exp(-linear)), 0.1, 0.9)


def _grid_z(n_grid: int) -> NDArray[np.float64]:
    return reference_quantiles(n_grid)


def draw_quantiles(
    X: NDArray[np.float64],
    arm: NDArray[np.int64],
    regime: str,
    n_grid: int,
    rng: np.random.Generator,
    configuration: DGPConfiguration = DGPConfiguration(),
) -> NDArray[np.float64]:
    """Return the observed monotone grid vectors, shape (n, K)."""

    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    a = np.asarray(arm, dtype=float)
    location = _prognostic(X) + a * _location_effect(X, regime)
    log_scale = (
        configuration.baseline_scale
        + _log_scale_prognostic(X)
        + a * _log_scale_effect(X, regime)
    )
    outer = configuration.sigma_outer * rng.normal(size=X.shape[0])
    return (
        (location + outer)[:, None] + np.exp(log_scale)[:, None] * _grid_z(n_grid)
    )


def sample_dataset(
    n_rows: int,
    regime: str,
    seed: int,
    *,
    n_grid: int = 5,
    configuration: DGPConfiguration = DGPConfiguration(),
) -> dict[str, NDArray[np.float64]]:
    """Sample one oracle dataset from the requested regime."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_rows, configuration.n_features))
    scores = propensity(X, configuration)
    treatment = rng.binomial(1, scores)
    if treatment.sum() < 2 or (1 - treatment).sum() < 2:
        treatment[:2] = [0, 1]
    quantiles = draw_quantiles(
        X, treatment, regime, n_grid, rng, configuration
    )
    return {
        "X": X,
        "treatment": treatment,
        "quantiles": quantiles,
        "propensity": scores,
    }


def true_target_means(
    X: NDArray[np.float64],
    arm: int,
    regime: str,
    manifest: TargetManifest,
    *,
    n_monte_carlo: int = 400,
    seed: int = 12345,
    configuration: DGPConfiguration = DGPConfiguration(),
) -> NDArray[np.float64]:
    """Monte Carlo evaluation of E[U(Y^a) | X=x], shape (n, D).

    The outer location noise enters the reference-distance coordinate
    nonlinearly, so this integral is taken numerically rather than in closed
    form.
    """

    rng = np.random.default_rng(seed)
    arms = np.full(X.shape[0], int(arm), dtype=np.int64)
    total = np.zeros((X.shape[0], manifest.dimension))
    for _ in range(n_monte_carlo):
        quantiles = draw_quantiles(
            X, arms, regime, manifest.n_grid, rng, configuration
        )
        total += manifest.build(quantiles)
    return total / n_monte_carlo


def true_target_contrast(
    X: NDArray[np.float64],
    regime: str,
    manifest: TargetManifest,
    *,
    n_monte_carlo: int = 400,
    seed: int = 12345,
    configuration: DGPConfiguration = DGPConfiguration(),
) -> NDArray[np.float64]:
    """Monte Carlo evaluation of the conditional target contrast, shape (n, D)."""

    treated = true_target_means(
        X,
        1,
        regime,
        manifest,
        n_monte_carlo=n_monte_carlo,
        seed=seed,
        configuration=configuration,
    )
    control = true_target_means(
        X,
        0,
        regime,
        manifest,
        n_monte_carlo=n_monte_carlo,
        seed=seed + 1,
        configuration=configuration,
    )
    return treated - control
