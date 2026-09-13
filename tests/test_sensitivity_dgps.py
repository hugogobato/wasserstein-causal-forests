"""Structural invariants of the SYM sensitivity DGP family.

The family exists to separate assignment alignment from confounding, so the
tests pin the properties that make that separation valid: one shared symmetric
outcome law, exact constant controls, null companions whose two arms coincide,
and an irrelevant-index regime that truly ignores x6. The Monte Carlo balance
numbers are reported by `research/checks/wcf_sensitivity_design_checks.py`;
here they are checked on a small draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.common.quantiles import validate_quantiles
from wasserstein_causal_forests.g3.dgps import DGPSample, GridSpec, build_dgp
from wasserstein_causal_forests.g3.sensitivity_dgps import (
    SENSITIVITY_ALIGNMENT_DGPS,
    SENSITIVITY_BASE_DGPS,
    SENSITIVITY_CONTROL_DGPS,
    SENSITIVITY_DGPS,
    SENSITIVITY_NULL_DGPS,
    SENSITIVITY_PRIMARY_DGPS,
    build_sensitivity_specs,
    register_sensitivity_dgps,
)

register_sensitivity_dgps()

_SAMPLE_CACHE: dict[tuple[str, int, int, int], DGPSample] = {}

_CONSTANT_PROPENSITY = {
    "SYM-RANDOM": 0.5,
    "SYM-RANDOM-NULL": 0.5,
    "SYM-RANDOM03": 0.3,
    "SYM-RANDOM03-NULL": 0.3,
}


def _sample(
    dgp_id: str, n_grid: int = 25, n_rows: int = 2000, seed: int = 20260912
) -> DGPSample:
    key = (dgp_id, n_grid, n_rows, seed)
    if key not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[key] = build_dgp(dgp_id, n_grid).sample(n_rows, seed=seed)
    return _SAMPLE_CACHE[key]


@pytest.mark.parametrize("dgp_id", SENSITIVITY_DGPS)
def test_every_regime_builds_on_the_declared_grids(dgp_id: str) -> None:
    for n_grid in (5, 25):
        dgp = build_dgp(dgp_id, n_grid)
        assert dgp.grid.n_grid == n_grid
        assert type(dgp.grid) is GridSpec
        assert dgp.spec.n_features == 6


@pytest.mark.parametrize("dgp_id", SENSITIVITY_DGPS)
def test_sampled_quantiles_are_strictly_increasing(dgp_id: str) -> None:
    sample = _sample(dgp_id)
    assert np.all(np.diff(sample.quantiles, axis=1) > 0.0)
    validate_quantiles(sample.quantiles, 25)
    assert sample.treatment.sum() >= 2
    assert (1 - sample.treatment).sum() >= 2


@pytest.mark.parametrize("dgp_id", SENSITIVITY_DGPS)
def test_inner_grid_vectors_are_pointwise_symmetric(dgp_id: str) -> None:
    dgp = build_dgp(dgp_id, 25)
    sample = _sample(dgp_id)
    weights = dgp.grid.weights
    for arm in (0, 1):
        quantiles = sample.quantiles[sample.treatment == arm]
        mean = (quantiles * weights).sum(axis=1, keepdims=True)
        odd_third_moment = ((quantiles - mean) ** 3 * weights).sum(axis=1)
        assert np.max(np.abs(odd_third_moment)) < 1e-8


@pytest.mark.parametrize("dgp_id", SENSITIVITY_NULL_DGPS)
def test_null_companions_share_both_arm_laws(dgp_id: str) -> None:
    dgp = build_dgp(dgp_id, 25)
    X = _sample(dgp_id).X
    assert dgp.spec.null_effect
    assert dgp.mean_quantiles(X, 0) == pytest.approx(
        dgp.mean_quantiles(X, 1), abs=1e-12
    )
    xi = np.linspace(-0.5, 0.5, X.shape[0])
    eta = np.zeros(X.shape[0])
    assert dgp._grid_at_latent(X, 0, xi, eta) == pytest.approx(
        dgp._grid_at_latent(X, 1, xi, eta), abs=1e-12
    )


@pytest.mark.parametrize("dgp_id", SENSITIVITY_DGPS)
def test_propensity_respects_the_declared_bounds(dgp_id: str) -> None:
    spec = build_sensitivity_specs()[dgp_id]
    X = np.random.default_rng(7).uniform(-1.0, 1.0, size=(4096, 6))
    propensity = spec.propensity(X)
    expected = _CONSTANT_PROPENSITY.get(dgp_id)
    if expected is None:
        assert np.all(propensity >= 0.05)
        assert np.all(propensity <= 0.95)
    else:
        assert np.array_equal(propensity, np.full(X.shape[0], expected))


def test_registration_is_idempotent_and_covers_every_id() -> None:
    register_sensitivity_dgps()
    register_sensitivity_dgps()
    assert set(build_sensitivity_specs()) == set(SENSITIVITY_DGPS)
    assert SENSITIVITY_NULL_DGPS == tuple(
        f"{dgp_id}-NULL" for dgp_id in SENSITIVITY_BASE_DGPS
    )
    assert set(SENSITIVITY_PRIMARY_DGPS) <= set(SENSITIVITY_BASE_DGPS)
    assert set(SENSITIVITY_ALIGNMENT_DGPS) <= set(SENSITIVITY_BASE_DGPS)
    assert set(SENSITIVITY_CONTROL_DGPS) <= set(SENSITIVITY_BASE_DGPS)
    for dgp_id in SENSITIVITY_DGPS:
        assert build_dgp(dgp_id, 5).spec.dgp_id == dgp_id


def test_irrelevant_regime_ignores_x6_in_every_outcome_surface() -> None:
    spec = build_sensitivity_specs()["SYM-IRREL"]
    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, size=(256, 6))
    permuted = X.copy()
    permuted[:, 5] = X[rng.permutation(X.shape[0]), 5]
    for arm in (0, 1):
        assert np.array_equal(spec.location(X, arm), spec.location(permuted, arm))
        assert np.array_equal(spec.shape(X, arm), spec.shape(permuted, arm))
    assert np.array_equal(spec.log_scale(X, 0), spec.log_scale(permuted, 0))
    # The irrelevant coordinate must still drive assignment, or the alignment
    # contrast would collapse into the aligned regime.
    assert not np.array_equal(spec.propensity(X), spec.propensity(permuted))


def test_aligned_regime_ignores_the_irrelevant_coordinate() -> None:
    spec = build_sensitivity_specs()["SYM-ALIGN"]
    rng = np.random.default_rng(13)
    X = rng.uniform(-1.0, 1.0, size=(256, 6))
    permuted = X.copy()
    permuted[:, 5] = X[rng.permutation(X.shape[0]), 5]
    assert np.array_equal(spec.propensity(X), spec.propensity(permuted))
