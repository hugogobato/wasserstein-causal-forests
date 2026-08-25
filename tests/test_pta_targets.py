"""WP2-B1 verification for the PTA target contract and scale manifest."""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.geometry import weighted_distance
from wasserstein_causal_forests.pta_bcf.targets import (
    ScaleManifest,
    TargetManifest,
    assert_disjoint,
    make_folds,
    uniform_grid_manifest,
)


def _sample_quantiles(n_rows: int, n_grid: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    location = rng.normal(size=n_rows)
    scale = np.exp(0.3 * rng.normal(size=n_rows))
    template = np.linspace(-1.5, 1.5, n_grid)
    return location[:, None] + scale[:, None] * template


def test_target_vector_matches_direct_recomputation():
    n_grid = 5
    manifest = uniform_grid_manifest(
        n_grid,
        functionals=("grid_mean", "grid_sd"),
        reference_quantiles=np.linspace(-1.0, 1.0, n_grid),
    )
    quantiles = _sample_quantiles(40, n_grid)
    U = manifest.build(quantiles)

    assert manifest.dimension == n_grid + 2 + 1
    assert U.shape == (40, manifest.dimension)
    np.testing.assert_array_equal(U, manifest.build(quantiles))
    np.testing.assert_allclose(U[:, manifest.quantile_slice], quantiles)

    weights = manifest.weights
    expected_mean = quantiles @ weights
    centered = quantiles - expected_mean[:, None]
    expected_sd = np.sqrt((centered**2) @ weights)
    np.testing.assert_allclose(U[:, n_grid], expected_mean)
    np.testing.assert_allclose(U[:, n_grid + 1], expected_sd)

    expected_reference = weighted_distance(
        quantiles,
        np.broadcast_to(manifest.reference_quantiles, quantiles.shape),
        weights,
    )
    np.testing.assert_allclose(U[:, manifest.reference_index], expected_reference)


def test_target_identifiers_separate_level_and_contrast():
    manifest = uniform_grid_manifest(
        3, functionals=("grid_sd",), reference_quantiles=np.array([-1.0, 0.0, 1.0])
    )
    assert manifest.column_names == ("q1", "q2", "q3", "grid_sd", "reference_distance")
    assert manifest.blocks == (
        "quantile",
        "quantile",
        "quantile",
        "functional",
        "reference",
    )
    assert manifest.level_target_ids[-2:] == ("TATE-K-j:grid_sd", "REF-A-K")
    assert manifest.contrast_target_ids[-2:] == (
        "TCATE-K-j:grid_sd",
        "REF-TCATE-K",
    )
    # No continuum identifier is emitted from a finite grid.
    assert all("OUT" not in name for name in manifest.level_target_ids)
    assert all(name not in ("REF-ATE", "REF-TCATE") for name in manifest.contrast_target_ids)


def test_scalar_reduction_is_the_identity():
    manifest = uniform_grid_manifest(1)
    assert manifest.dimension == 1
    assert manifest.n_functionals == 0
    assert not manifest.has_reference
    quantiles = _sample_quantiles(12, 1, seed=3)
    np.testing.assert_array_equal(manifest.build(quantiles), quantiles)


def test_manifest_rejects_invalid_declarations():
    with pytest.raises(ValueError):
        TargetManifest(weights=np.array([0.5, 0.6]))
    with pytest.raises(ValueError):
        TargetManifest(weights=np.array([0.5, 0.5]), functionals=("not_a_functional",))
    with pytest.raises(ValueError):
        TargetManifest(weights=np.array([0.5, 0.5]), functionals=("grid_sd", "grid_sd"))
    with pytest.raises(ValueError):
        uniform_grid_manifest(2).build(np.array([[1.0, 0.0]]))


def test_scale_manifest_uses_training_rows_only():
    manifest = uniform_grid_manifest(4, functionals=("grid_mean",))
    train = manifest.build(_sample_quantiles(60, 4, seed=1))
    evaluation = manifest.build(_sample_quantiles(30, 4, seed=2) * 25.0)

    scale_manifest = ScaleManifest.fit(train, manifest)
    np.testing.assert_allclose(scale_manifest.center, train.mean(axis=0))
    np.testing.assert_allclose(scale_manifest.scale, train.std(axis=0, ddof=1))
    assert scale_manifest.n_train == 60
    assert scale_manifest.was_fitted_on(train)
    assert not scale_manifest.was_fitted_on(np.vstack([train, evaluation]))

    # Transforming evaluation rows must not move the manifest.
    before = scale_manifest.to_dict()
    scale_manifest.transform(evaluation)
    assert scale_manifest.to_dict() == before

    pooled = ScaleManifest.fit(np.vstack([train, evaluation]), manifest)
    assert not np.allclose(pooled.center, scale_manifest.center)


def test_scale_manifest_round_trips_levels_and_contrasts():
    manifest = uniform_grid_manifest(3, functionals=("grid_sd",))
    train = manifest.build(_sample_quantiles(50, 3, seed=5))
    scale_manifest = ScaleManifest.fit(train, manifest)

    np.testing.assert_allclose(
        scale_manifest.inverse_transform(scale_manifest.transform(train)), train
    )

    # A contrast of standardized levels must not pick up the centering term.
    treated = manifest.build(_sample_quantiles(50, 3, seed=6))
    scaled_contrast = scale_manifest.transform(treated) - scale_manifest.transform(train)
    np.testing.assert_allclose(
        scale_manifest.inverse_transform_contrast(scaled_contrast), treated - train
    )

    restored = ScaleManifest.from_dict(scale_manifest.to_dict())
    np.testing.assert_allclose(restored.center, scale_manifest.center)
    np.testing.assert_allclose(restored.scale, scale_manifest.scale)
    assert restored.source_fingerprint == scale_manifest.source_fingerprint


def test_monotone_postprocessing_is_declared_and_local():
    manifest = uniform_grid_manifest(
        4,
        functionals=("grid_sd",),
        reference_quantiles=np.linspace(-1.0, 1.0, 4),
    )
    raw = np.array([[3.0, 1.0, 2.0, 4.0, 9.0, 7.0]])
    projected = manifest.project_quantile_block(raw)

    assert np.all(np.diff(projected[:, manifest.quantile_slice], axis=1) >= -1e-12)
    np.testing.assert_allclose(projected[:, manifest.n_grid :], raw[:, manifest.n_grid :])
    # The raw draw is untouched, so both versions remain reportable.
    np.testing.assert_array_equal(raw, np.array([[3.0, 1.0, 2.0, 4.0, 9.0, 7.0]]))
    np.testing.assert_allclose(
        manifest.project_quantile_block(projected), projected
    )


def test_manifest_serialization_is_deterministic():
    manifest = uniform_grid_manifest(
        5,
        functionals=("grid_mean", "grid_skewness"),
        reference_quantiles=np.linspace(-2.0, 2.0, 5),
    )
    payload = manifest.to_dict()
    restored = TargetManifest.from_dict(payload)
    assert restored.to_dict() == payload
    quantiles = _sample_quantiles(20, 5, seed=9)
    np.testing.assert_allclose(restored.build(quantiles), manifest.build(quantiles))


def test_folds_are_stratified_deterministic_and_leak_checked():
    rng = np.random.default_rng(4)
    treatment = rng.binomial(1, 0.4, size=120)
    plan = make_folds(120, treatment, n_folds=4, random_state=11)
    repeat = make_folds(120, treatment, n_folds=4, random_state=11)
    np.testing.assert_array_equal(plan.assignment, repeat.assignment)

    for fold in plan.fold_ids:
        held_out = plan.test_index(fold)
        assert held_out.size > 0
        assert np.unique(treatment[held_out]).size == 2
        assert_disjoint(plan.train_index(fold), held_out)

    with pytest.raises(ValueError):
        assert_disjoint(np.array([1, 2, 3]), np.array([3, 4]))
