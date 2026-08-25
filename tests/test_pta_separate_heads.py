"""WP2-B1 verification for the PTA-S scalar stochtree heads."""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.pta_bcf import dgps
from wasserstein_causal_forests.pta_bcf.separate_heads import (
    HeadBudget,
    PTASeparateHeads,
)
from wasserstein_causal_forests.pta_bcf.targets import (
    ScaleManifest,
    make_folds,
    uniform_grid_manifest,
)

TEST_BUDGET = HeadBudget(
    num_trees_prognostic=20,
    num_trees_treatment=10,
    num_gfr=5,
    num_burnin=10,
    num_mcmc=40,
)


def _fit_null_regime(n_rows: int = 240, seed: int = 0):
    manifest = dgps.pta_manifest(n_grid=4, functionals=("grid_mean", "grid_sd"))
    data = dgps.sample_dataset(n_rows, "null", seed, n_grid=manifest.n_grid)
    model = PTASeparateHeads(
        manifest, budget=TEST_BUDGET, n_folds=4, random_state=seed
    ).fit(data["X"], data["treatment"], data["quantiles"])
    return manifest, data, model


def test_scalar_reduction_matches_a_direct_bcf_fit():
    """With K=1, J=0 and no reference, PTA-S is one scalar BCFModel."""

    from stochtree import BCFModel

    manifest = uniform_grid_manifest(1)
    data = dgps.sample_dataset(200, "shared", 3, n_grid=1)
    propensity = data["propensity"]

    model = PTASeparateHeads(
        manifest, budget=TEST_BUDGET, n_folds=4, random_state=7
    ).fit(
        data["X"], data["treatment"], data["quantiles"], propensity=propensity
    )
    assert len(model.heads_) == 1

    scaling = ScaleManifest.fit(manifest.build(data["quantiles"]), manifest)
    response = scaling.transform(manifest.build(data["quantiles"]))[:, 0]
    direct = BCFModel()
    direct.sample(
        X_train=data["X"],
        Z_train=data["treatment"].astype(float),
        y_train=response,
        propensity_train=propensity,
        num_gfr=TEST_BUDGET.num_gfr,
        num_burnin=TEST_BUDGET.num_burnin,
        num_mcmc=TEST_BUDGET.num_mcmc,
        general_params={"random_seed": 7 * 1000 + 1, "keep_every": 1},
        prognostic_forest_params={"num_trees": TEST_BUDGET.num_trees_prognostic},
        treatment_effect_forest_params={"num_trees": TEST_BUDGET.num_trees_treatment},
    )
    expected = direct.predict(
        data["X"], np.zeros(data["X"].shape[0]), propensity
    )["cate"] * scaling.scale[0]
    observed = model.predict_contrast_draws(data["X"], propensity=propensity)[:, 0, :]
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_every_head_shares_folds_propensity_and_budget():
    manifest, data, model = _fit_null_regime()
    assert len(model.heads_) == manifest.dimension
    assert model.train_propensity_.shape == (data["X"].shape[0],)
    assert np.all(model.train_propensity_ > 0.0)
    assert np.all(model.train_propensity_ < 1.0)
    assert model.folds_.assignment.shape == (data["X"].shape[0],)
    for head in model.heads_:
        assert head.num_samples == TEST_BUDGET.num_mcmc

    supplied = make_folds(
        data["X"].shape[0], data["treatment"], n_folds=4, random_state=99
    )
    reused = PTASeparateHeads(
        manifest, budget=TEST_BUDGET, random_state=1
    ).fit(data["X"], data["treatment"], data["quantiles"], folds=supplied)
    np.testing.assert_array_equal(reused.folds_.assignment, supplied.assignment)


def test_null_treatment_effect_is_shrunk_toward_zero():
    manifest, data, model = _fit_null_regime()
    contrast = model.predict_contrast(data["X"])
    target_sd = manifest.build(data["quantiles"]).std(axis=0, ddof=1)
    relative = np.abs(contrast).mean(axis=0) / target_sd
    assert np.all(relative < 0.20), relative


def test_pure_functional_signal_is_recovered_without_location_leakage():
    """Treatment inflates spread only, so the grid mean must stay unaffected."""

    manifest = dgps.pta_manifest(
        n_grid=5, functionals=("grid_mean", "grid_sd"), with_reference=False
    )
    rng = np.random.default_rng(21)
    n_rows = 400
    X = rng.uniform(-1.0, 1.0, size=(n_rows, 4))
    treatment = rng.binomial(1, 0.5, size=n_rows)
    grid_z = dgps.reference_quantiles(manifest.n_grid)
    location = 0.6 * X[:, 0] + 0.2 * rng.normal(size=n_rows)
    log_scale = 0.1 * X[:, 1] + 0.7 * treatment
    quantiles = location[:, None] + np.exp(log_scale)[:, None] * grid_z

    model = PTASeparateHeads(
        manifest, budget=TEST_BUDGET, random_state=5
    ).fit(X, treatment, quantiles, propensity=np.full(n_rows, 0.5))
    contrast = model.predict_contrast(X, propensity=np.full(n_rows, 0.5)).mean(axis=0)

    mean_index = manifest.n_grid
    sd_index = manifest.n_grid + 1
    truth_sd = (np.exp(0.1 * X[:, 1] + 0.7) - np.exp(0.1 * X[:, 1])).mean() * np.sqrt(
        (grid_z**2) @ manifest.weights
    )
    assert contrast[sd_index] > 0.5 * truth_sd
    assert contrast[sd_index] < 1.5 * truth_sd
    assert abs(contrast[mean_index]) < 0.15 * truth_sd
    # The symmetric grid means the median coordinate also carries no effect.
    assert abs(contrast[manifest.n_grid // 2]) < 0.25 * truth_sd


def test_predictions_are_deterministic_and_json_serializable():
    manifest, data, model = _fit_null_regime(n_rows=180, seed=2)
    reference = model.predict_contrast(data["X"])

    repeat = PTASeparateHeads(
        manifest, budget=TEST_BUDGET, n_folds=4, random_state=2
    ).fit(data["X"], data["treatment"], data["quantiles"])
    np.testing.assert_allclose(
        repeat.predict_contrast(data["X"]), reference, rtol=0.0, atol=1e-12
    )

    payload = model.to_json_string()
    assert payload == model.to_json_string()
    restored = PTASeparateHeads.from_json_string(
        payload, propensity_model=model.propensity_model_
    )
    np.testing.assert_allclose(
        restored.predict_contrast(data["X"]), reference, rtol=0.0, atol=1e-12
    )
    assert restored.scale_manifest_.source_fingerprint == (
        model.scale_manifest_.source_fingerprint
    )


def test_scale_manifest_never_sees_evaluation_rows():
    manifest, data, model = _fit_null_regime(n_rows=200, seed=6)
    training_targets = manifest.build(data["quantiles"])
    assert model.scale_manifest_.was_fitted_on(training_targets)
    assert model.scale_manifest_.n_train == data["X"].shape[0]

    evaluation = dgps.sample_dataset(150, "null", 61, n_grid=manifest.n_grid)
    before = model.scale_manifest_.to_dict()
    model.predict_contrast(evaluation["X"])
    assert model.scale_manifest_.to_dict() == before


def test_monotone_projection_is_postprocessing_of_raw_arm_draws():
    manifest, data, model = _fit_null_regime(n_rows=180, seed=8)
    raw = model.predict_arm_draws(data["X"], 1, project=False)
    projected = model.predict_arm_draws(data["X"], 1, project=True)

    assert raw.shape == projected.shape
    quantile_block = projected[:, manifest.quantile_slice, :]
    assert np.all(np.diff(quantile_block, axis=1) >= -1e-9)
    np.testing.assert_allclose(
        projected[:, manifest.n_grid :, :], raw[:, manifest.n_grid :, :]
    )
    # The contrast is a difference of monotone vectors and is never projected.
    np.testing.assert_allclose(
        model.predict_contrast_draws(data["X"]),
        model.predict_draws(data["X"])["contrast"],
    )


def test_prediction_requires_consistent_inputs():
    manifest, data, model = _fit_null_regime(n_rows=150, seed=4)
    with pytest.raises(ValueError):
        model.predict_contrast(data["X"][:, :2])
    with pytest.raises(ValueError):
        model.predict_arm_draws(data["X"], 2)
