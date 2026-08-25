"""Contrast-level regularisers added after the G3 rule 1 failure.

Rule 1 failed because C-WDB-v1 manufactures a treatment contrast on D2, where
the true effect is exactly null. These tests pin the three repair mechanisms at
the unit level: what each rule does to a leaf, what the pooled initialisation
does to the initial contrast, and that the frozen v1 configuration is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.arm_shared_tree import ArmSharedTreeRegressor
from wasserstein_causal_forests.cwdb.cross_fitted import (
    CrossFittedCWDBRegressor,
    stratified_folds,
)
from wasserstein_causal_forests.cwdb.model import CWDBRegressor, compute_init_base


def leaf_data(
    gap: float, noise: float, seed: int = 3, n: int = 60, n_outputs: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-leaf data: a constant arm gap plus exchangeable noise."""

    generator = np.random.default_rng(seed)
    X = generator.uniform(-1.0, 1.0, size=(n, 2))
    treatment = np.arange(n) % 2
    gradients = generator.normal(scale=noise, size=(n, n_outputs))
    gradients += treatment[:, None] * gap
    return X, treatment, gradients


def single_leaf_tree(**parameters) -> ArmSharedTreeRegressor:
    return ArmSharedTreeRegressor(
        max_depth=0, min_samples_leaf=2, min_arm_leaf=2, **parameters
    )


# --------------------------------------------------------------- the ridge rule


def test_ridge_at_zero_strength_returns_the_raw_arm_means() -> None:
    X, treatment, gradients = leaf_data(gap=0.5, noise=0.2)
    tree = single_leaf_tree(contrast_rule="ridge", contrast_shrinkage=0.0)
    tree.fit(X, treatment, gradients)
    for arm in (0, 1):
        assert np.allclose(
            tree.predict(X, arm)[0], gradients[treatment == arm].mean(axis=0)
        )


def test_ridge_shrinks_only_the_contrast_and_never_the_pooled_component() -> None:
    X, treatment, gradients = leaf_data(gap=0.5, noise=0.2)
    pooled = gradients.mean(axis=0)
    share = float(np.mean(treatment))
    for strength in (0.0, 5.0, 50.0, 5000.0):
        tree = single_leaf_tree(contrast_rule="ridge", contrast_shrinkage=strength)
        tree.fit(X, treatment, gradients)
        values = {arm: tree.predict(X, arm)[0] for arm in (0, 1)}
        combined = share * values[1] + (1.0 - share) * values[0]
        assert np.allclose(combined, pooled)


def test_ridge_contrast_decreases_monotonically_in_the_strength() -> None:
    X, treatment, gradients = leaf_data(gap=0.5, noise=0.2)
    norms = []
    for strength in (0.0, 5.0, 50.0, 500.0, 50_000.0):
        tree = single_leaf_tree(contrast_rule="ridge", contrast_shrinkage=strength)
        tree.fit(X, treatment, gradients)
        norms.append(
            float(np.linalg.norm(tree.predict(X, 1)[0] - tree.predict(X, 0)[0]))
        )
    assert all(a > b for a, b in zip(norms[:-1], norms[1:], strict=True))
    assert norms[-1] == pytest.approx(0.0, abs=1e-3)


def test_ridge_uses_the_effective_sample_size_of_the_gap() -> None:
    X, treatment, gradients = leaf_data(gap=0.5, noise=0.2, n=60)
    tree = single_leaf_tree(contrast_rule="ridge", contrast_shrinkage=30.0)
    tree.fit(X, treatment, gradients)
    n_0, n_1 = 30, 30
    expected = (n_0 * n_1 / (n_0 + n_1)) / (n_0 * n_1 / (n_0 + n_1) + 30.0)
    raw = gradients[treatment == 1].mean(axis=0) - gradients[treatment == 0].mean(axis=0)
    assert np.allclose(tree.predict(X, 1)[0] - tree.predict(X, 0)[0], expected * raw)


# ----------------------------------------------------------- the threshold rule


def test_threshold_zeroes_a_contrast_that_is_pure_noise() -> None:
    X, treatment, gradients = leaf_data(gap=0.0, noise=0.5)
    tree = single_leaf_tree(contrast_rule="threshold", contrast_threshold_scale=1.0)
    tree.fit(X, treatment, gradients)
    assert np.allclose(tree.predict(X, 1)[0], tree.predict(X, 0)[0])


def test_threshold_passes_a_gap_far_above_the_noise_nearly_undamped() -> None:
    X, treatment, gradients = leaf_data(gap=4.0, noise=0.2)
    tree = single_leaf_tree(contrast_rule="threshold", contrast_threshold_scale=1.0)
    tree.fit(X, treatment, gradients)
    raw = gradients[treatment == 1].mean(axis=0) - gradients[treatment == 0].mean(axis=0)
    kept = tree.predict(X, 1)[0] - tree.predict(X, 0)[0]
    assert np.linalg.norm(kept) / np.linalg.norm(raw) > 0.99


def retained_contrast_fractions(scale: float, n_seeds: int = 80) -> np.ndarray:
    """Share of the raw null-leaf arm gap each rule setting lets through."""

    fractions = []
    for seed in range(n_seeds):
        X, treatment, gradients = leaf_data(gap=0.0, noise=0.5, seed=seed)
        tree = single_leaf_tree(
            contrast_rule="threshold", contrast_threshold_scale=scale
        )
        tree.fit(X, treatment, gradients)
        raw = (
            gradients[treatment == 1].mean(axis=0)
            - gradients[treatment == 0].mean(axis=0)
        )
        kept = tree.predict(X, 1)[0] - tree.predict(X, 0)[0]
        fractions.append(np.linalg.norm(kept) / np.linalg.norm(raw))
    return np.asarray(fractions)


def test_threshold_removes_most_of_a_null_leaf_gap() -> None:
    """The rule is calibrated so a pure-noise gap mostly does not survive."""

    fractions = retained_contrast_fractions(scale=1.0)
    assert fractions.mean() < 0.25
    assert np.mean(fractions == 0.0) > 0.4


def test_threshold_scale_controls_how_null_safe_the_rule_is() -> None:
    means = [retained_contrast_fractions(scale=s).mean() for s in (0.5, 1.0, 3.0)]
    assert all(a > b for a, b in zip(means[:-1], means[1:], strict=True))
    assert means[-1] < 0.02


def test_threshold_preserves_the_pooled_component() -> None:
    X, treatment, gradients = leaf_data(gap=2.0, noise=0.3)
    share = float(np.mean(treatment))
    tree = single_leaf_tree(contrast_rule="threshold")
    tree.fit(X, treatment, gradients)
    combined = (
        share * tree.predict(X, 1)[0] + (1.0 - share) * tree.predict(X, 0)[0]
    )
    assert np.allclose(combined, gradients.mean(axis=0))


# ------------------------------------------------------------ pooled initialisation


def confounded_sample(
    n: int = 400, seed: int = 5, effect: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Null-effect data with assignment driven by a prognostic covariate."""

    generator = np.random.default_rng(seed)
    X = generator.uniform(-1.0, 1.0, size=(n, 3))
    treatment = generator.binomial(1, 1.0 / (1.0 + np.exp(-2.0 * X[:, 0])))
    location = X[:, 0] + effect * treatment + generator.normal(scale=0.2, size=n)
    template = np.array([-1.0, -0.3, 0.3, 1.0])
    quantiles = location[:, None] + template
    weights = np.full(template.size, 1.0 / template.size)
    return X, treatment.astype(int), quantiles, weights


def test_per_arm_initialisation_carries_the_confounded_arm_gap() -> None:
    _, treatment, quantiles, _ = confounded_sample()
    per_arm = {
        arm: compute_init_base(quantiles[treatment == arm], 5) for arm in (0, 1)
    }
    offset = per_arm[1].mean(axis=0) - per_arm[0].mean(axis=0)
    assert np.linalg.norm(offset) > 0.5


def test_pooled_initialisation_starts_both_arms_at_the_same_law() -> None:
    X, treatment, quantiles, weights = confounded_sample()
    model = CWDBRegressor(
        architecture="v1",
        init_sharing="pooled",
        n_particles=5,
        n_estimators=0,
        max_depth=2,
        min_samples_leaf=10,
        min_arm_leaf=5,
        random_state=1,
    ).fit(X, treatment, quantiles, weights)
    assert np.array_equal(model.initial_particles_[0], model.initial_particles_[1])
    assert np.allclose(
        model.predict_mean_quantile_effect(X[:50]), 0.0, atol=1e-12
    )


def test_pooled_initialisation_also_reaches_the_independent_arm_architecture() -> None:
    X, treatment, quantiles, weights = confounded_sample()
    model = CWDBRegressor(
        architecture="v0",
        init_sharing="pooled",
        n_particles=5,
        n_estimators=0,
        random_state=1,
    ).fit(X, treatment, quantiles, weights)
    assert np.array_equal(
        model.arm_models_[0].initial_particles_,
        model.arm_models_[1].initial_particles_,
    )


# -------------------------------------------------- cross-fitted strength selection


def test_stratified_folds_balance_both_arms() -> None:
    treatment = np.array([0] * 31 + [1] * 29)
    folds = stratified_folds(treatment, n_folds=3, random_state=7)
    for arm in (0, 1):
        counts = np.bincount(folds[treatment == arm], minlength=3)
        assert counts.max() - counts.min() <= 1


def test_stratified_folds_are_deterministic_in_the_seed() -> None:
    treatment = np.array([0, 1] * 30)
    first = stratified_folds(treatment, 3, random_state=11)
    second = stratified_folds(treatment, 3, random_state=11)
    assert np.array_equal(first, second)


def fit_cross_fitted(effect: float, seed: int) -> CrossFittedCWDBRegressor:
    X, treatment, quantiles, weights = confounded_sample(
        n=300, seed=seed, effect=effect
    )
    return CrossFittedCWDBRegressor(
        contrast_candidates=(0.0, 500.0),
        n_folds=2,
        architecture="v1",
        init_sharing="pooled",
        n_particles=4,
        n_estimators=20,
        learning_rate=0.15,
        max_depth=2,
        min_samples_leaf=10,
        min_arm_leaf=5,
        random_state=seed,
    ).fit(X, treatment, quantiles, weights)


def test_cross_fitting_picks_strong_shrinkage_under_a_null_effect() -> None:
    model = fit_cross_fitted(effect=0.0, seed=5)
    assert model.selected_contrast_shrinkage_ == 500.0


def test_cross_fitting_picks_weak_shrinkage_under_a_large_effect() -> None:
    model = fit_cross_fitted(effect=1.5, seed=5)
    assert model.selected_contrast_shrinkage_ == 0.0


def test_cross_fitting_records_every_candidate_and_refits_at_the_winner() -> None:
    model = fit_cross_fitted(effect=0.0, seed=6)
    assert [record.contrast_shrinkage for record in model.selection_records_] == [
        0.0,
        500.0,
    ]
    assert all(np.isfinite(record.held_out_risk) for record in model.selection_records_)
    assert model.contrast_shrinkage == model.selected_contrast_shrinkage_
    assert hasattr(model, "fitted_architecture_")


# ---------------------------------------------------------- the frozen v1 defaults


def test_defaults_reproduce_the_frozen_v1_configuration() -> None:
    """No repair option may change C-WDB-v1 unless it is asked for."""

    X, treatment, quantiles, weights = confounded_sample(effect=0.8)
    parameters = {
        "architecture": "v1",
        "sharing": "partial",
        "arm_shrinkage": 5.0,
        "n_particles": 4,
        "n_estimators": 10,
        "learning_rate": 0.12,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "min_arm_leaf": 5,
        "random_state": 4,
    }
    frozen = CWDBRegressor(**parameters).fit(X, treatment, quantiles, weights)
    explicit = CWDBRegressor(
        init_sharing="per_arm", contrast_rule="arm_shrinkage", **parameters
    ).fit(X, treatment, quantiles, weights)
    for arm in (0, 1):
        assert np.array_equal(
            frozen.predict_particles(X[:40], arm),
            explicit.predict_particles(X[:40], arm),
        )
