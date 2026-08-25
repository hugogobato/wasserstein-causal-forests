from __future__ import annotations

import pickle

import numpy as np

from wasserstein_causal_forests.cwdb.energy import empirical_energy_risk
from wasserstein_causal_forests.cwdb.model import (
    ArmParticleBooster,
    CWDBRegressor,
)


def make_data(seed: int = 19, n: int = 120) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = 0.35 * X[:, 0] - 0.2 * X[:, 2]
    propensity = 1.0 / (1.0 + np.exp(-logits))
    treatment = rng.binomial(1, propensity)
    treatment[:2] = [0, 1]
    location = (
        0.8 * np.where(X[:, 0] > 0.0, 1.0, -1.0)
        + 0.3 * treatment
        + rng.normal(scale=0.35, size=n)
    )
    scale = np.exp(0.15 * X[:, 1] + 0.1 * treatment)
    template = np.array([-1.2, -0.4, 0.0, 0.5, 1.4])
    quantiles = location[:, None] + scale[:, None] * template
    weights = np.ones(template.size) / template.size
    return X, treatment, quantiles, weights


def model_parameters() -> dict[str, object]:
    return {
        "n_particles": 4,
        "n_estimators": 8,
        "learning_rate": 0.15,
        "max_depth": 2,
        "min_samples_leaf": 5,
        "collision_epsilon": 1e-3,
        "random_state": 81,
    }


def test_arm_booster_training_is_monotone_and_reduces_risk() -> None:
    X, treatment, quantiles, weights = make_data()
    mask = treatment == 0
    model = ArmParticleBooster(**model_parameters()).fit(
        X[mask], quantiles[mask], weights
    )
    prediction = model.predict_particles(X[mask])
    initial = np.broadcast_to(
        model.initial_particles_, prediction.shape
    ).copy()
    initial_risk = empirical_energy_risk(
        initial, quantiles[mask], weights, epsilon=model.collision_epsilon
    )
    assert model.train_risk_ <= initial_risk + 1e-12
    assert np.all(np.diff(prediction, axis=-1) >= -1e-12)
    assert all(
        step.loss_after <= step.loss_before + model.descent_tolerance
        for step in model.training_history_
    )
    assert all(step.projection_max >= 0.0 for step in model.training_history_)


def test_v0_equals_two_direct_arm_fits() -> None:
    X, treatment, quantiles, weights = make_data()
    parameters = model_parameters()
    causal = CWDBRegressor(architecture="v0", **parameters).fit(
        X, treatment, quantiles, weights
    )
    for arm in (0, 1):
        direct_parameters = dict(parameters)
        direct_parameters["random_state"] = int(parameters["random_state"]) + 10_000 * arm
        direct = ArmParticleBooster(**direct_parameters).fit(
            X[treatment == arm], quantiles[treatment == arm], weights
        )
        assert np.allclose(
            causal.predict_particles(X[:20], arm),
            direct.predict_particles(X[:20]),
            atol=0.0,
            rtol=0.0,
        )


def test_row_order_invariance() -> None:
    X, treatment, quantiles, weights = make_data()
    permutation = np.random.default_rng(902).permutation(X.shape[0])
    first = CWDBRegressor(architecture="v0", **model_parameters()).fit(
        X, treatment, quantiles, weights
    )
    second = CWDBRegressor(architecture="v0", **model_parameters()).fit(
        X[permutation],
        treatment[permutation],
        quantiles[permutation],
        weights,
    )
    for arm in (0, 1):
        assert np.allclose(
            first.predict_particles(X[:25], arm),
            second.predict_particles(X[:25], arm),
            atol=0.0,
            rtol=0.0,
        )


def test_fixed_seed_is_deterministic_and_pickle_roundtrips() -> None:
    X, treatment, quantiles, weights = make_data()
    first = CWDBRegressor(architecture="v0", **model_parameters()).fit(
        X, treatment, quantiles, weights
    )
    second = CWDBRegressor(architecture="v0", **model_parameters()).fit(
        X, treatment, quantiles, weights
    )
    restored = pickle.loads(pickle.dumps(first))
    for arm in (0, 1):
        expected = first.predict_particles(X[:15], arm)
        assert np.array_equal(expected, second.predict_particles(X[:15], arm))
        assert np.array_equal(expected, restored.predict_particles(X[:15], arm))


def test_public_summaries_are_law_invariant() -> None:
    X, treatment, quantiles, weights = make_data()
    model = CWDBRegressor(architecture="v0", **model_parameters()).fit(
        X, treatment, quantiles, weights
    )
    particles = model.predict_particles(X[:12], 1)
    mean = model.predict_mean_quantile(X[:12], 1)
    integral = model.predict_integral(
        X[:12], 1, lambda draws: np.sum(weights * draws**2, axis=-1)
    )
    assert np.allclose(mean, particles.mean(axis=1))
    assert np.allclose(
        integral, np.mean(np.sum(weights * particles**2, axis=-1), axis=1)
    )
    effect = model.predict_mean_quantile_effect(X[:12])
    expected = (
        model.predict_mean_quantile(X[:12], 1)
        - model.predict_mean_quantile(X[:12], 0)
    )
    assert np.allclose(effect, expected)


def test_particles_are_canonically_ordered_but_not_cross_arm_paired() -> None:
    X, treatment, quantiles, weights = make_data()
    model = CWDBRegressor(architecture="v0", **model_parameters()).fit(
        X, treatment, quantiles, weights
    )
    for arm in (0, 1):
        particles = model.predict_particles(X[:10], arm)
        for row in particles:
            tuples = [tuple(particle) for particle in row]
            assert tuples == sorted(tuples)

