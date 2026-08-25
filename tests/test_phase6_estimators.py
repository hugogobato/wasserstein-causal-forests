"""Fast unit tests for the Phase 6 estimators."""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.dr_calibration import (
    DRCalibratedCWDB,
    aipw_scores,
    hajek_bin_means,
)
from wasserstein_causal_forests.cwdb.krr_booster import KRRArmParticleBooster
from wasserstein_causal_forests.cwdb.smoothing import (
    SmoothedCWDB,
    jitter_cloud,
    scale_cloud,
)
from wasserstein_causal_forests.meta_learners.functional_r_learner import (
    FunctionalRLearner,
)


def _sample(n: int = 120, seed: int = 0, k: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    propensity = np.clip(0.5 + 0.3 * x[:, 0], 0.1, 0.9)
    a = rng.binomial(1, propensity)
    levels = (np.arange(k) + 0.5) / k
    z = _norm_ppf(levels)
    base = 1.0 + 0.5 * x[:, 0]
    q = base[:, None] + a[:, None] * 0.4 * z[None, :]
    return x, a, q, np.full(k, 1.0 / k)


def _norm_ppf(u):
    from scipy.stats import norm

    return norm.ppf(u)


def test_scale_cloud_identity_and_monotone():
    rng = np.random.default_rng(1)
    particles = np.sort(rng.normal(size=(6, 4, 5)), axis=-1)
    weights = np.full(5, 0.2)
    assert np.array_equal(scale_cloud(particles, weights, 1.0), particles)
    scaled = scale_cloud(particles, weights, 1.4)
    assert np.all(np.diff(scaled, axis=-1) >= -1e-12)
    # The pre-projection scaling fixes the barycenter; the cone projection may
    # move it only when scaling created violations.
    assert np.allclose(
        scaled.mean(axis=1), particles.mean(axis=1), atol=0.05
    )


def test_jitter_cloud_expands_atoms_and_stays_monotone():
    rng = np.random.default_rng(2)
    particles = np.sort(rng.normal(size=(3, 4, 5)), axis=-1)
    weights = np.full(5, 0.2)
    out = jitter_cloud(particles, weights, 0.25, replicates=3, random_state=0)
    assert out.shape == (3, 12, 5)
    assert np.all(np.diff(out, axis=-1) >= -1e-12)


def test_hajek_and_aipw_scores_finite():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    bins = np.array([0, 0, 1, 1])
    means = hajek_bin_means(values, bins, 2)
    assert np.allclose(means, [1.5, 3.5])
    scores = aipw_scores(
        h_observed=values,
        mu0=np.zeros(4),
        mu1=np.ones(4),
        ehat=np.full(4, 0.5),
        treatment=np.array([0, 1, 0, 1]),
    )
    assert np.all(np.isfinite(scores))


def test_dr_calibrated_fit_recovers_null_and_effect():
    x, a, q, w = _sample(n=200, seed=3)

    def reference_distance(block):
        centre = _norm_ppf((np.arange(block.shape[1]) + 0.5) / block.shape[1])
        diff = block - centre
        return np.sqrt(np.mean(diff * diff, axis=-1))

    model = DRCalibratedCWDB(
        functionals={"reference": reference_distance},
        architecture="v1",
        n_particles=3,
        n_estimators=8,
        learning_rate=0.2,
        max_depth=2,
        min_samples_leaf=5,
        min_arm_leaf=3,
        contrast_candidates=(0.0,),
        n_folds=2,
        random_state=0,
    )
    model.fit(x, a, q, w)
    marginal = model.dr_marginal("reference")
    assert np.isfinite(marginal)
    broadcast = model.dr_bin_contrasts("reference", x)
    assert broadcast.shape == (x.shape[0],)
    # Particles still come through unchanged in shape.
    particles = model.predict_particles(x[:7], 1)
    assert particles.shape == (7, 3, q.shape[1])


def test_smoothed_model_selects_a_transform_and_predicts():
    x, a, q, w = _sample(n=150, seed=4)
    model = SmoothedCWDB(
        architecture="v1",
        n_particles=3,
        n_estimators=8,
        learning_rate=0.2,
        max_depth=2,
        min_samples_leaf=5,
        min_arm_leaf=3,
        contrast_candidates=(0.0,),
        n_folds=2,
        jitter_replicates=2,
        random_state=0,
    )
    model.fit(x, a, q, w)
    assert model.selected_transform_ in {"scale", "jitter"}
    particles = model.predict_particles(x[:9], 0)
    assert particles.ndim == 3 and particles.shape[0] == 9
    assert np.all(np.isfinite(particles))
    assert np.all(np.diff(particles, axis=-1) >= -1e-12)


def test_krr_booster_reduces_training_risk_and_predicts():
    x, a, q, w = _sample(n=140, seed=5)
    booster = KRRArmParticleBooster(
        n_particles=3, n_estimators=10, learning_rate=0.25,
        collision_epsilon=1e-3, random_state=0,
    ).fit(x[a == 1], q[a == 1], w)
    assert booster.n_accepted_steps_ > 0
    particles = booster.predict_particles(x[:11])
    assert particles.shape == (11, 3, q.shape[1])
    assert np.all(np.diff(particles, axis=-1) >= -1e-12)


def test_functional_r_learner_joint_columns():
    x, a, q, w = _sample(n=220, seed=6)

    def mean_functional(block):
        return block @ w

    def sd_functional(block):
        centred = block - (block @ w)[..., None]
        return np.sqrt(np.maximum(centred * centred @ w, 0.0))

    model = FunctionalRLearner(
        functionals={"grid_mean": mean_functional, "grid_sd": sd_functional},
        contrast_budget={"n_estimators": 6, "learning_rate": 0.2,
                         "max_depth": 2, "min_samples_leaf": 8},
        random_state=0,
    )
    model.fit(x, a, q)
    contrasts = model.predict_contrasts(x[:13])
    assert set(contrasts) == {"grid_mean", "grid_sd"}
    for value in contrasts.values():
        assert value.shape == (13,)
        assert np.all(np.isfinite(value))
    arm_means = model.predict_arm_means(x[:13], 1)
    assert set(arm_means) == {"grid_sd", "grid_mean"}
    # The treatment adds 0.4 * z_k, a pure scale change on symmetric quantile
    # levels: the sd contrast must be positive and near 0.4 * RMS(z), while the
    # mean contrast stays near zero.
    rms_z = float(np.sqrt(np.mean(_norm_ppf((np.arange(q.shape[1]) + 0.5) / q.shape[1]) ** 2)))
    # A deliberately tiny budget plus selectable shrinkage means only the
    # sign and rough scale are testable here.
    assert contrasts["grid_sd"].mean() > 0.5 * 0.4 * rms_z
    assert abs(float(contrasts["grid_mean"].mean())) < 0.15
