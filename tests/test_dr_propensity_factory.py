"""Tests for the pluggable AIPW propensity factories and OOF retention."""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.cross_fitted import stratified_folds
from wasserstein_causal_forests.cwdb.dr_calibration import (
    PROPENSITY_CLIP,
    DRCalibratedCWDB,
    FunctionalAIPW,
    hist_gradient_boosting_propensity_factory,
    logistic_propensity_factory,
    random_forest_propensity_factory,
)

N, D, K, M = 200, 4, 5, 3
RANDOM_STATE = 7


def _synthetic(seed: int = 11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, D))
    logits = 0.7 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * x[:, 2]
    e_true = 1.0 / (1.0 + np.exp(-logits))
    a = rng.binomial(1, e_true)
    a[:2] = (0, 1)
    q = np.sort(
        rng.normal(size=(N, K))
        + 0.4 * a[:, None]
        + 0.3 * x[:, :1]
        + 0.2 * x[:, 1:2] * np.linspace(-1.0, 1.0, K)[None, :],
        axis=-1,
    )
    return x, a, q, np.full(K, 1.0 / K), e_true


def _functionals():
    return {
        "mean": lambda block: block.mean(axis=-1),
        "spread": lambda block: block.std(axis=-1),
    }


def _model(**overrides):
    parameters = {
        "functionals": _functionals(),
        "architecture": "v1",
        "n_particles": M,
        "n_estimators": 6,
        "learning_rate": 0.2,
        "max_depth": 2,
        "min_samples_leaf": 5,
        "min_arm_leaf": 3,
        "contrast_candidates": (0.0,),
        "n_folds": 2,
        "random_state": RANDOM_STATE,
    }
    parameters.update(overrides)
    return DRCalibratedCWDB(**parameters)


def test_default_and_explicit_logistic_factories_agree():
    x, a, q, w, _ = _synthetic()
    default = _model().fit(x, a, q, w)
    explicit = _model(propensity_factory="logistic").fit(x, a, q, w)
    assert np.array_equal(default.ehat_train_, explicit.ehat_train_)
    assert np.array_equal(default.aipw_.ehat_train_, explicit.aipw_.ehat_train_)


@pytest.mark.parametrize("name", ["hist_gradient_boosting", "random_forest"])
def test_flexible_factories_fit_end_to_end(name):
    x, a, q, w, _ = _synthetic()
    model = _model(propensity_factory=name).fit(x, a, q, w)
    assert np.all(np.isfinite(model.ehat_train_))
    assert np.all(model.ehat_train_ >= PROPENSITY_CLIP[0])
    assert np.all(model.ehat_train_ <= PROPENSITY_CLIP[1])
    for functional in model.aipw_.marginal_:
        assert np.isfinite(model.aipw_.marginal_[functional])
        bins = model.aipw_.bin_contrasts_[functional]
        assert bins.shape == (4,)
        assert np.all(np.isfinite(bins))
        broadcast = model.dr_bin_contrasts(functional, x)
        assert broadcast.shape == (N,)
        assert np.all(np.isfinite(broadcast))


def test_oracle_propensity_is_clipped_and_used_directly():
    x, a, q, w, e_true = _synthetic()
    e_oracle = e_true.copy()
    e_oracle[0], e_oracle[1] = 0.0, 1.0
    model = _model(oracle_propensity=True).fit(
        x, a, q, w, true_propensity=e_oracle
    )
    expected = np.clip(e_oracle, PROPENSITY_CLIP[0], PROPENSITY_CLIP[1])
    assert np.array_equal(model.ehat_train_, expected)
    assert np.array_equal(model.aipw_.ehat_train_, expected)


def test_oof_particles_cover_every_row_once():
    x, a, q, w, _ = _synthetic()
    model = _model().fit(x, a, q, w)
    assert np.array_equal(
        model.oof_folds_, stratified_folds(a, model.n_folds, RANDOM_STATE)
    )
    for arm in (0, 1):
        assert model.oof_particles_[arm].shape == (N, M, K)
        assert np.all(np.isfinite(model.oof_particles_[arm]))


def test_oracle_propensity_requires_true_propensity():
    x, a, q, w, _ = _synthetic()
    with pytest.raises(ValueError):
        _model(oracle_propensity=True).fit(x, a, q, w)


def test_unknown_propensity_factory_name_raises():
    with pytest.raises(ValueError, match="unknown propensity factory"):
        _model(propensity_factory="not_a_factory")


@pytest.mark.parametrize(
    "factory",
    [
        logistic_propensity_factory,
        hist_gradient_boosting_propensity_factory,
        random_forest_propensity_factory,
    ],
)
def test_factories_return_estimators_with_the_required_interface(factory):
    estimator = factory(3)
    assert callable(estimator.fit)
    assert callable(estimator.predict_proba)


class _ConstantPropensity:
    def fit(self, X, treatment):
        self.n_fit_ = X.shape[0]
        return self

    def predict_proba(self, X):
        return np.full((X.shape[0], 2), 0.5)


def test_functional_aipw_accepts_a_caller_supplied_factory():
    x, a, q, _, _ = _synthetic()
    observed = {"mean": q.mean(axis=-1)}
    oof = {"mean": {arm: np.zeros(N) for arm in (0, 1)}}
    seeds = []

    def factory(seed):
        seeds.append(seed)
        return _ConstantPropensity()

    layer = FunctionalAIPW(n_bins=4).fit(
        observed=observed,
        oof_arm_means=oof,
        X=x,
        treatment=a,
        random_state=RANDOM_STATE,
        propensity_factory=factory,
    )
    assert seeds == [RANDOM_STATE + fold for fold in range(5)]
    assert np.allclose(layer.ehat_train_, 0.5)
