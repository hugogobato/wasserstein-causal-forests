"""WP5.5-C unit checks: projection, descent, permutation, and collapse.

The particle ``mu/tau`` variant must satisfy the four WP5.5-C collapse checks
and the inherited C-WDB validity checks. The two load-bearing ones are:

1. With the leaf-share treatment basis and zero contrast penalty, the tree and
   the whole booster must reduce to the current reparameterized shared tree.
2. With the contrast field forced to zero, the method must reduce to a
   treatment-invariant pooled particle booster.
"""

from __future__ import annotations

import numpy as np

from wasserstein_causal_forests.common.quantiles import canonicalize_particles
from wasserstein_causal_forests.cwdb.arm_shared_tree import ArmSharedTreeRegressor
from wasserstein_causal_forests.cwdb.model import CWDBRegressor
from wasserstein_causal_forests.cwdb.mutau import (
    MutauCWDBRegressor,
    MutauSharedTreeRegressor,
)
from wasserstein_causal_forests.g3.dgps import build_dgp

MODEL_PARAMETERS = {
    "architecture": "v1",
    "sharing": "partial",
    "init_sharing": "pooled",
    "n_particles": 5,
    "n_estimators": 30,
    "learning_rate": 0.12,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "min_arm_leaf": 5,
}


def tree_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(300, 5))
    treatment = rng.binomial(1, 0.5 + 0.2 * X[:, 0])
    target = np.stack(
        [np.sin(np.pi * X[:, 0]) + treatment * 0.4 * X[:, 1]] * 3, axis=1
    )
    return X, treatment, target


def model_fixture(seed: int = 3, n: int = 300) -> tuple:
    dgp = build_dgp("D1", 25)
    train = dgp.sample(n, seed=seed)
    test = dgp.sample(200, seed=900_000 + seed)
    return dgp, train, test


# ------------------------------------------------------------- collapse checks


def test_tree_level_collapse_to_the_reparameterized_shared_tree() -> None:
    """WP5.5-C check 1 at the tree level: leaf-share basis, zero penalty."""

    X, treatment, target = tree_data()
    mutau = MutauSharedTreeRegressor(
        max_depth=3, min_samples_leaf=10, min_arm_leaf=5,
        ehat_basis="leaf_share", contrast_shrinkage=0.0,
    ).fit(X, treatment, target)
    shared = ArmSharedTreeRegressor(
        max_depth=3, min_samples_leaf=10, min_arm_leaf=5,
        sharing="partial", contrast_rule="ridge", contrast_shrinkage=0.0,
    ).fit(X, treatment, target)
    # Identical splits, because the split rule is inherited unchanged.
    assert np.array_equal(mutau.node_feature_, shared.node_feature_)
    assert np.allclose(
        mutau.node_threshold_, shared.node_threshold_, equal_nan=True
    )
    for arm in (0, 1):
        assert np.allclose(mutau.predict(X, arm), shared.predict(X, arm))


def test_model_level_collapse_to_the_reparameterized_shared_tree() -> None:
    """WP5.5-C check 1 at the model level: identical particles and risk."""

    dgp, train, test = model_fixture()
    weights = dgp.grid.weights
    mutau = MutauCWDBRegressor(
        ehat_basis="leaf_share", contrast_shrinkage=0.0, random_state=0,
        **MODEL_PARAMETERS,
    )
    mutau.fit(train.X, train.treatment, train.quantiles, weights)
    shared = CWDBRegressor(
        contrast_rule="ridge", contrast_shrinkage=0.0, random_state=0,
        **MODEL_PARAMETERS,
    )
    shared.fit(train.X, train.treatment, train.quantiles, weights)
    for arm in (0, 1):
        first = mutau.predict_particles(test.X, arm)
        second = shared.predict_particles(test.X, arm)
        assert np.max(np.abs(first - second)) < 1e-10
    assert mutau.train_risk_ == shared.train_risk_


def test_zero_contrast_field_gives_a_treatment_invariant_booster() -> None:
    """WP5.5-C check 4: with d forced to zero, the arms share one law."""

    dgp, train, test = model_fixture()
    weights = dgp.grid.weights
    model = MutauCWDBRegressor(
        ehat_basis="cross_fitted", contrast_shrinkage=0.0,
        force_zero_contrast=True, random_state=1, **MODEL_PARAMETERS,
    )
    model.fit(train.X, train.treatment, train.quantiles, weights)
    for arm in (0, 1):
        particles = model.predict_particles(test.X, arm)
        # Monotone cone: every particle is a valid quantile vector.
        assert np.all(np.diff(particles, axis=-1) >= -1e-12)
    assert np.allclose(
        model.predict_particles(test.X, 0),
        model.predict_particles(test.X, 1),
    )


def test_d2_contrast_field_collapses_while_the_pooled_law_stays_stable() -> None:
    """WP5.5-C check 2 on a null regime: no false contrast, finite risk."""

    dgp = build_dgp("D2", 25)
    train = dgp.sample(400, seed=2)
    test = dgp.sample(200, seed=900_002)
    weights = dgp.grid.weights
    model = MutauCWDBRegressor(
        ehat_basis="cross_fitted", contrast_candidates=(0.0, 500.0), n_folds=2,
        random_state=2, **MODEL_PARAMETERS,
    )
    model.fit(train.X, train.treatment, train.quantiles, weights)
    contrast = model.predict_mean_quantile_effect(test.X)
    # The pooled law is stable: training risk is finite and the arm mean
    # vectors stay close to the (null) truth.
    assert np.isfinite(model.train_risk_)
    assert np.sqrt(np.mean(contrast**2)) < 0.15
    # Held-out energy risk must select the strong regulariser on a null regime.
    assert model.selected_contrast_shrinkage_ >= 500.0


# --------------------------------------------------- validity and determinism


def test_particles_are_canonicalized_and_permutation_invariant() -> None:
    dgp, train, test = model_fixture(seed=4)
    weights = dgp.grid.weights
    base = MutauCWDBRegressor(random_state=5, **MODEL_PARAMETERS)
    base.fit(train.X, train.treatment, train.quantiles, weights)
    permuted = MutauCWDBRegressor(random_state=5, **MODEL_PARAMETERS)
    order = np.random.default_rng(11).permutation(train.X.shape[0])
    permuted.fit(
        train.X[order], train.treatment[order], train.quantiles[order], weights
    )
    for arm in (0, 1):
        first = base.predict_particles(test.X, arm)
        second = permuted.predict_particles(test.X, arm)
        # canonicalize_particles is the identity on canonical output; the
        # assertion that the two outputs are equal is the invariance claim.
        assert np.allclose(first, canonicalize_particles(first))
        assert np.allclose(first, second)


def test_boosting_is_a_monotone_descent() -> None:
    """Every accepted step must not raise the energy risk."""

    dgp, train, test = model_fixture(seed=6)
    weights = dgp.grid.weights
    model = MutauCWDBRegressor(random_state=7, **MODEL_PARAMETERS)
    model.fit(train.X, train.treatment, train.quantiles, weights)
    history = model.training_history_
    assert len(history) >= 1
    for step in history:
        assert step.loss_after <= step.loss_before + 1e-12
        assert step.step_size > 0.0


def test_cross_fitted_basis_uses_the_propensity_and_keeps_the_ridge_scale() -> None:
    """The cross-fitted basis must shrink with the same strength scale as R3.

    The ridge factor mass / (mass + lambda) with the leaf-share basis equals
    the repair track's n_eff / (n_eff + lambda), so lambda = 50 must pull a
    noisy single-leaf contrast toward zero without annihilating a strong one.
    """

    rng = np.random.default_rng(9)
    X = rng.uniform(-1.0, 1.0, size=(200, 5))
    treatment = rng.binomial(1, 0.5, size=200)
    ehat = np.full(200, 0.5)
    gap = 0.5
    target = rng.normal(size=(200, 3)) + treatment[:, None] * gap
    weak = MutauSharedTreeRegressor(
        max_depth=0, min_samples_leaf=2, min_arm_leaf=2,
        ehat_basis="cross_fitted", contrast_shrinkage=50.0,
    ).fit(X, treatment, target, ehat=ehat)
    strong = MutauSharedTreeRegressor(
        max_depth=0, min_samples_leaf=2, min_arm_leaf=2,
        ehat_basis="cross_fitted", contrast_shrinkage=0.0,
    ).fit(X, treatment, target, ehat=ehat)
    weak_gap = float(np.linalg.norm(
        weak.predict(X, 1, ehat=ehat)[0] - weak.predict(X, 0, ehat=ehat)[0]
    ))
    strong_gap = float(np.linalg.norm(
        strong.predict(X, 1, ehat=ehat)[0] - strong.predict(X, 0, ehat=ehat)[0]
    ))
    # 50 shrinks a leaf of 100 effective rows by ~100/(100 + 50) = 2/3.
    assert weak_gap < strong_gap
    assert weak_gap > 0.1 * strong_gap


def test_the_inherited_arm_shrinkage_knob_is_inert() -> None:
    """`arm_shrinkage` reaches the parent leaf rule, which mu/tau replaces.

    The Stage 1 manifest records `arm_shrinkage: 5.0` for `cwdb_mutau`, and a
    reader could reasonably believe R3's ridge was live alongside the contrast
    penalty. It was not: `_leaf_values` is overridden outright. This pins the
    fact so a future edit that starts consuming the knob has to say so, and so
    the Stage 2 manifest can drop the key knowing nothing depends on it.
    """

    X, treatment, target = tree_data(seed=11)
    ehat = np.full(X.shape[0], 0.5)
    predictions = []
    for arm_shrinkage in (0.0, 5.0, 500.0):
        tree = MutauSharedTreeRegressor(
            max_depth=3,
            min_samples_leaf=10,
            min_arm_leaf=5,
            arm_shrinkage=arm_shrinkage,
            ehat_basis="cross_fitted",
            contrast_shrinkage=0.0,
        ).fit(X, treatment, target, ehat=ehat)
        predictions.append(tree.predict(X, 1, ehat=ehat))
    for other in predictions[1:]:
        np.testing.assert_allclose(predictions[0], other, atol=0.0, rtol=0.0)
    assert "arm_shrinkage" in MutauSharedTreeRegressor.INERT_PARENT_PARAMETERS


def test_contrast_shrinkage_by_contrast_is_live() -> None:
    """The complement of the check above: the knob that is live must bite."""

    X, treatment, target = tree_data(seed=11)
    ehat = np.full(X.shape[0], 0.5)
    gaps = []
    for contrast_shrinkage in (0.0, 500.0):
        tree = MutauSharedTreeRegressor(
            max_depth=3,
            min_samples_leaf=10,
            min_arm_leaf=5,
            ehat_basis="cross_fitted",
            contrast_shrinkage=contrast_shrinkage,
        ).fit(X, treatment, target, ehat=ehat)
        gaps.append(
            float(np.linalg.norm(
                tree.predict(X, 1, ehat=ehat) - tree.predict(X, 0, ehat=ehat)
            ))
        )
    assert gaps[1] < 0.5 * gaps[0]
