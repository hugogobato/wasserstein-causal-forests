from __future__ import annotations

import pickle

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.arm_shared_tree import (
    ArmSharedTreeRegressor,
    pooled_split_gain,
)
from wasserstein_causal_forests.cwdb.model import CWDBRegressor
from wasserstein_causal_forests.cwdb.weak_learners import (
    IndependentArmTreeRegressor,
)


def balanced_tree_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.8, 1.0],
            [-1.5, -1.0],
            [-1.2, 0.5],
            [1.2, 0.0],
            [1.5, 1.0],
            [1.8, -1.0],
            [2.0, 0.5],
        ]
    )
    treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    gradients = np.column_stack(
        (
            np.where(X[:, 0] < 0.0, -2.0, 2.0) + 0.2 * treatment,
            np.where(X[:, 0] < 0.0, 1.0, -1.0) - 0.1 * treatment,
        )
    )
    return X, treatment, gradients


def test_pooled_split_gain_matches_hand_calculation() -> None:
    target = np.array([[0.0], [2.0], [10.0], [12.0]])
    left = np.array([True, True, False, False])
    # Parent SSE = 104, child SSEs = 2 + 2.
    assert pooled_split_gain(target, left) == pytest.approx(100.0)


def test_split_respects_arm_counts_and_uses_expected_feature() -> None:
    X, treatment, gradients = balanced_tree_data()
    tree = ArmSharedTreeRegressor(
        max_depth=1,
        min_samples_leaf=4,
        min_arm_leaf=2,
        arm_shrinkage=1.0,
    ).fit(X, treatment, gradients)
    assert tree.root_.feature == 0
    assert tree.root_.threshold == pytest.approx(0.0)
    assert tree.root_.left is not None and tree.root_.right is not None
    assert tree.root_.left.arm_counts == (2, 2)
    assert tree.root_.right.arm_counts == (2, 2)
    assert all(
        min(statistic["arm_counts"]) >= 2
        for statistic in tree.leaf_statistics_
    )


def test_forced_sharing_has_identical_arm_leaf_updates() -> None:
    X, treatment, gradients = balanced_tree_data()
    tree = ArmSharedTreeRegressor(
        max_depth=1,
        min_samples_leaf=4,
        min_arm_leaf=2,
        sharing="forced",
    ).fit(X, treatment, gradients)
    assert np.allclose(tree.predict(X, 0), tree.predict(X, 1))
    for statistic in tree.leaf_statistics_:
        assert np.allclose(
            statistic["arm_values"][0], statistic["arm_values"][1]
        )


def test_zero_shrinkage_preserves_a_null_arm_update() -> None:
    X, treatment, gradients = balanced_tree_data()
    gradients[treatment == 0] = 0.0
    tree = ArmSharedTreeRegressor(
        max_depth=1,
        min_samples_leaf=4,
        min_arm_leaf=2,
        arm_shrinkage=0.0,
        sharing="partial",
    ).fit(X, treatment, gradients)
    assert np.allclose(tree.predict(X, 0), 0.0)
    assert np.max(np.abs(tree.predict(X, 1))) > 0.0


def test_no_sharing_matches_two_independent_trees() -> None:
    X, treatment, gradients = balanced_tree_data()
    no_sharing = ArmSharedTreeRegressor(
        max_depth=2,
        min_samples_leaf=2,
        min_arm_leaf=1,
        sharing="none",
        random_state=10,
    ).fit(X, treatment, gradients)
    direct = IndependentArmTreeRegressor(
        max_depth=2, min_samples_leaf=2, random_state=10
    ).fit(X, treatment, gradients)
    for arm in (0, 1):
        assert np.array_equal(
            no_sharing.predict(X, arm), direct.predict(X, arm)
        )


def test_within_arm_row_permutation_invariance() -> None:
    X, treatment, gradients = balanced_tree_data()
    permutation = np.array([2, 0, 6, 4, 5, 1, 7, 3])
    first = ArmSharedTreeRegressor(
        max_depth=2,
        min_samples_leaf=2,
        min_arm_leaf=1,
        arm_shrinkage=2.0,
    ).fit(X, treatment, gradients)
    second = ArmSharedTreeRegressor(
        max_depth=2,
        min_samples_leaf=2,
        min_arm_leaf=1,
        arm_shrinkage=2.0,
    ).fit(X[permutation], treatment[permutation], gradients[permutation])
    for arm in (0, 1):
        assert np.allclose(first.predict(X, arm), second.predict(X, arm))


def make_causal_data(seed: int = 73, n: int = 100) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    treatment = np.arange(n) % 2
    rng.shuffle(treatment)
    location = (
        np.where(X[:, 0] > 0.0, 0.8, -0.8)
        + 0.25 * treatment
        + rng.normal(scale=0.3, size=n)
    )
    template = np.array([-1.0, -0.2, 0.4, 1.2])
    quantiles = location[:, None] + template
    weights = np.ones(template.size) / template.size
    return X, treatment, quantiles, weights


def test_model_no_sharing_limit_is_exactly_v0() -> None:
    X, treatment, quantiles, weights = make_causal_data()
    parameters = {
        "n_particles": 3,
        "n_estimators": 5,
        "learning_rate": 0.15,
        "max_depth": 2,
        "min_samples_leaf": 4,
        "min_arm_leaf": 2,
        "collision_epsilon": 1e-3,
        "random_state": 14,
    }
    v0 = CWDBRegressor(architecture="v0", **parameters).fit(
        X, treatment, quantiles, weights
    )
    no_sharing = CWDBRegressor(
        architecture="v1", sharing="none", **parameters
    ).fit(X, treatment, quantiles, weights)
    assert no_sharing.fitted_architecture_ == "v0"
    for arm in (0, 1):
        assert np.array_equal(
            v0.predict_particles(X[:20], arm),
            no_sharing.predict_particles(X[:20], arm),
        )


def test_shared_model_monotonicity_determinism_and_serialization() -> None:
    X, treatment, quantiles, weights = make_causal_data()
    parameters = {
        "architecture": "v1",
        "n_particles": 3,
        "n_estimators": 5,
        "learning_rate": 0.15,
        "max_depth": 2,
        "min_samples_leaf": 6,
        "min_arm_leaf": 2,
        "arm_shrinkage": 2.0,
        "collision_epsilon": 1e-3,
        "random_state": 22,
    }
    first = CWDBRegressor(**parameters).fit(X, treatment, quantiles, weights)
    second = CWDBRegressor(**parameters).fit(X, treatment, quantiles, weights)
    restored = pickle.loads(pickle.dumps(first))
    for arm in (0, 1):
        prediction = first.predict_particles(X[:20], arm)
        assert np.all(np.diff(prediction, axis=-1) >= -1e-12)
        assert np.array_equal(prediction, second.predict_particles(X[:20], arm))
        assert np.array_equal(prediction, restored.predict_particles(X[:20], arm))

