"""The vectorized split search must agree with the reference transcription.

`ArmSharedTreeRegressor._best_split` was rewritten for G3: the original
one-threshold-at-a-time scan costs O(n^2 M K p) and made the tournament
infeasible. `_best_split_reference` retains the original code and is the
semantic definition; these checks pin the fast path to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.arm_shared_tree import (
    ArmSharedTreeRegressor,
    pooled_split_gain,
)


def random_node(
    seed: int, n: int = 90, p: int = 4, n_outputs: int = 6, ties: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    if ties:
        # Repeated covariate values exercise the distinct-value guard.
        X = np.round(X * 3.0) / 3.0
    treatment = rng.binomial(1, 0.5, size=n)
    if treatment.sum() < 8 or (1 - treatment).sum() < 8:
        treatment = np.arange(n) % 2
    signal = np.where(X[:, 0] > 0.0, 1.0, -1.0) + 0.4 * treatment
    gradients = signal[:, None] * rng.normal(size=(1, n_outputs)) + 0.3 * rng.normal(
        size=(n, n_outputs)
    )
    return X, treatment, gradients


def prepared(
    tree: ArmSharedTreeRegressor,
    X: np.ndarray,
    treatment: np.ndarray,
    gradients: np.ndarray,
) -> np.ndarray:
    tree.n_features_in_ = X.shape[1]
    tree._X = X
    tree._A = treatment
    tree._G = gradients.reshape(gradients.shape[0], -1)
    return np.arange(X.shape[0])


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("ties", [False, True])
def test_fast_split_matches_reference(seed: int, ties: bool) -> None:
    X, treatment, gradients = random_node(seed, ties=ties)
    tree = ArmSharedTreeRegressor(min_samples_leaf=7, min_arm_leaf=3)
    indices = prepared(tree, X, treatment, gradients)

    fast = tree._best_split(indices)
    reference = tree._best_split_reference(indices)

    assert (fast is None) == (reference is None)
    if fast is None:
        return
    assert fast[0] == reference[0]
    assert fast[1] == pytest.approx(reference[1], abs=1e-12)
    assert fast[2] == pytest.approx(reference[2], rel=1e-9, abs=1e-9)
    assert np.array_equal(fast[3], reference[3])


@pytest.mark.parametrize("seed", range(6))
def test_fast_split_gain_matches_the_public_definition(seed: int) -> None:
    X, treatment, gradients = random_node(seed)
    tree = ArmSharedTreeRegressor(min_samples_leaf=7, min_arm_leaf=3)
    indices = prepared(tree, X, treatment, gradients)
    fast = tree._best_split(indices)
    assert fast is not None
    assert fast[2] == pytest.approx(
        pooled_split_gain(gradients, fast[3]), rel=1e-9, abs=1e-9
    )


def test_no_admissible_split_returns_none() -> None:
    X = np.zeros((20, 3))
    treatment = np.arange(20) % 2
    gradients = np.ones((20, 2))
    tree = ArmSharedTreeRegressor(min_samples_leaf=5, min_arm_leaf=2)
    indices = prepared(tree, X, treatment, gradients)
    assert tree._best_split(indices) is None
    assert tree._best_split_reference(indices) is None


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("sharing", ["partial", "forced"])
def test_batched_prediction_matches_row_by_row(seed: int, sharing: str) -> None:
    X, treatment, gradients = random_node(seed)
    tree = ArmSharedTreeRegressor(
        max_depth=3,
        min_samples_leaf=6,
        min_arm_leaf=2,
        arm_shrinkage=1.5,
        sharing=sharing,
    ).fit(X, treatment, gradients)

    for arm in (0, 1):
        batched = tree.predict(X, arm)
        row_wise = np.vstack([tree._predict_one(row, arm) for row in X])
        assert np.array_equal(batched, row_wise.reshape(batched.shape))

    mixed = tree.predict(X, treatment)
    row_wise = np.vstack(
        [tree._predict_one(row, int(a)) for row, a in zip(X, treatment)]
    )
    assert np.array_equal(mixed, row_wise.reshape(mixed.shape))
