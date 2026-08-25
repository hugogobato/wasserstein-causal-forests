"""Deterministic weak learners used by C-WDB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.tree import DecisionTreeRegressor


def _feature_target_order(
    X: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Canonicalize rows to remove input-order effects from tree tie breaking."""

    combined = np.column_stack((X, target))
    return np.lexsort(combined[:, ::-1].T)


@dataclass(frozen=True)
class TreeParameters:
    """Shared weak-learner controls."""

    max_depth: int = 2
    min_samples_leaf: int = 5
    random_state: int = 0


class MultiOutputTreeRegressor:
    """A small deterministic wrapper around a multi-output regression tree."""

    def __init__(
        self,
        *,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        random_state: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        target: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "MultiOutputTreeRegressor":
        x = np.asarray(X, dtype=float)
        y = np.asarray(target, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must have shape (n, p)")
        if y.ndim < 2 or y.shape[0] != x.shape[0]:
            raise ValueError("target must have shape (n, ...)")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("X and target must be finite")
        if x.shape[0] < self.min_samples_leaf:
            raise ValueError("not enough rows for min_samples_leaf")

        self.target_shape_ = y.shape[1:]
        flat_target = y.reshape(y.shape[0], -1)
        order = _feature_target_order(x, flat_target)
        ordered_weight = None
        if sample_weight is not None:
            ordered_weight = np.asarray(sample_weight, dtype=float)
            if ordered_weight.shape != (x.shape[0],):
                raise ValueError("sample_weight must have shape (n,)")
            ordered_weight = ordered_weight[order]

        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            splitter="best",
        )
        self.tree_.fit(x[order], flat_target[order], sample_weight=ordered_weight)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "tree_"):
            raise RuntimeError("the weak learner has not been fitted")
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have shape (n, {self.n_features_in_})"
            )
        prediction = np.asarray(self.tree_.predict(x), dtype=float)
        if prediction.ndim == 1:
            prediction = prediction[:, None]
        return prediction.reshape((x.shape[0],) + self.target_shape_)


class IndependentArmTreeRegressor:
    """Two separate observed-arm trees behind one arm-aware interface."""

    def __init__(
        self,
        *,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        random_state: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(
        self, X: ArrayLike, treatment: ArrayLike, target: ArrayLike
    ) -> "IndependentArmTreeRegressor":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        y = np.asarray(target, dtype=float)
        if x.ndim != 2 or a.shape != (x.shape[0],):
            raise ValueError("expected X (n,p) and treatment (n,)")
        if y.ndim < 2 or y.shape[0] != x.shape[0]:
            raise ValueError("target must have shape (n, ...)")
        if not np.all(np.isin(a, (0, 1))):
            raise ValueError("treatment must contain only 0 and 1")
        self.estimators_: dict[int, MultiOutputTreeRegressor] = {}
        for arm in (0, 1):
            mask = a == arm
            if np.sum(mask) < self.min_samples_leaf:
                raise ValueError(f"arm {arm} has too few observations")
            estimator = MultiOutputTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + arm,
            )
            estimator.fit(x[mask], y[mask])
            self.estimators_[arm] = estimator
        self.n_features_in_ = x.shape[1]
        self.target_shape_ = y.shape[1:]
        return self

    def predict(self, X: ArrayLike, arm: int) -> NDArray[np.float64]:
        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1")
        if not hasattr(self, "estimators_"):
            raise RuntimeError("the weak learner has not been fitted")
        return self.estimators_[arm].predict(X)

