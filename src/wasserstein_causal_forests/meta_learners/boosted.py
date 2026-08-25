"""Deterministic gradient-boosted multi-output regressor used by the Phase 5.5
nuisances and contrast regressions.

The weak learner is the certified deterministic tree from ``cwdb.weak_learners``,
so everything here inherits its canonical row ordering and tie-breaking. The
boosting loop is plain: each step fits a tree to the current residual and adds
``learning_rate`` times its prediction, with no line search. A line search would
make every nuisance fit measurably slower for a gain that only matters when the
weak learner is crude, and the nuisances are deliberately interchangeable parts
whose quality is audited by the WP5.5-A collapse and leakage tests rather than
by tuning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..cwdb.weak_learners import MultiOutputTreeRegressor


class BoostedMultiOutputRegressor:
    """Small deterministic gradient-boosted multi-output regressor."""

    def __init__(
        self,
        *,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        random_state: int = 0,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(
        self, X: ArrayLike, Y: ArrayLike
    ) -> "BoostedMultiOutputRegressor":
        x = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or y.shape[0] != x.shape[0]:
            raise ValueError("expected X (n, p) and Y (n, d)")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("X and Y must be finite")

        self.n_features_in_ = x.shape[1]
        self.base_mean_ = y.mean(axis=0)
        self.estimators_: list[MultiOutputTreeRegressor] = []
        current = np.tile(self.base_mean_, (x.shape[0], 1))
        for iteration in range(self.n_estimators):
            residual = y - current
            estimator = MultiOutputTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + iteration,
            )
            estimator.fit(x, residual)
            current = current + self.learning_rate * estimator.predict(x)
            self.estimators_.append(estimator)
        self.train_prediction_ = current
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("the regressor has not been fitted")
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        current = np.zeros((x.shape[0],) + self.estimators_[0].target_shape_)
        for estimator in self.estimators_:
            current = current + self.learning_rate * estimator.predict(x)
        return current + self.base_mean_
