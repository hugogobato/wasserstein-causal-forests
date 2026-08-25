"""WP5.5-B: the vector R-learner ``cwdb_rmean``.

The weak common-target comparison is the grid causal mean quantile vector
``MEANQ-A-K``. The R-learner's hypothesis is that the weak performance there is
driven by confounding and prognostic-surface leakage, not by the particle-law
representation. The semantic objective is the cross-fitted vector R-loss

    L_R(t) = (1 / n) sum_i || Z_i - mhat_{-i}(X_i)
                              - (A_i - ehat_{-i}(X_i)) t(X_i) ||_2^2,

with Z = W^{1/2} q(Y) in the rescaled coordinates. The implementation never
divides by A_i - ehat_i: the R-loss tree below computes each leaf value as
(sum w_i r_i) / (sum w_i^2), an aggregate ratio, so no single observation can
produce an unstable pseudo-outcome near an overlap boundary.

What this method is not: it estimates a conditional mean vector, so it does not
produce ``LAW-A-K`` and cannot be credited with D5 law separation, D6 mode
coverage, or D7 unseen-functional transfer. The adapter reports those targets
as ``not_applicable`` by contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..cwdb.geometry import from_rescaled, to_rescaled
from .nuisance import (
    CrossFittedNuisance,
    FoldPlan,
    NUISANCE_BUDGET,
    PROPENSITY_CLIP_HIGH,
    PROPENSITY_CLIP_LOW,
)

#: Frozen contrast budget for the Stage 1 mechanism screen. The pilot scan
#: (seeds 100-104, outside the Stage 1 manifest) showed that the R-loss trees
#: accumulate noise along the boosting path: the D8 contrast error is minimized
#: near 20 steps at learning rate 0.12 and rises monotonically afterwards, so
#: 20 steps is the frozen stopping rule and the ridge strength is the selected
#: hyperparameter. Depth 4 keeps the deterministic D0 surface recoverable.
CONTRAST_BUDGET = {
    "n_estimators": 20,
    "learning_rate": 0.12,
    "max_depth": 4,
    "min_samples_leaf": 10,
}

#: Below this weight mass a leaf cannot identify a contrast direction, so the
#: leaf value stays zero rather than dividing by a near-zero denominator.
MIN_WEIGHT_MASS = 1e-6


class RLossTree:
    """One depth-limited tree fitted to the vector R-loss.

    For a leaf with residuals r_i and treatment weights w_i = A_i - ehat_i,
    the leaf value minimizes sum || r_i - w_i v ||^2, whose closed form is
    v = (sum w_i r_i) / (sum w_i^2). The split search maximizes the exact gain
    SSE(node) - SSE(left) - SSE(right) of the same loss. Splits use only the
    residual and the treatment weight, never a division by an individual w_i.

    ``contrast_shrinkage`` multiplies each leaf value by ``mass / (mass + lam)``
    with ``mass = sum w_i^2``: the same leaf-size-adaptive ridge the G3 repair
    track calibrated for the particle booster, now applied to the R-loss leaf.
    A leaf with no identifiable contrast direction (tiny mass) is left at zero.
    """

    def __init__(
        self,
        *,
        max_depth: int = 4,
        min_samples_leaf: int = 10,
        min_weight_mass: float = MIN_WEIGHT_MASS,
        contrast_shrinkage: float | NDArray[np.float64] = 0.0,
        random_state: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_mass = min_weight_mass
        self.contrast_shrinkage = contrast_shrinkage
        self.random_state = random_state

    # ------------------------------------------------------------------- fitting

    def fit(
        self,
        X: ArrayLike,
        residual: ArrayLike,
        weight: ArrayLike,
    ) -> "RLossTree":
        x = np.asarray(X, dtype=float)
        r = np.asarray(residual, dtype=float)
        w = np.asarray(weight, dtype=float)
        if x.ndim != 2 or r.ndim != 2 or r.shape[0] != x.shape[0]:
            raise ValueError("expected X (n, p) and residual (n, K)")
        if w.shape != (x.shape[0],):
            raise ValueError("weight must have shape (n,)")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(r)):
            raise ValueError("X and residual must be finite")

        self.n_features_in_ = x.shape[1]
        self.n_coordinates_ = r.shape[1]
        # Canonical row order makes the tree invariant to input row permutation.
        combined = np.column_stack((x, r, w))
        order = np.lexsort(combined[:, ::-1].T)
        x, r, w = x[order], r[order], w[order]
        self._X, self._R, self._W = x, r, w
        indices = np.arange(x.shape[0])
        self._root_ = self._grow(indices, depth=0)
        self._flatten()
        del self._X, self._R, self._W
        return self

    def _leaf_value(
        self, indices: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], float, float]:
        """(value, SSE, weight mass) of the R-loss for one node's rows.

        ``contrast_shrinkage`` may be a per-column vector, in which case each
        coordinate receives its own mass ridge; the SSE used for split gains is
        always the unshrunk one, so the shrinkage regularises leaf values only.
        """
        r = self._R[indices]
        w = self._W[indices]
        mass = float(np.dot(w, w))
        if mass <= self.min_weight_mass:
            return np.zeros(self.n_coordinates_), float(np.sum(r * r)), mass
        value = np.einsum("i,ij->j", w, r) / mass
        sse = float(
            np.sum(r * r) - float(np.dot(value, value)) * mass
        )
        shrinkage = np.asarray(self.contrast_shrinkage, dtype=float)
        if shrinkage.ndim == 0:
            value = value * (mass / (mass + float(shrinkage)))
        else:
            value = value * (mass / (mass + shrinkage))
        return value, sse, mass

    def _grow(self, indices: NDArray[np.int64], depth: int) -> dict[str, object]:
        value, sse, mass = self._leaf_value(indices)
        node: dict[str, object] = {
            "value": value,
            "sse": sse,
            "mass": mass,
            "feature": None,
            "threshold": None,
            "gain": 0.0,
            "left": None,
            "right": None,
        }
        if depth >= self.max_depth or indices.size < 2 * self.min_samples_leaf:
            return node
        candidate = self._best_split(indices)
        if candidate is None:
            return node
        feature, threshold, gain, left = candidate
        if gain <= 1e-12:
            return node
        node["feature"] = feature
        node["threshold"] = threshold
        node["gain"] = gain
        node["left"] = self._grow(indices[left], depth + 1)
        node["right"] = self._grow(indices[~left], depth + 1)
        return node

    def _flatten(self) -> None:
        features: list[int] = []
        thresholds: list[float] = []
        left_child: list[int] = []
        right_child: list[int] = []
        values: list[NDArray[np.float64]] = []

        def visit(node: dict[str, object]) -> int:
            node_id = len(features)
            features.append(-1 if node["feature"] is None else int(node["feature"]))
            thresholds.append(
                np.nan if node["threshold"] is None else float(node["threshold"])
            )
            left_child.append(-1)
            right_child.append(-1)
            values.append(np.asarray(node["value"], dtype=float))
            if node["feature"] is not None:
                assert node["left"] is not None and node["right"] is not None
                left_child[node_id] = visit(node["left"])
                right_child[node_id] = visit(node["right"])
            return node_id

        visit(self._root_)
        self.node_feature_ = np.asarray(features, dtype=np.int64)
        self.node_threshold_ = np.asarray(thresholds, dtype=float)
        self.node_left_ = np.asarray(left_child, dtype=np.int64)
        self.node_right_ = np.asarray(right_child, dtype=np.int64)
        self.node_values_ = np.stack(values)

    def _best_split(
        self, indices: NDArray[np.int64]
    ) -> tuple[int, float, float, NDArray[np.bool_]] | None:
        """Vectorized exact-gain split search over the R-loss.

        For a set, SSE = sum ||r||^2 - ||sum w r||^2 / sum w^2. The first term
        is constant across splits of a node, so the gain only depends on the
        cumulative sums of w, wr, and w^2 along each sorted covariate, which
        one sort and one prefix pass evaluate for every admissible threshold.
        """

        n_rows = int(indices.size)
        r = self._R[indices]
        w = self._W[indices]
        design = self._X[indices]
        total_wr = np.einsum("i,ij->j", w, r)
        total_ww = float(np.dot(w, w))
        if total_ww <= self.min_weight_mass:
            return None
        parent_term = float(np.dot(total_wr, total_wr)) / total_ww

        n_left = np.arange(1, n_rows, dtype=float)
        n_right = n_rows - n_left
        feature_gains = np.full(self.n_features_in_, -np.inf)
        feature_positions = np.zeros(self.n_features_in_, dtype=np.int64)
        feature_orders: list[NDArray[np.int64] | None] = [None] * self.n_features_in_

        for feature in range(self.n_features_in_):
            values = design[:, feature]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            distinct = sorted_values[:-1] < sorted_values[1:]
            if not np.any(distinct):
                continue

            prefix_w = np.cumsum(w[order])[:-1]
            prefix_ww = np.cumsum(w[order] * w[order])[:-1]
            prefix_wr = np.cumsum(w[order, None] * r[order], axis=0)[:-1]
            left_mass = prefix_ww
            right_mass = total_ww - prefix_ww
            right_wr = total_wr - prefix_wr

            term_left = np.zeros(n_rows - 1)
            term_right = np.zeros(n_rows - 1)
            valid = (left_mass > self.min_weight_mass) & (right_mass > self.min_weight_mass)
            if np.any(valid):
                term_left[valid] = np.einsum(
                    "ij,ij->i", prefix_wr[valid], prefix_wr[valid]
                ) / left_mass[valid]
                term_right[valid] = np.einsum(
                    "ij,ij->i", right_wr[valid], right_wr[valid]
                ) / right_mass[valid]
            gains = term_left + term_right - parent_term

            admissible = (
                distinct
                & valid
                & (n_left >= self.min_samples_leaf)
                & (n_right >= self.min_samples_leaf)
            )
            position = self._first_strict_improvement(
                np.where(admissible, gains, -np.inf)
            )
            if position is None:
                continue
            feature_gains[feature] = gains[position]
            feature_positions[feature] = position
            feature_orders[feature] = order

        feature = self._first_strict_improvement(feature_gains)
        if feature is None:
            return None
        order = feature_orders[feature]
        assert order is not None
        position = int(feature_positions[feature])
        sorted_values = design[order, feature]
        threshold = float(
            (sorted_values[position] + sorted_values[position + 1]) / 2.0
        )
        left = design[:, feature] <= threshold
        return feature, threshold, float(feature_gains[feature]), left

    @staticmethod
    def _first_strict_improvement(gains: NDArray[np.float64]) -> int | None:
        if gains.size == 0 or not np.any(np.isfinite(gains)):
            return None
        running = np.maximum.accumulate(gains)
        previous = np.concatenate(([-np.inf], running[:-1]))
        improved = np.flatnonzero(gains > previous + 1e-15)
        return int(improved[-1]) if improved.size else None

    # ---------------------------------------------------------------- prediction

    def _leaf_ids(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        node = np.zeros(X.shape[0], dtype=np.int64)
        for _ in range(self.max_depth + 1):
            feature = self.node_feature_[node]
            internal = feature >= 0
            if not np.any(internal):
                break
            rows = np.flatnonzero(internal)
            current = node[rows]
            goes_right = X[rows, feature[rows]] > self.node_threshold_[current]
            node[rows] = np.where(
                goes_right, self.node_right_[current], self.node_left_[current]
            )
        return node

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "node_values_"):
            raise RuntimeError("the tree has not been fitted")
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        leaves = self._leaf_ids(x)
        return self.node_values_[leaves]

    # ------------------------------------------------------------------ reference

    def fit_reference(
        self,
        X: ArrayLike,
        residual: ArrayLike,
        weight: ArrayLike,
    ) -> "RLossTree":
        """Reference implementation: one threshold at a time, for the tests.

        This is the semantic definition of the split rule written out directly
        from the R-loss, in the same spirit as the shared tree's
        ``_best_split_reference``.
        """

        x = np.asarray(X, dtype=float)
        r = np.asarray(residual, dtype=float)
        w = np.asarray(weight, dtype=float)
        self.n_features_in_ = x.shape[1]
        self.n_coordinates_ = r.shape[1]
        combined = np.column_stack((x, r, w))
        order = np.lexsort(combined[:, ::-1].T)
        x, r, w = x[order], r[order], w[order]
        self._X, self._R, self._W = x, r, w
        indices = np.arange(x.shape[0])
        self._root_ = self._grow_reference(indices, depth=0)
        self._flatten()
        del self._X, self._R, self._W
        return self

    def _grow_reference(
        self, indices: NDArray[np.int64], depth: int
    ) -> dict[str, object]:
        value, sse, mass = self._leaf_value(indices)
        node: dict[str, object] = {
            "value": value,
            "sse": sse,
            "mass": mass,
            "feature": None,
            "threshold": None,
            "gain": 0.0,
            "left": None,
            "right": None,
        }
        if depth >= self.max_depth or indices.size < 2 * self.min_samples_leaf:
            return node
        parent_sse, _ = self._sse_of(indices)
        best: tuple[int, float, float, NDArray[np.bool_]] | None = None
        for feature in range(self.n_features_in_):
            values = self._X[indices, feature]
            unique = np.unique(values)
            if unique.size < 2:
                continue
            thresholds = (unique[:-1] + unique[1:]) / 2.0
            for threshold in thresholds:
                left = values <= threshold
                n_left = int(np.sum(left))
                n_right = indices.size - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                left_sse, left_mass = self._sse_of(indices[left])
                right_sse, right_mass = self._sse_of(indices[~left])
                if left_mass <= self.min_weight_mass or right_mass <= self.min_weight_mass:
                    continue
                gain = parent_sse - left_sse - right_sse
                if best is None or gain > best[2] + 1e-15:
                    best = (feature, float(threshold), float(gain), left)
        if best is None or best[2] <= 1e-12:
            return node
        node["feature"], node["threshold"], node["gain"], left = best
        node["left"] = self._grow_reference(indices[left], depth + 1)
        node["right"] = self._grow_reference(indices[~left], depth + 1)
        return node

    def _sse_of(self, indices: NDArray[np.int64]) -> tuple[float, float]:
        _, sse, mass = self._leaf_value(indices)
        return sse, mass


class VectorRLearner:
    """Cross-fitted vector R-learner for the rescaled quantile vector.

    ``ehat`` and ``mhat`` may be supplied explicitly (arrays of length n or
    callables) for the exact-nuisance tests and the collapse checks; when they
    are absent the learner cross-fits them internally.

    ``contrast_shrinkage`` is the fixed ridge strength on the contrast leaf
    values. When ``contrast_candidates`` is given instead, the strength is
    chosen on held-out R-loss over two folds (A15): each candidate is fitted on
    the complement of its fold and scored on the fold, and the final model is
    refitted on the whole sample at the winner. Ties break toward the stronger
    regulariser, the null-safe default.
    """

    def __init__(
        self,
        *,
        n_nuisance_folds: int = 5,
        contrast_budget: dict[str, int | float] | None = None,
        nuisance_budget: dict[str, int | float] | None = None,
        propensity_clip: tuple[float, float] = (PROPENSITY_CLIP_LOW, PROPENSITY_CLIP_HIGH),
        contrast_shrinkage: float = 0.0,
        contrast_candidates: tuple[float, ...] | None = None,
        n_selection_folds: int = 2,
        random_state: int = 0,
    ) -> None:
        self.n_nuisance_folds = n_nuisance_folds
        self.contrast_budget = dict(CONTRAST_BUDGET if contrast_budget is None else contrast_budget)
        self.nuisance_budget = dict(NUISANCE_BUDGET if nuisance_budget is None else nuisance_budget)
        self.propensity_clip = tuple(float(c) for c in propensity_clip)
        self.contrast_shrinkage = float(contrast_shrinkage)
        self.contrast_candidates = (
            None
            if contrast_candidates is None
            else tuple(float(c) for c in contrast_candidates)
        )
        self.n_selection_folds = n_selection_folds
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        grid_weights: ArrayLike,
        *,
        ehat: ArrayLike | None | callable = None,  # noqa: F821
        mhat: ArrayLike | None | callable = None,  # noqa: F821
    ) -> "VectorRLearner":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        q = np.asarray(quantiles, dtype=float)
        w = np.asarray(grid_weights, dtype=float)
        if x.ndim != 2 or a.shape != (x.shape[0],) or q.ndim != 2 or q.shape[0] != x.shape[0]:
            raise ValueError("expected X (n, p), treatment (n,), quantiles (n, K)")
        z = to_rescaled(q, w)
        self.n_features_in_ = x.shape[1]
        self.n_coordinates_ = q.shape[1]
        self.grid_weights_ = w.copy()

        if ehat is None or mhat is None:
            nuisance = CrossFittedNuisance(
                n_folds=self.n_nuisance_folds,
                propensity_clip=self.propensity_clip,
                prognostic_budget=self.nuisance_budget,
                random_state=self.random_state,
            )
            nuisance.fit(x, a, z)
            self.nuisance_ = nuisance
            mhat_values = nuisance.mhat_oof_
            ehat_values = nuisance.ehat_oof_
        else:
            self.nuisance_ = None
            mhat_raw = self._evaluate_nuisance(mhat, x)
            mhat_values = (
                np.broadcast_to(mhat_raw[:, None], z.shape).copy()
                if mhat_raw.ndim == 1
                else mhat_raw
            )
            ehat_values = np.clip(self._evaluate_nuisance(ehat, x), *self.propensity_clip)

        treatment_weight = a - ehat_values
        if self.contrast_candidates is not None:
            contrast_shrinkage = self._select_shrinkage(
                x, a, z, ehat_values, mhat_values
            )
            self.selected_contrast_shrinkage_ = contrast_shrinkage
            self.contrast_shrinkage = contrast_shrinkage
        else:
            contrast_shrinkage = self.contrast_shrinkage

        estimators, contrast, residual = self._fit_boosting(
            x, treatment_weight, z, mhat_values, contrast_shrinkage
        )
        self.estimators_ = estimators
        self.ehat_train_ = ehat_values
        self.mhat_train_ = mhat_values
        self.contrast_train_ = contrast
        self.train_risk_ = float(np.mean(np.sum(residual * residual, axis=1)))
        return self

    def _fit_boosting(
        self,
        x: NDArray[np.float64],
        treatment_weight: NDArray[np.float64],
        z: NDArray[np.float64],
        mhat_values: NDArray[np.float64],
        contrast_shrinkage: float,
    ) -> tuple[tuple[RLossTree, ...], NDArray[np.float64], NDArray[np.float64]]:
        contrast = np.zeros_like(z)
        residual = z - mhat_values - treatment_weight[:, None] * contrast
        estimators: list[RLossTree] = []
        learning_rate = float(self.contrast_budget["learning_rate"])
        for iteration in range(int(self.contrast_budget["n_estimators"])):
            estimator = RLossTree(
                max_depth=int(self.contrast_budget["max_depth"]),
                min_samples_leaf=int(self.contrast_budget["min_samples_leaf"]),
                contrast_shrinkage=contrast_shrinkage,
                random_state=self.random_state + iteration,
            )
            estimator.fit(x, residual, treatment_weight)
            direction = estimator.predict(x)
            contrast = contrast + learning_rate * direction
            residual = z - mhat_values - treatment_weight[:, None] * contrast
            estimators.append(estimator)
        return tuple(estimators), contrast, residual

    def _select_shrinkage(
        self,
        x: NDArray[np.float64],
        a: NDArray[np.int64],
        z: NDArray[np.float64],
        ehat_values: NDArray[np.float64],
        mhat_values: NDArray[np.float64],
    ) -> float:
        """Held-out R-loss over two folds for every candidate strength.

        The candidate models are fitted on each fold's complement and scored on
        the fold, and the nuisance values entering both sides are the ones the
        5-fold stack produced out of fold, so a scored row never sees itself in
        the contrast trees or in its own mhat/ehat.

        This is nested-approximate, not fully nested, and the difference is
        stated rather than glossed: the nuisance stack was fitted once on the
        whole sample, so a *training* row's residual carries an mhat whose fold
        model saw the selection holdout's outcomes. The path is second order (it
        reaches the selected lambda only through the training residuals, never
        through a scored row's own prediction), and the alternative is the fully
        nested refit `MutauCWDBRegressor._select_contrast_strength` performs,
        which costs the runtime flagged in the G3.5 memo. Any comparison of
        rmean's and mutau's selection cost must account for that asymmetry.
        """
        assert self.contrast_candidates is not None
        folds = FoldPlan.stratified(
            a, self.n_selection_folds, keys=x, random_state=self.random_state + 7
        )
        records: list[tuple[float, float]] = []
        for candidate in self.contrast_candidates:
            risks: list[float] = []
            for fold in range(self.n_selection_folds):
                train_rows = folds.labels != fold
                holdout = folds.labels == fold
                estimators, contrast, _ = self._fit_boosting(
                    x[train_rows],
                    a[train_rows] - ehat_values[train_rows],
                    z[train_rows],
                    mhat_values[train_rows],
                    candidate,
                )
                # Reconstruct the held-out contrast with the same trees.
                held = np.zeros((int(np.sum(holdout)), self.n_coordinates_))
                learning_rate = float(self.contrast_budget["learning_rate"])
                for estimator in estimators:
                    held = held + learning_rate * estimator.predict(x[holdout])
                residual = (
                    z[holdout] - mhat_values[holdout]
                    - (a[holdout] - ehat_values[holdout])[:, None] * held
                )
                risks.append(float(np.mean(np.sum(residual * residual, axis=1))))
            records.append((candidate, float(np.mean(risks))))
        self.selection_records_ = tuple(
            (float(candidate), float(risk)) for candidate, risk in records
        )
        # Ties break toward the stronger regulariser, the null-safe default.
        best = min(records, key=lambda item: (item[1], -item[0]))
        return float(best[0])

    @staticmethod
    def _evaluate_nuisance(
        nuisance: ArrayLike | callable,  # noqa: F821
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if callable(nuisance):
            return np.asarray(nuisance(X), dtype=float)
        values = np.asarray(nuisance, dtype=float)
        # e is a scalar field, m is a vector field over the grid coordinates.
        allowed = {(X.shape[0],)}
        if values.ndim == 2:
            allowed.add((X.shape[0], values.shape[1]))
        if values.shape not in allowed:
            raise ValueError("explicit nuisance values must have shape (n,) or (n, K)")
        return values

    def predict_contrast(self, X: ArrayLike) -> NDArray[np.float64]:
        """The fitted contrast t(X) in the rescaled coordinates."""
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        current = np.zeros((x.shape[0], self.n_coordinates_))
        learning_rate = float(self.contrast_budget["learning_rate"])
        for estimator in self.estimators_:
            current = current + learning_rate * estimator.predict(x)
        return current

    def predict_arm_mean(self, X: ArrayLike, arm: int) -> NDArray[np.float64]:
        """E[Z | X, A = arm] reconstructed from m, e, and the contrast.

        The R-loss identifies the contrast; the arm means are recovered from
        the identity m(x) = e(x) m_1(x) + (1 - e(x)) m_0(x):
        m_0 = m - e tau and m_1 = m + (1 - e) tau.
        """
        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1")
        x = np.asarray(X, dtype=float)
        contrast = self.predict_contrast(x)
        if self.nuisance_ is None:
            raise RuntimeError("arm means need a fitted nuisance stack")
        m = self.nuisance_.mhat(x)
        e = self.nuisance_.ehat(x)
        if arm == 0:
            return m - e[:, None] * contrast
        return m + (1.0 - e)[:, None] * contrast

    def predict_mean_quantiles(self, X: ArrayLike, arm: int) -> NDArray[np.float64]:
        """Arm mean quantile vector in the original (unscaled) coordinates."""
        return from_rescaled(self.predict_arm_mean(X, arm), self.grid_weights_)
