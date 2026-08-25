"""Honest shared covariate partitions with observed-arm leaf updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .weak_learners import IndependentArmTreeRegressor


@dataclass
class _SharedNode:
    depth: int
    n_rows: int
    arm_counts: tuple[int, int]
    pooled_value: NDArray[np.float64]
    arm_values: tuple[NDArray[np.float64], NDArray[np.float64]]
    feature: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: "_SharedNode | None" = None
    right: "_SharedNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def pooled_split_gain(target: ArrayLike, left_mask: ArrayLike) -> float:
    """Exact pooled multi-output SSE reduction for a proposed split."""

    y = np.asarray(target, dtype=float)
    left = np.asarray(left_mask, dtype=bool)
    if y.ndim < 2 or left.shape != (y.shape[0],):
        raise ValueError("expected target (n, ...) and left_mask (n,)")
    if not np.any(left) or np.all(left):
        raise ValueError("both split children must be nonempty")
    flat = y.reshape(y.shape[0], -1)

    def sse(rows: NDArray[np.float64]) -> float:
        centered = rows - rows.mean(axis=0)
        return float(np.sum(centered * centered))

    return sse(flat) - sse(flat[left]) - sse(flat[~left])


CONTRAST_RULES = ("arm_shrinkage", "ridge", "threshold")


class ArmSharedTreeRegressor:
    """One shared partition with arm-specific, partially pooled leaf vectors.

    Splits use only gradients observed for each unit's realized arm. Every
    candidate child must contain ``min_arm_leaf`` observations from both arms.

    ``contrast_rule`` selects how much of the raw arm gap survives into the leaf
    update. ``"arm_shrinkage"`` is the frozen G3 rule: each arm mean is pulled
    toward the pooled mean, which shrinks the contrast only as a side effect and
    by the factor ``n_a / (n_a + arm_shrinkage)`` at a balanced leaf. The other
    two rules act on the contrast itself, holding the pooled component fixed.
    """

    def __init__(
        self,
        *,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        min_arm_leaf: int = 2,
        arm_shrinkage: float = 5.0,
        sharing: str = "partial",
        contrast_rule: str = "arm_shrinkage",
        contrast_shrinkage: float = 0.0,
        contrast_threshold_scale: float = 1.0,
        contrast_damping: float = 1.0,
        min_gain: float = 1e-12,
        random_state: int = 0,
    ) -> None:
        if sharing not in {"partial", "forced", "none"}:
            raise ValueError("sharing must be 'partial', 'forced', or 'none'")
        if contrast_rule not in CONTRAST_RULES:
            raise ValueError(f"contrast_rule must be one of {CONTRAST_RULES}")
        if min_arm_leaf < 1:
            raise ValueError("min_arm_leaf must be positive")
        if arm_shrinkage < 0.0:
            raise ValueError("arm_shrinkage must be nonnegative")
        if contrast_shrinkage < 0.0:
            raise ValueError("contrast_shrinkage must be nonnegative")
        if contrast_threshold_scale < 0.0:
            raise ValueError("contrast_threshold_scale must be nonnegative")
        if not 0.0 <= contrast_damping <= 1.0:
            raise ValueError("contrast_damping must lie in [0, 1]")
        # The frozen rule reaches the contrast only through `arm_shrinkage`, so
        # a contrast-specific setting there would be a silent no-op. Refuse it
        # rather than let a manifest look as though it regularises something.
        if contrast_rule == "arm_shrinkage" and contrast_damping != 1.0:
            raise ValueError(
                "contrast_damping applies only to the 'ridge' and 'threshold' rules"
            )
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_arm_leaf = min_arm_leaf
        self.arm_shrinkage = arm_shrinkage
        self.sharing = sharing
        self.contrast_rule = contrast_rule
        self.contrast_shrinkage = contrast_shrinkage
        self.contrast_threshold_scale = contrast_threshold_scale
        self.contrast_damping = contrast_damping
        self.min_gain = min_gain
        self.random_state = random_state

    def fit(
        self, X: ArrayLike, treatment: ArrayLike, gradients: ArrayLike
    ) -> "ArmSharedTreeRegressor":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        g = np.asarray(gradients, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must have shape (n, p)")
        if a.shape != (x.shape[0],):
            raise ValueError("treatment must have shape (n,)")
        if g.ndim < 2 or g.shape[0] != x.shape[0]:
            raise ValueError("gradients must have shape (n, ...)")
        if not np.all(np.isin(a, (0, 1))):
            raise ValueError("treatment must contain only 0 and 1")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)):
            raise ValueError("X and gradients must be finite")
        for arm in (0, 1):
            if np.sum(a == arm) < self.min_arm_leaf:
                raise ValueError(f"arm {arm} has too few observations")

        self.n_features_in_ = x.shape[1]
        self.target_shape_ = g.shape[1:]
        self._X = x
        self._A = a
        self._G = g.reshape(g.shape[0], -1)

        if self.sharing == "none":
            self.independent_ = IndependentArmTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=max(self.min_samples_leaf, self.min_arm_leaf),
                random_state=self.random_state,
            )
            self.independent_.fit(x, a, g)
            self.leaf_statistics_ = []
            return self

        indices = np.arange(x.shape[0])
        self.root_ = self._grow(indices, depth=0)
        self.leaf_statistics_: list[dict[str, object]] = []
        self._collect_leaf_statistics(self.root_)
        self._flatten()
        del self._X, self._A, self._G
        return self

    def _flatten(self) -> None:
        """Store the grown tree as flat arrays so prediction can be batched."""

        features: list[int] = []
        thresholds: list[float] = []
        left_child: list[int] = []
        right_child: list[int] = []
        values: list[NDArray[np.float64]] = []

        def visit(node: _SharedNode) -> int:
            node_id = len(features)
            features.append(-1 if node.feature is None else int(node.feature))
            thresholds.append(
                np.nan if node.threshold is None else float(node.threshold)
            )
            left_child.append(-1)
            right_child.append(-1)
            values.append(np.stack(node.arm_values))
            if not node.is_leaf:
                assert node.left is not None and node.right is not None
                left_child[node_id] = visit(node.left)
                right_child[node_id] = visit(node.right)
            return node_id

        visit(self.root_)
        self.node_feature_ = np.asarray(features, dtype=np.int64)
        self.node_threshold_ = np.asarray(thresholds, dtype=float)
        self.node_left_ = np.asarray(left_child, dtype=np.int64)
        self.node_right_ = np.asarray(right_child, dtype=np.int64)
        # Shape (n_nodes, 2, D): one leaf vector per arm at every node.
        self.node_values_ = np.stack(values)

    def _leaf_ids(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Route every row to its leaf without a Python loop over rows."""

        node = np.zeros(X.shape[0], dtype=np.int64)
        for _ in range(self.max_depth + 1):
            feature = self.node_feature_[node]
            internal = feature >= 0
            if not np.any(internal):
                break
            rows = np.flatnonzero(internal)
            current = node[rows]
            goes_right = (
                X[rows, feature[rows]] > self.node_threshold_[current]
            )
            node[rows] = np.where(
                goes_right, self.node_right_[current], self.node_left_[current]
            )
        return node

    @staticmethod
    def _sse(rows: NDArray[np.float64]) -> float:
        if rows.shape[0] == 0:
            return 0.0
        centered = rows - rows.mean(axis=0)
        return float(np.sum(centered * centered))

    def _contrast_factor(
        self,
        delta: NDArray[np.float64],
        blocks: tuple[NDArray[np.float64], NDArray[np.float64]],
        counts: tuple[int, int],
    ) -> float:
        """Fraction of the raw arm gap the leaf is allowed to keep.

        ``"ridge"`` is a leaf-size-adaptive linear shrinkage: the contrast is
        estimated from ``n_eff = n_0 n_1 / (n_0 + n_1)`` effective observations,
        so the posterior-mean factor under a mean-zero Gaussian prior of
        precision ``contrast_shrinkage`` is ``n_eff / (n_eff + lambda)``.

        ``"threshold"`` is the positive-part James-Stein factor calibrated
        against the leaf's own noise level: ``sigma_squared`` is the plug-in
        variance of the arm gap, so under an exactly null effect the expected
        squared gap equals it and the factor is zero, while a gap far above the
        noise passes through undamped. Within-leaf covariate heterogeneity
        inflates ``sigma_squared``, which makes the rule conservative in the
        direction that matters for a null regime.
        """

        n_0, n_1 = counts
        if self.contrast_rule == "ridge":
            n_effective = n_0 * n_1 / (n_0 + n_1)
            factor = n_effective / (n_effective + self.contrast_shrinkage)
        else:
            squared_gap = float(np.dot(delta, delta))
            if squared_gap <= 0.0:
                return 0.0
            sigma_squared = 0.0
            for block, count in zip(blocks, counts, strict=True):
                if count > 1:
                    sigma_squared += float(np.sum(block.var(axis=0, ddof=1))) / count
            factor = max(
                0.0, 1.0 - self.contrast_threshold_scale * sigma_squared / squared_gap
            )
        return factor * self.contrast_damping

    def _leaf_values(
        self, indices: NDArray[np.int64]
    ) -> tuple[
        NDArray[np.float64],
        tuple[NDArray[np.float64], NDArray[np.float64]],
        tuple[int, int],
    ]:
        gradients = self._G[indices]
        pooled = gradients.mean(axis=0)
        blocks = tuple(gradients[self._A[indices] == arm] for arm in (0, 1))
        counts = (blocks[0].shape[0], blocks[1].shape[0])
        if self.sharing == "forced":
            return pooled, (pooled.copy(), pooled.copy()), counts
        if self.contrast_rule == "arm_shrinkage":
            arm_values = tuple(
                (count * block.mean(axis=0) + self.arm_shrinkage * pooled)
                / (count + self.arm_shrinkage)
                for block, count in zip(blocks, counts, strict=True)
            )
            return pooled, arm_values, counts

        # Reparameterized leaf: hold the pooled gradient fixed and shrink only
        # the arm gap. Writing the arm values as `pooled + (a - share) * factor *
        # delta` keeps the count-weighted average of the two exactly equal to
        # `pooled`, so a contrast rule can never move the pooled component.
        delta = blocks[1].mean(axis=0) - blocks[0].mean(axis=0)
        share = counts[1] / (counts[0] + counts[1])
        factor = self._contrast_factor(delta, blocks, counts)
        adjustment = factor * delta
        arm_values = (pooled - share * adjustment, pooled + (1.0 - share) * adjustment)
        return pooled, arm_values, counts

    def _best_split_reference(
        self, indices: NDArray[np.int64]
    ) -> tuple[int, float, float, NDArray[np.bool_]] | None:
        """Direct transcription of the split rule, one threshold at a time.

        Kept as the semantic definition of `_best_split`. It is
        :math:`O(n^2 M K p)` and is not used in fitting; the vectorized search
        below is checked against it.
        """

        parent_sse = self._sse(self._G[indices])
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
                if (
                    n_left < self.min_samples_leaf
                    or n_right < self.min_samples_leaf
                ):
                    continue
                left_arms = self._A[indices[left]]
                right_arms = self._A[indices[~left]]
                if any(
                    np.sum(child_arms == arm) < self.min_arm_leaf
                    for child_arms in (left_arms, right_arms)
                    for arm in (0, 1)
                ):
                    continue
                gain = (
                    parent_sse
                    - self._sse(self._G[indices[left]])
                    - self._sse(self._G[indices[~left]])
                )
                if best is None or gain > best[2] + 1e-15:
                    best = (feature, float(threshold), float(gain), left)
        return best

    @staticmethod
    def _first_strict_improvement(gains: NDArray[np.float64]) -> int | None:
        """Index the sequential ``gain > best + 1e-15`` scan would settle on."""

        if gains.size == 0 or not np.any(np.isfinite(gains)):
            return None
        running = np.maximum.accumulate(gains)
        previous = np.concatenate(([-np.inf], running[:-1]))
        improved = np.flatnonzero(gains > previous + 1e-15)
        return int(improved[-1]) if improved.size else None

    def _best_split(
        self, indices: NDArray[np.int64]
    ) -> tuple[int, float, float, NDArray[np.bool_]] | None:
        """Vectorized equivalent of `_best_split_reference`.

        For a multi-output SSE criterion the reduction achieved by a split has
        the closed form

            gain = ||S_L||^2 / n_L + ||S_R||^2 / n_R - ||S||^2 / n,

        with ``S`` the column sums of the node's gradients. Sorting each
        covariate once and taking a cumulative sum therefore scores every
        admissible threshold in a single pass, replacing the
        :math:`O(n^2 M K p)` rescan with :math:`O(p n (MK + \\log n))`.
        Gradients are centred at the node first; SSE is translation invariant,
        and centring keeps the cancellation in the identity above benign.
        """

        n_rows = int(indices.size)
        if n_rows < 2 * self.min_samples_leaf:
            return None
        gradients = self._G[indices]
        centered = gradients - gradients.mean(axis=0)
        arms = self._A[indices]
        design = self._X[indices]
        total = centered.sum(axis=0)
        parent_term = float(np.dot(total, total)) / n_rows
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

            left_sums = np.cumsum(centered[order], axis=0)[:-1]
            right_sums = total - left_sums
            gains = (
                np.einsum("ij,ij->i", left_sums, left_sums) / n_left
                + np.einsum("ij,ij->i", right_sums, right_sums) / n_right
                - parent_term
            )

            treated_left = np.cumsum(arms[order])[:-1].astype(float)
            control_left = n_left - treated_left
            treated_total = float(arms.sum())
            treated_right = treated_total - treated_left
            control_right = n_right - treated_right
            admissible = (
                distinct
                & (n_left >= self.min_samples_leaf)
                & (n_right >= self.min_samples_leaf)
                & (treated_left >= self.min_arm_leaf)
                & (control_left >= self.min_arm_leaf)
                & (treated_right >= self.min_arm_leaf)
                & (control_right >= self.min_arm_leaf)
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
        # The reference splits at the midpoint of two adjacent distinct values.
        threshold = float(
            (sorted_values[position] + sorted_values[position + 1]) / 2.0
        )
        left = design[:, feature] <= threshold
        return feature, threshold, float(feature_gains[feature]), left

    def _grow(self, indices: NDArray[np.int64], depth: int) -> _SharedNode:
        pooled, arm_values, counts = self._leaf_values(indices)
        node = _SharedNode(
            depth=depth,
            n_rows=indices.size,
            arm_counts=counts,
            pooled_value=pooled,
            arm_values=arm_values,
        )
        if depth >= self.max_depth:
            return node
        candidate = self._best_split(indices)
        if candidate is None:
            return node
        feature, threshold, gain, left = candidate
        if gain <= self.min_gain:
            return node
        node.feature = feature
        node.threshold = threshold
        node.gain = gain
        node.left = self._grow(indices[left], depth + 1)
        node.right = self._grow(indices[~left], depth + 1)
        return node

    def _collect_leaf_statistics(self, node: _SharedNode) -> None:
        if node.is_leaf:
            self.leaf_statistics_.append(
                {
                    "depth": node.depth,
                    "n_rows": node.n_rows,
                    "arm_counts": node.arm_counts,
                    "pooled_value": node.pooled_value.copy(),
                    "arm_values": (
                        node.arm_values[0].copy(),
                        node.arm_values[1].copy(),
                    ),
                }
            )
            return
        assert node.left is not None and node.right is not None
        self._collect_leaf_statistics(node.left)
        self._collect_leaf_statistics(node.right)

    def _predict_one(self, row: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        node = self.root_
        while not node.is_leaf:
            assert node.feature is not None
            assert node.threshold is not None
            assert node.left is not None and node.right is not None
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.arm_values[arm]

    def predict(
        self, X: ArrayLike, arm: int | ArrayLike
    ) -> NDArray[np.float64]:
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have shape (n, {self.n_features_in_})"
            )
        if np.isscalar(arm):
            arm_value = int(arm)
            if arm_value not in (0, 1):
                raise ValueError("arm must be 0 or 1")
            if self.sharing == "none":
                return self.independent_.predict(x, arm_value)
            arms = np.full(x.shape[0], arm_value, dtype=int)
        else:
            arms = np.asarray(arm, dtype=int)
            if arms.shape != (x.shape[0],) or not np.all(np.isin(arms, (0, 1))):
                raise ValueError("arm must be scalar or a binary vector of length n")
            if self.sharing == "none":
                prediction = np.empty((x.shape[0],) + self.target_shape_)
                for arm_value in (0, 1):
                    mask = arms == arm_value
                    prediction[mask] = self.independent_.predict(
                        x[mask], arm_value
                    )
                return prediction

        leaves = self._leaf_ids(x)
        flat = self.node_values_[leaves, arms]
        return flat.reshape((x.shape[0],) + self.target_shape_)

