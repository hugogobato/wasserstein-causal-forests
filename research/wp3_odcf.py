"""Finite-grid ODCF prototype for WP3.

This module deliberately stops at the pre-simulation boundary in the theory
plan.  It implements the finite-vector score, honest forest weights, the
composite squared-error split, an MMD-on-score split inspired by DRF, PAVA for
arm means, and a provisional inner-sample bootstrap correction.  It does not
implement official DRF/Causal-DRF and makes no consistency or inference claim.

The implementation uses region indices as the unit of cross-fitting.  Tree
split and populate indices are disjoint conditional on the supplied score
matrix.  Because cross-fitted nuisances can couple scores across those sets,
this is not an unconditional outcome-level honesty claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


Array = np.ndarray
DEFAULT_RANDOM_STATE = 20260727


def _as_2d(values: Array, name: str) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _as_1d(values: Array, name: str) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _as_index_array(values: Sequence[int], name: str) -> Array:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    numeric = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric != np.floor(numeric))
    ):
        raise ValueError(f"{name} must contain finite integers")
    return numeric.astype(int)


def _require_integer(value: object, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def trapezoidal_grid_weights(k: int) -> Array:
    """Frozen trapezoidal weights for an equally spaced [0.05, 0.95] grid."""
    k = _require_integer(k, "K", minimum=1)
    if k == 1:
        return np.ones(1, dtype=float)
    step = 0.9 / (k - 1)
    weights = np.full(k, step, dtype=float)
    weights[[0, -1]] = step / 2.0
    return weights


def _validate_grid_weights(k: int, quadrature_weights: Optional[Array]) -> Array:
    k = _require_integer(k, "K", minimum=1)
    if quadrature_weights is None:
        return trapezoidal_grid_weights(k)
    weights = _as_1d(quadrature_weights, "quadrature_weights")
    if len(weights) != k or np.any(weights <= 0):
        raise ValueError("quadrature weights must be positive and match K")
    return weights


def _mad(values: Array) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


@dataclass(frozen=True)
class CoordinateScaler:
    """Training-only scales for the finite target vector.

    The curve coordinates are left on their declared analysis scale.  Each
    nonlinear functional coordinate is multiplied by a positive scale that
    matches its robust training dispersion to the integrated curve dispersion.
    ``scales`` therefore contains one for every output coordinate, with ones in
    the first K positions.
    """

    K: int
    J: int
    scales: Array
    quadrature_weights: Array
    rule: str
    reference_dispersion: float
    denominators: Array

    @classmethod
    def fit(
        cls,
        scores: Array,
        K: int,
        quadrature_weights: Optional[Array] = None,
        rule: str = "robust_sd",
        treatment: Optional[Array] = None,
        propensity: Optional[Array] = None,
    ) -> "CoordinateScaler":
        scores = _as_2d(scores, "scores")
        K = _require_integer(K, "K", minimum=1)
        if not 1 <= K <= scores.shape[1]:
            raise ValueError("K must be between one and the number of outputs")
        J = scores.shape[1] - K
        if rule not in {"robust_sd", "mad", "null_score_se"}:
            raise ValueError("unknown scaling rule")
        curve_weights = _validate_grid_weights(K, quadrature_weights)
        curve_variances = np.var(scores[:, :K], axis=0, ddof=1 if len(scores) > 1 else 0)
        reference = float(
            np.sqrt(
                np.dot(curve_weights, curve_variances)
                / np.sum(curve_weights)
            )
        )
        # A calibration sample can have a constant curve (the pure-functional
        # falsification DGP is intentionally such a case).  Do not suppress a
        # genuinely varying scalar target merely because the curve happens to
        # have zero empirical dispersion; use the declared unit scale then.
        reference = reference if reference > 1e-12 else 1.0

        denominators = np.ones(J, dtype=float)
        for j in range(J):
            coordinate = scores[:, K + j]
            if rule == "robust_sd":
                denominator = 1.4826 * _mad(coordinate)
            elif rule == "mad":
                denominator = _mad(coordinate)
            else:
                if treatment is None or propensity is None:
                    raise ValueError(
                        "null_score_se requires treatment and propensity"
                    )
                treatment = _as_1d(treatment, "treatment")
                propensity = _as_1d(propensity, "propensity")
                if len(treatment) != len(scores) or len(propensity) != len(scores):
                    raise ValueError(
                        "treatment, propensity, and scores have different lengths"
                    )
                if not np.all(np.isin(treatment, [0.0, 1.0])):
                    raise ValueError("treatment must be binary")
                if np.any((propensity <= 0) | (propensity >= 1)):
                    raise ValueError("propensity must lie strictly inside (0, 1)")
                n_eff = float(np.sum(propensity * (1.0 - propensity)))
                n_eff = max(n_eff, 1.0)
                std = float(np.std(coordinate, ddof=1 if len(coordinate) > 1 else 0))
                denominator = std / np.sqrt(n_eff)
            if not np.isfinite(denominator) or denominator <= 1e-12:
                fallback = float(np.std(coordinate, ddof=1 if len(coordinate) > 1 else 0))
                denominator = fallback if np.isfinite(fallback) and fallback > 1e-12 else 1.0
            denominators[j] = denominator

        scales = np.ones(scores.shape[1], dtype=float)
        if J:
            scales[K:] = reference / denominators
        return cls(
            K=K,
            J=J,
            scales=scales,
            quadrature_weights=curve_weights,
            rule=rule,
            reference_dispersion=reference,
            denominators=denominators,
        )

    def transform(self, values: Array) -> Array:
        values = _as_2d(values, "values")
        if values.shape[1] != self.K + self.J:
            raise ValueError("values have the wrong number of target coordinates")
        return values * self.scales

    def inverse(self, values: Array) -> Array:
        values = _as_2d(values, "values")
        if values.shape[1] != self.K + self.J:
            raise ValueError("values have the wrong number of target coordinates")
        return values / self.scales


def coordinate_weights(
    K: int,
    J: int,
    quadrature_weights: Optional[Array] = None,
    active_coordinates: Optional[Sequence[int]] = None,
) -> Array:
    """Return the finite direct-sum squared-norm weights."""
    K = _require_integer(K, "K", minimum=1)
    J = _require_integer(J, "J", minimum=0)
    curve_weights = _validate_grid_weights(K, quadrature_weights)
    weights = np.r_[curve_weights, np.ones(J, dtype=float)]
    if active_coordinates is None:
        return weights
    active = _as_index_array(active_coordinates, "active_coordinates")
    if np.any(active < 0) or np.any(active >= K + J):
        raise ValueError("active coordinate out of range")
    return weights[active]


def weighted_sse(
    values: Array,
    indices: Sequence[int],
    weights: Array,
) -> float:
    """Within-node weighted squared error, using equal region weights."""
    values = _as_2d(values, "values")
    weights = _as_1d(weights, "weights")
    if len(weights) != values.shape[1] or np.any(weights <= 0):
        raise ValueError("weights must be positive and match value coordinates")
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return 0.0
    local = values[indices]
    mean = np.mean(local, axis=0)
    residual = local - mean
    return float(np.sum((residual * residual) * weights[None, :]))


def split_gain(
    values: Array,
    parent_indices: Sequence[int],
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    weights: Array,
) -> float:
    """Numerically stable finite-node CART gain."""
    parent = np.asarray(parent_indices, dtype=int)
    left = np.asarray(left_indices, dtype=int)
    right = np.asarray(right_indices, dtype=int)
    if len(left) == 0 or len(right) == 0 or len(parent) != len(left) + len(right):
        raise ValueError("split must partition a nonempty parent into two children")
    combined = np.r_[left, right]
    if (
        len(np.unique(parent)) != len(parent)
        or len(np.unique(combined)) != len(combined)
        or not np.array_equal(np.sort(parent), np.sort(combined))
    ):
        raise ValueError("left and right must be a disjoint partition of parent")
    return split_gain_from_means(
        np.mean(values[left], axis=0),
        np.mean(values[right], axis=0),
        len(left),
        len(right),
        weights,
    )


def split_gain_from_means(
    left_mean: Array,
    right_mean: Array,
    left_size: int,
    right_size: int,
    weights: Array,
) -> float:
    """Equivalent two-child gain, useful for direct algebra checks."""
    if left_size <= 0 or right_size <= 0:
        raise ValueError("both children must be nonempty")
    left_mean = _as_1d(left_mean, "left_mean")
    right_mean = _as_1d(right_mean, "right_mean")
    weights = _as_1d(weights, "weights")
    if left_mean.shape != right_mean.shape or len(weights) != len(left_mean):
        raise ValueError("means and weights must have matching coordinates")
    if np.any(weights <= 0):
        raise ValueError("weights must be positive")
    difference = left_mean - right_mean
    return float(
        left_size
        * right_size
        / (left_size + right_size)
        * np.sum(weights * difference * difference)
    )


def exhaustive_gain_identity(values: Array, weights: Array) -> float:
    """Maximum absolute error over all nontrivial small-node partitions."""
    values = _as_2d(values, "values")
    if len(values) > 8:
        raise ValueError("the exhaustive check is intentionally limited to eight rows")
    parent = np.arange(len(values))
    errors = []
    for size in range(1, len(values)):
        for left_tuple in combinations(parent, size):
            left = np.asarray(left_tuple, dtype=int)
            right = np.setdiff1d(parent, left, assume_unique=True)
            direct = weighted_sse(values, parent, weights)
            direct -= weighted_sse(values, left, weights)
            direct -= weighted_sse(values, right, weights)
            means = split_gain_from_means(
                np.mean(values[left], axis=0),
                np.mean(values[right], axis=0),
                len(left),
                len(right),
                weights,
            )
            errors.append(abs(direct - means))
    return float(max(errors, default=0.0))


def _gaussian_kernel(values: Array, bandwidth: float) -> Array:
    differences = values[:, None, :] - values[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    return np.exp(-squared_distances / (2.0 * bandwidth * bandwidth))


def mmd_split_gain(
    values: Array,
    parent_indices: Sequence[int],
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    bandwidth: float,
    metric_weights: Optional[Array] = None,
) -> float:
    """A finite MMD-on-cross-fitted-score split criterion.

    This is a DRF-inspired comparator hook, not official DRF or an
    implementation of its asymptotic theory.  The Gaussian kernel is computed
    on the finite orthogonal-score vector under the direct-sum geometry.
    """
    parent = np.asarray(parent_indices, dtype=int)
    left = np.asarray(left_indices, dtype=int)
    right = np.asarray(right_indices, dtype=int)
    combined = np.r_[left, right]
    if (
        len(left) == 0
        or len(right) == 0
        or len(parent) != len(combined)
        or len(np.unique(parent)) != len(parent)
        or len(np.unique(combined)) != len(combined)
        or not np.array_equal(np.sort(parent), np.sort(combined))
    ):
        raise ValueError("MMD children must be a disjoint partition of parent")
    if bandwidth <= 0:
        raise ValueError("MMD bandwidth must be positive")
    local = values[np.r_[left, right]]
    if metric_weights is not None:
        metric_weights = _as_1d(metric_weights, "metric_weights")
        if len(metric_weights) != local.shape[1] or np.any(metric_weights <= 0):
            raise ValueError("metric_weights must be positive and match values")
        local = local * np.sqrt(metric_weights)[None, :]
    kernel = _gaussian_kernel(local, bandwidth)
    n_left = len(left)
    k_ll = kernel[:n_left, :n_left]
    k_rr = kernel[n_left:, n_left:]
    k_lr = kernel[:n_left, n_left:]
    mmd2 = np.mean(k_ll) + np.mean(k_rr) - 2.0 * np.mean(k_lr)
    return float((len(left) * len(right) / (len(left) + len(right))) * mmd2)


@dataclass
class _Node:
    node_id: int
    split_indices: Array
    population_indices: Array
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    gain: float = 0.0
    leaf_id: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class HonestTree:
    """Tree with split/populate index separation conditional on fixed scores."""

    root: _Node
    split_indices: Array
    estimation_indices: Array
    leaf_populations: Dict[int, Array]
    bandwidth: Optional[float] = None
    coordinate_scales: Optional[Array] = None

    @classmethod
    def fit(
        cls,
        X: Array,
        values: Array,
        split_indices: Array,
        estimation_indices: Array,
        coordinate_weights_: Array,
        rng: np.random.Generator,
        split_rule: str = "sse",
        min_leaf: int = 5,
        max_depth: int = 8,
        mtry: Optional[int] = None,
        max_thresholds: int = 32,
        min_gain: float = 1e-12,
        min_child_fraction: float = 0.10,
        noise_variances: Optional[Array] = None,
        active_coordinates: Optional[Sequence[int]] = None,
        coordinate_scales: Optional[Array] = None,
    ) -> "HonestTree":
        X = _as_2d(X, "X")
        values = _as_2d(values, "values")
        coordinate_weights_ = _as_1d(
            coordinate_weights_, "coordinate_weights"
        )
        if (
            len(coordinate_weights_) != values.shape[1]
            or np.any(coordinate_weights_ <= 0)
        ):
            raise ValueError(
                "coordinate_weights must be positive and match value coordinates"
            )
        split_indices = np.asarray(split_indices, dtype=int)
        estimation_indices = np.asarray(estimation_indices, dtype=int)
        for name, indices in (
            ("split_indices", split_indices),
            ("estimation_indices", estimation_indices),
        ):
            if (
                indices.ndim != 1
                or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0)
                or np.any(indices >= len(X))
            ):
                raise ValueError(f"{name} must contain distinct valid row indices")
        if np.intersect1d(split_indices, estimation_indices).size:
            raise ValueError("honest tree split and estimation samples overlap")
        if len(split_indices) < 2 * min_leaf:
            raise ValueError("split sample is too small for min_leaf")
        if len(estimation_indices) < 2 * min_leaf:
            raise ValueError("estimation sample is too small for min_leaf")
        if values.shape[0] != X.shape[0]:
            raise ValueError("X and values have different numbers of rows")
        active = (
            np.arange(values.shape[1])
            if active_coordinates is None
            else _as_index_array(active_coordinates, "active_coordinates")
        )
        if (
            active.ndim != 1
            or len(active) == 0
            or len(np.unique(active)) != len(active)
            or np.any(active < 0)
            or np.any(active >= values.shape[1])
        ):
            raise ValueError("active coordinates must be distinct valid indices")
        active_weights = coordinate_weights_[active]
        if np.any(active_weights <= 0):
            raise ValueError("active coordinate weights must be positive")
        min_leaf = _require_integer(min_leaf, "min_leaf", minimum=1)
        max_depth = _require_integer(max_depth, "max_depth", minimum=0)
        max_thresholds = _require_integer(
            max_thresholds, "max_thresholds", minimum=1
        )
        if mtry is None:
            mtry = max(1, int(np.sqrt(X.shape[1])))
        mtry = min(_require_integer(mtry, "mtry", minimum=1), X.shape[1])
        if split_rule not in {"sse", "mmd"}:
            raise ValueError("split_rule must be 'sse' or 'mmd'")
        if not 0 < min_child_fraction <= 0.5:
            raise ValueError("min_child_fraction must lie in (0, 0.5]")
        if noise_variances is not None:
            noise_variances = _as_2d(noise_variances, "noise_variances")
            if noise_variances.shape != values.shape or np.any(noise_variances < 0):
                raise ValueError("noise_variances must be nonnegative and match values")
            if split_rule == "mmd":
                raise ValueError(
                    "noise_variances are unsupported for the MMD split rule"
                )
        bandwidth = None
        if split_rule == "mmd":
            pair_indices = split_indices[: min(len(split_indices), 128)]
            pair_values = values[pair_indices][:, active] * np.sqrt(active_weights)[None, :]
            differences = pair_values[:, None, :] - pair_values[None, :, :]
            distances = np.sqrt(np.sum(differences * differences, axis=2))
            positive_distances = distances[distances > 0]
            bandwidth = float(np.median(positive_distances)) if len(positive_distances) else 1.0
            bandwidth = max(bandwidth, 1e-8)

        next_node_id = 0
        leaf_populations: Dict[int, Array] = {}
        next_leaf_id = 0

        def assign_leaf(node: _Node) -> None:
            nonlocal next_leaf_id
            node.leaf_id = next_leaf_id
            leaf_populations[next_leaf_id] = node.population_indices.copy()
            next_leaf_id += 1

        def candidate_gain(parent: Array, left: Array, right: Array) -> float:
            if split_rule == "sse":
                gain = split_gain(
                    values[:, active], parent, left, right, active_weights
                )
                if noise_variances is not None:
                    variance = np.asarray(noise_variances, dtype=float)[np.ix_(parent, active)]
                    variance_left = np.asarray(noise_variances, dtype=float)[np.ix_(left, active)]
                    variance_right = np.asarray(noise_variances, dtype=float)[np.ix_(right, active)]
                    estimated_noise_gain = np.sum(active_weights * (
                        np.sum(variance_left, axis=0) / len(left)
                        + np.sum(variance_right, axis=0) / len(right)
                        - np.sum(variance, axis=0) / len(parent)
                    ))
                    gain -= float(estimated_noise_gain)
                return float(gain)
            return mmd_split_gain(
                values[:, active],
                parent,
                left,
                right,
                float(bandwidth),
                metric_weights=active_weights,
            )

        def grow(split_node_indices: Array, population_node_indices: Array, depth: int) -> _Node:
            nonlocal next_node_id
            node = _Node(
                node_id=next_node_id,
                split_indices=split_node_indices.copy(),
                population_indices=population_node_indices.copy(),
            )
            next_node_id += 1
            if (
                depth >= max_depth
                or len(split_node_indices) < 2 * min_leaf
                or len(population_node_indices) < 2 * min_leaf
            ):
                assign_leaf(node)
                return node

            features = rng.choice(X.shape[1], size=mtry, replace=False)
            best = None
            for feature in features:
                unique = np.unique(X[split_node_indices, feature])
                if len(unique) <= 1:
                    continue
                thresholds = (unique[:-1] + unique[1:]) / 2.0
                if len(thresholds) > max_thresholds:
                    chosen = np.linspace(0, len(thresholds) - 1, max_thresholds).astype(int)
                    thresholds = thresholds[np.unique(chosen)]
                for threshold in thresholds:
                    left = split_node_indices[X[split_node_indices, feature] <= threshold]
                    right = split_node_indices[X[split_node_indices, feature] > threshold]
                    if len(left) < min_leaf or len(right) < min_leaf:
                        continue
                    if min(len(left), len(right)) / len(split_node_indices) < min_child_fraction:
                        continue
                    left_population = population_node_indices[
                        X[population_node_indices, feature] <= threshold
                    ]
                    right_population = population_node_indices[
                        X[population_node_indices, feature] > threshold
                    ]
                    if len(left_population) < min_leaf or len(right_population) < min_leaf:
                        continue
                    if (
                        min(len(left_population), len(right_population))
                        / len(population_node_indices)
                        < min_child_fraction
                    ):
                        continue
                    gain = candidate_gain(split_node_indices, left, right)
                    if best is None or gain > best[0]:
                        best = (
                            gain,
                            int(feature),
                            float(threshold),
                            left,
                            right,
                            left_population,
                            right_population,
                        )
            if best is None or best[0] <= min_gain:
                assign_leaf(node)
                return node

            (
                gain,
                feature,
                threshold,
                left_split,
                right_split,
                left_population,
                right_population,
            ) = best
            node.feature = feature
            node.threshold = threshold
            node.gain = float(gain)
            node.left = grow(left_split, left_population, depth + 1)
            node.right = grow(right_split, right_population, depth + 1)
            return node

        root = grow(split_indices, estimation_indices, 0)
        return cls(
            root=root,
            split_indices=split_indices.copy(),
            estimation_indices=estimation_indices.copy(),
            leaf_populations=leaf_populations,
            bandwidth=bandwidth,
            coordinate_scales=None if coordinate_scales is None else np.asarray(coordinate_scales).copy(),
        )

    def leaf_for(self, x: Array) -> _Node:
        x = np.asarray(x, dtype=float)
        node = self.root
        while not node.is_leaf:
            if node.feature is None or node.threshold is None:
                raise RuntimeError("malformed honest tree")
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def prediction_and_weights(self, x: Array, values: Array) -> tuple[Array, Array]:
        leaf = self.leaf_for(x)
        population = self.leaf_populations[leaf.leaf_id]
        if len(population) == 0:
            raise RuntimeError("honest tree has an empty local estimation leaf")
        prediction = np.mean(values[population], axis=0)
        weights = np.zeros(values.shape[0], dtype=float)
        weights[population] = 1.0 / len(population)
        return prediction, weights


@dataclass
class CrossFitResult:
    scores: Array
    propensity: Array
    m0: Array
    m1: Array
    folds: list[tuple[Array, Array]]
    nuisance_backends: list[str]


def make_region_folds(
    n: int,
    n_folds: int,
    random_state: int = 0,
    stratify: Optional[Array] = None,
) -> list[tuple[Array, Array]]:
    """Create disjoint region-level folds, optionally stratified by a label."""
    if n_folds < 2 or n_folds > n:
        raise ValueError("n_folds must be between two and n")
    rng = np.random.default_rng(random_state)
    if stratify is None:
        validation_parts = list(np.array_split(rng.permutation(n), n_folds))
    else:
        labels = _as_1d(stratify, "stratify")
        if len(labels) != n:
            raise ValueError("stratification labels must have length n")
        validation_lists: list[list[int]] = [[] for _ in range(n_folds)]
        for label in np.unique(labels):
            label_indices = np.flatnonzero(labels == label)
            if len(label_indices) < n_folds:
                raise ValueError(
                    "each stratification group must contain at least n_folds observations"
                )
            for fold_number, part in enumerate(
                np.array_split(rng.permutation(label_indices), n_folds)
            ):
                validation_lists[fold_number].extend(part.tolist())
        validation_parts = [
            rng.permutation(np.asarray(part, dtype=int)) for part in validation_lists
        ]
    folds = []
    for validation in validation_parts:
        training = np.setdiff1d(np.arange(n), validation, assume_unique=True)
        folds.append((training, np.asarray(validation, dtype=int)))
    return folds


def _fit_default_nuisances(
    X_train: Array,
    z_train: Array,
    u_train: Array,
    X_valid: Array,
    seed: int,
    fit_propensity: bool = True,
) -> tuple[Optional[Array], Array, Array, str]:
    """Fit default nuisance models, falling back only if sklearn is unavailable."""
    if not np.all(np.isin(z_train, [0.0, 1.0])):
        raise ValueError("nuisance-training treatment must be binary")
    if not (np.any(z_train == 0) and np.any(z_train == 1)):
        raise ValueError("each nuisance-training fold must contain both treatment arms")
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError:
        e_valid = (
            np.full(len(X_valid), np.mean(z_train), dtype=float)
            if fit_propensity
            else None
        )
        m0_mean = np.mean(u_train[z_train == 0], axis=0)
        m1_mean = np.mean(u_train[z_train == 1], axis=0)
        return (
            e_valid,
            np.tile(m0_mean, (len(X_valid), 1)),
            np.tile(m1_mean, (len(X_valid), 1)),
            "constant_fallback",
        )

    leaf_size = max(2, min(10, len(X_train) // 20))
    e_valid = None
    if fit_propensity:
        classifier = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=leaf_size,
            random_state=seed,
            n_jobs=1,
        )
        classifier.fit(X_train, z_train)
        e_valid = classifier.predict_proba(X_valid)[:, 1]
    m0_model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=leaf_size,
        random_state=seed + 1,
        n_jobs=1,
    )
    m1_model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=leaf_size,
        random_state=seed + 2,
        n_jobs=1,
    )
    m0_model.fit(X_train[z_train == 0], u_train[z_train == 0])
    m1_model.fit(X_train[z_train == 1], u_train[z_train == 1])
    return (
        e_valid,
        m0_model.predict(X_valid),
        m1_model.predict(X_valid),
        "sklearn_random_forest",
    )


def dr_scores(
    U: Array,
    Z: Array,
    propensity: Array,
    m0: Array,
    m1: Array,
) -> Array:
    U = _as_2d(U, "U")
    Z = _as_1d(Z, "Z")
    propensity = _as_1d(propensity, "propensity")
    m0 = _as_2d(m0, "m0")
    m1 = _as_2d(m1, "m1")
    if not (len(U) == len(Z) == len(propensity) == len(m0) == len(m1)):
        raise ValueError("DR score inputs have different lengths")
    if m0.shape != U.shape or m1.shape != U.shape:
        raise ValueError("m0 and m1 must have exactly the same shape as U")
    if not np.all(np.isin(Z, [0.0, 1.0])):
        raise ValueError("Z must be binary")
    if np.any((propensity <= 0) | (propensity >= 1)):
        raise ValueError("propensities must be strictly inside (0, 1)")
    return (
        m1
        - m0
        + (Z / propensity)[:, None] * (U - m1)
        - ((1.0 - Z) / (1.0 - propensity))[:, None] * (U - m0)
    )


def arm_dr_scores(
    U: Array,
    Z: Array,
    propensity: Array,
    m0: Array,
    m1: Array,
) -> tuple[Array, Array]:
    """Return control- and treatment-arm AIPW pseudo-outcomes."""
    U = _as_2d(U, "U")
    Z = _as_1d(Z, "Z")
    propensity = _as_1d(propensity, "propensity")
    m0 = _as_2d(m0, "m0")
    m1 = _as_2d(m1, "m1")
    if not (len(U) == len(Z) == len(propensity) == len(m0) == len(m1)):
        raise ValueError("arm-score inputs have different lengths")
    if m0.shape != U.shape or m1.shape != U.shape:
        raise ValueError("m0 and m1 must have exactly the same shape as U")
    if not np.all(np.isin(Z, [0.0, 1.0])):
        raise ValueError("Z must be binary")
    if np.any((propensity <= 0) | (propensity >= 1)):
        raise ValueError("propensities must be strictly inside (0, 1)")
    score0 = m0 + ((1.0 - Z) / (1.0 - propensity))[:, None] * (U - m0)
    score1 = m1 + (Z / propensity)[:, None] * (U - m1)
    return score0, score1


def oracle_dr_scores(
    U: Array,
    Z: Array,
    true_propensity: float | Array,
    true_m0: Array,
    true_m1: Array,
) -> Array:
    """Construct the WP3-T4 score with supplied true nuisances."""
    U = _as_2d(U, "U")
    propensity = np.asarray(true_propensity, dtype=float)
    if propensity.ndim == 0:
        propensity = np.full(len(U), float(propensity))
    return dr_scores(U, Z, propensity, true_m0, true_m1)


def cross_fitted_dr_scores(
    X: Array,
    Z: Array,
    U: Array,
    n_folds: int = 5,
    random_state: int = 0,
    known_propensity: Optional[float | Array] = None,
) -> CrossFitResult:
    X = _as_2d(X, "X")
    Z = _as_1d(Z, "Z")
    U = _as_2d(U, "U")
    if not (len(X) == len(Z) == len(U)):
        raise ValueError("X, Z, and U have different lengths")
    if not np.all(np.isin(Z, [0.0, 1.0])):
        raise ValueError("Z must be binary")
    known = None if known_propensity is None else np.asarray(known_propensity, dtype=float)
    if known is not None:
        if known.ndim == 0:
            if not 0 < float(known) < 1:
                raise ValueError("known propensities must be strictly inside (0, 1)")
        elif known.ndim != 1 or len(known) != len(X):
            raise ValueError("known_propensity must be scalar or have one value per region")
        elif np.any((known <= 0) | (known >= 1)):
            raise ValueError("known propensities must be strictly inside (0, 1)")
    folds = make_region_folds(len(X), n_folds, random_state, stratify=Z)
    e = np.empty(len(X), dtype=float)
    m0 = np.empty_like(U)
    m1 = np.empty_like(U)
    nuisance_backends: list[str] = []
    for fold_number, (training, validation) in enumerate(folds):
        if known_propensity is None:
            e_valid, m0_valid, m1_valid, backend = _fit_default_nuisances(
                X[training],
                Z[training],
                U[training],
                X[validation],
                random_state + fold_number,
                fit_propensity=True,
            )
            if e_valid is None:
                raise RuntimeError("the propensity nuisance was not fitted")
            e[validation] = np.clip(e_valid, 0.02, 0.98)
            nuisance_backends.append(f"{backend}:estimated_propensity")
        else:
            e[validation] = known if known.ndim == 0 else known[validation]
            _, m0_valid, m1_valid, backend = _fit_default_nuisances(
                X[training],
                Z[training],
                U[training],
                X[validation],
                random_state + fold_number,
                fit_propensity=False,
            )
            nuisance_backends.append(f"{backend}:known_propensity")
        m0[validation] = m0_valid
        m1[validation] = m1_valid
    scores = dr_scores(U, Z, e, m0, m1)
    result = CrossFitResult(
        scores=scores,
        propensity=e,
        m0=m0,
        m1=m1,
        folds=folds,
        nuisance_backends=nuisance_backends,
    )
    assert_cross_fit_disjointness(result, len(X))
    return result


def assert_cross_fit_disjointness(result: CrossFitResult, n: int) -> None:
    seen = []
    for training, validation in result.folds:
        if np.intersect1d(training, validation).size:
            raise AssertionError("cross-fitting training and validation indices overlap")
        expected_training = np.setdiff1d(
            np.arange(n), validation, assume_unique=False
        )
        if not np.array_equal(np.sort(training), expected_training):
            raise AssertionError(
                "cross-fitting training indices are not the validation complement"
            )
        seen.extend(validation.tolist())
    if sorted(seen) != list(range(n)):
        raise AssertionError("cross-fitting validation folds do not partition region indices")


@dataclass
class ODCFEstimator:
    """Finite-grid score-honest CART forest with a common prediction API.

    Tree split and populate indices are disjoint conditional on the supplied
    score matrix.  Cross-fitted nuisance estimation can couple those scores
    across the two index sets, so this finite prototype does not claim full
    outcome-level honesty or the asymptotic regularity conditions of A10.
    """

    K: int
    J: int
    variant: str = "composite"
    n_trees: int = 200
    subsample_fraction: float = 0.7
    honesty_fraction: float = 0.5
    min_leaf: int = 5
    min_child_fraction: float = 0.10
    max_depth: int = 8
    mtry: Optional[int] = None
    max_thresholds: int = 32
    min_gain: float = 1e-12
    random_state: int = DEFAULT_RANDOM_STATE
    scaling_rule: str = "robust_sd"
    quadrature_weights: Optional[Array] = None
    active_coordinates: Optional[Sequence[int]] = None
    inner_noise_correction: bool = False
    trees: list[HonestTree] = field(default_factory=list, init=False)
    scaler: Optional[CoordinateScaler] = field(default=None, init=False)
    X_train: Optional[Array] = field(default=None, init=False)
    scores_unscaled: Optional[Array] = field(default=None, init=False)
    noise_variances: Optional[Array] = field(default=None, init=False)
    selected_coordinates: Optional[Array] = field(default=None, init=False)

    def fit(
        self,
        X: Array,
        scores: Array,
        treatment: Optional[Array] = None,
        propensity: Optional[Array] = None,
        noise_variances: Optional[Array] = None,
    ) -> "ODCFEstimator":
        X = _as_2d(X, "X")
        scores = _as_2d(scores, "scores")
        _require_integer(self.K, "K", minimum=1)
        _require_integer(self.J, "J", minimum=0)
        if len(X) != len(scores):
            raise ValueError("X and scores have different lengths")
        if scores.shape[1] != self.K + self.J:
            raise ValueError("scores do not match K+J")
        _require_integer(self.n_trees, "n_trees", minimum=1)
        _require_integer(self.min_leaf, "min_leaf", minimum=1)
        _require_integer(self.max_depth, "max_depth", minimum=0)
        _require_integer(self.max_thresholds, "max_thresholds", minimum=1)
        _require_integer(self.random_state, "random_state", minimum=0)
        if not np.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError(
                "min_gain must be finite and nonnegative"
            )
        if not 0 < self.min_child_fraction <= 0.5:
            raise ValueError("min_child_fraction must lie in (0, 0.5]")
        if self.variant not in {"curve_only", "composite", "mmd_score"}:
            raise ValueError(
                "ODCFEstimator variant must be curve_only, composite, or mmd_score"
            )
        if self.variant == "mmd_score" and self.inner_noise_correction:
            raise ValueError(
                "inner_noise_correction is an SSE-only experimental heuristic "
                "and is unsupported for mmd_score"
            )
        self.quadrature_weights = _validate_grid_weights(self.K, self.quadrature_weights)
        # This full-sample scaler is diagnostic only.  Every tree below fits a
        # separate scaler on its split subsample, so populate outcomes cannot
        # affect the tree structure through coordinate calibration.
        self.scaler = CoordinateScaler.fit(
            scores,
            self.K,
            self.quadrature_weights,
            self.scaling_rule,
            treatment=treatment,
            propensity=propensity,
        )
        if noise_variances is not None:
            noise_variances = _as_2d(noise_variances, "noise_variances")
            if noise_variances.shape != scores.shape or np.any(noise_variances < 0):
                raise ValueError("noise_variances must be nonnegative and match scores")
        if self.inner_noise_correction and noise_variances is None:
            raise ValueError(
                "inner_noise_correction=True requires score-scale noise_variances"
            )
        self.X_train = X.copy()
        self.scores_unscaled = scores.copy()
        self.noise_variances = None if noise_variances is None else noise_variances.copy()
        if self.active_coordinates is None:
            if self.variant == "curve_only":
                active = np.arange(self.K)
            else:
                active = np.arange(self.K + self.J)
        else:
            active = _as_index_array(
                self.active_coordinates, "active_coordinates"
            )
        if (
            len(active) == 0
            or len(np.unique(active)) != len(active)
            or np.any(active < 0)
            or np.any(active >= self.K + self.J)
        ):
            raise ValueError("active_coordinates are invalid")
        if self.variant == "curve_only" and np.any(active >= self.K):
            raise ValueError("curve_only cannot activate functional coordinates")
        self.selected_coordinates = active
        if not (0 < self.subsample_fraction <= 1 and 0 < self.honesty_fraction < 1):
            raise ValueError("subsample_fraction and honesty_fraction are invalid")
        subsample_size = max(
            4 * self.min_leaf,
            int(np.ceil(self.subsample_fraction * len(X))),
        )
        subsample_size = min(subsample_size, len(X))
        if subsample_size < 4 * self.min_leaf:
            raise ValueError("sample is too small for the requested honest tree")
        split_size = int(np.floor(self.honesty_fraction * subsample_size))
        split_size = max(2 * self.min_leaf, split_size)
        split_size = min(split_size, subsample_size - 2 * self.min_leaf)
        self.trees = []
        split_rule = "mmd" if self.variant == "mmd_score" else "sse"
        all_weights = coordinate_weights(
            self.K, self.J, self.quadrature_weights, active_coordinates=None
        )
        tree_sequences = np.random.SeedSequence(self.random_state).spawn(self.n_trees)
        for tree_sequence in tree_sequences:
            sampling_sequence, node_sequence = tree_sequence.spawn(2)
            sampling_rng = np.random.default_rng(sampling_sequence)
            node_rng = np.random.default_rng(node_sequence)
            subsample = sampling_rng.choice(
                len(X), size=subsample_size, replace=False
            )
            split_indices = sampling_rng.choice(
                subsample, size=split_size, replace=False
            )
            estimation_indices = np.setdiff1d(subsample, split_indices, assume_unique=False)
            split_treatment = None if treatment is None else np.asarray(treatment)[split_indices]
            split_propensity = None if propensity is None else np.asarray(propensity)[split_indices]
            tree_scaler = CoordinateScaler.fit(
                scores[split_indices],
                self.K,
                self.quadrature_weights,
                self.scaling_rule,
                treatment=split_treatment,
                propensity=split_propensity,
            )
            tree_values = tree_scaler.transform(scores)
            tree_noise = None
            if self.inner_noise_correction:
                tree_noise = noise_variances * (tree_scaler.scales[None, :] ** 2)
            tree = HonestTree.fit(
                X,
                tree_values,
                split_indices,
                estimation_indices,
                all_weights,
                node_rng,
                split_rule=split_rule,
                min_leaf=self.min_leaf,
                min_child_fraction=self.min_child_fraction,
                max_depth=self.max_depth,
                mtry=self.mtry,
                max_thresholds=self.max_thresholds,
                min_gain=self.min_gain,
                noise_variances=tree_noise,
                active_coordinates=active,
                coordinate_scales=tree_scaler.scales,
            )
            self.trees.append(tree)
        return self

    def _check_fitted(self) -> None:
        if (
            self.scaler is None
            or self.X_train is None
            or self.scores_unscaled is None
            or not self.trees
        ):
            raise RuntimeError("fit must be called before prediction")

    def predict_scaled(self, X_new: Array) -> Array:
        """Return predictions under the diagnostic full-sample scale."""
        self._check_fitted()
        return self.scaler.transform(self.predict(X_new))

    def predict(self, X_new: Array) -> Array:
        """Return predictions on the unscaled scientific target coordinates."""
        self._check_fitted()
        X_new = _as_2d(X_new, "X_new")
        if X_new.shape[1] != self.X_train.shape[1]:
            raise ValueError("X_new has the wrong feature dimension")
        predictions = []
        for x in X_new:
            tree_predictions = [
                tree.prediction_and_weights(x, self.scores_unscaled)[0]
                for tree in self.trees
            ]
            predictions.append(np.mean(tree_predictions, axis=0))
        return np.asarray(predictions)

    def weights_at(self, x: Array) -> Array:
        self._check_fitted()
        x = _as_1d(x, "x")
        if len(x) != self.X_train.shape[1]:
            raise ValueError("x has the wrong feature dimension")
        weights = np.zeros(len(self.X_train), dtype=float)
        for tree in self.trees:
            _, tree_weights = tree.prediction_and_weights(x, self.scores_unscaled)
            weights += tree_weights / len(self.trees)
        total = np.sum(weights)
        if total <= 0:
            raise RuntimeError("forest weights are empty")
        return weights / total

    def honesty_report(self) -> dict[str, object]:
        self._check_fitted()
        disjoint = []
        leaf_subset = []
        balanced_children = []
        for tree in self.trees:
            disjoint.append(np.intersect1d(tree.split_indices, tree.estimation_indices).size == 0)
            for population in tree.leaf_populations.values():
                leaf_subset.append(np.setdiff1d(population, tree.estimation_indices).size == 0)

            pending = [tree.root]
            while pending:
                node = pending.pop()
                if node.is_leaf:
                    continue
                if node.left is None or node.right is None:
                    raise RuntimeError("malformed internal tree node")
                split_sizes = (
                    len(node.left.split_indices),
                    len(node.right.split_indices),
                )
                population_sizes = (
                    len(node.left.population_indices),
                    len(node.right.population_indices),
                )
                balanced_children.append(
                    min(split_sizes) >= self.min_leaf
                    and min(population_sizes) >= self.min_leaf
                    and min(split_sizes) / len(node.split_indices)
                    >= self.min_child_fraction
                    and min(population_sizes) / len(node.population_indices)
                    >= self.min_child_fraction
                )
                pending.extend([node.left, node.right])
        return {
            "n_trees": len(self.trees),
            "claim_scope": "index/score honesty conditional on supplied scores",
            "all_split_estimation_disjoint": bool(all(disjoint)),
            "all_leaf_populations_from_estimation": bool(all(leaf_subset)),
            "all_children_satisfy_balance": bool(all(balanced_children)),
            "all_local_leaves_nonempty": bool(
                all(
                    len(population) >= self.min_leaf
                    for tree in self.trees
                    for population in tree.leaf_populations.values()
                )
            ),
            "min_tree_estimation_size": int(min(len(tree.estimation_indices) for tree in self.trees)),
        }


@dataclass
class SpecializedForest:
    """Separate target-specific forests exposed through the ODCF API."""

    models: Dict[str, ODCFEstimator]
    K: int
    J: int

    def predict(self, X_new: Array) -> Array:
        if not self.models:
            raise RuntimeError("no specialized forests were fitted")
        X_new = _as_2d(X_new, "X_new")
        output = np.full((len(X_new), self.K + self.J), np.nan, dtype=float)
        for model in self.models.values():
            coordinates = model.selected_coordinates
            output[:, coordinates] = model.predict(X_new)[:, coordinates]
        if np.any(~np.isfinite(output)):
            raise RuntimeError("specialized forests do not cover every target coordinate")
        return output

    def honesty_report(self) -> dict[str, object]:
        return {name: model.honesty_report() for name, model in self.models.items()}


def fit_specialized_forests(
    X: Array,
    scores: Array,
    K: int,
    J: int,
    groups: Mapping[str, Sequence[int]],
    **kwargs,
) -> SpecializedForest:
    if not groups:
        raise ValueError("groups must contain at least one coordinate block")
    flattened = np.concatenate(
        [
            _as_index_array(coordinates, f"groups[{name!r}]")
            for name, coordinates in groups.items()
        ]
    )
    expected = np.arange(K + J)
    if (
        len(flattened) != K + J
        or len(np.unique(flattened)) != len(flattened)
        or not np.array_equal(np.sort(flattened), expected)
    ):
        raise ValueError(
            "specialized groups must form an exact, nonoverlapping cover of all coordinates"
        )
    models: Dict[str, ODCFEstimator] = {}
    for offset, (name, coordinates) in enumerate(groups.items()):
        model_kwargs = dict(kwargs)
        base_seed = kwargs.get("random_state", DEFAULT_RANDOM_STATE)
        model_kwargs["random_state"] = (
            _require_integer(base_seed, "random_state", minimum=0) + offset
        )
        model_kwargs["active_coordinates"] = _as_index_array(
            coordinates, f"groups[{name!r}]"
        )
        model_kwargs["variant"] = "composite"
        models[name] = ODCFEstimator(K=K, J=J, **model_kwargs).fit(X, scores)
    return SpecializedForest(models=models, K=K, J=J)


def pava(values: Array, weights: Optional[Array] = None) -> Array:
    """Weighted nondecreasing isotonic projection by the PAVA algorithm."""
    values = _as_1d(values, "values")
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    weights = _as_1d(weights, "weights")
    if len(weights) != len(values) or np.any(weights <= 0):
        raise ValueError("PAVA weights must be positive and match values")
    block_values: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    block_ends: list[int] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(index)
        block_ends.append(index)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_weights[-2] * block_values[-2]
                + block_weights[-1] * block_values[-1]
            ) / total_weight
            block_values[-2] = merged_value
            block_weights[-2] = total_weight
            block_ends[-2] = block_ends[-1]
            block_values.pop()
            block_weights.pop()
            block_starts.pop()
            block_ends.pop()
    projected = np.empty_like(values)
    for value, start, end in zip(block_values, block_starts, block_ends):
        projected[start : end + 1] = value
    return projected


def project_arm_mean_curves(
    arm_curves: Array,
    quadrature_weights: Optional[Array] = None,
) -> Array:
    """Project arm-specific mean curves under the declared grid geometry."""
    curves = _as_2d(arm_curves, "arm_curves")
    weights = (
        trapezoidal_grid_weights(curves.shape[1])
        if quadrature_weights is None
        else _as_1d(quadrature_weights, "quadrature_weights")
    )
    if len(weights) != curves.shape[1] or np.any(weights <= 0):
        raise ValueError("quadrature_weights must be positive and match curve width")
    return np.vstack([pava(curve, weights=weights) for curve in curves])


def effect_curve_from_arms(arm_1: Array, arm_0: Array) -> Array:
    """Return an unconstrained effect curve; no PAVA is applied."""
    arm_1 = _as_2d(arm_1, "arm_1")
    arm_0 = _as_2d(arm_0, "arm_0")
    if arm_1.shape != arm_0.shape:
        raise ValueError("arm curves have different shapes")
    return arm_1 - arm_0


@dataclass
class ArmCurveForest:
    """Separate score-honest forests for the two arm mean quantile curves."""

    control_model: ODCFEstimator
    treated_model: ODCFEstimator
    quadrature_weights: Array

    def predict_arms(
        self,
        X_new: Array,
        project: bool = True,
    ) -> tuple[Array, Array]:
        control = self.control_model.predict(X_new)
        treated = self.treated_model.predict(X_new)
        if project:
            control = project_arm_mean_curves(
                control, self.quadrature_weights
            )
            treated = project_arm_mean_curves(
                treated, self.quadrature_weights
            )
        return control, treated

    def predict_effect(self, X_new: Array, project_arms: bool = True) -> Array:
        """Subtract arm curves without projecting the resulting effect."""
        control, treated = self.predict_arms(X_new, project=project_arms)
        return effect_curve_from_arms(treated, control)

    def honesty_report(self) -> dict[str, object]:
        return {
            "control": self.control_model.honesty_report(),
            "treated": self.treated_model.honesty_report(),
        }


def fit_arm_curve_forests(
    X: Array,
    U: Array,
    Z: Array,
    propensity: Array,
    m0: Array,
    m1: Array,
    K: int,
    **kwargs,
) -> ArmCurveForest:
    """Fit separate arm-mean curve forests from cross-fitted AIPW scores."""
    K = _require_integer(K, "K", minimum=1)
    score0, score1 = arm_dr_scores(U, Z, propensity, m0, m1)
    if K > score0.shape[1]:
        raise ValueError("K exceeds the arm-score coordinate count")
    reserved = {"variant", "active_coordinates", "J"}.intersection(kwargs)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(
            f"fit_arm_curve_forests fixes {names}; do not supply them"
        )
    model_kwargs = dict(kwargs)
    base_seed = _require_integer(
        model_kwargs.pop("random_state", DEFAULT_RANDOM_STATE),
        "random_state",
        minimum=0,
    )
    control_model = ODCFEstimator(
        K=K,
        J=0,
        variant="curve_only",
        random_state=base_seed,
        **model_kwargs,
    ).fit(X, score0[:, :K])
    treated_model = ODCFEstimator(
        K=K,
        J=0,
        variant="curve_only",
        random_state=base_seed + 1,
        **model_kwargs,
    ).fit(X, score1[:, :K])
    return ArmCurveForest(
        control_model=control_model,
        treated_model=treated_model,
        quadrature_weights=np.asarray(
            control_model.quadrature_weights, dtype=float
        ).copy(),
    )


def empirical_unit_vector(
    raw_sample: Array,
    quantile_probabilities: Array,
    functional_grid_size: int = 400,
) -> Array:
    """Compute the provisional finite U-vector from one inner sample."""
    raw_sample = _as_1d(raw_sample, "raw_sample")
    if np.any(raw_sample < 0):
        raise ValueError("raw income must be nonnegative")
    p = _as_1d(quantile_probabilities, "quantile_probabilities")
    if np.any((p <= 0) | (p >= 1)) or np.any(np.diff(p) <= 0):
        raise ValueError(
            "quantile probabilities must be strictly increasing and interior"
        )
    analysis_quantiles = np.log1p(
        np.quantile(raw_sample, p, method="inverted_cdf")
    )
    functional_probabilities = (np.arange(functional_grid_size) + 0.5) / functional_grid_size
    raw_quantiles = np.quantile(
        raw_sample,
        functional_probabilities,
        method="inverted_cdf",
    )
    weights = np.full(functional_grid_size, 1.0 / functional_grid_size)
    mean = float(np.dot(weights, raw_quantiles))
    if mean <= 0:
        raise ValueError("finite functionals require positive mean income")
    gini = 1.0 - 2.0 * float(np.dot(weights, (1.0 - functional_probabilities) * raw_quantiles)) / mean
    positive = raw_quantiles > 0
    ratios = np.zeros_like(raw_quantiles)
    ratios[positive] = raw_quantiles[positive] / mean
    theil = float(np.dot(weights[positive], ratios[positive] * np.log(ratios[positive])))
    atkinson = 1.0 - float(np.dot(weights, np.sqrt(raw_quantiles)) ** 2) / mean
    return np.r_[analysis_quantiles, gini, theil, atkinson]


@dataclass
class InnerBootstrapResult:
    corrected_vector: Array
    estimated_noise_variance: Array
    plug_in_vector: Array
    bootstrap_vectors: Array


def bootstrap_bias_corrected_unit(
    raw_sample: Array,
    quantile_probabilities: Array,
    n_replicates: int = 100,
    random_state: int = 0,
) -> InnerBootstrapResult:
    """Experimental within-region bootstrap correction for U-hat.

    The correction is the ordinary bootstrap bias estimate
    ``2 * U_hat - mean(U_boot)``.  It is deliberately not asserted to be valid
    for empirical quantile functionals or survey samples; WP5 must analyze it.
    """
    raw_sample = _as_1d(raw_sample, "raw_sample")
    if n_replicates < 2 or len(raw_sample) < 2:
        raise ValueError("bootstrap requires at least two observations and replicates")
    rng = np.random.default_rng(random_state)
    plug_in = empirical_unit_vector(raw_sample, quantile_probabilities)
    bootstrap_vectors = np.vstack([
        empirical_unit_vector(
            raw_sample[rng.integers(0, len(raw_sample), size=len(raw_sample))],
            quantile_probabilities,
        )
        for _ in range(n_replicates)
    ])
    corrected = 2.0 * plug_in - np.mean(bootstrap_vectors, axis=0)
    variance = np.var(bootstrap_vectors, axis=0, ddof=1)
    return InnerBootstrapResult(
        corrected_vector=corrected,
        estimated_noise_variance=variance,
        plug_in_vector=plug_in,
        bootstrap_vectors=bootstrap_vectors,
    )


def fit_odcf_from_inner_samples(
    X: Array,
    Z: Array,
    inner_samples: Sequence[Array],
    quantile_probabilities: Array,
    estimator: ODCFEstimator,
    nuisance_folds: int = 5,
    random_state: int = 0,
    n_bootstrap_replicates: int = 100,
    known_propensity: Optional[float | Array] = None,
) -> tuple[ODCFEstimator, CrossFitResult, list[InnerBootstrapResult]]:
    """Fit the provisional inner-sample path before nuisance scoring.

    If requested, the experimental correction propagates only the direct
    empirical-outcome component of inner-sample variance onto the DR-score
    scale.  It does not include nuisance-induced covariance or establish a
    measurement-error theorem.
    """
    if len(inner_samples) != len(X):
        raise ValueError("inner_samples and X have different lengths")
    bootstrap_results = [
        bootstrap_bias_corrected_unit(
            sample,
            quantile_probabilities,
            n_replicates=n_bootstrap_replicates,
            random_state=random_state + i,
        )
        for i, sample in enumerate(inner_samples)
    ]
    U = np.vstack([result.corrected_vector for result in bootstrap_results])
    noise = np.vstack([result.estimated_noise_variance for result in bootstrap_results])
    cross_fit = cross_fitted_dr_scores(
        X,
        Z,
        U,
        n_folds=nuisance_folds,
        random_state=random_state,
        known_propensity=known_propensity,
    )
    score_noise = None
    if estimator.inner_noise_correction:
        Z_array = _as_1d(Z, "Z")
        direct_coefficient = (
            Z_array / cross_fit.propensity
            + (1.0 - Z_array) / (1.0 - cross_fit.propensity)
        )
        score_noise = noise * (direct_coefficient[:, None] ** 2)
    estimator.fit(
        X,
        cross_fit.scores,
        treatment=Z,
        propensity=cross_fit.propensity,
        noise_variances=score_noise,
    )
    return estimator, cross_fit, bootstrap_results


__all__ = [
    "ArmCurveForest",
    "CoordinateScaler",
    "CrossFitResult",
    "DEFAULT_RANDOM_STATE",
    "HonestTree",
    "InnerBootstrapResult",
    "ODCFEstimator",
    "SpecializedForest",
    "assert_cross_fit_disjointness",
    "arm_dr_scores",
    "bootstrap_bias_corrected_unit",
    "coordinate_weights",
    "cross_fitted_dr_scores",
    "dr_scores",
    "effect_curve_from_arms",
    "empirical_unit_vector",
    "exhaustive_gain_identity",
    "fit_odcf_from_inner_samples",
    "fit_arm_curve_forests",
    "fit_specialized_forests",
    "make_region_folds",
    "mmd_split_gain",
    "oracle_dr_scores",
    "pava",
    "project_arm_mean_curves",
    "split_gain",
    "split_gain_from_means",
    "trapezoidal_grid_weights",
    "weighted_sse",
]
