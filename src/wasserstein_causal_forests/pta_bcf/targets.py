"""Fixed PTA target vector U(Y) and its training-only scale manifest.

The target contract implements

    U(Y) = {q(Y), T_1(Y), ..., T_J(Y), d_W(q(Y), q(nu_star))}

with every coordinate a declared measurable function of the frozen grid
representation q(Y). Grid functionals carry grid target identifiers from
`research/estimand_contract.md`; no continuum identifier is emitted here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common.quantiles import validate_quantiles, validate_weights
from ..cwdb.geometry import project_quantiles, weighted_distance

TARGET_CONTRACT_ID = "PTA-U-v1"

QUANTILE_BLOCK = "quantile"
FUNCTIONAL_BLOCK = "functional"
REFERENCE_BLOCK = "reference"


def _grid_mean(q: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    return q @ w


def _grid_sd(q: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = q - (q @ w)[..., None]
    return np.sqrt(np.maximum(centered * centered @ w, 0.0))


def _grid_skewness(
    q: NDArray[np.float64], w: NDArray[np.float64]
) -> NDArray[np.float64]:
    centered = q - (q @ w)[..., None]
    variance = np.maximum(centered * centered @ w, 0.0)
    third = centered**3 @ w
    scale = np.power(variance, 1.5)
    return np.where(scale > 0.0, third / np.where(scale > 0.0, scale, 1.0), 0.0)


def _grid_upper_tail_mean(
    q: NDArray[np.float64], w: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Weighted mean over the upper half of the declared grid."""

    half = w.size // 2
    tail = np.zeros_like(w)
    tail[half:] = w[half:]
    total = tail.sum()
    if total <= 0.0:
        raise ValueError("upper tail of the declared grid carries no weight")
    return q @ (tail / total)


#: Registry of admissible grid functionals h_j with T_j^K = h_j o q.
GRID_FUNCTIONALS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "grid_mean": _grid_mean,
    "grid_sd": _grid_sd,
    "grid_skewness": _grid_skewness,
    "grid_upper_tail_mean": _grid_upper_tail_mean,
}


@dataclass(frozen=True)
class TargetManifest:
    """Frozen description of the PTA target vector U(Y).

    The manifest is fixed before outcome analysis. It never depends on the
    treatment assignment, the observed responses, or any fold split.
    """

    weights: NDArray[np.float64]
    functionals: tuple[str, ...] = ()
    reference_quantiles: NDArray[np.float64] | None = None
    contract_id: str = TARGET_CONTRACT_ID
    grid_points: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        weights = validate_weights(self.weights, require_normalized=True)
        object.__setattr__(self, "weights", weights)
        unknown = [name for name in self.functionals if name not in GRID_FUNCTIONALS]
        if unknown:
            raise ValueError(f"unknown grid functionals: {sorted(unknown)}")
        if len(set(self.functionals)) != len(self.functionals):
            raise ValueError("grid functionals must be declared at most once")
        if self.reference_quantiles is not None:
            reference = validate_quantiles(
                self.reference_quantiles, weights.size, check_monotone=True
            )
            if reference.ndim != 1:
                raise ValueError("reference_quantiles must be one-dimensional")
            object.__setattr__(self, "reference_quantiles", reference)
        if self.grid_points is not None:
            grid = np.asarray(self.grid_points, dtype=float)
            if grid.shape != weights.shape:
                raise ValueError("grid_points must match the weight length")
            if np.any(np.diff(grid) <= 0.0) or grid[0] <= 0.0 or grid[-1] >= 1.0:
                raise ValueError("grid_points must be increasing inside (0, 1)")
            object.__setattr__(self, "grid_points", grid)

    @property
    def n_grid(self) -> int:
        """K, the number of declared quantile coordinates."""

        return int(self.weights.size)

    @property
    def n_functionals(self) -> int:
        """J, the number of declared grid functionals."""

        return len(self.functionals)

    @property
    def has_reference(self) -> bool:
        return self.reference_quantiles is not None

    @property
    def dimension(self) -> int:
        """D = K + J + 1{reference}."""

        return self.n_grid + self.n_functionals + int(self.has_reference)

    @property
    def quantile_slice(self) -> slice:
        return slice(0, self.n_grid)

    @property
    def functional_slice(self) -> slice:
        return slice(self.n_grid, self.n_grid + self.n_functionals)

    @property
    def reference_index(self) -> int | None:
        return self.dimension - 1 if self.has_reference else None

    @property
    def column_names(self) -> tuple[str, ...]:
        names = [f"q{index + 1}" for index in range(self.n_grid)]
        names.extend(self.functionals)
        if self.has_reference:
            names.append("reference_distance")
        return tuple(names)

    @property
    def blocks(self) -> tuple[str, ...]:
        blocks = [QUANTILE_BLOCK] * self.n_grid
        blocks.extend([FUNCTIONAL_BLOCK] * self.n_functionals)
        if self.has_reference:
            blocks.append(REFERENCE_BLOCK)
        return tuple(blocks)

    @property
    def level_target_ids(self) -> tuple[str, ...]:
        ids = ["MEANQ-A-K"] * self.n_grid
        ids.extend(f"TATE-K-j:{name}" for name in self.functionals)
        if self.has_reference:
            ids.append("REF-A-K")
        return tuple(ids)

    @property
    def contrast_target_ids(self) -> tuple[str, ...]:
        ids = ["MEANQ-A-K"] * self.n_grid
        ids.extend(f"TCATE-K-j:{name}" for name in self.functionals)
        if self.has_reference:
            ids.append("REF-TCATE-K")
        return tuple(ids)

    def build(self, quantiles: ArrayLike) -> NDArray[np.float64]:
        """Return U(Y) with shape (n, D) for monotone grid vectors."""

        q = validate_quantiles(quantiles, self.n_grid, check_monotone=True)
        if q.ndim != 2:
            raise ValueError("quantiles must have shape (n, K)")
        columns = [q]
        if self.functionals:
            columns.append(
                np.column_stack(
                    [
                        GRID_FUNCTIONALS[name](q, self.weights)
                        for name in self.functionals
                    ]
                )
            )
        if self.reference_quantiles is not None:
            distance = weighted_distance(
                q, np.broadcast_to(self.reference_quantiles, q.shape), self.weights
            )
            columns.append(np.asarray(distance, dtype=float)[:, None])
        return np.column_stack(columns)

    def project_quantile_block(self, U: ArrayLike) -> NDArray[np.float64]:
        """Apply weighted isotonic postprocessing to the quantile block only.

        This is a declared postprocessing step on arm-level predictions. It is
        never applied to treatment contrasts, which are differences of monotone
        vectors and need not be monotone.
        """

        array = np.array(U, dtype=float, copy=True)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(f"U must have shape (n, {self.dimension})")
        array[:, self.quantile_slice] = project_quantiles(
            array[:, self.quantile_slice], self.weights
        )
        return array

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "weights": self.weights.tolist(),
            "functionals": list(self.functionals),
            "reference_quantiles": (
                None
                if self.reference_quantiles is None
                else self.reference_quantiles.tolist()
            ),
            "grid_points": (
                None if self.grid_points is None else self.grid_points.tolist()
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TargetManifest":
        reference = payload.get("reference_quantiles")
        grid = payload.get("grid_points")
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            functionals=tuple(payload.get("functionals", ())),
            reference_quantiles=(
                None if reference is None else np.asarray(reference, dtype=float)
            ),
            contract_id=str(payload.get("contract_id", TARGET_CONTRACT_ID)),
            grid_points=None if grid is None else np.asarray(grid, dtype=float),
        )


def uniform_grid_manifest(
    n_grid: int,
    *,
    functionals: Sequence[str] = (),
    reference_quantiles: ArrayLike | None = None,
) -> TargetManifest:
    """Return a manifest on the midpoint grid with equal quadrature weights."""

    if n_grid < 1:
        raise ValueError("n_grid must be positive")
    points = (np.arange(n_grid, dtype=float) + 0.5) / n_grid
    weights = np.full(n_grid, 1.0 / n_grid)
    return TargetManifest(
        weights=weights,
        functionals=tuple(functionals),
        reference_quantiles=(
            None
            if reference_quantiles is None
            else np.asarray(reference_quantiles, dtype=float)
        ),
        grid_points=points,
    )


def _fingerprint(array: NDArray[np.float64]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array, dtype=float).tobytes())
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ScaleManifest:
    """Coordinatewise centering and scaling estimated on training rows only.

    The manifest records the fingerprint and row count of the exact array it
    was estimated from, so a later audit can verify that no evaluation row
    contributed to the scale.
    """

    center: NDArray[np.float64]
    scale: NDArray[np.float64]
    n_train: int
    source_fingerprint: str
    column_names: tuple[str, ...] = ()
    scale_floor: float = 1e-8

    @classmethod
    def fit(
        cls,
        U_train: ArrayLike,
        manifest: TargetManifest | None = None,
        *,
        scale_floor: float = 1e-8,
    ) -> "ScaleManifest":
        array = np.asarray(U_train, dtype=float)
        if array.ndim != 2 or array.shape[0] < 2:
            raise ValueError("U_train must have shape (n, D) with at least two rows")
        if not np.all(np.isfinite(array)):
            raise ValueError("U_train must be finite")
        if manifest is not None and array.shape[1] != manifest.dimension:
            raise ValueError("U_train width does not match the target manifest")
        center = array.mean(axis=0)
        scale = array.std(axis=0, ddof=1)
        scale = np.where(scale > scale_floor, scale, 1.0)
        return cls(
            center=center,
            scale=scale,
            n_train=int(array.shape[0]),
            source_fingerprint=_fingerprint(array),
            column_names=() if manifest is None else manifest.column_names,
            scale_floor=float(scale_floor),
        )

    def _check_width(self, array: NDArray[np.float64]) -> None:
        if array.ndim < 1 or array.shape[-1] != self.center.size:
            raise ValueError(
                f"expected {self.center.size} target coordinates, "
                f"got {array.shape[-1] if array.ndim else 0}"
            )

    def transform(self, U: ArrayLike) -> NDArray[np.float64]:
        """Standardize levels. Never mutates the manifest."""

        array = np.asarray(U, dtype=float)
        self._check_width(array)
        return (array - self.center) / self.scale

    def inverse_transform(self, U_scaled: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(U_scaled, dtype=float)
        self._check_width(array)
        return array * self.scale + self.center

    def inverse_transform_contrast(self, delta_scaled: ArrayLike) -> NDArray[np.float64]:
        """Return contrasts on the original scale.

        Differences of levels lose the centering term, so only the scale is
        reapplied. Using `inverse_transform` here would add a spurious offset.
        """

        array = np.asarray(delta_scaled, dtype=float)
        self._check_width(array)
        return array * self.scale

    def was_fitted_on(self, U: ArrayLike) -> bool:
        return _fingerprint(np.asarray(U, dtype=float)) == self.source_fingerprint

    def to_dict(self) -> dict[str, object]:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "n_train": self.n_train,
            "source_fingerprint": self.source_fingerprint,
            "column_names": list(self.column_names),
            "scale_floor": self.scale_floor,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ScaleManifest":
        return cls(
            center=np.asarray(payload["center"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            n_train=int(payload["n_train"]),
            source_fingerprint=str(payload["source_fingerprint"]),
            column_names=tuple(payload.get("column_names", ())),
            scale_floor=float(payload.get("scale_floor", 1e-8)),
        )


@dataclass(frozen=True)
class FoldPlan:
    """Common cross-fitting folds shared by every target coordinate."""

    assignment: NDArray[np.int64]
    n_folds: int
    random_state: int
    fold_ids: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        assignment = np.asarray(self.assignment, dtype=np.int64)
        if assignment.ndim != 1:
            raise ValueError("fold assignment must be one-dimensional")
        object.__setattr__(self, "assignment", assignment)
        object.__setattr__(self, "fold_ids", tuple(range(self.n_folds)))

    def train_index(self, fold: int) -> NDArray[np.int64]:
        return np.flatnonzero(self.assignment != fold)

    def test_index(self, fold: int) -> NDArray[np.int64]:
        return np.flatnonzero(self.assignment == fold)


def make_folds(
    n_rows: int,
    treatment: ArrayLike,
    *,
    n_folds: int = 5,
    random_state: int = 0,
) -> FoldPlan:
    """Return treatment-stratified folds reused by every PTA component."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    a = np.asarray(treatment, dtype=int)
    if a.size != n_rows:
        raise ValueError("treatment length must equal n_rows")
    rng = np.random.default_rng(random_state)
    assignment = np.empty(n_rows, dtype=np.int64)
    for arm in np.unique(a):
        index = np.flatnonzero(a == arm)
        shuffled = rng.permutation(index)
        assignment[shuffled] = np.arange(shuffled.size) % n_folds
    return FoldPlan(
        assignment=assignment, n_folds=n_folds, random_state=int(random_state)
    )


def assert_disjoint(
    train_index: ArrayLike, evaluation_index: ArrayLike
) -> None:
    """Raise when an evaluation row also appears in the fitting rows."""

    train = np.asarray(train_index, dtype=np.int64)
    evaluation = np.asarray(evaluation_index, dtype=np.int64)
    overlap = np.intersect1d(train, evaluation)
    if overlap.size:
        raise ValueError(
            f"{overlap.size} evaluation rows leak into the fitting rows"
        )
