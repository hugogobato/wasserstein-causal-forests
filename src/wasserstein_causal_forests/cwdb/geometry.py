"""Weighted finite-grid Wasserstein geometry and monotone projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common.quantiles import validate_weights


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Aggregated diagnostics for weighted isotonic projection."""

    n_vectors: int
    n_changed: int
    input_violations: int
    max_absolute_adjustment: float
    max_weighted_l2_adjustment: float
    output_max_violation: float


def to_rescaled(
    quantiles: ArrayLike, weights: ArrayLike
) -> NDArray[np.float64]:
    """Map q to z = diag(sqrt(w)) q."""

    q = np.asarray(quantiles, dtype=float)
    if q.ndim == 0:
        raise ValueError("quantiles must have a coordinate dimension")
    w = validate_weights(weights, q.shape[-1])
    if not np.all(np.isfinite(q)):
        raise ValueError("quantiles must be finite")
    return q * np.sqrt(w)


def from_rescaled(
    coordinates: ArrayLike, weights: ArrayLike
) -> NDArray[np.float64]:
    """Invert the rescaling z = diag(sqrt(w)) q."""

    z = np.asarray(coordinates, dtype=float)
    if z.ndim == 0:
        raise ValueError("coordinates must have a final dimension")
    w = validate_weights(weights, z.shape[-1])
    if not np.all(np.isfinite(z)):
        raise ValueError("coordinates must be finite")
    return z / np.sqrt(w)


def weighted_squared_distance(
    first: ArrayLike, second: ArrayLike, weights: ArrayLike
) -> NDArray[np.float64]:
    """Squared weighted Euclidean distance along the final dimension."""

    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("both inputs must have the same final dimension")
    w = validate_weights(weights, left.shape[-1])
    difference = left - right
    return np.sum(w * difference * difference, axis=-1)


def weighted_distance(
    first: ArrayLike, second: ArrayLike, weights: ArrayLike
) -> NDArray[np.float64]:
    """Weighted finite-grid Wasserstein distance."""

    return np.sqrt(weighted_squared_distance(first, second, weights))


def _weighted_pava(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Weighted pool-adjacent-violators algorithm for one vector."""

    means: list[float] = []
    masses: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for coordinate, (value, mass) in enumerate(zip(values, weights, strict=True)):
        means.append(float(value))
        masses.append(float(mass))
        starts.append(coordinate)
        ends.append(coordinate + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            merged_mass = masses[-2] + masses[-1]
            merged_mean = (
                masses[-2] * means[-2] + masses[-1] * means[-1]
            ) / merged_mass
            means[-2:] = [merged_mean]
            masses[-2:] = [merged_mass]
            starts[-2:] = [starts[-2]]
            ends[-2:] = [ends[-1]]

    projected = np.empty_like(values)
    for mean, start, end in zip(means, starts, ends, strict=True):
        projected[start:end] = mean
    return projected


def project_quantiles(
    values: ArrayLike,
    weights: ArrayLike,
    *,
    return_diagnostics: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], ProjectionDiagnostics]:
    """Project vectors onto the monotone cone in the weighted norm."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError("values must have a nonempty coordinate dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    w = validate_weights(weights, array.shape[-1])

    flat = array.reshape(-1, array.shape[-1])
    # A vector already in the monotone cone is its own projection: PAVA pools
    # nothing, so every block is a singleton and the output is bit-identical to
    # the input. Skipping those rows is exact and, during boosting, skips
    # essentially all of them.
    violating = (
        np.any(np.diff(flat, axis=1) < 0.0, axis=1)
        if flat.shape[1] > 1
        else np.zeros(flat.shape[0], dtype=bool)
    )
    projected_flat = flat.copy()
    violating_rows = np.flatnonzero(violating)
    for row in violating_rows:
        projected_flat[row] = _weighted_pava(flat[row], w)
    projected = projected_flat.reshape(array.shape)
    if not return_diagnostics:
        return projected

    raw_differences = np.diff(flat, axis=1)
    input_violations = int(np.sum(raw_differences < 0.0))
    adjustment = projected_flat - flat
    row_absolute = np.max(np.abs(adjustment), axis=1)
    row_weighted = np.sqrt(np.sum(w * adjustment * adjustment, axis=1))
    output_differences = np.diff(projected_flat, axis=1)
    output_max_violation = (
        float(max(0.0, -np.min(output_differences)))
        if output_differences.size
        else 0.0
    )
    diagnostics = ProjectionDiagnostics(
        n_vectors=flat.shape[0],
        n_changed=int(np.count_nonzero(row_absolute > 0.0)),
        input_violations=input_violations,
        max_absolute_adjustment=float(np.max(row_absolute, initial=0.0)),
        max_weighted_l2_adjustment=float(np.max(row_weighted, initial=0.0)),
        output_max_violation=output_max_violation,
    )
    return projected, diagnostics

