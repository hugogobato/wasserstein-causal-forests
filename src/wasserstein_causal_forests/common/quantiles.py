"""Validation helpers for finite monotone quantile vectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_weights(
    weights: ArrayLike,
    n_coordinates: int | None = None,
    *,
    require_normalized: bool = False,
    normalization_tolerance: float = 1e-12,
) -> NDArray[np.float64]:
    """Return a validated one-dimensional vector of positive finite weights."""

    result = np.asarray(weights, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional array")
    if n_coordinates is not None and result.size != n_coordinates:
        raise ValueError(
            f"weights has length {result.size}, expected {n_coordinates}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("weights must be finite")
    if np.any(result <= 0.0):
        raise ValueError("all quadrature weights must be strictly positive")
    if require_normalized and not np.isclose(
        result.sum(), 1.0, atol=normalization_tolerance, rtol=0.0
    ):
        raise ValueError("quadrature weights must sum to one")
    return result


def is_monotone(values: ArrayLike, *, atol: float = 1e-12) -> NDArray[np.bool_]:
    """Check nondecreasing order along the final array dimension."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("quantile values must have at least one dimension")
    if array.shape[-1] <= 1:
        return np.ones(array.shape[:-1], dtype=bool)
    return np.all(np.diff(array, axis=-1) >= -atol, axis=-1)


def validate_quantiles(
    values: ArrayLike,
    n_coordinates: int | None = None,
    *,
    check_monotone: bool = True,
    atol: float = 1e-12,
) -> NDArray[np.float64]:
    """Return finite quantile vectors and optionally enforce monotonicity."""

    result = np.asarray(values, dtype=float)
    if result.ndim == 0 or result.shape[-1] == 0:
        raise ValueError("quantiles must have a nonempty coordinate dimension")
    if n_coordinates is not None and result.shape[-1] != n_coordinates:
        raise ValueError(
            f"quantiles has {result.shape[-1]} coordinates, "
            f"expected {n_coordinates}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("quantiles must be finite")
    if check_monotone and not np.all(is_monotone(result, atol=atol)):
        raise ValueError("quantile vectors must be nondecreasing")
    return result


def canonicalize_particles(particles: ArrayLike) -> NDArray[np.float64]:
    """Lexicographically sort particles within every predicted empirical law.

    The sort removes externally visible particle labels. It has no statistical
    meaning and is never used to pair particles across treatment arms.
    """

    result = np.asarray(particles, dtype=float)
    if result.ndim < 2:
        raise ValueError("particles must have shape (..., n_particles, K)")
    n_particles, n_coordinates = result.shape[-2:]
    flat = result.reshape(-1, n_particles, n_coordinates)
    ordered = np.empty_like(flat)
    for row_index, row in enumerate(flat):
        keys = row[:, ::-1].T
        order = np.lexsort(keys)
        ordered[row_index] = row[order]
    return ordered.reshape(result.shape)


def canonical_training_order(
    X: ArrayLike, treatment: ArrayLike, quantiles: ArrayLike
) -> NDArray[np.int64]:
    """Return a deterministic lexicographic order for a complete training row."""

    x = np.asarray(X, dtype=float)
    a = np.asarray(treatment, dtype=int)
    q = np.asarray(quantiles, dtype=float)
    if x.ndim != 2 or q.ndim != 2 or a.ndim != 1:
        raise ValueError("expected X (n,p), treatment (n,), and quantiles (n,K)")
    if not (x.shape[0] == a.size == q.shape[0]):
        raise ValueError("X, treatment, and quantiles must have the same rows")
    combined = np.column_stack((x, a, q))
    return np.lexsort(combined[:, ::-1].T)

