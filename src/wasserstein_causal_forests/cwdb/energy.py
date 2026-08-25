"""Certified ensemble-coupled empirical energy score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common.quantiles import validate_weights


@dataclass(frozen=True)
class EnergyScoreComponents:
    """Attraction, repulsion, and total loss for one or more observations."""

    attraction: NDArray[np.float64]
    repulsion: NDArray[np.float64]
    total: NDArray[np.float64]


def _validate_inputs(
    particles: ArrayLike, observation: ArrayLike, weights: ArrayLike, epsilon: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    p = np.asarray(particles, dtype=float)
    y = np.asarray(observation, dtype=float)
    if p.ndim < 2:
        raise ValueError("particles must have shape (..., M, K)")
    if y.ndim < 1:
        raise ValueError("observation must have shape (..., K)")
    if p.shape[-1] != y.shape[-1]:
        raise ValueError("particles and observation must use the same K")
    if p.shape[:-2] != y.shape[:-1]:
        raise ValueError("leading particle and observation dimensions must match")
    if p.shape[-2] < 1:
        raise ValueError("at least one particle is required")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
        raise ValueError("particles and observation must be finite")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    w = validate_weights(weights, p.shape[-1])
    return p, y, w


def _smoothed_norm(
    difference: NDArray[np.float64],
    weights: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    squared = np.sum(weights * difference * difference, axis=-1)
    return np.sqrt(squared + epsilon * epsilon) - epsilon


# The pairwise tensor has shape (rows, M, M, K) and several temporaries of that
# size are live at once, so a full pass costs roughly 4 * 8 * M^2 * K bytes per
# row. At n = 2000, M = 50, K = 49 that is about 8 GB, which the tournament
# machine does not have. Splitting the leading axis is exact: every quantity
# below is computed independently per leading index, so concatenating the
# chunks reproduces the unchunked array bit for bit.
_ENERGY_CHUNK_BYTES = 128 * 1024 * 1024


def _chunk_rows(p: NDArray[np.float64]) -> int:
    """Rows per pass, or 0 to run the whole batch in one pass."""

    if p.ndim != 3:
        return 0
    per_row = 4 * 8 * p.shape[-2] * p.shape[-2] * p.shape[-1]
    if p.shape[0] * per_row <= _ENERGY_CHUNK_BYTES:
        return 0
    return max(1, _ENERGY_CHUNK_BYTES // per_row)


def _energy_score_components_dense(
    p: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    epsilon: float,
) -> EnergyScoreComponents:
    attraction_distances = _smoothed_norm(p - y[..., None, :], w, epsilon)
    pair_differences = p[..., :, None, :] - p[..., None, :, :]
    pair_distances = _smoothed_norm(pair_differences, w, epsilon)
    attraction = np.mean(attraction_distances, axis=-1)
    repulsion = 0.5 * np.mean(pair_distances, axis=(-2, -1))
    total = attraction - repulsion
    return EnergyScoreComponents(
        attraction=np.asarray(attraction),
        repulsion=np.asarray(repulsion),
        total=np.asarray(total),
    )


def energy_score_components(
    particles: ArrayLike,
    observation: ArrayLike,
    weights: ArrayLike,
    *,
    epsilon: float = 0.0,
) -> EnergyScoreComponents:
    """Compute both terms of the loss-oriented empirical energy score."""

    p, y, w = _validate_inputs(particles, observation, weights, epsilon)
    step = _chunk_rows(p)
    if step == 0:
        return _energy_score_components_dense(p, y, w, epsilon)
    parts = [
        _energy_score_components_dense(p[start : start + step], y[start : start + step], w, epsilon)
        for start in range(0, p.shape[0], step)
    ]
    return EnergyScoreComponents(
        attraction=np.concatenate([part.attraction for part in parts]),
        repulsion=np.concatenate([part.repulsion for part in parts]),
        total=np.concatenate([part.total for part in parts]),
    )


def energy_score(
    particles: ArrayLike,
    observation: ArrayLike,
    weights: ArrayLike,
    *,
    epsilon: float = 0.0,
) -> NDArray[np.float64]:
    """Return the certified score, lower being better."""

    return energy_score_components(
        particles, observation, weights, epsilon=epsilon
    ).total


def empirical_energy_risk(
    particles: ArrayLike,
    observations: ArrayLike,
    weights: ArrayLike,
    *,
    epsilon: float = 0.0,
) -> float:
    """Average score over observations with row-specific particle laws."""

    scores = energy_score(particles, observations, weights, epsilon=epsilon)
    return float(np.mean(scores))


def _safe_ratio(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    result = np.zeros_like(numerator)
    np.divide(
        numerator,
        denominator[..., None],
        out=result,
        where=denominator[..., None] > 0.0,
    )
    return result


def _energy_gradient_dense(
    p: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    n_particles = p.shape[-2]

    outcome_difference = p - y[..., None, :]
    outcome_denominator = np.sqrt(
        np.sum(w * outcome_difference * outcome_difference, axis=-1)
        + epsilon * epsilon
    )
    attraction_gradient = _safe_ratio(
        w * outcome_difference, outcome_denominator
    ) / n_particles

    pair_difference = p[..., :, None, :] - p[..., None, :, :]
    pair_denominator = np.sqrt(
        np.sum(w * pair_difference * pair_difference, axis=-1)
        + epsilon * epsilon
    )
    pair_ratios = _safe_ratio(w * pair_difference, pair_denominator)
    repulsion_gradient = np.sum(pair_ratios, axis=-2) / (n_particles**2)
    return attraction_gradient - repulsion_gradient


def energy_gradient(
    particles: ArrayLike,
    observation: ArrayLike,
    weights: ArrayLike,
    *,
    epsilon: float = 0.0,
) -> NDArray[np.float64]:
    """Gradient with respect to every particle and quantile coordinate.

    Both ordered appearances of each particle in the repulsion term are
    included. At exact collisions with ``epsilon=0``, the declared zero
    subgradient is returned.
    """

    p, y, w = _validate_inputs(particles, observation, weights, epsilon)
    step = _chunk_rows(p)
    if step == 0:
        return _energy_gradient_dense(p, y, w, epsilon)
    return np.concatenate(
        [
            _energy_gradient_dense(
                p[start : start + step], y[start : start + step], w, epsilon
            )
            for start in range(0, p.shape[0], step)
        ]
    )

