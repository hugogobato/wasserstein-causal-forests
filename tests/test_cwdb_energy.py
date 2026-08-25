from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.energy import (
    energy_gradient,
    energy_score,
    energy_score_components,
)
from wasserstein_causal_forests.cwdb.geometry import (
    from_rescaled,
    project_quantiles,
    to_rescaled,
    weighted_distance,
)


@pytest.fixture
def score_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    particles = np.array(
        [
            [-1.2, -0.3, 0.8, 1.7],
            [-0.4, 0.1, 0.9, 2.1],
            [0.2, 0.7, 1.5, 2.8],
        ]
    )
    observation = np.array([-0.6, 0.0, 1.1, 2.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    return particles, observation, weights


@pytest.mark.parametrize("epsilon", [0.0, 1e-3])
def test_full_gradient_matches_central_differences(
    score_case: tuple[np.ndarray, np.ndarray, np.ndarray], epsilon: float
) -> None:
    particles, observation, weights = score_case
    analytic = energy_gradient(
        particles, observation, weights, epsilon=epsilon
    )
    numeric = np.empty_like(analytic)
    step = 1e-6
    for particle in range(particles.shape[0]):
        for coordinate in range(particles.shape[1]):
            upper = particles.copy()
            lower = particles.copy()
            upper[particle, coordinate] += step
            lower[particle, coordinate] -= step
            numeric[particle, coordinate] = (
                energy_score(upper, observation, weights, epsilon=epsilon)
                - energy_score(lower, observation, weights, epsilon=epsilon)
            ) / (2.0 * step)
    assert np.max(np.abs(analytic - numeric)) < 1e-6


def test_score_components_use_full_pairwise_repulsion(
    score_case: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    particles, observation, weights = score_case
    components = energy_score_components(particles, observation, weights)
    direct_attraction = np.mean(
        [weighted_distance(p, observation, weights) for p in particles]
    )
    direct_repulsion = 0.5 * np.mean(
        [
            weighted_distance(first, second, weights)
            for first in particles
            for second in particles
        ]
    )
    assert components.attraction == pytest.approx(direct_attraction)
    assert components.repulsion == pytest.approx(direct_repulsion)
    assert components.total == pytest.approx(
        direct_attraction - direct_repulsion
    )


def test_particle_permutation_invariance_and_gradient_equivariance(
    score_case: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    particles, observation, weights = score_case
    permutation = np.array([2, 0, 1])
    original_score = energy_score(particles, observation, weights, epsilon=1e-3)
    permuted_score = energy_score(
        particles[permutation], observation, weights, epsilon=1e-3
    )
    original_gradient = energy_gradient(
        particles, observation, weights, epsilon=1e-3
    )
    permuted_gradient = energy_gradient(
        particles[permutation], observation, weights, epsilon=1e-3
    )
    assert abs(float(original_score - permuted_score)) < 1e-12
    assert np.max(
        np.abs(original_gradient[permutation] - permuted_gradient)
    ) < 1e-12


@pytest.mark.parametrize("epsilon", [0.0, 1e-3])
def test_collisions_have_finite_score_and_gradient(epsilon: float) -> None:
    particles = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 2.0]])
    observation = np.array([0.0, 1.0])
    weights = np.array([0.25, 0.75])
    score = energy_score(particles, observation, weights, epsilon=epsilon)
    gradient = energy_gradient(
        particles, observation, weights, epsilon=epsilon
    )
    assert np.isfinite(score)
    assert np.all(np.isfinite(gradient))


def test_rescaling_is_exact_and_invertible() -> None:
    quantiles = np.array([[-2.0, -0.3, 1.7], [0.0, 1.0, 4.0]])
    weights = np.array([0.2, 0.3, 0.5])
    rescaled = to_rescaled(quantiles, weights)
    assert np.max(np.abs(from_rescaled(rescaled, weights) - quantiles)) < 1e-12
    direct = weighted_distance(quantiles[0], quantiles[1], weights)
    euclidean = np.linalg.norm(rescaled[0] - rescaled[1])
    assert abs(float(direct - euclidean)) < 1e-12


def test_weighted_isotonic_projection_and_diagnostics() -> None:
    values = np.array([[3.0, 0.0, 2.0], [-1.0, 0.0, 4.0]])
    weights = np.array([0.25, 0.5, 0.25])
    projected, diagnostics = project_quantiles(
        values, weights, return_diagnostics=True
    )
    assert np.allclose(projected[0], [1.0, 1.0, 2.0])
    assert np.allclose(projected[1], values[1])
    assert np.all(np.diff(projected, axis=1) >= 0.0)
    assert diagnostics.n_changed == 1
    assert diagnostics.input_violations == 1
    assert diagnostics.output_max_violation == 0.0


def test_projected_preconditioned_step_descends(
    score_case: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    particles, observation, weights = score_case
    before = energy_score(particles, observation, weights, epsilon=1e-3)
    gradient = energy_gradient(
        particles, observation, weights, epsilon=1e-3
    )
    candidate = project_quantiles(
        particles - 1e-2 * gradient / weights, weights
    )
    after = energy_score(candidate, observation, weights, epsilon=1e-3)
    assert after < before


def test_proper_score_distinguishes_two_mode_law_from_barycenter() -> None:
    weights = np.array([1.0])
    observations = np.array([[-1.0], [1.0]])
    correct = np.broadcast_to(
        np.array([[-1.0], [1.0]]), (2, 2, 1)
    ).copy()
    collapsed = np.zeros((2, 2, 1))
    correct_risk = np.mean(energy_score(correct, observations, weights))
    collapsed_risk = np.mean(energy_score(collapsed, observations, weights))
    assert correct_risk < collapsed_risk


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        energy_score(np.zeros((2, 2)), np.zeros(2), [0.5, 0.0])

