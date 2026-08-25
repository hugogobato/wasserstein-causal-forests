"""Chunking the leading axis of the energy kernels must change nothing.

The pairwise tensor is (n, M, M, K); at the tournament's largest cells a single
dense pass needs several gigabytes. `energy_score_components` and
`energy_gradient` therefore split the leading axis. Every quantity they compute
is independent per leading index, so the split is expected to be bit-exact, and
these checks pin that.
"""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb import energy


def batch(seed: int, n: int, n_particles: int, n_grid: int):
    rng = np.random.default_rng(seed)
    particles = np.sort(rng.normal(size=(n, n_particles, n_grid)), axis=-1)
    observations = np.sort(rng.normal(size=(n, n_grid)), axis=-1)
    weights = np.full(n_grid, 1.0 / n_grid)
    return particles, observations, weights


@pytest.mark.parametrize("epsilon", [0.0, 1e-3])
@pytest.mark.parametrize("seed", range(4))
def test_chunked_score_is_bit_identical(seed: int, epsilon: float) -> None:
    particles, observations, weights = batch(seed, n=57, n_particles=6, n_grid=5)

    dense = energy.energy_score_components(
        particles, observations, weights, epsilon=epsilon
    )
    original = energy._ENERGY_CHUNK_BYTES
    try:
        # Force a split into many uneven chunks, including a short final one.
        energy._ENERGY_CHUNK_BYTES = 4 * 8 * 6 * 6 * 5 * 7
        chunked = energy.energy_score_components(
            particles, observations, weights, epsilon=epsilon
        )
    finally:
        energy._ENERGY_CHUNK_BYTES = original

    assert np.array_equal(chunked.attraction, dense.attraction)
    assert np.array_equal(chunked.repulsion, dense.repulsion)
    assert np.array_equal(chunked.total, dense.total)


@pytest.mark.parametrize("epsilon", [0.0, 1e-3])
@pytest.mark.parametrize("seed", range(4))
def test_chunked_gradient_is_bit_identical(seed: int, epsilon: float) -> None:
    particles, observations, weights = batch(seed, n=57, n_particles=6, n_grid=5)

    dense = energy.energy_gradient(
        particles, observations, weights, epsilon=epsilon
    )
    original = energy._ENERGY_CHUNK_BYTES
    try:
        energy._ENERGY_CHUNK_BYTES = 4 * 8 * 6 * 6 * 5 * 7
        chunked = energy.energy_gradient(
            particles, observations, weights, epsilon=epsilon
        )
    finally:
        energy._ENERGY_CHUNK_BYTES = original

    assert chunked.shape == dense.shape
    assert np.array_equal(chunked, dense)


def test_chunking_is_skipped_for_a_single_unbatched_law() -> None:
    rng = np.random.default_rng(0)
    particles = np.sort(rng.normal(size=(6, 5)), axis=-1)
    observation = np.sort(rng.normal(size=5))
    weights = np.full(5, 0.2)
    assert energy._chunk_rows(particles) == 0
    assert np.isscalar(
        float(energy.energy_score(particles, observation, weights))
    )
