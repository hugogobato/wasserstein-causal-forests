"""The common law representation must score every method identically.

`LawPrediction` exists so one metric implementation covers both C-WDB's
row-specific particles and the forest baselines' shared training bank. These
checks pin the two paths to each other and pin the weighted energy risk to the
certified uniform-weight score in `wasserstein_causal_forests.cwdb.energy`.
"""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.energy import energy_score
from wasserstein_causal_forests.g3.laws import (
    LawPrediction,
    energy_risk_against_truth,
    kernel_law_error,
    median_heuristic_bandwidth,
    mode_coverage,
    tail_probability,
)


def particles(seed: int, n: int = 40, n_particles: int = 6, n_grid: int = 5):
    rng = np.random.default_rng(seed)
    return np.sort(rng.normal(size=(n, n_particles, n_grid)), axis=-1)


def truth(seed: int, n: int = 40, n_nodes: int = 9, n_grid: int = 5):
    rng = np.random.default_rng(seed + 500)
    nodes = np.sort(rng.normal(size=(n, n_nodes, n_grid)), axis=-1)
    weights = rng.uniform(0.5, 1.5, size=n_nodes)
    return nodes, weights / weights.sum()


def grid_weights(n_grid: int = 5) -> np.ndarray:
    return np.full(n_grid, 1.0 / n_grid)


@pytest.mark.parametrize("seed", range(4))
def test_energy_risk_reduces_to_the_certified_score_at_one_truth_node(
    seed: int,
) -> None:
    """With the truth a point mass, the risk is the certified score itself."""

    block = particles(seed)
    observations = np.sort(np.random.default_rng(seed + 9).normal(size=(40, 5)), axis=-1)
    w = grid_weights()
    prediction = LawPrediction.from_particles(block)

    risk = energy_risk_against_truth(
        prediction, observations[:, None, :], np.ones(1), w, epsilon=1e-3
    )
    certified = energy_score(block, observations, w, epsilon=1e-3)
    assert risk == pytest.approx(certified, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_shared_and_row_specific_paths_agree(seed: int) -> None:
    """A shared bank broadcast to every row must score like row-specific atoms."""

    rng = np.random.default_rng(seed)
    bank = np.sort(rng.normal(size=(7, 5)), axis=-1)
    weights = rng.uniform(0.2, 1.0, size=(40, 7))
    weights = weights / weights.sum(axis=1, keepdims=True)
    w = grid_weights()
    nodes, node_weights = truth(seed)

    shared = LawPrediction.from_forest_weights(bank, weights)
    expanded = LawPrediction(
        atoms=np.broadcast_to(bank, (40, 7, 5)).copy(),
        weights=weights,
        shared_atoms=False,
    )

    for metric in (
        lambda p: energy_risk_against_truth(p, nodes, node_weights, w, epsilon=1e-3),
        lambda p: kernel_law_error(p, nodes, node_weights, w, bandwidth=0.8),
        lambda p: p.mean_quantiles(),
        lambda p: tail_probability(p, level_index=4, threshold=0.5),
    ):
        assert metric(shared) == pytest.approx(metric(expanded), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_kernel_law_error_vanishes_only_at_the_truth(seed: int) -> None:
    nodes, node_weights = truth(seed)
    w = grid_weights()
    exact = LawPrediction(
        atoms=nodes, weights=np.tile(node_weights, (nodes.shape[0], 1)),
        shared_atoms=False,
    )
    bandwidth = median_heuristic_bandwidth(nodes, w)

    at_truth = kernel_law_error(exact, nodes, node_weights, w, bandwidth=bandwidth)
    assert at_truth == pytest.approx(0.0, abs=1e-12)

    wrong = LawPrediction.from_particles(particles(seed + 100))
    assert np.min(kernel_law_error(wrong, nodes, node_weights, w, bandwidth=bandwidth)) > 0.0


@pytest.mark.parametrize("seed", range(4))
def test_energy_risk_is_minimised_at_the_truth(seed: int) -> None:
    """The score is proper, so no competitor beats the true law on average."""

    nodes, node_weights = truth(seed, n=60, n_nodes=12)
    w = grid_weights()
    exact = LawPrediction(
        atoms=nodes, weights=np.tile(node_weights, (nodes.shape[0], 1)),
        shared_atoms=False,
    )
    best = energy_risk_against_truth(exact, nodes, node_weights, w, epsilon=0.0)

    rng = np.random.default_rng(seed)
    for _ in range(5):
        perturbed = LawPrediction(
            atoms=np.sort(nodes + 0.4 * rng.normal(size=nodes.shape), axis=-1),
            weights=np.tile(node_weights, (nodes.shape[0], 1)),
            shared_atoms=False,
        )
        competitor = energy_risk_against_truth(
            perturbed, nodes, node_weights, w, epsilon=0.0
        )
        assert competitor.mean() > best.mean()


def test_mode_coverage_detects_collapse() -> None:
    """A law sitting on one mode of a bimodal truth scores exactly one half."""

    w = grid_weights()
    base = np.sort(np.random.default_rng(1).normal(size=(30, 5)), axis=-1)
    centres = np.stack([base - 3.0, base + 3.0], axis=1)

    collapsed = LawPrediction.from_particles(
        np.repeat((base - 3.0)[:, None, :], 8, axis=1)
    )
    assert mode_coverage(
        collapsed, centres, w, radius=0.5, mass_floor=0.1
    ) == pytest.approx(0.5)

    spread = np.concatenate(
        [np.repeat((base - 3.0)[:, None, :], 4, axis=1),
         np.repeat((base + 3.0)[:, None, :], 4, axis=1)],
        axis=1,
    )
    assert mode_coverage(
        LawPrediction.from_particles(spread), centres, w, radius=0.5, mass_floor=0.1
    ) == pytest.approx(1.0)


def test_effective_support_counts_atoms() -> None:
    uniform = LawPrediction.from_particles(particles(0, n_particles=8))
    assert uniform.effective_support() == pytest.approx(8.0)

    concentrated = np.zeros((3, 8))
    concentrated[:, 0] = 1.0
    degenerate = LawPrediction(
        atoms=particles(0, n=3, n_particles=8),
        weights=concentrated,
        shared_atoms=False,
    )
    assert degenerate.effective_support() == pytest.approx(1.0)


def test_malformed_predictions_are_rejected() -> None:
    good_atoms = particles(0, n=4, n_particles=3)
    with pytest.raises(ValueError):
        LawPrediction(atoms=good_atoms, weights=np.ones((4, 3)), shared_atoms=False)
    with pytest.raises(ValueError):
        LawPrediction(
            atoms=good_atoms,
            weights=np.full((4, 3), 1.0 / 3.0) * np.array([1.0, 1.0, -1.0]),
            shared_atoms=False,
        )
    with pytest.raises(ValueError):
        LawPrediction(atoms=good_atoms[0], weights=np.full((4, 3), 1 / 3), shared_atoms=False)
