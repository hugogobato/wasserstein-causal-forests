"""A common representation for every method's conditional law estimate.

The six tournament methods produce conditional laws in two different shapes.
C-WDB emits M particles per test row, so its atoms are row specific and its
weights uniform. The two forest baselines emit weights over the training
sample, so their atoms are one shared bank of n_train grid vectors and only the
weights vary by row. `LawPrediction` covers both, which is what lets one metric
implementation score all of them without target substitution.

The distinction is not cosmetic. With a shared bank the pairwise atom distance
matrix is computed once for the whole test set rather than once per row, which
turns the energy score's repulsion term from an O(n A^2 K) loop into a single
matrix product. At n_train = 1000 that is the difference between a metric that
runs and one that does not.

Methods that estimate a conditional mean rather than a conditional law, namely
the two PTA endpoints, do not produce a `LawPrediction` at all. Under
`research/estimand_contract.md` section 4 a posterior draw of a mean surface
may not be relabelled as a draw from the outcome law, so those methods report
`status = "not_applicable"` on every law-level metric instead of supplying a
substitute.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Bytes a single pairwise temporary may occupy. `_squared_distances` holds
#: about four arrays of that shape at once, so the real peak per call is a few
#: times this. Keep it small: the tournament runs one worker per core, and every
#: worker can be inside this code at the same time.
CHUNK_BUDGET_BYTES = 32 * 1024 * 1024


def chunk_rows(n_atoms: int, n_paired: int) -> int:
    """Test rows per pass so that (rows, n_atoms, n_paired) stays in budget.

    A fixed row count is wrong here because the atom count is the training
    sample size for the forest baselines but the particle count for C-WDB, a
    range of two orders of magnitude. At n_train = 2000 against a 288-node
    truth, sixty-four rows would reserve nearly 300 MB per array and over a
    gigabyte per call.
    """

    per_row = max(1, 8 * int(n_atoms) * int(n_paired))
    return max(1, CHUNK_BUDGET_BYTES // per_row)


@dataclass(frozen=True)
class LawPrediction:
    """A conditional law estimate: weighted atoms of Q_K per test row."""

    atoms: NDArray[np.float64]
    weights: NDArray[np.float64]
    shared_atoms: bool

    def __post_init__(self) -> None:
        atoms = np.asarray(self.atoms, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 2:
            raise ValueError("weights must have shape (n, A)")
        expected = (weights.shape[1], atoms.shape[-1])
        if self.shared_atoms:
            if atoms.shape != expected:
                raise ValueError("shared atoms must have shape (A, K)")
        elif atoms.shape != (weights.shape[0],) + expected:
            raise ValueError("row-specific atoms must have shape (n, A, K)")
        if not np.all(np.isfinite(atoms)) or not np.all(np.isfinite(weights)):
            raise ValueError("atoms and weights must be finite")
        if np.any(weights < 0.0):
            raise ValueError("weights must be nonnegative")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
            raise ValueError("each row's weights must sum to one")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "weights", weights)

    @property
    def n_rows(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.weights.shape[1])

    @property
    def n_grid(self) -> int:
        return int(self.atoms.shape[-1])

    @classmethod
    def from_particles(cls, particles: NDArray[np.float64]) -> "LawPrediction":
        """Wrap an (n, M, K) particle block carrying uniform mass."""

        particles = np.asarray(particles, dtype=float)
        if particles.ndim != 3:
            raise ValueError("particles must have shape (n, M, K)")
        n_rows, n_particles = particles.shape[:2]
        weights = np.full((n_rows, n_particles), 1.0 / n_particles)
        return cls(atoms=particles, weights=weights, shared_atoms=False)

    @classmethod
    def from_forest_weights(
        cls, training_quantiles: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> "LawPrediction":
        """Wrap forest weights over a shared bank of training grid vectors."""

        return cls(
            atoms=np.asarray(training_quantiles, dtype=float),
            weights=np.asarray(weights, dtype=float),
            shared_atoms=True,
        )

    def row_atoms(self, rows: slice) -> NDArray[np.float64]:
        """Atoms for a block of rows, always shaped (block, A, K)."""

        if self.shared_atoms:
            block = self.weights[rows].shape[0]
            return np.broadcast_to(self.atoms, (block, *self.atoms.shape))
        return self.atoms[rows]

    # ----------------------------------------------------------- expectations

    def mean_quantiles(self) -> NDArray[np.float64]:
        """E_Phat[q], the estimate of `MEANQ-A-K`, shape (n, K)."""

        if self.shared_atoms:
            return self.weights @ self.atoms
        return np.einsum("na,nak->nk", self.weights, self.atoms)

    def scalar_expectation(
        self, statistic: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """E_Phat[statistic(q)] for a statistic mapping (m, K) to (m,)."""

        if self.shared_atoms:
            return self.weights @ np.asarray(statistic(self.atoms), dtype=float)
        flat = self.atoms.reshape(-1, self.n_grid)
        values = np.asarray(statistic(flat), dtype=float)
        return np.einsum("na,na->n", self.weights, values.reshape(self.weights.shape))

    def effective_support(self) -> NDArray[np.float64]:
        """Participation ratio of the weights, an atom-count diagnostic."""

        return 1.0 / np.sum(self.weights * self.weights, axis=1)


def smoothed_norm(
    difference: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    """The collision-smoothed weighted norm used by the certified score."""

    squared = np.sum(grid_weights * difference * difference, axis=-1)
    return np.sqrt(squared + epsilon * epsilon) - epsilon


def _squared_distances(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pairwise squared W_{2,K} distances, via the inner-product identity.

    Expanding ||a - b||^2 as ||a||^2 + ||b||^2 - 2 a.b turns the cross term
    into a batched matrix product, which BLAS runs an order of magnitude faster
    than the broadcast difference tensor and without materialising an
    (n, A, J, K) temporary. The cancellation this trades for is harmless here:
    the absolute error is at machine epsilon on quantities the callers either
    smooth by epsilon or exponentiate.

    Accepts (..., A, K) and (..., J, K) and returns (..., A, J).
    """

    scale = np.sqrt(grid_weights)
    a = left * scale
    b = right * scale
    cross = np.matmul(a, np.swapaxes(b, -1, -2))
    squared = (
        np.sum(a * a, axis=-1)[..., :, None]
        + np.sum(b * b, axis=-1)[..., None, :]
        - 2.0 * cross
    )
    return np.maximum(squared, 0.0)


def _self_repulsion(
    prediction: LawPrediction, grid_weights: NDArray[np.float64], epsilon: float
) -> NDArray[np.float64]:
    """0.5 * sum_{j,l} v_j v_l d_eps(p_j, p_l), shape (n,)."""

    if prediction.shared_atoms:
        # One (A, A) distance matrix serves every row, so the row loop becomes
        # a matrix product.
        squared = _squared_distances(prediction.atoms, prediction.atoms, grid_weights)
        distances = np.sqrt(squared + epsilon * epsilon) - epsilon
        return 0.5 * np.einsum(
            "na,ab,nb->n", prediction.weights, distances, prediction.weights
        )

    total = np.empty(prediction.n_rows)
    step = chunk_rows(prediction.n_atoms, prediction.n_atoms)
    for start in range(0, prediction.n_rows, step):
        rows = slice(start, start + step)
        atoms = prediction.atoms[rows]
        squared = _squared_distances(atoms, atoms, grid_weights)
        distances = np.sqrt(squared + epsilon * epsilon) - epsilon
        block_weights = prediction.weights[rows]
        total[rows] = 0.5 * np.einsum(
            "na,nab,nb->n", block_weights, distances, block_weights
        )
    return total


def energy_risk_against_truth(
    prediction: LawPrediction,
    truth_nodes: NDArray[np.float64],
    truth_weights: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
    *,
    epsilon: float,
) -> NDArray[np.float64]:
    """`arm_energy_risk`: E_{q ~ P_a(x)} S_eps(Phat_a(x), q), shape (n,).

    The expectation is taken against the DGP's quadrature representation of the
    true conditional law rather than against a single observed draw, so the
    metric carries quadrature error rather than one draw's worth of sampling
    noise. Lower is better, and the minimiser over all laws is the truth.
    """

    truth_nodes = np.asarray(truth_nodes, dtype=float)
    truth_weights = np.asarray(truth_weights, dtype=float)
    if truth_nodes.ndim != 3 or truth_nodes.shape[0] != prediction.n_rows:
        raise ValueError("truth_nodes must have shape (n, J, K)")
    if truth_weights.shape != (truth_nodes.shape[1],):
        raise ValueError("truth_weights must have shape (J,)")

    attraction = np.empty(prediction.n_rows)
    step = chunk_rows(prediction.n_atoms, truth_nodes.shape[1])
    for start in range(0, prediction.n_rows, step):
        rows = slice(start, start + step)
        atoms = prediction.row_atoms(rows)
        squared = _squared_distances(atoms, truth_nodes[rows], grid_weights)
        distances = np.sqrt(squared + epsilon * epsilon) - epsilon
        attraction[rows] = np.einsum(
            "na,naj,j->n", prediction.weights[rows], distances, truth_weights
        )
    return attraction - _self_repulsion(prediction, grid_weights, epsilon)


def _gaussian_kernel_gram(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    """Gaussian kernel on the rescaled coordinates z = diag(sqrt(w)) q."""

    squared = _squared_distances(left, right, grid_weights)
    return np.exp(-squared / (2.0 * bandwidth * bandwidth))


def kernel_law_error(
    prediction: LawPrediction,
    truth_nodes: NDArray[np.float64],
    truth_weights: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
    *,
    bandwidth: float,
) -> NDArray[np.float64]:
    """`kernel_law_error`: squared MMD to the true conditional law, shape (n,).

    The kernel is Gaussian in the rescaled coordinates, so distances in the
    kernel are the declared W_{2,K} distances and the discrepancy is scale
    equivariant in the units of the outcome distributions. A Gaussian kernel is
    characteristic, so the discrepancy vanishes only at the true law.
    """

    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    truth_nodes = np.asarray(truth_nodes, dtype=float)
    truth_weights = np.asarray(truth_weights, dtype=float)

    truth_term = np.einsum(
        "i,nij,j->n",
        truth_weights,
        _gaussian_kernel_gram(truth_nodes, truth_nodes, grid_weights, bandwidth),
        truth_weights,
    )

    result = np.empty(prediction.n_rows)
    if prediction.shared_atoms:
        gram = _gaussian_kernel_gram(
            prediction.atoms, prediction.atoms, grid_weights, bandwidth
        )
        own_term = np.einsum("na,ab,nb->n", prediction.weights, gram, prediction.weights)
    else:
        own_term = np.empty(prediction.n_rows)

    step = chunk_rows(prediction.n_atoms, max(truth_nodes.shape[1], prediction.n_atoms))
    for start in range(0, prediction.n_rows, step):
        rows = slice(start, start + step)
        atoms = prediction.row_atoms(rows)
        block_weights = prediction.weights[rows]
        if not prediction.shared_atoms:
            gram = _gaussian_kernel_gram(atoms, atoms, grid_weights, bandwidth)
            own_term[rows] = np.einsum("na,nab,nb->n", block_weights, gram, block_weights)
        cross = _gaussian_kernel_gram(
            atoms, truth_nodes[rows], grid_weights, bandwidth
        )
        cross_term = np.einsum("na,naj,j->n", block_weights, cross, truth_weights)
        result[rows] = own_term[rows] - 2.0 * cross_term + truth_term[rows]
    return np.maximum(result, 0.0)


def median_heuristic_bandwidth(
    nodes: NDArray[np.float64], grid_weights: NDArray[np.float64]
) -> float:
    """Median pairwise W_{2,K} distance over a pooled atom sample.

    This mirrors `bandwidth_rule = "median_distance"` in the Causal-DRF driver:
    a rule tied to the outcome's own scale rather than to arbitrary units.
    """

    flat = np.asarray(nodes, dtype=float).reshape(-1, nodes.shape[-1])
    if flat.shape[0] > 512:
        index = np.linspace(0, flat.shape[0] - 1, 512).astype(int)
        flat = flat[index]
    distances = np.sqrt(_squared_distances(flat, flat, grid_weights))
    upper = distances[np.triu_indices(flat.shape[0], k=1)]
    median = float(np.median(upper)) if upper.size else 1.0
    return median if median > 0.0 else 1.0


def tail_probability(
    prediction: LawPrediction, *, level_index: int, threshold: float
) -> NDArray[np.float64]:
    """`tail_calibration` estimate: Phat{q_{level_index} > threshold}."""

    return prediction.scalar_expectation(
        lambda block: (block[:, level_index] > threshold).astype(float)
    )


def mode_coverage(
    prediction: LawPrediction,
    mode_centres: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
    *,
    radius: float,
    mass_floor: float,
) -> NDArray[np.float64]:
    """`mode_coverage`: fraction of true modes carrying mass, shape (n,).

    A mode counts as covered when the atoms within `radius` of its centre carry
    at least `mass_floor` of the predicted mass. A law that has collapsed onto
    one mode of a bimodal outer law scores 0.5 however well it fits that mode,
    which is the collapse signature the D6 claim is about.
    """

    mode_centres = np.asarray(mode_centres, dtype=float)
    if mode_centres.ndim != 3 or mode_centres.shape[0] != prediction.n_rows:
        raise ValueError("mode_centres must have shape (n, R, K)")
    covered = np.empty((prediction.n_rows, mode_centres.shape[1]))
    step = chunk_rows(prediction.n_atoms, mode_centres.shape[1])
    for start in range(0, prediction.n_rows, step):
        rows = slice(start, start + step)
        atoms = prediction.row_atoms(rows)
        distances = np.sqrt(_squared_distances(atoms, mode_centres[rows], grid_weights))
        mass = np.einsum(
            "na,nar->nr", prediction.weights[rows], (distances <= radius).astype(float)
        )
        covered[rows] = (mass >= mass_floor).astype(float)
    return covered.mean(axis=1)
