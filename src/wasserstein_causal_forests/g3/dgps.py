"""The G3 tournament DGP suite D0 through D9, with quadrature oracle truth.

Every unit carries a distribution-valued outcome under the `ORACLE-V1`
observation regime: the observed response is the exact grid quantile vector
q(Y) of the unit's own outcome law. All ten regimes share one generative form,

    q(Y^a)_k = m_a(x) + xi + exp{s_a(x) + eta} * psi(z_k; gamma_a(x)),

with z_k the standard normal quantile at the declared grid level u_k, the outer
latent pair (xi, eta) independent of x and A given the arm, and

    psi(z; g) = z + g He_2(z) / 2 + g He_3(z) / 6
              = z + g (z^2 - 1) / 2 + g (z^3 - 3 z) / 6

a reshaping built from the second and third probabilists' Hermite polynomials.
Both are orthogonal to the identity under the standard normal, so E{psi(Z)} = 0
and the reshaping leaves the mean alone while moving skewness (the quadratic
term) and the upper tail (the cubic term). Its derivative is
1 + g (z + (z^2 - 1) / 2), minimised at z = -1 with value 1 - g, so psi is
strictly increasing for every 0 <= g < 1 whatever the grid: the monotone-cone
guarantee does not weaken as K grows. The regimes differ only in the surfaces
m_a, s_a, gamma_a, the outer law, and the propensity. Writing every regime in one form is what makes the oracle truth
below uniform: because (xi, eta) is low dimensional with a known Gaussian (or
Gaussian-mixture) law, the conditional expectation of any grid functional is a
Gauss-Hermite quadrature sum rather than a Monte Carlo average, so the truth
that enters an RMSE denominator carries quadrature error rather than sampling
noise. `research/checks/g3_dgp_truth_accuracy.py` measures that error against
a large Monte Carlo draw.

Target identifiers follow `research/estimand_contract.md` (`G0-WP0-A-v1`).
Only grid identifiers are emitted here: the stored grid does not determine an
arbitrary continuum functional.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..pta_bcf.targets import GRID_FUNCTIONALS

#: Regime identifiers in the frozen order used by every manifest and result row.
DGP_IDS = ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")

#: Equal-probability bins of X_0 defining the discrete moderator V = g(X).
#: A discrete moderator keeps every `TCATE-K-j` statement a finite collection of
#: conditional means, which avoids the contract's prohibition on claims at every
#: continuous v.
MODERATOR_EDGES = (-0.5, 0.0, 0.5)
N_MODERATOR_BINS = len(MODERATOR_EDGES) + 1

#: Default nodes per outer latent dimension for the quadrature truth. Twelve
#: agrees with a twenty-node rule to 7.5e-3 across every regime and target,
#: which is inside the Monte Carlo noise floor of
#: `research/checks/g3_dgp_truth_accuracy.py`. The law-level metrics cost grows
#: with the square of the node count, so the smallest converged rule matters.
DEFAULT_QUADRATURE_NODES = 12

#: Largest admissible shape parameter. `psi(.; g)` has minimum slope 1 - g, so
#: anything below one keeps every sampled vector strictly inside the monotone
#: cone; the margin guards against reaching a numerically flat segment.
MAX_SHAPE = 0.85


def moderator_bins(X: NDArray[np.float64]) -> NDArray[np.int64]:
    """V = g(X), the pre-treatment moderator, as a bin index in 0..3."""

    return np.searchsorted(np.asarray(MODERATOR_EDGES), X[:, 0], side="right")


def _psi(z: NDArray[np.float64], gamma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Monotone Hermite reshaping, broadcast over (n, K)."""

    hermite = (z * z - 1.0) / 2.0 + (z * z * z - 3.0 * z) / 6.0
    return z + gamma[..., None] * hermite


@dataclass(frozen=True)
class GridSpec:
    """The declared quantile grid: levels, base shape, and quadrature weights."""

    n_grid: int

    def __post_init__(self) -> None:
        if self.n_grid < 2:
            raise ValueError("n_grid must be at least 2")

    @property
    def levels(self) -> NDArray[np.float64]:
        return (np.arange(self.n_grid, dtype=float) + 0.5) / self.n_grid

    @property
    def base_z(self) -> NDArray[np.float64]:
        return norm.ppf(self.levels)

    @property
    def weights(self) -> NDArray[np.float64]:
        return np.full(self.n_grid, 1.0 / self.n_grid)

    @property
    def max_abs_z(self) -> float:
        return float(np.max(np.abs(self.base_z)))

    def reference_quantiles(self) -> NDArray[np.float64]:
        """The frozen external reference law nu_star: the standard normal."""

        return self.base_z


@dataclass(frozen=True)
class OuterLaw:
    """Law of the outer latent (xi, eta), with its Gauss-Hermite rule.

    `xi` is the outer location shift and `eta` the outer log-scale shift. When
    `mixture_shift` is nonzero, xi is an equal-weight two-component mixture
    centred at plus and minus that shift, which is what makes the D6 outer law
    multimodal.
    """

    location_sd: float = 0.0
    log_scale_sd: float = 0.0
    mixture_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.location_sd < 0.0 or self.log_scale_sd < 0.0:
            raise ValueError("outer standard deviations must be nonnegative")
        if self.mixture_shift < 0.0:
            raise ValueError("mixture_shift must be nonnegative")

    @property
    def is_multimodal(self) -> bool:
        return self.mixture_shift > 0.0

    @property
    def location_modes(self) -> NDArray[np.float64]:
        """Mode locations of the outer location law, for `mode_coverage`."""

        if not self.is_multimodal:
            return np.zeros(1)
        return np.array([-self.mixture_shift, self.mixture_shift])

    def sample(
        self, rng: np.random.Generator, n_rows: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Draw (xi, eta), always consuming the generator in a fixed order."""

        base_location = rng.normal(size=n_rows)
        component = rng.integers(0, 2, size=n_rows) * 2 - 1
        base_log_scale = rng.normal(size=n_rows)
        xi = self.location_sd * base_location
        if self.is_multimodal:
            xi = xi + component * self.mixture_shift
        return xi, self.log_scale_sd * base_log_scale

    def location_survival(self, cutoff: NDArray[np.float64]) -> NDArray[np.float64]:
        """P(xi > cutoff), in closed form.

        Threshold events in the location are indicators, which no Gaussian
        quadrature rule integrates: the discontinuity destroys the polynomial
        exactness the rule is built on. Handling the location analytically and
        quadraturing only the smooth log-scale latent keeps `tail_calibration`
        truth exact.
        """

        cutoff = np.asarray(cutoff, dtype=float)
        shifts = self.location_modes
        component_weight = 1.0 / shifts.size
        survival = np.zeros_like(cutoff)
        for shift in shifts:
            centred = cutoff - shift
            if self.location_sd > 0.0:
                survival += component_weight * norm.sf(centred / self.location_sd)
            else:
                survival += component_weight * (centred < 0.0)
        return survival

    def log_scale_quadrature(
        self, n_nodes: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Nodes and weights integrating the log-scale latent alone."""

        if self.log_scale_sd <= 0.0:
            return np.zeros(1), np.ones(1)
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
        weights = weights / weights.sum()
        return self.log_scale_sd * nodes, weights

    def quadrature(
        self, n_nodes: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Nodes and weights integrating against the outer law.

        A degenerate coordinate collapses to a single node, so a regime with no
        outer randomness costs one evaluation rather than `n_nodes` squared.
        """

        if n_nodes < 1:
            raise ValueError("n_nodes must be positive")
        raw_nodes, raw_weights = np.polynomial.hermite_e.hermegauss(n_nodes)
        raw_weights = raw_weights / np.sqrt(2.0 * np.pi)

        if self.location_sd > 0.0:
            location, location_weight = self.location_sd * raw_nodes, raw_weights
        else:
            location, location_weight = np.zeros(1), np.ones(1)
        if self.is_multimodal:
            location = np.concatenate(
                [location - self.mixture_shift, location + self.mixture_shift]
            )
            location_weight = np.concatenate(
                [0.5 * location_weight, 0.5 * location_weight]
            )
        if self.log_scale_sd > 0.0:
            log_scale, log_scale_weight = self.log_scale_sd * raw_nodes, raw_weights
        else:
            log_scale, log_scale_weight = np.zeros(1), np.ones(1)

        xi = np.repeat(location, log_scale.size)
        eta = np.tile(log_scale, location.size)
        weights = np.outer(location_weight, log_scale_weight).ravel()
        return xi, eta, weights / weights.sum()


Surface = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


@dataclass(frozen=True)
class DGPSpec:
    """Frozen description of one regime."""

    dgp_id: str
    description: str
    location: Surface
    log_scale: Surface
    shape: Surface
    outer: Callable[[int], OuterLaw]
    propensity: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    n_features: int = 5
    #: True when the two arm conditional laws coincide for every x.
    null_effect: bool = False


@dataclass(frozen=True)
class DGPSample:
    """One oracle dataset: covariates, assignment, and observed grid vectors."""

    X: NDArray[np.float64]
    treatment: NDArray[np.int64]
    quantiles: NDArray[np.float64]
    propensity: NDArray[np.float64]
    dgp_id: str
    seed: int

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def moderator(self) -> NDArray[np.int64]:
        return moderator_bins(self.X)


class DistributionalDGP:
    """A regime paired with a grid, exposing sampling and oracle truth."""

    def __init__(
        self,
        spec: DGPSpec,
        grid: GridSpec,
        *,
        n_quadrature_nodes: int = DEFAULT_QUADRATURE_NODES,
    ) -> None:
        self.spec = spec
        self.grid = grid
        self.n_quadrature_nodes = n_quadrature_nodes
        self._check_monotone_shape()

    def _check_monotone_shape(self) -> None:
        """psi(.; g) has minimum slope 1 - g, so 0 <= g <= MAX_SHAPE suffices."""

        rng = np.random.default_rng(0)
        probe = rng.uniform(-1.0, 1.0, size=(8192, self.spec.n_features))
        for arm in (0, 1):
            gamma = self.spec.shape(probe, arm)
            if np.min(gamma, initial=0.0) < 0.0 or np.max(gamma, initial=0.0) > MAX_SHAPE:
                raise ValueError(
                    f"{self.spec.dgp_id} arm {arm}: shape parameter must lie in "
                    f"[0, {MAX_SHAPE}] to keep every draw in the monotone cone"
                )

    # ---------------------------------------------------------------- sampling

    def sample(self, n_rows: int, seed: int) -> DGPSample:
        """Draw one oracle dataset. Deterministic given (regime, n, seed, K)."""

        if n_rows < 4:
            raise ValueError("n_rows must be at least 4")
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1.0, 1.0, size=(n_rows, self.spec.n_features))
        scores = self.spec.propensity(X)
        treatment = rng.binomial(1, scores).astype(np.int64)
        if treatment.sum() < 2 or (n_rows - treatment.sum()) < 2:
            treatment[:4] = np.array([0, 0, 1, 1], dtype=np.int64)

        quantiles = np.empty((n_rows, self.grid.n_grid))
        for arm in (0, 1):
            rows = np.flatnonzero(treatment == arm)
            if rows.size == 0:
                continue
            xi, eta = self.spec.outer(arm).sample(rng, rows.size)
            quantiles[rows] = self._grid_at_latent(X[rows], arm, xi, eta)
        return DGPSample(
            X=X,
            treatment=treatment,
            quantiles=quantiles,
            propensity=scores,
            dgp_id=self.spec.dgp_id,
            seed=seed,
        )

    def _grid_at_latent(
        self,
        X: NDArray[np.float64],
        arm: int,
        xi: NDArray[np.float64],
        eta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate q(Y^a) at row-specific outer latents, shape (n, K)."""

        location = self.spec.location(X, arm) + xi
        scale = np.exp(self.spec.log_scale(X, arm) + eta)
        gamma = self.spec.shape(X, arm)
        return location[:, None] + scale[:, None] * _psi(self.grid.base_z, gamma)

    # ------------------------------------------------------------------- truth

    def iter_law_nodes(
        self, X: NDArray[np.float64], arm: int
    ) -> Iterator[tuple[float, NDArray[np.float64]]]:
        """Yield (weight, grid vectors) for the discretized law of q(Y^a)|X.

        Iterating rather than materializing keeps peak memory at one (n, K)
        block: the full node tensor is (n, J, K) and reaches nearly a gigabyte
        at the tournament's largest cells.
        """

        xi_nodes, eta_nodes, weights = self.spec.outer(arm).quadrature(
            self.n_quadrature_nodes
        )
        n_rows = X.shape[0]
        for xi, eta, weight in zip(xi_nodes, eta_nodes, weights, strict=True):
            yield float(weight), self._grid_at_latent(
                X, arm, np.full(n_rows, xi), np.full(n_rows, eta)
            )

    def conditional_expectation(
        self,
        X: NDArray[np.float64],
        arm: int,
        statistic: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """E[statistic(q(Y^a)) | X=x] by quadrature over the outer law."""

        total: NDArray[np.float64] | None = None
        for weight, block in self.iter_law_nodes(X, arm):
            contribution = weight * np.asarray(statistic(block), dtype=float)
            total = contribution if total is None else total + contribution
        assert total is not None  # the rule always has at least one node
        return total

    def mean_quantiles(self, X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        """`MEANQ-A-K`: the conditional mean quantile vector, shape (n, K).

        In one dimension this is also the grid representation of `BARY-A`, but
        the two identifiers stay distinct in every result row.
        """

        return self.conditional_expectation(X, arm, lambda block: block)

    def mean_quantile_contrast(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """The grid causal mean tau_q^K(x), the mandatory common target."""

        return self.mean_quantiles(X, 1) - self.mean_quantiles(X, 0)

    def functional(
        self, X: NDArray[np.float64], arm: int, name: str
    ) -> NDArray[np.float64]:
        """E[h_j(q(Y^a)) | X=x] for a declared grid functional, shape (n,)."""

        if name not in GRID_FUNCTIONALS:
            raise ValueError(f"unknown grid functional {name!r}")
        h = GRID_FUNCTIONALS[name]
        w = self.grid.weights
        return self.conditional_expectation(X, arm, lambda block: h(block, w))

    def functional_contrast(
        self, X: NDArray[np.float64], name: str
    ) -> NDArray[np.float64]:
        """The pointwise ingredient of `TATE-K-j` and `TCATE-K-j`."""

        return self.functional(X, 1, name) - self.functional(X, 0, name)

    def reference_distance(
        self, X: NDArray[np.float64], arm: int
    ) -> NDArray[np.float64]:
        """`REF-A-K`: E[d_W(q(Y^a), q(nu_star)) | X=x], shape (n,)."""

        w = self.grid.weights
        reference = self.grid.reference_quantiles()

        def distance(block: NDArray[np.float64]) -> NDArray[np.float64]:
            difference = block - reference
            return np.sqrt(np.sum(w * difference * difference, axis=-1))

        return self.conditional_expectation(X, arm, distance)

    def reference_contrast(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """The pointwise ingredient of `REF-ATE-K` and `REF-TCATE-K`."""

        return self.reference_distance(X, 1) - self.reference_distance(X, 0)

    def tail_probability(
        self, X: NDArray[np.float64], arm: int, *, level_index: int, threshold: float
    ) -> NDArray[np.float64]:
        """`tail_calibration` truth: P{q_{level_index}(Y^a) > threshold | X=x}.

        Unlike the other targets this one is not a quadrature sum over both
        latents. The event is an indicator in the outer location, so the
        location is integrated in closed form and only the smooth log-scale
        latent goes through the quadrature rule. See `OuterLaw.location_survival`.
        """

        if not 0 <= level_index < self.grid.n_grid:
            raise ValueError("level_index is outside the declared grid")
        outer = self.spec.outer(arm)
        eta_nodes, eta_weights = outer.log_scale_quadrature(self.n_quadrature_nodes)
        location = self.spec.location(X, arm)
        scale = np.exp(self.spec.log_scale(X, arm))
        coordinate = _psi(self.grid.base_z, self.spec.shape(X, arm))[:, level_index]

        total = np.zeros(X.shape[0])
        for eta, weight in zip(eta_nodes, eta_weights, strict=True):
            cutoff = threshold - location - scale * np.exp(eta) * coordinate
            total += weight * outer.location_survival(cutoff)
        return total

    def location_modes(self, arm: int) -> NDArray[np.float64]:
        """Outer location modes, the reference set for `mode_coverage`."""

        return self.spec.outer(arm).location_modes

    def conditional_mode_centres(
        self, X: NDArray[np.float64], arm: int
    ) -> NDArray[np.float64]:
        """`mode_coverage` reference: one grid vector per mode, shape (n, R, K).

        Each centre is the conditional law evaluated at an outer location mode
        with the outer log-scale at its own mode, zero. A unimodal regime
        returns a single centre, so the metric degenerates gracefully.
        """

        modes = self.location_modes(arm)
        n_rows = X.shape[0]
        centres = np.empty((n_rows, modes.size, self.grid.n_grid))
        for index, mode in enumerate(modes):
            centres[:, index] = self._grid_at_latent(
                X, arm, np.full(n_rows, float(mode)), np.zeros(n_rows)
            )
        return centres


# --------------------------------------------------------------------- regimes


def _clipped_logistic(index: NDArray[np.float64], low: float, high: float):
    return np.clip(1.0 / (1.0 + np.exp(-index)), low, high)


def _mild_propensity(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return _clipped_logistic(0.8 * (X[:, 0] + 0.5 * X[:, 1]), 0.1, 0.9)


def _strong_propensity(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return _clipped_logistic(
        2.5 * (X[:, 0] + 0.7 * X[:, 1] - 0.5 * X[:, 2]), 0.05, 0.95
    )


def _deteriorating_propensity(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return _clipped_logistic(4.0 * (X[:, 0] + 0.5 * X[:, 1]), 0.01, 0.99)


def _zero(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    return np.zeros(X.shape[0])


def _smooth_prognostic(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.6 * np.sin(np.pi * X[:, 0]) + 0.4 * X[:, 1] * X[:, 2]


def _baseline_location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    return _smooth_prognostic(X) + arm * 0.8 * X[:, 0]


def _baseline_log_scale(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    return 0.20 * X[:, 3] + arm * 0.25 * X[:, 1]


def _build_specs() -> dict[str, DGPSpec]:
    quiet = OuterLaw()
    stochastic = OuterLaw(location_sd=0.35, log_scale_sd=0.20)

    def separate_location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        # Disjoint covariate supports and different shapes, so one shared
        # partition must carry the union of both arms' split structure.
        if arm == 0:
            return 0.9 * np.sin(np.pi * X[:, 0])
        return 0.9 * (X[:, 3] * X[:, 3] - 1.0 / 3.0) + 0.6 * X[:, 4]

    def separate_log_scale(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        return 0.30 * X[:, 1] if arm == 0 else -0.30 * X[:, 2]

    def shared_location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        # One moderator drives prognosis and effect in both arms, so a shared
        # partition spends its splits on the only axis that matters.
        return 1.0 * np.sin(np.pi * X[:, 0]) + arm * 0.9 * X[:, 0]

    def shared_log_scale(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        return 0.25 * X[:, 0] + arm * 0.30 * X[:, 0]

    # D5 matches the two arms' mean scales exactly: arm 0 carries outer
    # log-scale noise of standard deviation rho, contributing exp(rho^2 / 2) to
    # E[scale], and arm 1 carries none but adds rho^2 / 2 to its log scale.
    d5_rho = 0.45
    d5_log_scale_offset = 0.5 * d5_rho * d5_rho

    def equal_barycentre_log_scale(
        X: NDArray[np.float64], arm: int
    ) -> NDArray[np.float64]:
        base = 0.20 * X[:, 3]
        return base if arm == 0 else base + d5_log_scale_offset

    def equal_barycentre_outer(arm: int) -> OuterLaw:
        # Both outer location laws are centred at zero with different spreads,
        # so the arms share every mean but no higher law feature.
        if arm == 0:
            return OuterLaw(location_sd=0.40, log_scale_sd=d5_rho)
        return OuterLaw(location_sd=0.15, log_scale_sd=0.0)

    def transfer_shape(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        # The only treatment effect lives in the shape of the inner law. Both
        # arms share m_a and s_a, and psi is mean-free, so grid_mean carries
        # almost no signal while grid_skewness and grid_upper_tail_mean carry
        # all of it. Arm 0 stays in [0.10, 0.30] and arm 1 in [0.40, 0.70].
        return 0.20 + 0.10 * X[:, 2] + arm * (0.30 + 0.10 * X[:, 0])

    def weak_location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        strong_prognostic = 1.2 * np.sin(np.pi * X[:, 0]) + 0.9 * X[:, 1] * X[:, 2]
        return strong_prognostic + arm * (0.12 * X[:, 0] + 0.06)

    def weak_log_scale(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        return 0.20 * X[:, 3] + arm * 0.05 * X[:, 1]

    specs = [
        DGPSpec(
            dgp_id="D0",
            description="Deterministic conditional distributions",
            location=_baseline_location,
            log_scale=_baseline_log_scale,
            shape=_zero,
            outer=lambda arm: quiet,
            propensity=_mild_propensity,
        ),
        DGPSpec(
            dgp_id="D1",
            description="Smooth stochastic location-scale law",
            location=_baseline_location,
            log_scale=_baseline_log_scale,
            shape=_zero,
            outer=lambda arm: stochastic,
            propensity=_mild_propensity,
        ),
        DGPSpec(
            dgp_id="D2",
            description="Null treatment effect",
            location=lambda X, arm: _smooth_prognostic(X),
            log_scale=lambda X, arm: 0.20 * X[:, 3],
            shape=_zero,
            outer=lambda arm: stochastic,
            propensity=_mild_propensity,
            null_effect=True,
        ),
        DGPSpec(
            dgp_id="D3",
            description="Separate-head favorable",
            location=separate_location,
            log_scale=separate_log_scale,
            shape=_zero,
            outer=lambda arm: stochastic,
            propensity=_mild_propensity,
        ),
        DGPSpec(
            dgp_id="D4",
            description="Shared-structure favorable",
            location=shared_location,
            log_scale=shared_log_scale,
            shape=_zero,
            outer=lambda arm: stochastic,
            propensity=_mild_propensity,
        ),
        DGPSpec(
            dgp_id="D5",
            description="Equal barycenters, different random-measure laws",
            location=lambda X, arm: _smooth_prognostic(X),
            log_scale=equal_barycentre_log_scale,
            shape=_zero,
            outer=equal_barycentre_outer,
            propensity=_mild_propensity,
        ),
        DGPSpec(
            dgp_id="D6",
            description="Multimodal outer law",
            location=lambda X, arm: _smooth_prognostic(X) + arm * 0.7 * X[:, 0],
            log_scale=lambda X, arm: 0.20 * X[:, 3],
            shape=_zero,
            outer=lambda arm: OuterLaw(
                location_sd=0.25, log_scale_sd=0.15, mixture_shift=1.5
            ),
            propensity=_mild_propensity,
        ),
        DGPSpec(
            dgp_id="D7",
            description="Unseen functional transfer",
            location=lambda X, arm: _smooth_prognostic(X),
            log_scale=lambda X, arm: 0.20 * X[:, 3],
            shape=transfer_shape,
            outer=lambda arm: stochastic,
            propensity=_mild_propensity,
        ),
        DGPSpec(
            dgp_id="D8",
            description="Strong confounding, weak effect",
            location=weak_location,
            log_scale=weak_log_scale,
            shape=_zero,
            outer=lambda arm: stochastic,
            propensity=_strong_propensity,
        ),
        DGPSpec(
            dgp_id="D9",
            description="Overlap deterioration",
            location=lambda X, arm: _smooth_prognostic(X) + arm * 0.7 * X[:, 0],
            log_scale=_baseline_log_scale,
            shape=_zero,
            outer=lambda arm: stochastic,
            propensity=_deteriorating_propensity,
        ),
    ]
    return {spec.dgp_id: spec for spec in specs}


_SPECS = _build_specs()

#: Additional regimes registered after the frozen suite (Phase 5.5 imbalance
#: variants). They reuse the frozen outcome surfaces and differ only in the
#: propensity, so no oracle-truth code path changes.
_EXTRA_SPECS: dict[str, DGPSpec] = {}


def register_specs(specs: dict[str, DGPSpec]) -> None:
    """Register extra regimes under new identifiers without touching the frozen
    suite. ``build_dgp`` resolves the frozen registry first, so a Phase 5.5
    variant can never shadow a frozen regime."""

    for dgp_id, spec in specs.items():
        if dgp_id in _SPECS:
            raise ValueError(f"cannot register {dgp_id!r}: it is a frozen regime")
        _EXTRA_SPECS[dgp_id] = spec


def _imbalance_propensity(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Treatment probability pushed toward 0.8 on average.

    The outcome surfaces are untouched; only the assignment mechanism changes.
    The X-learner's rationale (one arm more informative than the other) can
    only appear where the arm sizes and overlap differ, which a balanced
    regime never contains.
    """
    return _clipped_logistic(
        2.5 * (X[:, 0] + 0.7 * X[:, 1] - 0.6 * X[:, 2] - 0.55), 0.05, 0.95
    )


def build_imbalance_specs() -> dict[str, DGPSpec]:
    """The Phase 5.5 imbalance stress regimes: D2-imb, D7-imb, D8-imb.

    Each keeps its frozen outcome surfaces and replaces only the propensity.
    """

    frozen = _build_specs()
    specs: dict[str, DGPSpec] = {}
    for base_id in ("D2", "D7", "D8"):
        base = frozen[base_id]
        specs[f"{base_id}-imb"] = DGPSpec(
            dgp_id=f"{base_id}-imb",
            description=f"{base_id} with an imbalanced propensity",
            location=base.location,
            log_scale=base.log_scale,
            shape=base.shape,
            outer=base.outer,
            propensity=_imbalance_propensity,
            n_features=base.n_features,
            null_effect=base.null_effect,
        )
    return specs


register_specs(build_imbalance_specs())


#: Builders for regimes whose grid object is not the plain ``GridSpec``: a
#: track may need its own reference law or weights while keeping every other
#: piece of the oracle machinery. Registered beside the extra specs so that a
#: later phase can never shadow a frozen regime.
_EXTRA_BUILDERS: dict[str, Callable[[int], "DistributionalDGP"]] = {}


def register_builders(builders: dict[str, Callable[[int], "DistributionalDGP"]]) -> None:
    """Register whole-DGP builders under new identifiers."""

    for dgp_id, builder in builders.items():
        if dgp_id in _SPECS or dgp_id in _EXTRA_BUILDERS:
            raise ValueError(f"cannot register builder for {dgp_id!r}: already present")
        _EXTRA_BUILDERS[dgp_id] = builder


def build_dgp(
    dgp_id: str,
    n_grid: int,
    *,
    n_quadrature_nodes: int = DEFAULT_QUADRATURE_NODES,
) -> DistributionalDGP:
    """Instantiate one regime on the declared grid."""

    if dgp_id in _SPECS:
        return DistributionalDGP(
            _SPECS[dgp_id], GridSpec(n_grid), n_quadrature_nodes=n_quadrature_nodes
        )
    spec = _EXTRA_SPECS.get(dgp_id)
    builder = _EXTRA_BUILDERS.get(dgp_id)
    if spec is None and builder is None:
        raise ValueError(f"unknown DGP {dgp_id!r}; expected one of {DGP_IDS}")
    if builder is not None:
        return builder(n_grid)
    return DistributionalDGP(
        spec, GridSpec(n_grid), n_quadrature_nodes=n_quadrature_nodes
    )
