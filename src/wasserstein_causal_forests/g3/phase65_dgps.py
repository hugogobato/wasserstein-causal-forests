"""Phase 6.5 regimes: realism ingredient ablations and zero inflation.

Track D isolates one realism ingredient at a time. Each DA regime is the IC1
surface family with exactly one switch off, so a paired difference against the
frozen IC1 rows attributes the Phase 6 Track B reversal to an ingredient rather
than to "realism" wholesale:

* ``DAskew``  inner Hermite shape parameters at zero (symmetric inner law);
* ``DArand``  propensities constant at one half (assignment exogenous);
* ``DAunit``  every outcome surface divided by the frozen IC1 population scale,
  which makes the whole problem unit-variance without touching shape;
* ``DAref``   the standard-normal reference instead of the benchmark economy;
* ``DAdim``   covariates collapsed to the two dimensions that carry the
  surfaces, with the frozen coefficient layout remapped onto them.

Track E registers four zero-inflated regimes. The unit of this project is a
panel observation whose outcome IS a distribution, and zero inflation enters at
that level: with probability ``1 - p_a(x)`` the observation's entire outcome law
is degenerate at zero (a county where nobody buys insurance, a region with no
waiting list), and otherwise it is a strictly positive income-like
distribution. The conditional law of ``q(Y^a) | X = x`` is therefore a mixture
over law space,

    P_a^K(x) = (1 - p_a(x)) * delta_{0_K}
               + p_a(x) * Law^+_a(x),

with component weights that depend on x. That single structural fact is why
this module exists: every downstream metric contract assumes truth node weights
shared across rows, and the mixture breaks it, so `law_node_weights` supplies
an ``(n, J)`` weight matrix and the metric layer contracts accordingly. The
frozen suite returns None there and its code paths are byte-for-byte unchanged.

The generative form of the positive part is the project's Hermite form shifted
to insurance-like levels, and its surfaces are chosen so the positive branch is
strictly greater than zero over the whole covariate cube and latent range,
which is verified at construction time: a pinned lower tail followed by negative
"positive" coordinates would put the point mass inside the support of the other
component and make every zero-mass metric meaningless.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dgps import (
    DEFAULT_QUADRATURE_NODES,
    DistributionalDGP,
    DGPSpec,
    OuterLaw,
    register_builders,
    register_specs,
)
from .phase6_dgps import (
    IncomeGridSpec,
    _adoption_deteriorating,
    _income_log_scale,
    _income_prognostic,
)

PHASE65_CONTRACT_ID = "G3-PHASE65-v1"

#: Frozen divisor for the unit-scale ablation: the pooled population standard
#: deviation of IC1 outcomes, measured on a 30,000-row draw at seed 12345
#: before any decisive cell ran. Dividing by it turns DAunit into an exact
#: unit-variance rendering of IC1.
UNIT_SCALE_DIVISOR = 1.274

ABLATED_DGPS: tuple[str, ...] = (
    "DAskew", "DArand", "DAunit", "DAref", "DAdim",
)
ZERO_INFLATED_DGPS: tuple[str, ...] = ("ZI0", "ZI1", "ZI2", "ZI3")


# --------------------------------------------------------------------- Track D


def _scaled_location(
    divide: float,
):
    def location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        return _ic1_location(X, arm) / divide

    return location


def _ic1_location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    """IC1's location surface: prognostic plus the EITC-like credit."""

    return _income_prognostic(X) + arm * (0.20 + 0.15 * X[:, 3])


def _ic1_shape(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    base = 0.55 + 0.10 * X[:, 3] - 0.08 * X[:, 1]
    shift = arm * (0.16 + 0.06 * X[:, 3])
    return np.clip(base - shift, 0.05, 0.85)


def build_ablated_specs() -> dict[str, DGPSpec]:
    """Five one-switch-off renderings of IC1."""

    log_divided = np.log(UNIT_SCALE_DIVISOR)
    quiet_outer = OuterLaw(location_sd=0.20 / UNIT_SCALE_DIVISOR,
                           log_scale_sd=0.12)

    def symmetric_shape(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        return np.zeros(X.shape[0])

    def half_propensity(X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(X.shape[0], 0.5)

    def scaled_log_scale(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        return _income_log_scale(X, arm) - log_divided

    # Two-covariate remap of the six-role layout: poverty/moderator stays X_0,
    # education takes X_1, and urbanisation, non-employment, growth, and
    # manufacturing fold onto the pair with their frozen signs.
    def dim_location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        prognostic = 0.55 * X[:, 1] + 0.35 * X[:, 0] - 0.45 * X[:, 1] \
            + 0.30 * X[:, 0]
        return prognostic + arm * (0.20 + 0.15 * X[:, 1])

    def dim_log_scale(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        return 0.16 + 0.08 * X[:, 1] - 0.04 * X[:, 0]

    def dim_shape(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        base = 0.55 + 0.10 * X[:, 1] - 0.08 * X[:, 0]
        return np.clip(base - arm * (0.16 + 0.06 * X[:, 1]), 0.05, 0.85)

    def dim_adoption(X: NDArray[np.float64]) -> NDArray[np.float64]:
        index = 1.4 * X[:, 1] - 1.1 * X[:, 0]
        return np.clip(1.0 / (1.0 + np.exp(-index)), 0.10, 0.90)

    return {
        "DAskew": DGPSpec(
            dgp_id="DAskew",
            description="IC1 with the inner Hermite shape switched off",
            location=_ic1_location,
            log_scale=_income_log_scale,
            shape=symmetric_shape,
            outer=lambda arm: OuterLaw(location_sd=0.20, log_scale_sd=0.12),
            propensity=_adoption_mild65,
            n_features=6,
        ),
        "DArand": DGPSpec(
            dgp_id="DArand",
            description="IC1 with endogenous adoption switched off",
            location=_ic1_location,
            log_scale=_income_log_scale,
            shape=_ic1_shape,
            outer=lambda arm: OuterLaw(location_sd=0.20, log_scale_sd=0.12),
            propensity=half_propensity,
            n_features=6,
        ),
        "DAunit": DGPSpec(
            dgp_id="DAunit",
            description=(
                "IC1 rescaled to unit variance by the frozen divisor "
                f"{UNIT_SCALE_DIVISOR}"
            ),
            location=_scaled_location(UNIT_SCALE_DIVISOR),
            log_scale=scaled_log_scale,
            shape=_ic1_shape,
            outer=lambda arm: quiet_outer,
            propensity=_adoption_mild65,
            n_features=6,
        ),
        "DAref": DGPSpec(
            dgp_id="DAref",
            description=(
                "IC1 outcomes with the standard-normal reference law; the "
                "benchmark-economy reference is the only switched-off piece"
            ),
            location=_ic1_location,
            log_scale=_income_log_scale,
            shape=_ic1_shape,
            outer=lambda arm: OuterLaw(location_sd=0.20, log_scale_sd=0.12),
            propensity=_adoption_mild65,
            n_features=6,
        ),
        "DAdim": DGPSpec(
            dgp_id="DAdim",
            description="IC1 surfaces remapped onto two covariates",
            location=dim_location,
            log_scale=dim_log_scale,
            shape=dim_shape,
            outer=lambda arm: OuterLaw(location_sd=0.20, log_scale_sd=0.12),
            propensity=dim_adoption,
            n_features=2,
        ),
    }


def _adoption_mild65(X: NDArray[np.float64]) -> NDArray[np.float64]:
    index = 1.4 * X[:, 3] - 1.1 * X[:, 4] + 0.4 * X[:, 5]
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.10, 0.90)


# --------------------------------------------------------------------- Track E


def _zi_positive_location(shift: float, slope: float):
    def location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        return (
            # The base clears the worst corner of the cube and the 12-node
            # quadrature range (|node| up to about 5.5 standard deviations):
            # with the frozen log-scale and shape surfaces the minimum
            # positive-part coordinate stays above roughly 0.45.
            4.80
            + 0.35 * X[:, 1]
            - 0.25 * X[:, 3]
            + arm * (shift + slope * X[:, 1])
        )

    return location


def _zi_positive_log_scale(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    return 0.10 + 0.06 * X[:, 5] - 0.03 * X[:, 1]


def _zi_positive_shape(compression: float):
    def shape(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
        base = 0.50 + 0.06 * X[:, 3]
        return np.clip(base - arm * compression, 0.05, 0.85)

    return shape


def _participation_base(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Baseline share of positive-component units, driven by non-employment."""

    index = 1.2 * X[:, 3] - 0.8 * X[:, 1]
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.05, 0.95)


def _participation_policy(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Treated-arm participation: a credit that moves units off the spike."""

    index = 1.2 * X[:, 3] - 0.8 * X[:, 1] + 0.90 + 0.20 * X[:, 3]
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.05, 0.95)


def _participation_deteriorating(X: NDArray[np.float64]) -> NDArray[np.float64]:
    index = 3.0 * (X[:, 3] - 0.9 * X[:, 4])
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.01, 0.99)


class ZeroInflatedDGP(DistributionalDGP):
    """A regime whose conditional law mixes a point mass on the zero law.

    ``participation(x, arm)`` returns ``P(positive component | X = x, arm)``,
    the probability that the unit's whole outcome distribution is non-degenerate.
    Sampling, oracle truth, and every derived target follow that mixture
    exactly; the row-dependent component weights are exposed through
    `law_node_weights` so the metric layer stays generic.
    """

    def __init__(
        self,
        spec: DGPSpec,
        grid: IncomeGridSpec,
        *,
        participation: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
        n_quadrature_nodes: int = DEFAULT_QUADRATURE_NODES,
    ) -> None:
        super().__init__(spec, grid, n_quadrature_nodes=n_quadrature_nodes)
        self.participation = participation
        self._check_positive_support()

    def _check_positive_support(self) -> None:
        """The positive branch must clear zero over the cube and latent range.

        A negative "positive-part" coordinate would overlap the spike's
        neighbourhood and turn every zero-mass statement into an artefact of
        the surface parameterisation, which is a construction defect rather
        than a result.
        """

        rng = np.random.default_rng(0)
        probe = rng.uniform(-1.0, 1.0, size=(4096, self.spec.n_features))
        minimum = np.inf
        for arm in (0, 1):
            xi_nodes, eta_nodes, _ = self.spec.outer(arm).quadrature(
                self.n_quadrature_nodes
            )
            for xi in (float(np.min(xi_nodes)), float(np.max(xi_nodes))):
                for eta in (float(np.min(eta_nodes)), float(np.max(eta_nodes))):
                    block = self._grid_at_latent(
                        probe, arm,
                        np.full(probe.shape[0], xi),
                        np.full(probe.shape[0], eta),
                    )
                    minimum = min(minimum, float(block.min()))
        if minimum <= 1e-6:
            raise ValueError(
                f"{self.spec.dgp_id}: positive-part minimum {minimum:.4f} "
                "crosses zero; raise the location surface"
            )

    # ---------------------------------------------------------------- sampling

    def sample(self, n_rows: int, seed: int):
        from .dgps import DGPSample

        rng = np.random.default_rng(seed)
        X = rng.uniform(-1.0, 1.0, size=(n_rows, self.spec.n_features))
        scores = self.spec.propensity(X)
        treatment = rng.binomial(1, scores).astype(np.int64)
        if treatment.sum() < 2 or (n_rows - treatment.sum()) < 2:
            treatment[:4] = np.array([0, 0, 1, 1], dtype=np.int64)

        quantiles = np.zeros((n_rows, self.grid.n_grid))
        for arm in (0, 1):
            rows = np.flatnonzero(treatment == arm)
            if rows.size == 0:
                continue
            positive = rng.binomial(1, self.participation(X[rows], arm))
            drawn = rows[positive == 1]
            if drawn.size == 0:
                continue
            xi, eta = self.spec.outer(arm).sample(rng, drawn.size)
            quantiles[drawn] = self._grid_at_latent(X[drawn], arm, xi, eta)
        return DGPSample(
            X=X,
            treatment=treatment,
            quantiles=quantiles,
            propensity=scores,
            dgp_id=self.spec.dgp_id,
            seed=seed,
        )

    # ------------------------------------------------------------------- truth

    def iter_law_nodes(self, X, arm):
        """Blocks in the order `(zero block, positive quadrature blocks...)`.

        The scalar weights yielded here are column means of the row-dependent
        matrix and are only meaningful alongside `law_node_weights`; every
        truth consumer in the evaluation layer reads the matrix instead.
        """

        weights_matrix = self.law_node_weights(X, arm)
        n_rows = X.shape[0]
        yield float(np.mean(weights_matrix[:, 0])), np.zeros((n_rows,
                                                              self.grid.n_grid))
        xi_nodes, eta_nodes, shared = self.spec.outer(arm).quadrature(
            self.n_quadrature_nodes
        )
        for index, (xi, eta) in enumerate(
            zip(xi_nodes, eta_nodes, strict=True)
        ):
            yield float(np.mean(weights_matrix[:, index + 1])), (
                self._grid_at_latent(
                    X, arm,
                    np.full(n_rows, xi), np.full(n_rows, eta),
                )
            )

    def law_node_weights(self, X, arm):
        positive = self.participation(X, arm)
        _, _, shared = self.spec.outer(arm).quadrature(self.n_quadrature_nodes)
        return np.column_stack([1.0 - positive, positive[:, None] * shared[None, :]])

    def conditional_expectation(self, X, arm, statistic):
        total: NDArray[np.float64] | None = None
        row_weights = self.law_node_weights(X, arm)
        for index, (_, block) in enumerate(self.iter_law_nodes(X, arm)):
            contribution = row_weights[:, [index]] * np.asarray(
                statistic(block), dtype=float
            )
            total = contribution if total is None else total + contribution
        assert total is not None
        return total

    def tail_probability(self, X, arm, *, level_index, threshold):
        positive = super().tail_probability(
            X, arm, level_index=level_index, threshold=threshold
        )
        return self.participation(X, arm) * positive

    def zero_type_probability(self, X, arm):
        return 1.0 - self.participation(X, arm)


def build_zero_inflated_specs() -> dict[str, DGPSpec]:
    """Four two-part regimes: placebo, participation effect, intensity effect,
    and both effects under deteriorating overlap."""

    return {
        "ZI0": DGPSpec(
            dgp_id="ZI0",
            description="Zero-inflated placebo: no policy effect anywhere",
            location=_zi_positive_location(0.0, 0.0),
            log_scale=_zi_positive_log_scale,
            shape=_zi_positive_shape(0.0),
            outer=lambda arm: OuterLaw(location_sd=0.15, log_scale_sd=0.10),
            propensity=_adoption_mild65,
            n_features=6,
            null_effect=True,
        ),
        "ZI1": DGPSpec(
            dgp_id="ZI1",
            description="Participation effect only: the policy moves the spike",
            location=_zi_positive_location(0.0, 0.0),
            log_scale=_zi_positive_log_scale,
            shape=_zi_positive_shape(0.0),
            outer=lambda arm: OuterLaw(location_sd=0.15, log_scale_sd=0.10),
            propensity=_adoption_mild65,
            n_features=6,
        ),
        "ZI2": DGPSpec(
            dgp_id="ZI2",
            description=("Intensity effect only: the policy shifts the "
                         "positive part and compresses its shape"),
            location=_zi_positive_location(0.28, 0.08),
            log_scale=_zi_positive_log_scale,
            shape=_zi_positive_shape(0.10),
            outer=lambda arm: OuterLaw(location_sd=0.15, log_scale_sd=0.10),
            propensity=_adoption_mild65,
            n_features=6,
        ),
        "ZI3": DGPSpec(
            dgp_id="ZI3",
            description="Both effects under deteriorating overlap",
            location=_zi_positive_location(0.28, 0.08),
            log_scale=_zi_positive_log_scale,
            shape=_zi_positive_shape(0.10),
            outer=lambda arm: OuterLaw(location_sd=0.15, log_scale_sd=0.10),
            propensity=_participation_deteriorating,
            n_features=6,
        ),
    }


_ZI_PARTICIPATIONS: dict[str, Callable[[NDArray[np.float64], int],
                                       NDArray[np.float64]]] = {
    "ZI0": lambda X, arm: _participation_base(X),
    "ZI1": lambda X, arm: (_participation_base(X) if arm == 0
                           else _participation_policy(X)),
    "ZI2": lambda X, arm: _participation_base(X),
    "ZI3": lambda X, arm: (_participation_deteriorating(X) if arm == 0
                           else _zi3_treated_participation(X)),
}


def _zi3_treated_participation(X: NDArray[np.float64]) -> NDArray[np.float64]:
    index = 3.0 * (X[:, 3] - 0.9 * X[:, 4]) + 0.90 + 0.60 * X[:, 3]
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.01, 0.99)


class ZeroInflatedGridSpec(IncomeGridSpec):
    """The ZI grids share the benchmark-economy reference of the income track."""


_PHASE65_SPECS: dict[str, DGPSpec] = {}


def register_phase65_dgps() -> None:
    """Register the DA and ZI regimes exactly once per process."""

    if _PHASE65_SPECS:
        return
    _PHASE65_SPECS.update(build_ablated_specs())
    _PHASE65_SPECS.update(build_zero_inflated_specs())
    try:
        register_specs(_PHASE65_SPECS)
    except ValueError:
        return
    builders: dict[str, Callable[[int], DistributionalDGP]] = {}
    for name in ABLATED_DGPS:
        grid_type: Any = IncomeGridSpec
        if name == "DAref":
            # The reference law is the switched-off ingredient, so this regime
            # uses the plain standard-normal grid object.
            from .dgps import GridSpec

            grid_type = GridSpec
        builders[name] = (
            lambda k, s=name, g=grid_type: DistributionalDGP(
                _PHASE65_SPECS[s], g(k)
            )
        )
    for name in ZERO_INFLATED_DGPS:
        builders[name] = (
            lambda k, s=name: ZeroInflatedDGP(
                _PHASE65_SPECS[s], ZeroInflatedGridSpec(k),
                participation=_ZI_PARTICIPATIONS[s],
            )
        )
    register_builders(builders)
