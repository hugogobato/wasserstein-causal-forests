"""Phase 6 realism track: income-distribution regimes IC0 through IC3.

The applied study under discussion observes, for each state and year, the
distribution of household income, and asks how a policy moved that
distribution toward an explicit benchmark economy. No regime in the frozen
suite looks like that: every one of them is symmetric-tailed at moderate shape,
references the standard normal, and treats adoption as unrelated to the
economy it later acts on. The four regimes here close that gap while keeping
the generative form, so the entire quadrature oracle machinery applies
unchanged:

    q(Y^a)_k = m_a(x) + xi + exp{s_a(x) + eta} * psi(z_k; gamma_a(x)).

What changes is only ever the surfaces, the outer law, the propensity, the
reference vector, and the covariate count. The inner shape parameters now sit
in the upper part of the admissible range [0, 0.85], which is what produces a
right-skewed income-like quantile contour; psi's minimum slope is 1 - gamma,
so monotonicity is preserved exactly as in the frozen suite.

The reference law nu_star is a benchmark economy: a fixed quantile vector with
a higher location, a wider scale, and much less skewness than any regime arm.
`IncomeGridSpec` supplies it; everything downstream (law metrics, PTA target
manifests, forest drivers) reads the reference from the grid object, so no
other code path knows this track exists.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .dgps import (
    DistributionalDGP,
    DGPSpec,
    GridSpec,
    OuterLaw,
    _psi,
    register_builders,
    register_specs,
)

PHASE6_CONTRACT_ID = "G3-PHASE6-v1"

#: Benchmark-economy reference: higher median, wider scale, mild skew. Frozen
#: before any Phase 6 cell ran; it is the applied study's "Sweden", not a
#: quantity any method sees during fitting.
REFERENCE_LOCATION = 1.10
REFERENCE_LOG_SCALE = 0.60
REFERENCE_SHAPE = 0.30


class IncomeGridSpec(GridSpec):
    """The income track's grid: same levels and weights, benchmark reference."""

    def reference_quantiles(self) -> NDArray[np.float64]:
        z = self.base_z
        hermite = (z * z - 1.0) / 2.0 + (z * z * z - 3.0 * z) / 6.0
        return REFERENCE_LOCATION + REFERENCE_LOG_SCALE * (
            z + REFERENCE_SHAPE * hermite
        )


def _income_prognostic(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Median log-income proxy: education and urbanisation raise it,
    non-employment and weak growth lower it."""

    return (
        0.55 * X[:, 1]
        + 0.35 * X[:, 2]
        - 0.45 * X[:, 3]
        + 0.30 * X[:, 4]
    )


def _income_log_scale(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    return 0.16 + 0.08 * X[:, 5] - 0.04 * X[:, 1]


def _income_shape(X: NDArray[np.float64], shift: NDArray[np.float64]):
    """Baseline inequality surface minus an arm-specific compression."""

    base = 0.55 + 0.10 * X[:, 3] - 0.08 * X[:, 1]
    return np.clip(base - shift, 0.05, 0.85)


def _eitc_location_shift(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """EITC-like: a credit that is larger where non-employment is high."""
    return 0.20 + 0.15 * X[:, 3]


def _floor_location_shift(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimum-wage-like: a small uniform floor."""
    return 0.10 + 0.05 * X[:, 2]


def _adoption_mild(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Endogenous adoption: states with weak growth and high non-employment
    are more likely to adopt, which confounds both surfaces."""
    index = 1.4 * X[:, 3] - 1.1 * X[:, 4] + 0.4 * X[:, 5]
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.10, 0.90)


def _adoption_moderate(X: NDArray[np.float64]) -> NDArray[np.float64]:
    index = 1.8 * X[:, 3] - 0.9 * X[:, 4]
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.05, 0.95)


def _adoption_deteriorating(X: NDArray[np.float64]) -> NDArray[np.float64]:
    index = 3.5 * (X[:, 3] - 0.9 * X[:, 4])
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.01, 0.99)


_OUTER = OuterLaw(location_sd=0.20, log_scale_sd=0.12)
_QUIET = OuterLaw(location_sd=0.20, log_scale_sd=0.12)


def build_income_specs() -> dict[str, DGPSpec]:
    """The four IC regimes. Covariate layout:
    x0 poverty moderator, x1 education, x2 urbanisation, x3 non-employment,
    x4 growth, x5 manufacturing share."""

    def ic0_location(X, arm):
        return _income_prognostic(X)

    def ic1_location(X, arm):
        return _income_prognostic(X) + arm * _eitc_location_shift(X)

    def ic2_location(X, arm):
        return _income_prognostic(X) + arm * _floor_location_shift(X)

    def ic0_shape(X, arm):
        return _income_shape(X, np.zeros(X.shape[0]))

    def ic1_shape(X, arm):
        return _income_shape(X, arm * (0.16 + 0.06 * X[:, 3]))

    def ic2_shape(X, arm):
        return _income_shape(X, arm * 0.08 * np.ones(X.shape[0]))

    def ic2_log_scale(X, arm):
        # A wage floor compresses the whole quantile contour toward its
        # centre: shrinking the log-scale raises the bottom (where psi is
        # negative) and lowers the top, which a shape compression alone
        # cannot do - psi's Hermite reshaping fattens both tails together.
        return _income_log_scale(X, arm) - arm * 0.12

    specs = {
        "IC0": DGPSpec(
            dgp_id="IC0",
            description="Income baseline: endogenous-free placebo, no policy effect",
            location=ic0_location,
            log_scale=_income_log_scale,
            shape=ic0_shape,
            outer=lambda arm: _QUIET,
            propensity=_adoption_mild,
            n_features=6,
            null_effect=True,
        ),
        "IC1": DGPSpec(
            dgp_id="IC1",
            description="EITC-like credit under mildly endogenous adoption",
            location=ic1_location,
            log_scale=_income_log_scale,
            shape=ic1_shape,
            outer=lambda arm: _OUTER,
            propensity=_adoption_mild,
            n_features=6,
        ),
        "IC2": DGPSpec(
            dgp_id="IC2",
            description="Wage-floor policy under moderately endogenous adoption",
            location=ic2_location,
            log_scale=ic2_log_scale,
            shape=ic2_shape,
            outer=lambda arm: _OUTER,
            propensity=_adoption_moderate,
            n_features=6,
        ),
        "IC3": DGPSpec(
            dgp_id="IC3",
            description="IC1 outcomes under deteriorating overlap",
            location=ic1_location,
            log_scale=_income_log_scale,
            shape=ic1_shape,
            outer=lambda arm: _OUTER,
            propensity=_adoption_deteriorating,
            n_features=6,
        ),
    }
    return specs


INCOME_DGPS: tuple[str, ...] = ("IC0", "IC1", "IC2", "IC3")


def register_phase6_dgps() -> None:
    """Register the IC regimes exactly once per process."""

    specs = build_income_specs()
    try:
        register_specs(specs)
    except ValueError:
        return
    register_builders(
        {
            name: (
                lambda k, s=name: DistributionalDGP(
                    specs[s], IncomeGridSpec(k)
                )
            )
            for name in INCOME_DGPS
        }
    )
