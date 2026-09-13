"""Symmetric inner-law regimes with progressively nonlinear assignment.

The regimes in this module share one outcome law and differ only in the
assignment mechanism, so a score difference across them isolates how the
propensity's relation to prognosis affects an estimator. The inner law is
symmetric by construction: the shape coefficient is zero and the outer
log-scale shock is switched off, leaving only a location shock,

    q_k(Y^a) = mu(x) + a tau(x) + xi + exp{s(x)} z_k,   xi ~ N(0, 0.25^2),

with six independent U[-1, 1] covariates. `SYM-NL` and `SYM-MU` make the
assignment a nonlinear function of the same surfaces that drive the outcome;
`SYM-ALIGN` and `SYM-IRREL` push a nonlinear index through x1 or through x6, a
coordinate that enters no outcome surface, so their propensity marginals agree
while their relation to prognosis does not. Every regime has a `<ID>-NULL`
companion with tau(x) = 0.

See `report/sensitivity_and_dgp_experiments.md`, sections "Symmetric and
nonlinear assignment regimes" and "Isolate assignment alignment". The design
checks live in `research/checks/wcf_sensitivity_design_checks.py`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .dgps import (
    DGPSpec,
    DistributionalDGP,
    GridSpec,
    OuterLaw,
    register_builders,
    register_specs,
)


def outcome_mu(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Baseline location surface mu(x), shared by every regime in the family."""

    return (
        0.70 * np.sin(np.pi * X[:, 0])
        + 0.35 * X[:, 1] * X[:, 2]
        + 0.20 * np.cos(np.pi * X[:, 3])
    )


def outcome_tau(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Treatment-effect surface tau(x); x6 is absent from every surface."""

    return 0.25 + 0.15 * np.sin(np.pi * X[:, 1])


def outcome_log_scale(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Deterministic log-scale surface s(x), identical in both arms."""

    return 0.12 + 0.08 * X[:, 4] ** 2 - 0.05 * X[:, 2]


def outcome_location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    """location(x, arm) = mu(x) + arm * tau(x)."""

    return outcome_mu(X) + arm * outcome_tau(X)


def _null_location(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    return outcome_mu(X)


def _log_scale_surface(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    return outcome_log_scale(X)


def _symmetric_shape(X: NDArray[np.float64], arm: int) -> NDArray[np.float64]:
    return np.zeros(X.shape[0])


def _clip_logistic(index: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(1.0 / (1.0 + np.exp(-index)), 0.05, 0.95)


def _assign_random(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.full(X.shape[0], 0.5)


def _assign_random03(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.full(X.shape[0], 0.3)


def _assign_linear(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return _clip_logistic(1.4 * X[:, 0] - 1.1 * X[:, 1] + 0.4 * X[:, 2])


def _assign_nonlinear(X: NDArray[np.float64]) -> NDArray[np.float64]:
    index = (
        1.6 * np.sin(np.pi * X[:, 0])
        + 0.8 * X[:, 1] * X[:, 2]
        - 0.7 * X[:, 3] ** 2
        + 0.35 * X[:, 4]
    )
    return _clip_logistic(index)


def _assign_mu(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return _clip_logistic(1.2 * outcome_mu(X) + 0.5 * X[:, 4])


def _assign_aligned(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return _clip_logistic(1.6 * np.sin(np.pi * X[:, 0]))


def _assign_irrelevant(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return _clip_logistic(1.6 * np.sin(np.pi * X[:, 5]))


_OUTER = OuterLaw(location_sd=0.25, log_scale_sd=0.0)

#: Regimes compared against the baselines in the sensitivity study.
SENSITIVITY_PRIMARY_DGPS: tuple[str, ...] = (
    "SYM-RANDOM",
    "SYM-LIN",
    "SYM-NL",
    "SYM-MU",
)

#: Same outcome surfaces and nonlinear index, aligned through x1 or through x6.
SENSITIVITY_ALIGNMENT_DGPS: tuple[str, ...] = ("SYM-ALIGN", "SYM-IRREL")

#: Constant-propensity controls: unequal allocation without confounding.
SENSITIVITY_CONTROL_DGPS: tuple[str, ...] = ("SYM-RANDOM", "SYM-RANDOM03")

SENSITIVITY_BASE_DGPS: tuple[str, ...] = (
    "SYM-RANDOM",
    "SYM-RANDOM03",
    "SYM-LIN",
    "SYM-NL",
    "SYM-MU",
    "SYM-ALIGN",
    "SYM-IRREL",
)

#: One tau(x) = 0 placebo companion per base regime.
SENSITIVITY_NULL_DGPS: tuple[str, ...] = tuple(
    f"{dgp_id}-NULL" for dgp_id in SENSITIVITY_BASE_DGPS
)

SENSITIVITY_DGPS: tuple[str, ...] = SENSITIVITY_BASE_DGPS + SENSITIVITY_NULL_DGPS


def build_sensitivity_specs() -> dict[str, DGPSpec]:
    """One spec per assignment mechanism plus its null companion."""

    mechanisms = (
        ("SYM-RANDOM", "constant propensity e(x)=0.5", _assign_random),
        ("SYM-RANDOM03", "constant propensity e(x)=0.3", _assign_random03),
        ("SYM-LIN", "linear logistic propensity", _assign_linear),
        ("SYM-NL", "nonlinear logistic propensity", _assign_nonlinear),
        (
            "SYM-MU",
            "propensity driven by the baseline outcome surface mu(x)",
            _assign_mu,
        ),
        (
            "SYM-ALIGN",
            "nonlinear index aligned with prognosis through x1",
            _assign_aligned,
        ),
        (
            "SYM-IRREL",
            "nonlinear index through x6, absent from every outcome surface",
            _assign_irrelevant,
        ),
    )
    specs: dict[str, DGPSpec] = {}
    for base_id, description, propensity in mechanisms:
        specs[base_id] = DGPSpec(
            dgp_id=base_id,
            description=f"Symmetric inner law, {description}",
            location=outcome_location,
            log_scale=_log_scale_surface,
            shape=_symmetric_shape,
            outer=lambda arm: _OUTER,
            propensity=propensity,
            n_features=6,
        )
        null_id = f"{base_id}-NULL"
        specs[null_id] = DGPSpec(
            dgp_id=null_id,
            description=f"Symmetric null effect, {description}",
            location=_null_location,
            log_scale=_log_scale_surface,
            shape=_symmetric_shape,
            outer=lambda arm: _OUTER,
            propensity=propensity,
            n_features=6,
            null_effect=True,
        )
    return specs


_SENSITIVITY_REGISTERED = False


def register_sensitivity_dgps() -> None:
    """Register the SYM specs and builders exactly once per process."""

    global _SENSITIVITY_REGISTERED
    if _SENSITIVITY_REGISTERED:
        return
    specs = build_sensitivity_specs()
    try:
        register_specs(specs)
    except ValueError:
        _SENSITIVITY_REGISTERED = True
        return
    register_builders(
        {
            name: (lambda k, s=name: DistributionalDGP(specs[s], GridSpec(k)))
            for name in specs
        }
    )
    _SENSITIVITY_REGISTERED = True
