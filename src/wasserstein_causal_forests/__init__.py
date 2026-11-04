"""Wasserstein Causal Forests (WCF) package.

Provides Causal Wasserstein Distribution Boosting (CWDB), cross-fitted shrinkage
selection, doubly robust (AIPW) functional calibration, two-part zero-inflated
distributional estimation, and the applied-study adapter.
"""

from .applied import (
    AppliedDataset,
    aggregate_replications,
    ecdf_moderator,
    energy_score_on,
    grid_weights,
    make_functionals,
    midpoint_levels,
    plugin_contrasts,
    quantile_matrix,
    run_placebo,
    run_wcf,
    run_wcf_replications,
    weighted_quantiles,
)
from .cwdb import (
    ArmParticleBooster,
    ArmSharedTreeRegressor,
    CrossFittedCWDBRegressor,
    CWDBRegressor,
    DRCalibratedCWDB,
    FunctionalAIPW,
)

__all__ = [
    "AppliedDataset",
    "ArmParticleBooster",
    "ArmSharedTreeRegressor",
    "CWDBRegressor",
    "CrossFittedCWDBRegressor",
    "DRCalibratedCWDB",
    "FunctionalAIPW",
    "aggregate_replications",
    "ecdf_moderator",
    "energy_score_on",
    "grid_weights",
    "make_functionals",
    "midpoint_levels",
    "plugin_contrasts",
    "quantile_matrix",
    "run_placebo",
    "run_wcf",
    "run_wcf_replications",
    "weighted_quantiles",
]
