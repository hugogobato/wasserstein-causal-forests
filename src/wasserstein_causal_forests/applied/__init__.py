"""Applied-data interface for the Wasserstein Causal Forest estimator."""

from .adapter import (
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

__all__ = [
    "AppliedDataset",
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
