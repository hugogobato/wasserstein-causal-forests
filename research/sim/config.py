from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_K = 49
DEFAULT_J = 3
L_FUNCTIONAL = 400
QUANTILE_GRID = np.linspace(0.05, 0.95, DEFAULT_K)
FUNCTIONAL_GRID = (np.arange(L_FUNCTIONAL) + 0.5) / L_FUNCTIONAL

DEFAULT_N_FOLDS = 5
DEFAULT_N_TREES = 200
DEFAULT_MIN_LEAF = 5
DEFAULT_MAX_DEPTH = 8

OBSERVATION_REGIMES = (
    "oracle_latent",
    "feasible_growing_inner",
    "identified_measurement_model",
    "empirical_proxy",
)

METHOD_NAMES = (
    "odcf_composite",
    "odcf_curve_only",
    "odcf_mmd_score",
    "odcf_composite_bootstrap",
    "pointwise_causal_forest",
    "scalar_causal_forest",
    "multi_output_dr_forest",
    "specialized_forest",
    "two_arm_frechet_forest",
    "global_dr_estimator",
    "drf_inspired_arm_mmd",
    "causal_drf_port",
    "focal_dr_meta_learner",
    "wasserstein_random_forest",
)

# Prior-art incumbents added for the second pilot.  The first pilot compared
# only against internal ablations and homebrew comparators, so the closest
# published competitors had never been run.
PRIOR_ART_METHODS = (
    "causal_drf_port",
    "focal_dr_meta_learner",
    "wasserstein_random_forest",
)

# Frozen worst-coordinate standardizers, declared before the second pilot.
#
# The first pilot standardized by np.std(vstack(true_m0, true_m1), axis=0).
# On D4 most coordinates are constant across units, so that empirical scale
# collapsed to ~0, was floored at 1e-8, and inflated the metric to 1e6-1e7.
# worst_standardized_error is the declared primary metric for D5 and D8, so a
# realization-dependent standardizer cannot be used at the gate.  The constants
# below are fixed across every DGP, regime, method, seed, and sample size, and
# are chosen as the magnitude of a scientifically meaningful effect on each
# coordinate: 0.10 in log1p income units for the quantile curve, and 0.05 on
# the bounded Gini/Theil/Atkinson coordinates.
WORST_COORDINATE_CURVE_SCALE = 0.10
WORST_COORDINATE_FUNCTIONAL_SCALES = (0.05, 0.05, 0.05)


def frozen_coordinate_scales(K: int = DEFAULT_K, J: int = DEFAULT_J) -> np.ndarray:
    """Return the frozen (K+J,) standardizer for worst-coordinate error."""
    if K < 1 or J < 0:
        raise ValueError("K must be positive and J nonnegative")
    if J > len(WORST_COORDINATE_FUNCTIONAL_SCALES):
        raise ValueError(
            "no frozen scale is declared for more than "
            f"{len(WORST_COORDINATE_FUNCTIONAL_SCALES)} functional coordinates"
        )
    scales = np.r_[
        np.full(K, WORST_COORDINATE_CURVE_SCALE, dtype=float),
        np.asarray(WORST_COORDINATE_FUNCTIONAL_SCALES[:J], dtype=float),
    ]
    if np.any(scales <= 0):
        raise ValueError("frozen coordinate scales must be strictly positive")
    return scales

@dataclass
class SimConfig:
    n_regions: int = 500
    K: int = DEFAULT_K
    J: int = DEFAULT_J
    seed: int = 20260727
    n_folds: int = DEFAULT_N_FOLDS
    n_trees: int = DEFAULT_N_TREES
    min_leaf: int = DEFAULT_MIN_LEAF
    max_depth: int = DEFAULT_MAX_DEPTH
    n_eval: int = 200
    d_covariates: int = 5
    d_noise: int = 0
    inner_sample_min: int = 100
    inner_sample_max: int = 500
    observation_regime: str = "oracle_latent"
    methods: tuple[str, ...] = field(default_factory=lambda: METHOD_NAMES)
    dgp_names: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4", "D5", "D8")
