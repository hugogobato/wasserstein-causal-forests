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
)

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
