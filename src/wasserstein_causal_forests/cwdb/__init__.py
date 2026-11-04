"""Causal Wasserstein Distribution Boosting."""

from .arm_shared_tree import CONTRAST_RULES, ArmSharedTreeRegressor
from .cross_fitted import (
    DEFAULT_CONTRAST_CANDIDATES,
    CrossFittedCWDBRegressor,
    stratified_folds,
)
from .dr_calibration import DRCalibratedCWDB, FunctionalAIPW, aipw_scores, hajek_bin_means
from .energy import (
    EnergyScoreComponents,
    empirical_energy_risk,
    energy_gradient,
    energy_score,
    energy_score_components,
)
from .model import ArmParticleBooster, CWDBRegressor

__all__ = [
    "CONTRAST_RULES",
    "DEFAULT_CONTRAST_CANDIDATES",
    "ArmParticleBooster",
    "ArmSharedTreeRegressor",
    "CWDBRegressor",
    "CrossFittedCWDBRegressor",
    "DRCalibratedCWDB",
    "EnergyScoreComponents",
    "FunctionalAIPW",
    "aipw_scores",
    "hajek_bin_means",
    "stratified_folds",
    "empirical_energy_risk",
    "energy_gradient",
    "energy_score",
    "energy_score_components",
]

