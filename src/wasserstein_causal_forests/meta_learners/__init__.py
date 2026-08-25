"""Phase 5.5 meta-learners: vector R-learner and vector X-learner.

These are the two mean-contrast variants of the G3.5 pivot. Neither produces a
conditional law; only the particle ``mu/tau`` variant in ``cwdb.mutau`` can
claim law-level targets, and then only after the same validity checks C-WDB
passes.
"""

from .r_learner import CONTRAST_BUDGET, RLossTree, VectorRLearner
from .x_learner import EFFECT_BUDGET, VectorXLearner
from .nuisance import (
    CrossFittedNuisance,
    FoldPlan,
    NUISANCE_BUDGET,
    PROPENSITY_CLIP_HIGH,
    PROPENSITY_CLIP_LOW,
    clip_propensity,
)

__all__ = [
    "CONTRAST_BUDGET",
    "EFFECT_BUDGET",
    "NUISANCE_BUDGET",
    "PROPENSITY_CLIP_HIGH",
    "PROPENSITY_CLIP_LOW",
    "RLossTree",
    "VectorRLearner",
    "VectorXLearner",
    "CrossFittedNuisance",
    "FoldPlan",
    "clip_propensity",
]
