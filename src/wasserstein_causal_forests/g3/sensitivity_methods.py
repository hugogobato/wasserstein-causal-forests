"""Method identifiers for the flexible and oracle propensity variants.

`cwdb_dr_flex` and `cwdb_dr_oracle` reuse the frozen `cwdb_dr` adapter with a
non-logistic propensity factory and with the sampled true propensity,
respectively, and both request the dense common-grid calibration payload. The
static entries mirror `wcf_sensitivity.build_wcf_method_registry` so the
identifiers resolve before a frozen manifest is applied; an applied manifest
still wins because registration writes with `setdefault`.

`cwdb_dr_oracle` is a DIAGNOSTIC, not a feasible competitor: it consumes the
true propensity and cannot be estimated from data. `cwdb_dr_flex` uses fixed
classifier hyperparameters with no held-out tuning, so its contrast against
`cwdb_dr` measures the propensity model family rather than a tuned alternative.
See `report/sensitivity_and_dgp_experiments.md`, sections "Propensity-model
sensitivity" and "Concrete implementation work packages".
"""

from __future__ import annotations

from typing import Any

from .manifest import METHOD_REGISTRY
from .phase6 import PHASE6_METHOD_REGISTRY
from .wcf_sensitivity import (
    COMMON_GRID_LEVELS,
    CONTRAST_CANDIDATES,
    SELECTION_FOLDS,
)

flexible_propensity_note = (
    "cwdb_dr_oracle is a DIAGNOSTIC, not a feasible competitor: it consumes "
    "the true propensity and cannot be estimated from data. cwdb_dr_flex uses "
    "fixed hyperparameters with no held-out tuning, so its contrast against "
    "cwdb_dr measures the propensity model family, not a tuned alternative."
)

_TARGET_IDS = list(PHASE6_METHOD_REGISTRY["cwdb_dr"]["target_ids"])

SENSITIVITY_METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "cwdb_dr_flex": {
        "role": "variant",
        "adapter": "cwdb_dr",
        "produces_law": True,
        "cross_fitted": True,
        "target_ids": list(_TARGET_IDS),
        "parameters": {
            "contrast_candidates": list(CONTRAST_CANDIDATES),
            "n_folds": SELECTION_FOLDS,
            "propensity_factory": "hist_gradient_boosting",
            "common_grid_levels": COMMON_GRID_LEVELS,
        },
    },
    "cwdb_dr_oracle": {
        "role": "diagnostic",
        "adapter": "cwdb_dr",
        "produces_law": True,
        "cross_fitted": True,
        "target_ids": list(_TARGET_IDS),
        "parameters": {
            "contrast_candidates": list(CONTRAST_CANDIDATES),
            "n_folds": SELECTION_FOLDS,
            "oracle_propensity": True,
            "common_grid_levels": COMMON_GRID_LEVELS,
        },
    },
}


def register_sensitivity_methods() -> None:
    """Install the two identifiers without overwriting a frozen registry."""

    for name, entry in SENSITIVITY_METHOD_REGISTRY.items():
        METHOD_REGISTRY.setdefault(name, entry)
