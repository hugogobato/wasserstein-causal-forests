"""Phase 6: reference-effect repair mechanisms and the income realism track.

The frozen record leaves three open problems. On the reference-distribution
targets the project cares most about (`REF-ATE-K`, `REF-TCATE-K`), the
recommended claimant loses to both forest baselines on D5 by a factor of four
to five; the diagnostic recorded in the phase report attributes this to
arm-specific under-dispersion of the fitted particle cloud, which biases every
convex spread-sensitive functional. The D7 pure-shape transfer is damaged by
exactly the contrast regularisation that fixes the null regime. And no regime
in the suite resembles the applied study under discussion: state-year panels,
income-shaped right-skewed inner distributions, an explicit benchmark-economy
reference, endogenous policy adoption.

Phase 6 tests one mechanism per variant, all on the frozen main coordinates so
every new cell pairs seed by seed with the existing roster rows:

* ``cwdb_dr`` keeps the cross-fitted R3 law untouched and adds a doubly-robust
  calibration layer for declared functional contrasts, built from out-of-fold
  arm-law means and a cross-fitted propensity.
* ``cwdb_smooth`` repairs dispersion after fitting, choosing between radial
  scaling and Gaussian jitter on held-out energy score.
* ``cwdb_krr`` swaps the weak learner: kernel ridge regression instead of a
  regression tree, independent arms, everything else byte-for-byte the booster.
* ``cwdb_frl`` generalises the Phase 5.5 vector R-learner to arbitrary grid
  functionals, including ones no training manifest contains.

Track B registers four income regimes (IC0 through IC3) built on the same
generative form, so the quadrature oracle machinery applies unchanged; only
the surfaces, propensities, outer law, reference vector, and covariate count
move. Everything here is frozen before the first decisive seed; see
`research/simulation_preregistration_phase6.md`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from . import repair as _repair  # noqa: F401  (side-effecting import: registry)
from .dgps import MODERATOR_EDGES
from .manifest import (
    BOOSTING_BUDGET,
    ESTIMAND_CONTRACT_ID,
    EVALUATION_MANIFEST,
    N_TEST,
    TEST_SEED_OFFSET,
    TRAINING_FUNCTIONALS,
    Cell,
    GridSpec,
)

PHASE6_CONTRACT_ID = "G3-PHASE6-v1"
PARENT_CONTRACT_ID = "G3-PHASE55-v1"

#: Frozen candidate strengths for every cross-fitted variant in this phase,
#: identical to the repair track's scan, and the repair track's own fold count.
#: The first Track A screen ran these variants at three folds, which is the
#: mutau setting but not R3's; the resulting laws were not the published R3
#: estimator, so the screen was rerun at two folds before any claim was written.
PHASE6_CONTRAST_CANDIDATES = (0.0, 50.0, 500.0)
PHASE6_SELECTION_FOLDS = 2

#: The frozen variant roster. ``target_ids`` follow the WP5.5-A schema.
PHASE6_METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "cwdb_dr": {
        "role": "variant",
        "adapter": "cwdb_dr",
        "produces_law": True,
        "target_ids": [
            "LAW-A-M-K", "LAW-A-K", "MEANQ-A-K",
            "TATE-K-grid_mean", "TCATE-K-grid_mean",
            "REF-ATE-K", "REF-TCATE-K",
        ],
        "inference": None,
        "cross_fitted": True,
        "parameters": {
            "contrast_candidates": list(PHASE6_CONTRAST_CANDIDATES),
            "n_folds": PHASE6_SELECTION_FOLDS,
        },
    },
    "cwdb_smooth": {
        "role": "variant",
        "adapter": "cwdb_smooth",
        "produces_law": True,
        "target_ids": ["LAW-A-M-K", "LAW-A-K", "MEANQ-A-K"],
        "inference": None,
        "cross_fitted": True,
        "parameters": {
            "contrast_candidates": list(PHASE6_CONTRAST_CANDIDATES),
            "n_folds": PHASE6_SELECTION_FOLDS,
        },
    },
    "cwdb_krr": {
        "role": "variant",
        "adapter": "cwdb_krr",
        "produces_law": True,
        "target_ids": ["LAW-A-M-K", "LAW-A-K", "MEANQ-A-K"],
        "inference": None,
        "cross_fitted": False,
        "parameters": {},
    },
    "cwdb_frl": {
        "role": "variant",
        "adapter": "cwdb_frl",
        "produces_law": False,
        "target_ids": [
            "MEANQ-A-K", "TATE-K-grid_sd", "TCATE-K-grid_sd",
            "TATE-K-grid_skewness", "TCATE-K-grid_skewness",
            "REF-ATE-K", "REF-TCATE-K",
        ],
        "inference": None,
        "cross_fitted": True,
        "parameters": {},
    },
}

PHASE6_METHODS = tuple(PHASE6_METHOD_REGISTRY)

#: Track A: mechanism screen on frozen main coordinates.
TRACK_A_DGPS = ("D0", "D2", "D5", "D6", "D7", "D8")
TRACK_A_METHODS = PHASE6_METHODS
TRACK_B_DGPS = ("IC0", "IC1", "IC2", "IC3")
TRACK_B_METHODS = (
    "cwdb_v1", "cwdb_r3_cvridge", "cwdb_dr", "causal_drf", "drf", "pta_s",
)
PHASE6_SEEDS = tuple(range(10))


def register_phase6_methods() -> None:
    """Make the variants visible to the runner's adapter factory."""

    for name, entry in PHASE6_METHOD_REGISTRY.items():
        from .manifest import METHOD_REGISTRY

        METHOD_REGISTRY.setdefault(name, entry)


def build_phase6_grids(
    tracks: tuple[str, ...] = ("a", "b"),
) -> tuple[GridSpec, ...]:
    """The frozen Phase 6 grids: Track A mechanisms, Track B realism."""

    result: list[GridSpec] = []
    if "a" in tracks:
        result.append(
            GridSpec(
                grid="main",
                purpose="Phase 6 mechanism screen on the frozen coordinates",
                dgps=TRACK_A_DGPS,
                n_train=(500, 1000),
                n_grid=(25,),
                n_particles=(10,),
                methods=TRACK_A_METHODS,
                seeds=PHASE6_SEEDS,
                notes=(
                    "Variant cells only. Coordinates and seeds copy the frozen "
                    "main grid, so every row pairs seed by seed with the "
                    "existing tournament, repair, and original-code baseline "
                    "rows."
                ),
            )
        )
    if "b" in tracks:
        result.append(
            GridSpec(
                grid="income",
                purpose=(
                    "Income realism track: state-year panels with a "
                    "benchmark-economy reference"
                ),
                dgps=TRACK_B_DGPS,
                n_train=(500, 1000),
                n_grid=(25,),
                n_particles=(10,),
                methods=TRACK_B_METHODS,
                seeds=PHASE6_SEEDS,
                notes=(
                    "New regimes; every method runs fresh here. The moderator "
                    "stays the four-bin discretisation of X_0 and the oracle "
                    "machinery is the frozen one, because the generative form "
                    "is unchanged."
                ),
            )
        )
    return tuple(result)


def enumerate_phase6_cells(tracks: tuple[str, ...] = ("a", "b")) -> list[Cell]:
    cells: list[Cell] = []
    seen: set[str] = set()
    for grid in build_phase6_grids(tracks):
        for cell in grid.cells():
            if cell.key in seen:
                raise ValueError(f"duplicate cell key for {cell}")
            seen.add(cell.key)
            cells.append(cell)
    return cells


def build_phase6_manifest(tracks: tuple[str, ...] = ("a", "b")) -> dict[str, Any]:
    grids = build_phase6_grids(tracks)
    cells = enumerate_phase6_cells(tracks)
    document: dict[str, Any] = {
        "manifest_contract_id": PHASE6_CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "parent_manifest_contract_id": PARENT_CONTRACT_ID,
        "moderator_edges": list(MODERATOR_EDGES),
        "n_cells": len(cells),
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "boosting_budget": BOOSTING_BUDGET,
        "contrast_candidates": list(PHASE6_CONTRAST_CANDIDATES),
        "selection_folds": PHASE6_SELECTION_FOLDS,
        "evaluation_manifest": {
            **EVALUATION_MANIFEST,
            "functionals": list(EVALUATION_MANIFEST["functionals"]),
        },
        "method_registry": {
            **{name: PHASE6_METHOD_REGISTRY[name] for name in TRACK_A_METHODS},
            "cwdb_r3_cvridge": _repair.REPAIR_METHOD_REGISTRY["cwdb_r3_cvridge"],
        },
        "grids": [
            {
                **{
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in asdict(grid).items()
                },
                "n_cells": len(grid.cells()),
            }
            for grid in grids
        ],
        "cells": [cell.to_dict() for cell in cells],
    }
    document["manifest_checksum"] = hashlib.sha256(
        json.dumps(document["cells"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return document


register_phase6_methods()
