"""Phase 5.5 (G3.5): orthogonalized and meta-learner C-WDB variants.

The G3 repair track left one uncomfortable comparison open: on the grid causal
mean, PTA-S beats the R3 claimant in five regimes and loses in two. Phase 5.5
tests whether treatment-effect orthogonalization improves the weak common-target
comparison without discarding C-WDB's law-level capability. This module owns
the frozen contract: the method registry with explicit ``produces_law``,
``target_ids``, ``inference``, and ``cross_fitted`` fields, the imbalance stress
regimes, and the staged manifest. Nothing here chooses a threshold after seeing
a result; everything below is frozen before the first Stage 1 seed.

The registry entries are the WP5.5-A schema:

* ``cwdb_rmean``: vector R-learner on rescaled quantiles, mean contrast only.
* ``cwdb_xmean``: cross-fitted vector X-learner, mean contrast only.
* ``cwdb_mutau``: particle booster with prognostic/contrast leaf fields, the
  only variant that can claim law-level targets, and then only after the same
  validity checks C-WDB R3 passes.

A mean-only variant must be rejected if a law metric is requested; the adapter
layer enforces that by contract, and the evaluation layer reports
``not_applicable`` rows rather than substitute quantities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from ..meta_learners.r_learner import CONTRAST_BUDGET
from ..meta_learners.x_learner import EFFECT_BUDGET
from .dgps import build_imbalance_specs
from .manifest import (
    BOOSTING_BUDGET,
    ESTIMAND_CONTRACT_ID,
    EVALUATION_MANIFEST,
    METHOD_REGISTRY,
    N_TEST,
    TEST_SEED_OFFSET,
    TRAINING_FUNCTIONALS,
    Cell,
    GridSpec,
)

PHASE55_CONTRACT_ID = "G3-PHASE55-v1"
PARENT_CONTRACT_ID = "G3-REPAIR-v1"

#: The frozen variant roster. ``target_ids`` names the estimand contract
#: identifiers each method may supply; ``inference`` stays null because no
#: Phase 5.5 variant has an interval construction; ``cross_fitted`` records
#: that every variant's nuisances, propensity, and tuning respect A15.
PHASE55_METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "cwdb_rmean": {
        "role": "variant",
        "adapter": "rmean",
        "produces_law": False,
        "target_ids": ["MEANQ-A-K"],
        "inference": None,
        "cross_fitted": True,
        "parameters": {},
    },
    "cwdb_xmean": {
        "role": "variant",
        "adapter": "xmean",
        "produces_law": False,
        "target_ids": ["MEANQ-A-K"],
        "inference": None,
        "cross_fitted": True,
        "parameters": {},
    },
    "cwdb_mutau": {
        "role": "variant",
        "adapter": "mutau",
        "produces_law": True,
        "target_ids": ["LAW-A-M-K", "LAW-A-K", "MEANQ-A-K"],
        "inference": None,
        "cross_fitted": True,
        "parameters": {
            #: Contrast strengths the particle mu/tau booster scans, and the
            #: folds it scans them over. Three folds score every training row
            #: exactly once; the pilot showed the two-fold estimate was too
            #: noisy to separate the strengths on the null regime, which is
            #: precisely the regime where the choice matters.
            "contrast_candidates": [0.0, 50.0, 500.0],
            "n_folds": 3,
        },
    },
}

PHASE55_METHODS = tuple(PHASE55_METHOD_REGISTRY)

#: Frozen candidate strengths and folds for the R-learner's ridge selection.
#: The same strengths the G3 repair track scanned, and three folds so every
#: training row is scored exactly once (the pilot showed two folds could not
#: separate the strengths on the null regime).
RMEAN_SHRINKAGE_CANDIDATES = (0.0, 50.0, 500.0)
RMEAN_SELECTION_FOLDS = 3

#: Frozen hyperparameter budgets the variants share. The contrast budget is the
#: G3 boosting budget; the mutau variant inherits the frozen C-WDB budget from
#: the manifest's ``boosting_budget``.
CONTRAST_BUDGET_FROZEN = dict(CONTRAST_BUDGET)
EFFECT_BUDGET_FROZEN = dict(EFFECT_BUDGET)

#: Stage 1 grid. The `main` cells reuse the frozen G3 coordinates (grid, n,
#: K, M, seeds), so every Stage 1 row pairs seed by seed with the existing
#: PTA-S and R3 rows in `results/merged*`. Ten seeds is the pilot-replication
#: count of the mechanism screen, declared here before the first run.
STAGE1_DGPS = ("D0", "D2", "D7", "D8")
STAGE1_SEEDS = tuple(range(10))
IMBALANCE_DGPS = ("D2-imb", "D7-imb", "D8-imb")

#: Stage 2 scope. Only `cwdb_mutau` continues: `cwdb_rmean` cannot supply a law
#: target by contract and was retained as a benchmark, and `cwdb_xmean` failed
#: its Stage 1 mechanism screen. Stage 2 runs the claimant on the whole frozen
#: main grid at the incumbents' own replication count, because the phase's
#: numerical hook for the mu/tau variant names D3, D5, D6 and D9 (sharing, law
#: separation, mode coverage, and the unseen-functional regime) and none of
#: those was inside the Stage 1 screen.
STAGE2_METHODS = ("cwdb_mutau",)
STAGE2_SEEDS = tuple(range(20))


IMBALANCE_SPECS = build_imbalance_specs()


def register_phase55_methods() -> None:
    """Make the variants visible to the runner's adapter factory.

    Registry entries cannot disturb the frozen manifest: its checksum is taken
    over the enumerated cells, and a method enters the cell list only by
    appearing in a ``GridSpec``, which ``build_grids()`` does not do.
    """

    for name, entry in PHASE55_METHOD_REGISTRY.items():
        METHOD_REGISTRY.setdefault(name, entry)


register_phase55_methods()


def build_phase55_grids(
    dgps: tuple[str, ...] | None = None,
    methods: tuple[str, ...] = PHASE55_METHODS,
    include_imbalance: bool = True,
    seeds: tuple[int, ...] = STAGE1_SEEDS,
) -> tuple[GridSpec, ...]:
    """The frozen Stage 1 grid set: `main` coordinates plus the imbalance suite.

    The `main` grid copies the frozen coordinates so every cell pairs with the
    existing G3 rows; the `imbalance` grid is the X-learner stress suite, which
    has no frozen counterpart and is compared internally (xmean against rmean
    on identical cells).
    """

    frozen = _frozen_grids()
    unknown = set(methods) - set(PHASE55_METHOD_REGISTRY)
    if unknown:
        raise ValueError(f"unknown Phase 5.5 methods: {sorted(unknown)}")
    result: list[GridSpec] = []
    main = frozen["main"]
    selected_main = main.dgps if dgps is None else tuple(
        d for d in main.dgps if d in dgps
    )
    if selected_main:
        result.append(
            GridSpec(
                grid="main",
                purpose="Phase 5.5 mechanism screen on the frozen coordinates",
                dgps=selected_main,
                n_train=main.n_train,
                n_grid=main.n_grid,
                n_particles=main.n_particles,
                methods=methods,
                seeds=seeds,
                notes=(
                    "Variant cells only. Coordinates and seeds are copied from "
                    "the frozen main grid so every row pairs seed by seed with "
                    "the existing PTA-S, Causal-DRF, and R3 rows."
                ),
            )
        )
    if include_imbalance:
        result.append(
            GridSpec(
                grid="imbalance",
                purpose="X-learner stress suite: same outcomes, imbalanced assignment",
                dgps=IMBALANCE_DGPS,
                n_train=(500,),
                n_grid=(25,),
                n_particles=(10,),
                methods=("cwdb_rmean", "cwdb_xmean"),
                seeds=seeds,
                notes=(
                    "Internal comparison only: xmean against rmean on identical "
                    "cells, because no frozen baseline exists on these regimes."
                ),
            )
        )
    return tuple(result)


def _frozen_grids():
    from .manifest import build_grids

    return {grid.grid: grid for grid in build_grids()}


def enumerate_phase55_cells(
    dgps: tuple[str, ...] | None = None,
    methods: tuple[str, ...] = PHASE55_METHODS,
    include_imbalance: bool = True,
    seeds: tuple[int, ...] = STAGE1_SEEDS,
) -> list[Cell]:
    cells: list[Cell] = []
    seen: set[str] = set()
    for grid in build_phase55_grids(dgps, methods, include_imbalance, seeds):
        for cell in grid.cells():
            if cell.key in seen:
                raise ValueError(f"duplicate cell key for {cell}")
            seen.add(cell.key)
            cells.append(cell)
    return cells


def build_phase55_manifest(
    dgps: tuple[str, ...] | None = STAGE1_DGPS,
    methods: tuple[str, ...] = PHASE55_METHODS,
    include_imbalance: bool = True,
    seeds: tuple[int, ...] = STAGE1_SEEDS,
) -> dict[str, Any]:
    """The frozen Phase 5.5 Stage 1 manifest document.

    The default scope is the Stage 1 mechanism screen: the four regimes the
    phase cares about, at the frozen coordinates. Passing ``dgps=None`` widens
    the main grid to every frozen regime, which is how a later Stage 2 would
    extend the screen.
    """

    grids = build_phase55_grids(dgps, methods, include_imbalance, seeds)
    cells = enumerate_phase55_cells(dgps, methods, include_imbalance, seeds)
    document: dict[str, Any] = {
        "manifest_contract_id": PHASE55_CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "parent_manifest_contract_id": PARENT_CONTRACT_ID,
        "stage": 1,
        "n_cells": len(cells),
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "boosting_budget": BOOSTING_BUDGET,
        "contrast_budget": CONTRAST_BUDGET_FROZEN,
        "effect_budget": EFFECT_BUDGET_FROZEN,
        "rmean_shrinkage_candidates": list(RMEAN_SHRINKAGE_CANDIDATES),
        "rmean_selection_folds": RMEAN_SELECTION_FOLDS,
        "evaluation_manifest": {
            **EVALUATION_MANIFEST,
            "functionals": list(EVALUATION_MANIFEST["functionals"]),
        },
        "method_registry": {
            name: PHASE55_METHOD_REGISTRY[name] for name in methods
        },
        "imbalance_specs": {
            name: {
                "base": name[:-4],
                "propensity": "clipped logistic, target assignment ~ 0.8",
                "null_effect": spec.null_effect,
            }
            for name, spec in IMBALANCE_SPECS.items()
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


def build_stage2_grids(
    methods: tuple[str, ...] = STAGE2_METHODS,
    seeds: tuple[int, ...] = STAGE2_SEEDS,
) -> tuple[GridSpec, ...]:
    """The Stage 2 grid set: exactly the cells Stage 1 did not already run.

    Two sub-grids, both on the frozen `main` coordinates. The first is the six
    regimes Stage 1 never touched, at the full seed count. The second is the
    seed top-up on the four screen regimes, which Stage 1 ran at ten seeds while
    every incumbent carries twenty; without it, a pooled comparison would have
    to restrict the incumbents to the screen's replication, which is the
    convention that made an R3 sample-size artefact appear in the report.

    Nothing already computed is enumerated here. Cell keys are content-addressed
    over the coordinates alone, so the Stage 1 and Stage 2 key sets are disjoint
    by construction and the two merged tables union without ambiguity.
    """

    unknown = set(methods) - set(PHASE55_METHOD_REGISTRY)
    if unknown:
        raise ValueError(f"unknown Phase 5.5 methods: {sorted(unknown)}")
    carried = tuple(seed for seed in STAGE1_SEEDS if seed in seeds)
    if carried != STAGE1_SEEDS:
        raise ValueError(
            "Stage 2 must contain the Stage 1 seeds so the top-up is well defined"
        )
    main = _frozen_grids()["main"]
    new_dgps = tuple(d for d in main.dgps if d not in STAGE1_DGPS)
    screen_dgps = tuple(d for d in main.dgps if d in STAGE1_DGPS)
    top_up_seeds = tuple(seed for seed in seeds if seed not in STAGE1_SEEDS)
    result: list[GridSpec] = []
    if new_dgps:
        result.append(
            GridSpec(
                grid="main",
                purpose="Phase 5.5 Stage 2: the regimes the mechanism screen omitted",
                dgps=new_dgps,
                n_train=main.n_train,
                n_grid=main.n_grid,
                n_particles=main.n_particles,
                methods=methods,
                seeds=seeds,
                notes=(
                    "Frozen main coordinates. These regimes carry the phase's "
                    "open law-level questions: D3 sharing, D5 law separation, "
                    "D6 mode coverage, and D9."
                ),
            )
        )
    if screen_dgps and top_up_seeds:
        result.append(
            GridSpec(
                grid="main",
                purpose="Phase 5.5 Stage 2: seed top-up on the four screen regimes",
                dgps=screen_dgps,
                n_train=main.n_train,
                n_grid=main.n_grid,
                n_particles=main.n_particles,
                methods=methods,
                seeds=top_up_seeds,
                notes=(
                    "Seeds the Stage 1 screen did not run. Merged with the "
                    "Stage 1 rows, these regimes reach the twenty seeds every "
                    "incumbent already carries."
                ),
            )
        )
    return tuple(result)


def enumerate_stage2_cells(
    methods: tuple[str, ...] = STAGE2_METHODS,
    seeds: tuple[int, ...] = STAGE2_SEEDS,
) -> list[Cell]:
    cells: list[Cell] = []
    seen: set[str] = set()
    for grid in build_stage2_grids(methods, seeds):
        for cell in grid.cells():
            if cell.key in seen:
                raise ValueError(f"duplicate cell key for {cell}")
            seen.add(cell.key)
            cells.append(cell)
    return cells


def build_phase55_stage2_manifest(
    methods: tuple[str, ...] = STAGE2_METHODS,
    seeds: tuple[int, ...] = STAGE2_SEEDS,
) -> dict[str, Any]:
    """The frozen Phase 5.5 Stage 2 manifest.

    Same contract identifiers, same budgets, same evaluation manifest as Stage
    1. The only thing that moves is the scope, and it moves by widening the
    regime set and the seed count, never by changing a coordinate a Stage 1 row
    was produced at. The Stage 1 checksum is recorded so an auditor can verify
    which frozen table this stage is meant to be unioned with.
    """

    grids = build_stage2_grids(methods, seeds)
    cells = enumerate_stage2_cells(methods, seeds)
    stage1 = build_phase55_manifest()
    document: dict[str, Any] = {
        "manifest_contract_id": PHASE55_CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "parent_manifest_contract_id": PARENT_CONTRACT_ID,
        "stage": 2,
        "stage1_manifest_checksum": stage1["manifest_checksum"],
        "union_note": (
            "Stage 2 enumerates only the cells Stage 1 did not run. The analysis "
            "surface for the claimant is the union of this table with the Stage 1 "
            "cwdb_mutau rows in results/merged_phase55; the two key sets are "
            "disjoint by construction."
        ),
        "n_cells": len(cells),
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "boosting_budget": BOOSTING_BUDGET,
        "contrast_budget": CONTRAST_BUDGET_FROZEN,
        "effect_budget": EFFECT_BUDGET_FROZEN,
        "rmean_shrinkage_candidates": list(RMEAN_SHRINKAGE_CANDIDATES),
        "rmean_selection_folds": RMEAN_SELECTION_FOLDS,
        "evaluation_manifest": {
            **EVALUATION_MANIFEST,
            "functionals": list(EVALUATION_MANIFEST["functionals"]),
        },
        "method_registry": {
            name: PHASE55_METHOD_REGISTRY[name] for name in methods
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
