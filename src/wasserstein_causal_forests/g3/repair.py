"""The rule-1 repair track: new C-WDB variants on the frozen coordinates.

The G3 tournament returned `NOT-GO` on one rule. Rule 1 asks whether the method
stays quiet on D2, where the true treatment effect is exactly null, and C-WDB-v1
produced a false contrast 2.69 times the best baseline's against a cap of 1.25.
`research/gates/G3_simulation_memo.md` Section 7 names the repair: a
contrast-level regulariser, then a re-run of the same manifest.

This module is that re-run's manifest. Three properties matter.

*Nothing already computed is recomputed.* The repair manifest contains cells for
the new methods only. Every baseline, ablation, and C-WDB-v1 row comes from
`results/merged/main_results.parquet` unchanged, so the comparison is against
the same numbers the memo reports rather than against a fresh draw of them.

*The coordinates are the frozen ones.* Grid labels, sample sizes, grid sizes,
particle counts, and seeds are copied from `build_grids()`, and a repair cell
carries the same `grid` label as the frozen cells it will be compared against.
Paired comparisons match on (grid, dgp, n, K, M, seed), so a repair variant and
a frozen baseline pair seed by seed exactly as two frozen methods do.

*The thresholds are the frozen ones.* Nothing here restates a gate rule. The
analysis reads `GATE_RULES` with the repair variant named as claimant, and rule
1's baseline reference stays the best of `FROZEN_G3_METHODS`, so a repair
variant cannot move the bar it is judged against by entering the pool.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

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
    build_grids,
)

REPAIR_CONTRACT_ID = "G3-REPAIR-v1"

#: Candidate strengths the cross-fitted variant scans, and the number of folds
#: it scans them over. Declared here rather than chosen per regime, so the
#: variant's adaptivity is in the selection rule and not in a strength picked by
#: hand after seeing a regime's result.
#:
#: The size of this scan is a cost decision, not an accuracy one. Selection
#: costs `n_folds * len(candidates)` extra fits, and gate rule 6 caps the
#: claimant's median runtime at 60 times Causal-DRF's. Three folds over four
#: strengths measured 6.9 times C-WDB-v1's fit time on the pilot seeds, which
#: projects past that ceiling; two folds over three strengths is about four
#: ordinary fits and stays inside it.
CONTRAST_CANDIDATES = (0.0, 50.0, 500.0)
CONTRAST_SELECTION_FOLDS = 2

#: Strength for the fixed-ridge variant. Unlike the other two repairs this one
#: has a constant that had to be chosen, and it was chosen on pilot seeds
#: 100-104, which the tournament manifest does not enumerate. It is therefore
#: the hand-tuned reference: the adaptive rules get no such tuning, so any
#: advantage they hold over it is an advantage held against a tuned opponent.
CONTRAST_RIDGE_STRENGTH = 50.0

#: Calibrations of the adaptive threshold rule. One is the null-calibrated
#: value: under an exactly null arm gap the expected squared gap equals its
#: plug-in variance, so the retained fraction is zero in expectation. Three
#: thresholds at a higher quantile of the same reference distribution and enters
#: as a declared conservative sensitivity, not as a value chosen after the fact.
#: Both are entered and both are reported; the stage-1 screen decides which
#: continues, on a criterion fixed before the first decisive seed.
CONTRAST_THRESHOLD_SCALE = 1.0
CONTRAST_THRESHOLD_SCALE_CONSERVATIVE = 3.0

#: The repair roster. Every entry keeps C-WDB-v1's architecture, boosting budget,
#: and repulsion term; they differ only in how the arm contrast is regularised.
#:
#: `cwdb_v1_pooledinit` is the mechanism ablation for the initialisation, not a
#: candidate method: a per-arm initial law is a marginal quantity, so under
#: confounding it seeds the contrast with the arm gap in the covariate
#: distribution before a single tree is fitted. Isolating it says how much of
#: the D2 failure the initialisation alone explains.
REPAIR_METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "cwdb_v1_pooledinit": {
        "role": "ablation",
        "adapter": "cwdb",
        "produces_law": True,
        "parameters": {
            "architecture": "v1",
            "sharing": "partial",
            "arm_shrinkage": 5.0,
            "init_sharing": "pooled",
        },
    },
    "cwdb_r1_ridge": {
        "role": "repair",
        "adapter": "cwdb",
        "produces_law": True,
        "parameters": {
            "architecture": "v1",
            "sharing": "partial",
            "arm_shrinkage": 5.0,
            "init_sharing": "pooled",
            "contrast_rule": "ridge",
            "contrast_shrinkage": CONTRAST_RIDGE_STRENGTH,
        },
    },
    "cwdb_r2_threshold": {
        "role": "repair",
        "adapter": "cwdb",
        "produces_law": True,
        "parameters": {
            "architecture": "v1",
            "sharing": "partial",
            "arm_shrinkage": 5.0,
            "init_sharing": "pooled",
            "contrast_rule": "threshold",
            "contrast_threshold_scale": CONTRAST_THRESHOLD_SCALE,
        },
    },
    "cwdb_r2_threshold3": {
        "role": "repair",
        "adapter": "cwdb",
        "produces_law": True,
        "parameters": {
            "architecture": "v1",
            "sharing": "partial",
            "arm_shrinkage": 5.0,
            "init_sharing": "pooled",
            "contrast_rule": "threshold",
            "contrast_threshold_scale": CONTRAST_THRESHOLD_SCALE_CONSERVATIVE,
        },
    },
    "cwdb_r3_cvridge": {
        "role": "repair",
        "adapter": "cwdb",
        "produces_law": True,
        "parameters": {
            "architecture": "v1",
            "sharing": "partial",
            "arm_shrinkage": 5.0,
            "init_sharing": "pooled",
            "contrast_rule": "ridge",
            "contrast_candidates": list(CONTRAST_CANDIDATES),
            "n_folds": CONTRAST_SELECTION_FOLDS,
        },
    },
}

REPAIR_METHODS = tuple(REPAIR_METHOD_REGISTRY)

#: Grids a repaired claimant must occupy for every gate rule and preregistered
#: ablation to be computable. `main` carries rules 1 to 5, `smallk` carries the
#: repulsion ablation against the squared-W2 booster, and `shrinkage` carries
#: the causal-regularisation ablation against `arm_shrinkage = 0`.
REPAIR_GRIDS = ("main", "smallk", "shrinkage")

#: Methods that cleared the stage-1 screen and continue, plus the initialisation
#: ablation, which is exempt because its job is attribution rather than
#: candidacy. `cwdb_r2_threshold` at the null-calibrated `c = 1` missed the
#: stage-1 cap at a ratio of 1.45 and stops after D2; its rows stay in the track
#: and stay reported.
STAGE_TWO_METHODS = (
    "cwdb_v1_pooledinit",
    "cwdb_r1_ridge",
    "cwdb_r2_threshold3",
    "cwdb_r3_cvridge",
)


@dataclass(frozen=True)
class RepairStage:
    """One staged slice of the repair track."""

    stage: int
    purpose: str
    dgps: tuple[str, ...] | None
    methods: tuple[str, ...]
    grids: tuple[str, ...]


#: The track as run, in order. The manifest is the union of these, so a stage
#: that stops early keeps its cells in the reconciliation instead of turning
#: into rows the merge cannot account for. A screened-out variant's results are
#: evidence about the screen and must survive in the audit.
REPAIR_STAGES: tuple[RepairStage, ...] = (
    RepairStage(
        stage=1,
        purpose="screen every candidate on the regime rule 1 failed",
        dgps=("D2",),
        methods=REPAIR_METHODS,
        grids=("main",),
    ),
    RepairStage(
        stage=2,
        purpose="every regime and both ablation grids, for what cleared stage 1",
        dgps=None,
        methods=STAGE_TWO_METHODS,
        grids=REPAIR_GRIDS,
    ),
)


def register_repair_methods() -> None:
    """Make the repair methods visible to the runner's adapter factory.

    Adding registry entries cannot disturb the frozen manifest: its checksum is
    taken over the enumerated cells, and a method enters the cell list only by
    appearing in a `GridSpec`, which `build_grids()` still does not do.
    """

    for name, entry in REPAIR_METHOD_REGISTRY.items():
        METHOD_REGISTRY.setdefault(name, entry)


register_repair_methods()


def build_repair_grids(
    dgps: tuple[str, ...] | None = None,
    methods: tuple[str, ...] = REPAIR_METHODS,
    grids: tuple[str, ...] = REPAIR_GRIDS,
) -> tuple[GridSpec, ...]:
    """Frozen grids restricted to the repair methods, and optionally to some DGPs.

    `dgps` narrows the regimes, which is how the staged run works: D2 first,
    because that is the rule the repair exists to fix, and the rest only for a
    variant that fixed it.
    """

    unknown = set(methods) - set(REPAIR_METHOD_REGISTRY)
    if unknown:
        raise ValueError(f"unknown repair methods: {sorted(unknown)}")
    frozen = {grid.grid: grid for grid in build_grids()}
    missing = set(grids) - set(frozen)
    if missing:
        raise ValueError(f"unknown frozen grids: {sorted(missing)}")

    result: list[GridSpec] = []
    for name in grids:
        source = frozen[name]
        selected = source.dgps if dgps is None else tuple(
            d for d in source.dgps if d in dgps
        )
        if not selected:
            continue
        result.append(
            GridSpec(
                grid=source.grid,
                purpose=f"rule 1 repair: {source.purpose}",
                dgps=selected,
                n_train=source.n_train,
                n_grid=source.n_grid,
                n_particles=source.n_particles,
                methods=methods,
                seeds=source.seeds,
                notes=(
                    "Repair cells only. Coordinates and seeds are copied from the "
                    "frozen grid of the same name so every repair row pairs with "
                    "the existing rows of that grid."
                ),
            )
        )
    return tuple(result)


def enumerate_repair_cells(
    dgps: tuple[str, ...] | None = None,
    methods: tuple[str, ...] = REPAIR_METHODS,
    grids: tuple[str, ...] = REPAIR_GRIDS,
) -> list[Cell]:
    cells: list[Cell] = []
    seen: set[str] = set()
    for grid in build_repair_grids(dgps, methods, grids):
        for cell in grid.cells():
            if cell.key in seen:
                raise ValueError(f"duplicate cell key for {cell}")
            seen.add(cell.key)
            cells.append(cell)
    return cells


def staged_specs_and_cells(
    stages: tuple[RepairStage, ...],
) -> tuple[list[GridSpec], list[Cell]]:
    """Union of several stages' grids and cells, deduplicated by cell key.

    Stages overlap: stage 2 re-declares D2 on the `main` grid for the variants
    that continued, which stage 1 already ran. The union keeps one copy, so the
    manifest reconciles against every row the track produced, and a cell already
    computed is skipped by `--resume` rather than recomputed.
    """

    specs: list[GridSpec] = []
    cells: list[Cell] = []
    seen: set[str] = set()
    for stage in stages:
        for spec in build_repair_grids(stage.dgps, stage.methods, stage.grids):
            specs.append(spec)
            for cell in spec.cells():
                if cell.key in seen:
                    continue
                seen.add(cell.key)
                cells.append(cell)
    return specs, cells


def build_staged_repair_manifest(
    stages: tuple[RepairStage, ...] = REPAIR_STAGES,
) -> dict[str, Any]:
    """The manifest for the repair track as staged, covering every stage run."""

    specs, cells = staged_specs_and_cells(stages)
    document = _manifest_document(specs, cells)
    document["stages"] = [asdict(stage) for stage in stages]
    return document


def build_repair_manifest(
    dgps: tuple[str, ...] | None = None,
    methods: tuple[str, ...] = REPAIR_METHODS,
    grids: tuple[str, ...] = REPAIR_GRIDS,
) -> dict[str, Any]:
    """The manifest for one stage, in the same shape as the frozen one."""

    return _manifest_document(
        list(build_repair_grids(dgps, methods, grids)),
        enumerate_repair_cells(dgps, methods, grids),
    )


def _manifest_document(
    specs: list[GridSpec], cells: list[Cell]
) -> dict[str, Any]:
    methods = tuple(dict.fromkeys(m for spec in specs for m in spec.methods))
    document: dict[str, Any] = {
        "manifest_contract_id": REPAIR_CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "parent_manifest_contract_id": "G3-MAIN-v1",
        "n_cells": len(cells),
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "boosting_budget": BOOSTING_BUDGET,
        "evaluation_manifest": {
            **EVALUATION_MANIFEST,
            "functionals": list(EVALUATION_MANIFEST["functionals"]),
        },
        "method_registry": {
            name: REPAIR_METHOD_REGISTRY[name] for name in methods
        },
        "grids": [
            {
                **{
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in asdict(spec).items()
                },
                "n_cells": len(spec.cells()),
            }
            for spec in specs
        ],
        "cells": [cell.to_dict() for cell in cells],
    }
    document["manifest_checksum"] = hashlib.sha256(
        json.dumps(document["cells"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return document
