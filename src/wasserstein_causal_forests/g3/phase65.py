"""Phase 6.5: adversarial controls, ingredient ablation, zero inflation.

The frozen Phase 6 record leaves three attribution problems open, and this
phase's grids are built so each cell pairs seed by seed either with an existing
merged row or with its own track's base point:

* Track C (``c_controls``): the incumbents' best shot on the income track.
  ``causal_drf_log`` and ``drf_log`` fit on ``log Y`` and map quantiles back
  through the monotone inverse, which leaves the atom bank unchanged and moves
  only the geometry; ``causal_drf_retn`` runs the incumbent at a held-out-
  selected multiple of its data-driven kernel bandwidth, implemented as exact
  outcome rescaling because every kernel quantity is homogeneous in that scale,
  so the published driver runs unmodified. ``c_scaling`` probes whether the
  forest deficit shrinks as the training bank grows.
* Track D (``d_ablation``): five DA regimes, each the IC1 family with exactly
  one realism ingredient switched off, against the frozen IC1 rows as base
  point.
* Track E (``e_zi``): four zero-inflated regimes whose conditional laws mix a
  point mass on the degenerate law with covariate-dependent weight, plus the
  two-part claimant ``cwdb_zipt``.

Nothing here replaces the frozen Phase 6 record; every number lands beside it.
Freeze order matters: the bandwidth-selection document must exist before any
decisive retune cell, and no candidate outside the declared grid may be tried
after seeing a decisive result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

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
from .phase6 import PHASE6_CONTRAST_CANDIDATES, PHASE6_SELECTION_FOLDS

PHASE65_CONTRACT_ID = "G3-PHASE65-v1"
PARENT_CONTRACT_ID = "G3-PHASE6-v1"

#: Frozen evaluation additions for this phase: the loose tolerance that defines
#: a degenerate atom. Recorded here so the manifest carries what the metric
#: layer enforces.
PHASE65_ZERO_MASS_TOLERANCE = 0.05

#: The retune roster. ``cwdb_r3_cvridge`` resolves through the repair registry;
#: its entry is spliced into the manifest document rather than redefined here.
TRACK_C_CONTROL_METHODS = ("causal_drf_log", "drf_log", "causal_drf_retn")
TRACK_C_SCALING_METHODS = ("cwdb_r3_cvridge", "causal_drf", "drf")
TRACK_D_METHODS = ("cwdb_r3_cvridge", "cwdb_dr", "causal_drf", "drf")
TRACK_E_METHODS = (
    "cwdb_r3_cvridge", "cwdb_dr", "cwdb_zipt", "causal_drf", "drf",
)

INCOME_DGPS = ("IC0", "IC1", "IC2", "IC3")
SCALING_DGPS = ("IC0", "IC1")
ABLATION_DGPS = ("DAskew", "DArand", "DAunit", "DAref", "DAdim")
ZERO_INFLATED_DGPS = ("ZI0", "ZI1", "ZI2", "ZI3")

PHASE65_SEEDS = tuple(range(10))
SCALING_SEEDS = tuple(range(5))

PHASE65_METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "causal_drf_log": {
        "role": "baseline-control",
        "adapter": "forest_log",
        "produces_law": True,
        "target_ids": [
            "LAW-A-M-K", "LAW-A-K", "MEANQ-A-K",
            "TATE-K-grid_mean", "TCATE-K-grid_mean",
            "REF-ATE-K", "REF-TCATE-K",
        ],
        "inference": None,
        "cross_fitted": False,
        "parameters": {"method": "causal_drf"},
    },
    "drf_log": {
        "role": "baseline-control",
        "adapter": "forest_log",
        "produces_law": True,
        "target_ids": [
            "LAW-A-M-K", "LAW-A-K", "MEANQ-A-K",
            "TATE-K-grid_mean", "TCATE-K-grid_mean",
            "REF-ATE-K", "REF-TCATE-K",
        ],
        "inference": None,
        "cross_fitted": False,
        "parameters": {"method": "drf"},
    },
    "causal_drf_retn": {
        "role": "baseline-control",
        "adapter": "forest_retn",
        "produces_law": True,
        "target_ids": [
            "LAW-A-M-K", "LAW-A-K", "MEANQ-A-K",
            "TATE-K-grid_mean", "TCATE-K-grid_mean",
            "REF-ATE-K", "REF-TCATE-K",
        ],
        "inference": None,
        "cross_fitted": False,
        "parameters": {},
    },
    "cwdb_zipt": {
        "role": "variant",
        "adapter": "zipt",
        "produces_law": True,
        "target_ids": [
            "LAW-A-M-K", "LAW-A-K", "MEANQ-A-K",
            "TATE-K-grid_mean", "TCATE-K-grid_mean",
            "REF-ATE-K", "REF-TCATE-K",
        ],
        "inference": None,
        "cross_fitted": True,
        "parameters": {
            "classifier_c": 1.0,
            "contrast_candidates": list(PHASE6_CONTRAST_CANDIDATES),
            "n_folds": PHASE6_SELECTION_FOLDS,
        },
    },
}

METHODS = tuple(PHASE65_METHOD_REGISTRY)


def register_phase65_methods() -> None:
    """Make the variants visible to the runner's adapter factory."""

    for name, entry in PHASE65_METHOD_REGISTRY.items():
        from .manifest import METHOD_REGISTRY

        METHOD_REGISTRY.setdefault(name, entry)


def build_phase65_grids() -> tuple[GridSpec, ...]:
    """The four frozen Phase 6.5 grids."""

    return (
        GridSpec(
            grid="c_controls",
            purpose=(
                "Adversarial baseline controls on the income track: log "
                "geometry and a held-out-selected kernel bandwidth"
            ),
            dgps=INCOME_DGPS,
            n_train=(500, 1000),
            n_grid=(25,),
            n_particles=(10,),
            methods=TRACK_C_CONTROL_METHODS,
            seeds=PHASE65_SEEDS,
            notes=(
                "Every row pairs seed by seed with the frozen Phase 6 Track B "
                "record; R3 and cwdb_dr are reused from that record, not rerun."
            ),
        ),
        GridSpec(
            grid="c_scaling",
            purpose="Does the forest deficit shrink as the training bank grows?",
            dgps=SCALING_DGPS,
            n_train=(2000, 4000),
            n_grid=(25,),
            n_particles=(10,),
            methods=TRACK_C_SCALING_METHODS,
            seeds=SCALING_SEEDS,
            notes="Descriptive probe, five seeds, no gate attached.",
        ),
        GridSpec(
            grid="d_ablation",
            purpose="One realism ingredient switched off at a time",
            dgps=ABLATION_DGPS,
            n_train=(1000,),
            n_grid=(25,),
            n_particles=(10,),
            methods=TRACK_D_METHODS,
            seeds=PHASE65_SEEDS,
            notes=(
                "Base point is the frozen IC1 record at the same seeds; the "
                "paired difference against it names the load-bearing ingredient."
            ),
        ),
        GridSpec(
            grid="e_zi",
            purpose="Zero-inflated outcomes and the two-part assembly",
            dgps=ZERO_INFLATED_DGPS,
            n_train=(500, 1000),
            n_grid=(25,),
            n_particles=(10,),
            methods=TRACK_E_METHODS,
            seeds=PHASE65_SEEDS,
            notes=(
                "PTA-S excluded by design: it produces no law and cannot carry "
                "a zero-mass statement. Adding it would need a new frozen "
                "manifest, not an afterthought."
            ),
        ),
    )


def enumerate_phase65_cells() -> list[Cell]:
    cells: list[Cell] = []
    seen: set[str] = set()
    for grid in build_phase65_grids():
        for cell in grid.cells():
            if cell.key in seen:
                raise ValueError(f"duplicate cell key for {cell}")
            seen.add(cell.key)
            cells.append(cell)
    return cells


def build_phase65_manifest() -> dict[str, Any]:
    grids = build_phase65_grids()
    cells = enumerate_phase65_cells()
    from .repair import REPAIR_METHOD_REGISTRY

    document: dict[str, Any] = {
        "manifest_contract_id": PHASE65_CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "parent_manifest_contract_id": PARENT_CONTRACT_ID,
        "moderator_edges": list(MODERATOR_EDGES),
        "n_cells": len(cells),
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "boosting_budget": BOOSTING_BUDGET,
        "zero_mass_tolerance": PHASE65_ZERO_MASS_TOLERANCE,
        "bandwidth_candidates": [0.25, 0.5, 1.0, 2.0, 4.0],
        "selection_seeds": [100, 101],
        "evaluation_manifest": {
            **EVALUATION_MANIFEST,
            "functionals": list(EVALUATION_MANIFEST["functionals"]),
            "zero_mass_tolerance": PHASE65_ZERO_MASS_TOLERANCE,
        },
        "method_registry": {
            **PHASE65_METHOD_REGISTRY,
            "cwdb_r3_cvridge": REPAIR_METHOD_REGISTRY["cwdb_r3_cvridge"],
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


register_phase65_methods()
