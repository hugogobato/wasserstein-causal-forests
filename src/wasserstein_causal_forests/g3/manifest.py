"""Cell enumeration, hyperparameter registry, and manifest construction.

A tournament cell is one (grid, DGP, n, K, M, method, seed) combination. The
manifest enumerates every cell exactly once and gives each a stable key derived
from its own content, so a shard can be re-executed, split, or merged without a
central counter and a duplicate is detectable by key alone.

Hyperparameters live here rather than in the runner because the preregistration
freezes them: the registry below is the frozen budget, and the runner may only
look budgets up, never choose them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .dgps import DGP_IDS

MANIFEST_CONTRACT_ID = "G3-MAIN-v1"
ESTIMAND_CONTRACT_ID = "G0-WP0-A-v1"

#: Grid functionals every method is trained on. `grid_skewness` and
#: `grid_upper_tail_mean` are deliberately excluded: they are the unseen
#: functionals of the D7 transfer claim, evaluated but never trained on.
TRAINING_FUNCTIONALS = ("grid_mean", "grid_sd")

#: The roster the first G3 tournament ran. Frozen: rule 1 measures the claimant
#: against the best of these, and a repair variant added later must not be able
#: to move that reference by entering the pool.
FROZEN_G3_METHODS = (
    "cwdb_v1",
    "cwdb_v0",
    "cwdb_v1_noshrink",
    "sqw2_booster",
    "pta_s",
    "pta_f",
    "wdrft",
    "causal_drf",
)

#: Methods and their frozen constructor arguments. `role` separates the claimant
#: from its ablations and from the baselines it must beat.
METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "cwdb_v1": {
        "role": "claimant",
        "adapter": "cwdb",
        "produces_law": True,
        "parameters": {"architecture": "v1", "sharing": "partial", "arm_shrinkage": 5.0},
    },
    "cwdb_v0": {
        "role": "ablation",
        "adapter": "cwdb",
        "produces_law": True,
        "parameters": {"architecture": "v0", "sharing": "none", "arm_shrinkage": 5.0},
    },
    "cwdb_v1_noshrink": {
        "role": "ablation",
        "adapter": "cwdb",
        "produces_law": True,
        "parameters": {"architecture": "v1", "sharing": "partial", "arm_shrinkage": 0.0},
    },
    "sqw2_booster": {
        "role": "ablation",
        "adapter": "sqw2",
        "produces_law": True,
        "parameters": {},
    },
    "pta_s": {
        "role": "baseline",
        "adapter": "pta_s",
        "produces_law": False,
        "parameters": {},
    },
    "pta_f": {
        "role": "baseline",
        "adapter": "pta_f",
        "produces_law": False,
        "parameters": {},
    },
    "wdrft": {
        "role": "baseline",
        "adapter": "forest",
        "produces_law": True,
        "parameters": {"method": "wdrft"},
    },
    "causal_drf": {
        "role": "baseline",
        "adapter": "forest",
        "produces_law": True,
        "parameters": {"method": "causal_drf"},
    },
    "drf": {
        "role": "baseline",
        "adapter": "forest",
        "produces_law": True,
        "parameters": {"method": "drf"},
    },
}

#: Frozen boosting budget shared by every C-WDB variant and the comparator.
BOOSTING_BUDGET = {
    "n_estimators": 100,
    "learning_rate": 0.12,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "min_arm_leaf": 5,
    "collision_epsilon": 1e-3,
}

#: Frozen evaluation-manifest parameters. The tail event and mode radius are
#: fixed before any decisive run so no threshold can be chosen after seeing a
#: ranking.
EVALUATION_MANIFEST = {
    "manifest_id": "G3-EVAL-v1",
    "functionals": TRAINING_FUNCTIONALS,
    "tail_threshold": 1.5,
    "mode_radius": 1.0,
    "mode_mass_floor": 0.15,
    "collision_epsilon": 1e-3,
    "n_law_rows": 200,
}

N_TEST = 1000
TEST_SEED_OFFSET = 900_000

#: Metrics where a larger value is better. Everything else is an error.
HIGHER_IS_BETTER = ("mode_coverage",)

#: The primary law metric for gate rule 2. Fixed here, before any decisive run,
#: so it cannot be swapped for whichever metric happens to rank C-WDB best.
PRIMARY_LAW_METRIC = "kernel_law_error"

#: Number of paired standard errors a difference must clear to count as a win.
#: Two-sided, so this is roughly a 5 percent paired test across replications.
DECISION_SE_MULTIPLE = 2.0

#: Gate rules from the Phase G3 decision list, in machine-checkable form. Every
#: threshold here is frozen before the first decisive seed; the cost pilot ran
#: on a single regime at a seed outside the manifest and produced no ranking.
GATE_RULES: dict[str, Any] = {
    "rule_1_correctness": {
        "statement": "passes D0 through D2 correctness and null checks",
        "d0_max_mean_quantile_rmse": 0.15,
        "d2_max_mean_quantile_rmse": 0.15,
        "d2_max_false_effect_ratio": 1.25,
        "note": (
            "D0 is deterministic, so its conditional mean is recoverable; D2 "
            "has an exactly null effect, so any estimated contrast is false "
            "heterogeneity. The ratio compares C-WDB-v1's null-regime error "
            "against the best baseline's, so a shared difficulty is not "
            "charged to C-WDB alone."
        ),
    },
    "rule_2_law_advantage": {
        "statement": (
            "beats Causal-DRF on the primary law metric in at least two "
            "scientifically relevant mechanisms"
        ),
        "metric": PRIMARY_LAW_METRIC,
        "comparator": "causal_drf",
        "eligible_dgps": ["D1", "D5", "D6", "D7"],
        "min_wins": 2,
    },
    "rule_3_transfer": {
        "statement": (
            "transfers the advantage to at least one predeclared functional or "
            "reference target"
        ),
        "metrics": [
            "tcate_functional_rmse",
            "reference_tcate_rmse",
            "reference_effect_rmse",
        ],
        "comparator": "causal_drf",
        "eligible_dgps": ["D5", "D6", "D7"],
        "min_wins": 1,
    },
    "rule_4_beats_direct_learner": {
        "statement": "beats PTA-S on the transferred target of rule 3",
        "comparator": "pta_s",
        "min_wins": 1,
    },
    "rule_5_no_collapse": {
        "statement": "no systematic particle collapse or excessive projection",
        "d6_min_mode_coverage": 0.90,
        "min_effective_support_fraction": 0.60,
        "note": (
            "Effective support is the participation ratio of the particle "
            "weights, which is M for a spread law and 1 for a collapsed one; "
            "the floor is a fraction of M."
        ),
    },
    "rule_6_cost": {
        "statement": "compute cost commensurate with the gain",
        "max_runtime_ratio_to_causal_drf": 60.0,
        "note": (
            "Causal-DRF is the cheapest law-producing incumbent. A claimant "
            "costing more than this multiple of it must be justified by the "
            "size of its advantage, which the memo states explicitly."
        ),
    },
}


@dataclass(frozen=True)
class Cell:
    """One unit of tournament work."""

    grid: str
    dgp: str
    n_train: int
    n_grid: int
    n_particles: int
    method: str
    seed: int

    @property
    def key(self) -> str:
        """Content-addressed cell identifier, stable across shard layouts."""

        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def test_seed(self) -> int:
        """Test design seed. Shared by every method in the cell's replication,
        and disjoint from every training seed, so methods are compared on
        identical test points and never evaluated on their own training design.
        """

        return TEST_SEED_OFFSET + self.seed

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cell_key": self.key, "test_seed": self.test_seed}


@dataclass(frozen=True)
class GridSpec:
    """One preregistered sub-grid and the claim it serves."""

    grid: str
    purpose: str
    dgps: tuple[str, ...]
    n_train: tuple[int, ...]
    n_grid: tuple[int, ...]
    n_particles: tuple[int, ...]
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    notes: str = ""

    def cells(self) -> list[Cell]:
        return [
            Cell(self.grid, dgp, n, k, m, method, seed)
            for dgp in self.dgps
            for n in self.n_train
            for k in self.n_grid
            for m in self.n_particles
            for method in self.methods
            for seed in self.seeds
        ]


def _seeds(count: int) -> tuple[int, ...]:
    return tuple(range(count))


LAW_METHODS = ("cwdb_v1", "cwdb_v0", "wdrft", "causal_drf")


def build_grids() -> tuple[GridSpec, ...]:
    """The frozen G3 grid set.

    PTA-F appears only in `smallk`. Its cost accelerates in the target
    dimension D = K + J + 1, which the Phase 3 report measured at 20.4 s for
    D = 2, 27.4 s for D = 4, and 51.0 s for D = 8; the dense residual
    covariance makes D = 28 unaffordable within any safe compute budget. The
    Phase G3 computational notes anticipate this and cap PTA-F at D in
    {2, 4, 8}. Restricting it to K = 5 (so D = 8) keeps it in the tournament on
    a grid where every other method also runs, which is what a fair comparison
    needs; excluding it from the K = 25 grids is a declared cost limitation and
    is reported as such rather than as a loss.
    """

    return (
        GridSpec(
            grid="main",
            purpose="primary tournament at the working grid resolution",
            dgps=DGP_IDS,
            n_train=(500, 1000),
            n_grid=(25,),
            n_particles=(10,),
            methods=("cwdb_v1", "cwdb_v0", "wdrft", "causal_drf", "pta_s"),
            seeds=_seeds(20),
            notes=(
                "PTA-F excluded: at K = 25 its target dimension is D = 28, and "
                "its cost accelerates in D because of the dense residual "
                "covariance. PTA-S is included: it fits D independent scalar "
                "heads, so its cost is linear in D and the pilot measured it at "
                "1.14 s per head."
            ),
        ),
        GridSpec(
            grid="smallk",
            purpose="full roster including PTA-F, at D = K + J + 1 = 8",
            dgps=DGP_IDS,
            n_train=(500,),
            n_grid=(5,),
            n_particles=(10,),
            methods=(
                "cwdb_v1", "cwdb_v0", "sqw2_booster",
                "wdrft", "causal_drf", "pta_s", "pta_f",
            ),
            seeds=_seeds(20),
        ),
        GridSpec(
            grid="particles",
            purpose="claim 2: finite-particle approximation controlled",
            dgps=("D1", "D6"),
            n_train=(1000,),
            n_grid=(25,),
            n_particles=(2, 5, 10, 25),
            methods=("cwdb_v1", "sqw2_booster"),
            seeds=_seeds(20),
        ),
        GridSpec(
            grid="resolution",
            purpose="grid-resolution sensitivity of the law-level comparison",
            dgps=("D1", "D5", "D6", "D7"),
            n_train=(1000,),
            n_grid=(49,),
            n_particles=(10,),
            methods=LAW_METHODS,
            seeds=_seeds(10),
        ),
        GridSpec(
            grid="shrinkage",
            purpose="claim 6: causal regularization controls noise",
            dgps=("D2", "D8"),
            n_train=(1000,),
            n_grid=(25,),
            n_particles=(10,),
            methods=("cwdb_v1", "cwdb_v1_noshrink"),
            seeds=_seeds(20),
        ),
        GridSpec(
            grid="scaling",
            purpose="runtime and memory scaling at the largest sample size",
            dgps=("D1", "D4", "D6"),
            n_train=(2000,),
            n_grid=(25,),
            n_particles=(10,),
            methods=LAW_METHODS + ("pta_s",),
            seeds=_seeds(10),
        ),
    )


def enumerate_cells() -> list[Cell]:
    """Every cell in the frozen manifest, each appearing exactly once."""

    cells: list[Cell] = []
    seen: set[str] = set()
    for grid in build_grids():
        for cell in grid.cells():
            if cell.key in seen:
                raise ValueError(f"duplicate cell key for {cell}")
            seen.add(cell.key)
            cells.append(cell)
    return cells


def build_manifest(cost_pilot: dict[str, float] | None = None) -> dict[str, Any]:
    """The frozen manifest document, optionally carrying pilot cost estimates."""

    grids = build_grids()
    cells = enumerate_cells()
    document: dict[str, Any] = {
        "manifest_contract_id": MANIFEST_CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "n_cells": len(cells),
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "boosting_budget": BOOSTING_BUDGET,
        "evaluation_manifest": {
            **EVALUATION_MANIFEST,
            "functionals": list(EVALUATION_MANIFEST["functionals"]),
        },
        "method_registry": METHOD_REGISTRY,
        "primary_law_metric": PRIMARY_LAW_METRIC,
        "higher_is_better": list(HIGHER_IS_BETTER),
        "decision_se_multiple": DECISION_SE_MULTIPLE,
        "gate_rules": GATE_RULES,
        "grids": [
            {
                **{k: list(v) if isinstance(v, tuple) else v
                   for k, v in asdict(grid).items()},
                "n_cells": len(grid.cells()),
            }
            for grid in grids
        ],
        "cells": [cell.to_dict() for cell in cells],
    }
    if cost_pilot is not None:
        document["cost_pilot_seconds_per_cell"] = cost_pilot
        document["estimated_cpu_hours"] = round(
            sum(cost_pilot.get(cell.method, 0.0) for cell in cells) / 3600.0, 2
        )
    document["manifest_checksum"] = hashlib.sha256(
        json.dumps(document["cells"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return document
