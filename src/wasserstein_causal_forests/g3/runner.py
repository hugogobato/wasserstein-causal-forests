"""Execute manifest cells and write result rows.

One cell is one training fit and one evaluation against oracle truth. A cell
that raises is recorded as a failure row with its reason and is never retried
with a different seed: the manifest fixes which seeds exist, and quietly
replacing a seed that a method failed on would turn a robustness result into a
selection artefact.

Parallelism is at the cell level with one thread per worker. The energy kernels
and the BLAS calls in the metric layer would otherwise each try to use every
core, and the resulting oversubscription is slower than the serial code.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..pta_bcf.separate_heads import HeadBudget
from .dgps import build_dgp
from .evaluation import EvaluationManifest, evaluate
from .manifest import (
    BOOSTING_BUDGET,
    ESTIMAND_CONTRACT_ID,
    EVALUATION_MANIFEST,
    MANIFEST_CONTRACT_ID,
    METHOD_REGISTRY,
    N_TEST,
    TRAINING_FUNCTIONALS,
    Cell,
)
from .methods import (
    CWDBAdapter,
    ForestBaselineAdapter,
    PTAForcedAdapter,
    PTASeparateAdapter,
    SquaredW2BoosterAdapter,
)

# Imported for its side effect: the repair variants add themselves to
# METHOD_REGISTRY so `build_adapter` can resolve them in any worker process,
# whether the pool was forked or spawned. It cannot affect the frozen manifest,
# whose cells come from `build_grids()` alone.
from . import repair as _repair  # noqa: E402,F401  (side-effecting import)

# The Phase 5.5 variants register themselves the same way, including their
# imbalance regimes under the frozen outcome surfaces.
from . import phase55 as _phase55  # noqa: E402,F401  (side-effecting import)

# Phase 6 registers its variants, and the income realism track registers its
# IC regimes with whole-DGP builders so `build_dgp` resolves them by name.
from . import phase6 as _phase6  # noqa: E402,F401  (side-effecting import)
from . import phase6_dgps as _phase6_dgps  # noqa: E402,F401

# Phase 6.5 adds the control adapters, the two-part claimant, and the ablation
# and zero-inflated regimes.
from . import phase65 as _phase65  # noqa: E402,F401  (side-effecting import)
from . import phase65_dgps as _phase65_dgps  # noqa: E402,F401

_phase6_dgps.register_phase6_dgps()
_phase65_dgps.register_phase65_dgps()

SINGLE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_to_one_thread() -> None:
    """Cap every numerical library at one thread. Call before importing BLAS."""

    for variable in SINGLE_THREAD_VARIABLES:
        os.environ[variable] = "1"


def build_adapter(cell: Cell, cache_directory: Path | None):
    """Instantiate the frozen adapter for a cell's method."""

    entry = METHOD_REGISTRY[cell.method]
    kind = entry["adapter"]
    parameters = dict(entry["parameters"])
    if kind == "cwdb":
        return CWDBAdapter(
            n_particles=cell.n_particles, **BOOSTING_BUDGET, **parameters
        )
    if kind == "sqw2":
        budget = {k: v for k, v in BOOSTING_BUDGET.items() if k != "collision_epsilon"}
        return SquaredW2BoosterAdapter(
            n_particles=cell.n_particles, arm_shrinkage=5.0, **budget
        )
    if kind == "pta_s":
        return PTASeparateAdapter(budget=HeadBudget())
    if kind == "pta_f":
        return PTAForcedAdapter()
    if kind == "forest":
        return ForestBaselineAdapter(cache_directory=cache_directory, **parameters)
    if kind == "rmean":
        from .phase55_methods import RMetaLearnerAdapter

        return RMetaLearnerAdapter()
    if kind == "xmean":
        from .phase55_methods import XMetaLearnerAdapter

        return XMetaLearnerAdapter()
    if kind == "mutau":
        from .phase55_methods import MutauAdapter

        return MutauAdapter(
            n_particles=cell.n_particles, **BOOSTING_BUDGET, **parameters
        )
    if kind == "cwdb_dr":
        from .phase6_methods import DRAdapter

        return DRAdapter(
            n_particles=cell.n_particles, **BOOSTING_BUDGET, **parameters
        )
    if kind == "cwdb_smooth":
        from .phase6_methods import SmoothAdapter

        return SmoothAdapter(
            n_particles=cell.n_particles, **BOOSTING_BUDGET, **parameters
        )
    if kind == "cwdb_krr":
        from .phase6_methods import KRRAdapter

        return KRRAdapter(n_particles=cell.n_particles, **BOOSTING_BUDGET)
    if kind == "cwdb_frl":
        from .phase6_methods import FRLAdapter

        return FRLAdapter()
    if kind == "forest_log":
        from .phase65_methods import LogForestAdapter

        return LogForestAdapter(
            parameters["method"], cache_directory=cache_directory
        )
    if kind == "forest_retn":
        from .phase65_methods import RetunedCausalDRFAdapter

        return RetunedCausalDRFAdapter(cache_directory=cache_directory)
    if kind == "zipt":
        from .phase65_methods import ZIPTAdapter

        return ZIPTAdapter(
            n_particles=cell.n_particles, **BOOSTING_BUDGET, **parameters
        )
    raise ValueError(f"unknown adapter kind {kind!r}")


def evaluation_manifest(n_grid: int) -> EvaluationManifest:
    """The frozen evaluation manifest, resolved against a grid size."""

    settings = dict(EVALUATION_MANIFEST)
    return EvaluationManifest(
        manifest_id=settings["manifest_id"],
        functionals=tuple(settings["functionals"]),
        # The tail event is at the top grid coordinate whatever K is, so the
        # declared event means the same thing at every resolution.
        tail_level_index=n_grid - 1,
        tail_threshold=float(settings["tail_threshold"]),
        mode_radius=float(settings["mode_radius"]),
        mode_mass_floor=float(settings["mode_mass_floor"]),
        collision_epsilon=float(settings["collision_epsilon"]),
        n_law_rows=int(settings["n_law_rows"]),
        zero_mass_tolerance=float(settings.get("zero_mass_tolerance", 0.05)),
    )


def run_cell(
    cell: Cell,
    *,
    cache_directory: Path | None = None,
    manifest_contract_id: str = MANIFEST_CONTRACT_ID,
) -> list[dict[str, Any]]:
    """Execute one cell and return its result rows, successful or failed."""

    started = time.perf_counter()
    common = {
        **asdict(cell),
        "cell_key": cell.key,
        "test_seed": cell.test_seed,
        "manifest_contract_id": manifest_contract_id,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "evaluation_manifest_id": EVALUATION_MANIFEST["manifest_id"],
        "method_role": METHOD_REGISTRY[cell.method]["role"],
        "n_test": N_TEST,
    }
    try:
        dgp = build_dgp(cell.dgp, cell.n_grid)
        train = dgp.sample(cell.n_train, seed=cell.seed)
        test = dgp.sample(N_TEST, seed=cell.test_seed)
        adapter = build_adapter(cell, cache_directory)
        output = adapter.fit_predict(
            train, test.X, dgp, TRAINING_FUNCTIONALS, seed=cell.seed
        )
        rows = evaluate(
            output,
            dgp,
            test.X,
            evaluation_manifest(cell.n_grid),
            # Every method in this replication sees the same test design, so
            # they share one oracle truth.
            cache_key=(cell.test_seed,),
        )
    except Exception as error:  # noqa: BLE001 - a failed cell is a result
        reason = f"{type(error).__name__}: {error}"
        return [
            {
                **common,
                "metric": "cell_failure",
                "target_id": "NONE_OPERATIONAL",
                "arm": None,
                "detail": traceback.format_exc(limit=3)[-800:],
                "value": None,
                "status": "failed",
                "failure_reason": reason[:500],
                "wall_seconds": time.perf_counter() - started,
            }
        ]
    wall = time.perf_counter() - started
    return [{**common, **row, "wall_seconds": wall} for row in rows]


def run_shard(
    cells: list[Cell],
    output_path: Path,
    *,
    cache_directory: Path | None = None,
    log_path: Path | None = None,
    manifest_contract_id: str = MANIFEST_CONTRACT_ID,
) -> dict[str, Any]:
    """Run a shard of cells, checkpointing each one as it completes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    n_failed = 0
    started = time.perf_counter()
    for index, cell in enumerate(cells):
        cell_rows = run_cell(
            cell,
            cache_directory=cache_directory,
            manifest_contract_id=manifest_contract_id,
        )
        rows.extend(cell_rows)
        failed = cell_rows[0]["status"] == "failed"
        n_failed += int(failed)
        if log_path is not None:
            # Checkpoint before the next cell starts, so an interrupted shard
            # still reconciles against the manifest.
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "cell_key": cell.key,
                            **asdict(cell),
                            "status": "failed" if failed else "ok",
                            "wall_seconds": round(cell_rows[0]["wall_seconds"], 3),
                            "n_rows": len(cell_rows),
                            "finished_at": time.time(),
                        }
                    )
                    + "\n"
                )
        if (index + 1) % 20 == 0 or index + 1 == len(cells):
            write_rows(rows, output_path)
    write_rows(rows, output_path)
    return {
        "n_cells": len(cells),
        "n_failed": n_failed,
        "n_rows": len(rows),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "output": str(output_path),
    }


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write result rows to parquet, falling back to JSON Lines without pyarrow."""

    if not rows:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        path.with_suffix(".jsonl").write_text(
            "\n".join(json.dumps(row, default=_json_default) for row in rows) + "\n",
            encoding="utf-8",
        )
        return
    columns = sorted({key for row in rows for key in row})
    table = pa.table({name: [row.get(name) for row in rows] for name in columns})
    pq.write_table(table, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"cannot serialise {type(value)!r}")
