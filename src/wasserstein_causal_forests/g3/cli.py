"""Command line entry points for the G3 tournament.

    python -m wasserstein_causal_forests.g3.cli freeze
    python -m wasserstein_causal_forests.g3.cli run --workers 6
    python -m wasserstein_causal_forests.g3.cli merge

`run` dispatches manifest cells across worker processes, each pinned to one
thread and each writing its own shard file and execution-log lines. Sharding by
worker rather than by task keeps the checkpoint granularity at one cell without
sending result rows between processes, and lets an interrupted run resume by
skipping cells whose keys are already in the log.
"""

from __future__ import annotations

import os

# Pin every numerical library to one thread BEFORE any of them is imported.
# OpenMP reads these at initialisation, so setting them after `import numpy` or
# `import stochtree` is silently ineffective: the pilot originally did that and
# spent 20 minutes on a cell that takes 32 seconds single-threaded, because
# eight OpenMP threads spin-waited on work too small to divide.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
    os.environ[_variable] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402

from .manifest import Cell, build_manifest, enumerate_cells  # noqa: E402
from .runner import pin_to_one_thread, run_shard  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "results" / "manifests" / "main_manifest.json"
COST_PILOT_PATH = ROOT / "results" / "manifests" / "cost_pilot.json"
EXECUTION_LOG = ROOT / "results" / "manifests" / "execution_log.jsonl"
RESULTS_DIRECTORY = ROOT / "results" / "main"
CACHE_DIRECTORY = ROOT / "results" / "rcpp_cache"

#: The repair track writes beside the frozen one and never into it. Its shards,
#: execution log, and manifest are separate files, so re-running the repair can
#: never overwrite or invalidate a result the G3 memo already reports.
REPAIR_MANIFEST_PATH = ROOT / "results" / "manifests" / "repair_manifest.json"
REPAIR_EXECUTION_LOG = ROOT / "results" / "manifests" / "repair_execution_log.jsonl"
REPAIR_RESULTS_DIRECTORY = ROOT / "results" / "repair"

#: The Phase 5.5 track, isolated from both earlier tracks so that a rerun or a
#: re-freeze can never touch a frozen or repaired result row.
PHASE55_MANIFEST_PATH = ROOT / "results" / "manifests" / "phase55_manifest.json"
PHASE55_EXECUTION_LOG = ROOT / "results" / "manifests" / "phase55_execution_log.jsonl"
PHASE55_RESULTS_DIRECTORY = ROOT / "results" / "phase55"
PHASE55_CONFIG_PATH = ROOT / "configs" / "simulation_phase55.yaml"

#: Stage 2 writes beside Stage 1 and never into it, for the same reason the
#: Phase 5.5 track sits beside the repair track: a stage that can overwrite the
#: rows an earlier memo reports is not a stage, it is a revision.
STAGE2_MANIFEST_PATH = ROOT / "results" / "manifests" / "phase55_stage2_manifest.json"
STAGE2_EXECUTION_LOG = (
    ROOT / "results" / "manifests" / "phase55_stage2_execution_log.jsonl"
)
STAGE2_RESULTS_DIRECTORY = ROOT / "results" / "phase55_stage2"
STAGE2_CONFIG_PATH = ROOT / "configs" / "simulation_phase55_stage2.yaml"


def _track_paths(track: str) -> tuple[Path, Path, Path]:
    """Manifest, execution log, and shard directory for a track."""

    if track == "repair":
        return REPAIR_MANIFEST_PATH, REPAIR_EXECUTION_LOG, REPAIR_RESULTS_DIRECTORY
    if track == "phase55":
        return PHASE55_MANIFEST_PATH, PHASE55_EXECUTION_LOG, PHASE55_RESULTS_DIRECTORY
    if track == "phase55_stage2":
        return STAGE2_MANIFEST_PATH, STAGE2_EXECUTION_LOG, STAGE2_RESULTS_DIRECTORY
    return MANIFEST_PATH, EXECUTION_LOG, RESULTS_DIRECTORY


def _pilot_costs() -> dict[str, float] | None:
    if not COST_PILOT_PATH.exists():
        return None
    document = json.loads(COST_PILOT_PATH.read_text(encoding="utf-8"))
    costs: dict[str, list[float]] = {}
    for row in document["measurements"]:
        if row["status"] == "ok":
            costs.setdefault(row["method"], []).append(float(row["wall_seconds"]))
    # The manifest's estimate multiplies a per-method cost by that method's cell
    # count, so the representative value must be the mean over the shapes the
    # method actually runs at, not its cheapest one.
    return {method: float(np.mean(values)) for method, values in costs.items()}


def freeze(_: argparse.Namespace) -> int:
    """Write the frozen manifest, including pilot cost estimates when present."""

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = build_manifest(cost_pilot=_pilot_costs())
    MANIFEST_PATH.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")
    print(f"  cells:    {document['n_cells']}")
    print(f"  checksum: {document['manifest_checksum']}")
    if "estimated_cpu_hours" in document:
        print(f"  estimated CPU hours: {document['estimated_cpu_hours']}")
    return 0


def _completed_keys(log_path: Path) -> set[str]:
    """Cell keys already finished, across every per-worker log.

    Workers each append to their own `<stem>_NNN.jsonl`, so a resume that read
    only the single-file path would see nothing and rerun the whole manifest.
    """

    keys: set[str] = set()
    for path in log_path.parent.glob(f"{log_path.stem}*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(json.loads(line)["cell_key"])
    return keys


def shard_suffix(index: int, tag: str = "") -> str:
    """Filename suffix for one worker's shard and execution log.

    A shard file is overwritten wholesale by the worker that owns it, so a
    staged run must not reuse an earlier stage's filenames: under `--resume` the
    second run holds only the new cells, and writing them to `shard_000.parquet`
    would delete the first stage's results. A tag keeps the stages in separate
    files, and both the merge and the resume scan glob across them.
    """

    return f"{tag}_{index:03d}" if tag else f"{index:03d}"


def _worker(payload: tuple[list[dict], int, str, str]) -> dict:
    cell_dicts, index, track, tag = payload
    pin_to_one_thread()
    _, log_path, results_directory = _track_paths(track)
    contract = {
        "main": "G3-MAIN-v1",
        "repair": "G3-REPAIR-v1",
        "phase55": "G3-PHASE55-v1",
        "phase55_stage2": "G3-PHASE55-v1",
    }[track]
    cells = [
        Cell(**{k: v for k, v in item.items()
                if k not in {"cell_key", "test_seed"}})
        for item in cell_dicts
    ]
    suffix = shard_suffix(index, tag)
    return run_shard(
        cells,
        results_directory / f"shard_{suffix}.parquet",
        cache_directory=CACHE_DIRECTORY,
        log_path=log_path.with_name(f"{log_path.stem}_{suffix}.jsonl"),
        manifest_contract_id=contract,
    )


def _shard_by_replication(cells: list[dict], workers: int) -> list[list[dict]]:
    """Split cells into shards, keeping each replication whole.

    A replication is one (grid, dgp, n_train, n_grid, n_particles, seed) with
    every method attached. Keeping it in one worker lets the methods share the
    cached oracle truth, which costs more than most fits. Replications are then
    dealt round-robin rather than in contiguous blocks, because per-method costs
    span an order of magnitude and a block partition would leave one worker
    holding every PTA-F cell.
    """

    groups: dict[tuple, list[dict]] = {}
    for cell in cells:
        key = (
            cell["grid"], cell["dgp"], cell["n_train"],
            cell["n_grid"], cell["n_particles"], cell["seed"],
        )
        groups.setdefault(key, []).append(cell)
    shards: list[list[dict]] = [[] for _ in range(workers)]
    for index, key in enumerate(sorted(groups)):
        shards[index % workers].extend(groups[key])
    return shards


def run(arguments: argparse.Namespace) -> int:
    """Dispatch the manifest across worker processes."""

    from concurrent.futures import ProcessPoolExecutor

    from . import r_bridge

    track = getattr(arguments, "track", "main")
    manifest_path, execution_log, results_directory = _track_paths(track)
    if not manifest_path.exists():
        print("manifest not frozen; run `freeze` first", file=sys.stderr)
        return 1
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = [dict(item) for item in document["cells"]]
    if arguments.grid:
        wanted = set(arguments.grid.split(","))
        cells = [item for item in cells if item["grid"] in wanted]
    if getattr(arguments, "dgp", ""):
        wanted = set(arguments.dgp.split(","))
        cells = [item for item in cells if item["dgp"] in wanted]
    if getattr(arguments, "method", ""):
        wanted = set(arguments.method.split(","))
        cells = [item for item in cells if item["method"] in wanted]
    # Resume is on by default for the repair track: it runs in stages, and the
    # point of the track is that nothing already computed is computed again.
    if arguments.resume:
        done = _completed_keys(execution_log)
        cells = [item for item in cells if item["cell_key"] not in done]
    if arguments.limit:
        cells = cells[: arguments.limit]
    if not cells:
        print("nothing to run")
        return 0

    results_directory.mkdir(parents=True, exist_ok=True)
    execution_log.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Causal-DRF now uses the authors' causal-clean package through the
    # original-code driver, so no project-local Rcpp translation unit is built.

    workers = arguments.workers
    shards = _shard_by_replication(cells, workers)
    print(
        f"running {len(cells)} cells across {workers} workers "
        f"({min(len(s) for s in shards)}-{max(len(s) for s in shards)} cells each)",
        flush=True,
    )

    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for summary in pool.map(
            _worker,
            [
                (shard, index, track, arguments.shard_tag)
                for index, shard in enumerate(shards)
                if shard
            ],
        ):
            summaries.append(summary)
            print(
                f"shard done: {summary['n_cells']} cells, "
                f"{summary['n_failed']} failed, {summary['wall_seconds']}s",
                flush=True,
            )

    total_failed = sum(s["n_failed"] for s in summaries)
    print(
        f"\n{sum(s['n_cells'] for s in summaries)} cells, {total_failed} failed, "
        f"{sum(s['n_rows'] for s in summaries)} rows"
    )
    return 0


def freeze_phase55(arguments: argparse.Namespace) -> int:
    """Write the frozen Phase 5.5 manifest, as JSON for the runner and as YAML
    in `configs/simulation_phase55.yaml` for the record.

    The YAML is a rendering of the same document, so the manifest checksum in
    the two files always agrees.
    """

    from .phase55 import PHASE55_METHODS, build_phase55_manifest

    document = build_phase55_manifest(
        methods=PHASE55_METHODS,
        include_imbalance=True,
    )
    PHASE55_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHASE55_MANIFEST_PATH.write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    PHASE55_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a declared dependency
        print("PyYAML not installed; skipping the YAML rendering", file=sys.stderr)
    else:
        PHASE55_CONFIG_PATH.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
    print(f"wrote {PHASE55_MANIFEST_PATH}")
    print(f"wrote {PHASE55_CONFIG_PATH}")
    print(f"  cells:    {document['n_cells']}")
    print(f"  stage:    {document['stage']}")
    print(f"  methods:  {', '.join(document['method_registry'])}")
    print(f"  checksum: {document['manifest_checksum']}")
    return 0


def freeze_phase55_stage2(_: argparse.Namespace) -> int:
    """Write the frozen Phase 5.5 Stage 2 manifest and its YAML rendering."""

    from .phase55 import build_phase55_stage2_manifest

    document = build_phase55_stage2_manifest()
    STAGE2_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE2_MANIFEST_PATH.write_text(json.dumps(document, indent=2), encoding="utf-8")
    STAGE2_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a declared dependency
        print("PyYAML not installed; skipping the YAML rendering", file=sys.stderr)
    else:
        STAGE2_CONFIG_PATH.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
    print(f"wrote {STAGE2_MANIFEST_PATH}")
    print(f"wrote {STAGE2_CONFIG_PATH}")
    print(f"  cells:    {document['n_cells']}")
    print(f"  stage:    {document['stage']}")
    print(f"  methods:  {', '.join(document['method_registry'])}")
    for grid in document["grids"]:
        print(
            f"  grid:     {grid['purpose']} "
            f"({len(grid['dgps'])} dgps x {len(grid['seeds'])} seeds "
            f"= {grid['n_cells']} cells)"
        )
    print(f"  stage 1:  {document['stage1_manifest_checksum']}")
    print(f"  checksum: {document['manifest_checksum']}")
    return 0


def merge(arguments: argparse.Namespace) -> int:
    from .merge import merge_results

    track = getattr(arguments, "track", "main")
    manifest_path, _, results_directory = _track_paths(track)
    destination = ROOT / "results" / {
        "main": "merged",
        "repair": "merged_repair",
        "phase55": "merged_phase55",
        "phase55_stage2": "merged_phase55_stage2",
    }[track]
    audit = merge_results(results_directory, manifest_path, destination)
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "PASS" else 1


def freeze_repair(arguments: argparse.Namespace) -> int:
    """Write the repair manifest for a chosen stage.

    Staged on purpose. Stage one is D2 alone, because rule 1 is the only rule
    the repair exists to fix and a variant that does not fix it should cost
    nothing further. Stage two widens to every regime, for variants that did.
    """

    from .repair import (
        REPAIR_METHODS,
        REPAIR_STAGES,
        build_repair_manifest,
        build_staged_repair_manifest,
    )

    if arguments.stage:
        # The union of every stage up to the one named. A stage that stopped
        # early keeps its cells here, so the merge reconciles against all of
        # them rather than reporting a screened-out variant's rows as unknown.
        stages = tuple(s for s in REPAIR_STAGES if s.stage <= arguments.stage)
        if not stages:
            print(f"no stage numbered {arguments.stage}", file=sys.stderr)
            return 1
        document = build_staged_repair_manifest(stages)
        described = f"stages 1-{arguments.stage}"
    else:
        dgps = tuple(arguments.dgps.split(",")) if arguments.dgps else None
        methods = (
            tuple(arguments.methods.split(",")) if arguments.methods else REPAIR_METHODS
        )
        document = build_repair_manifest(dgps, methods, tuple(arguments.grids.split(",")))
        described = f"regimes {'all' if dgps is None else ', '.join(dgps)}"

    REPAIR_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPAIR_MANIFEST_PATH.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"wrote {REPAIR_MANIFEST_PATH}")
    print(f"  cells:    {document['n_cells']}")
    print(f"  scope:    {described}")
    print(f"  methods:  {', '.join(document['method_registry'])}")
    print(f"  checksum: {document['manifest_checksum']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="g3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("freeze").set_defaults(handler=freeze)

    repair_freeze = subparsers.add_parser("freeze-repair")
    repair_freeze.add_argument(
        "--stage",
        type=int,
        default=0,
        help="freeze the union of every declared stage up to this one",
    )
    repair_freeze.add_argument("--dgps", type=str, default="")
    repair_freeze.add_argument("--methods", type=str, default="")
    repair_freeze.add_argument("--grids", type=str, default="main")
    repair_freeze.set_defaults(handler=freeze_repair)

    runner = subparsers.add_parser("run")
    runner.add_argument("--workers", type=int, default=6)
    runner.add_argument("--grid", type=str, default="")
    runner.add_argument("--dgp", type=str, default="")
    runner.add_argument("--method", type=str, default="")
    runner.add_argument("--limit", type=int, default=0)
    runner.add_argument("--resume", action="store_true")
    runner.add_argument("--track", choices=("main", "repair", "phase55", "phase55_stage2"), default="main")
    runner.add_argument(
        "--shard-tag",
        dest="shard_tag",
        type=str,
        default="",
        help="distinct shard and log filenames for a staged run",
    )
    runner.set_defaults(handler=run)

    merger = subparsers.add_parser("merge")
    merger.add_argument("--track", choices=("main", "repair", "phase55", "phase55_stage2"), default="main")
    merger.set_defaults(handler=merge)

    subparsers.add_parser("freeze-phase55").set_defaults(handler=freeze_phase55)
    subparsers.add_parser("freeze-phase55-stage2").set_defaults(
        handler=freeze_phase55_stage2
    )

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
