#!/usr/bin/env python3
"""Launcher for the Phase 6.5 tracks (controls + ablation + zero inflation).

    python research/run_phase65.py freeze
    python research/run_phase65.py run --workers 10 [--track c|d|e] [--limit N]
    python research/run_phase65.py merge

Same discipline as `research/run_phase6.py`: thread-pinning environment
variables are set before anything imports NumPy. Cells resume by key from the
per-worker execution logs; a failed cell is recorded as failed and never
retried under a different seed. The decisive retune cells additionally require
`results/manifests/phase65_bandwidth_selection.json`, which the selection pilot
writes.
"""

from __future__ import annotations

import os
import sys

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "R_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.manifest import Cell  # noqa: E402
from wasserstein_causal_forests.g3.phase65 import (  # noqa: E402
    PHASE65_CONTRACT_ID,
    build_phase65_manifest,
)
from wasserstein_causal_forests.g3.runner import (  # noqa: E402
    pin_to_one_thread,
    run_shard,
    write_rows,
)

MANIFEST_PATH = ROOT / "results" / "manifests" / "phase65_manifest.json"
EXECUTION_LOG = ROOT / "results" / "manifests" / "phase65_execution_log.jsonl"
RESULTS_DIRECTORY = ROOT / "results" / "phase65"
CACHE_DIRECTORY = ROOT / "results" / "rcpp_cache"
MERGED_PATH = ROOT / "results" / "merged_phase65" / "phase65_results.parquet"

TRACK_GRIDS = {
    "c": {"c_controls", "c_scaling"},
    "d": {"d_ablation"},
    "e": {"e_zi"},
}


def _completed_keys() -> set[str]:
    keys: set[str] = set()
    if not EXECUTION_LOG.parent.exists():
        return keys
    for path in EXECUTION_LOG.parent.glob(f"{EXECUTION_LOG.stem}*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(json.loads(line)["cell_key"])
    return keys


def _worker(payload: tuple[list[dict], int]) -> dict:
    cell_dicts, index = payload
    pin_to_one_thread()
    cells = [
        Cell(**{k: v for k, v in item.items() if k not in {"cell_key", "test_seed"}})
        for item in cell_dicts
    ]
    # A per-launch tag keeps successive invocations from overwriting one
    # another's shard files wholesale.
    tag = os.environ.get("PHASE65_SHARD_TAG", "")
    suffix = f"{tag}_{index:03d}" if tag else f"{index:03d}"
    return run_shard(
        cells,
        RESULTS_DIRECTORY / f"shard_{suffix}.parquet",
        cache_directory=CACHE_DIRECTORY,
        log_path=EXECUTION_LOG.with_name(f"{EXECUTION_LOG.stem}_{suffix}.jsonl"),
        manifest_contract_id=PHASE65_CONTRACT_ID,
    )


def _shard_by_replication(cells: list[dict], workers: int) -> list[list[dict]]:
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


def freeze(_: argparse.Namespace) -> int:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = build_phase65_manifest()
    MANIFEST_PATH.write_text(json.dumps(document, indent=2), encoding="utf-8")
    try:
        import yaml

        (ROOT / "configs" / "simulation_phase65.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
    except ImportError:
        print("PyYAML not installed; skipped the YAML rendering")
    print(f"wrote {MANIFEST_PATH}")
    print(f"  cells:    {document['n_cells']}")
    print(f"  checksum: {document['manifest_checksum']}")
    return 0


def run(arguments: argparse.Namespace) -> int:
    if not MANIFEST_PATH.exists():
        print("manifest not frozen; run `freeze` first", file=sys.stderr)
        return 1
    if arguments.track and arguments.track == "c":
        selection = ROOT / "results" / "manifests" / \
            "phase65_bandwidth_selection.json"
        if not selection.exists():
            print(
                "the bandwidth-selection document does not exist; run the "
                "selection pilot before any decisive retune cell",
                file=sys.stderr,
            )
            return 1
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cells = [dict(item) for item in document["cells"]]
    if arguments.track:
        keep = TRACK_GRIDS[arguments.track]
        cells = [item for item in cells if item["grid"] in keep]
    if arguments.dgp:
        wanted = set(arguments.dgp.split(","))
        cells = [item for item in cells if item["dgp"] in wanted]
    done = _completed_keys()
    cells = [item for item in cells if item["cell_key"] not in done]
    if arguments.limit:
        cells = cells[: arguments.limit]
    if not cells:
        print("nothing to run")
        return 0

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    workers = arguments.workers
    shards = _shard_by_replication(cells, workers)
    print(
        f"running {len(cells)} cells across {workers} workers "
        f"({min(len(s) for s in shards)}-{max(len(s) for s in shards)} each)",
        flush=True,
    )
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for summary in pool.map(
            _worker,
            [(shard, i) for i, shard in enumerate(shards) if shard],
        ):
            summaries.append(summary)
            print(
                f"shard done: {summary['n_cells']} cells, "
                f"{summary['n_failed']} failed, {summary['wall_seconds']}s",
                flush=True,
            )
    total_failed = sum(s["n_failed"] for s in summaries)
    print(
        f"\n{sum(s['n_cells'] for s in summaries)} cells, {total_failed} failed"
    )
    return 0


def merge(_: argparse.Namespace) -> int:
    """Union every shard into one parquet, refusing duplicates or gaps."""

    import pyarrow.parquet as pq
    import pandas as pd

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    declared = {item["cell_key"] for item in manifest["cells"]}
    frames = []
    seen: set[str] = set()
    n_failed = 0
    shard_paths = sorted(
        list(RESULTS_DIRECTORY.glob("shard_*.parquet"))
        + list((RESULTS_DIRECTORY / "colab_shards").glob("shard_*.parquet"))
    )
    for path in shard_paths:
        frame = pq.read_table(path).to_pandas()
        duplicates = seen & set(frame["cell_key"])
        if duplicates:
            raise SystemExit(
                f"duplicate cell keys across shards: {sorted(duplicates)[:4]}"
            )
        seen |= set(frame["cell_key"])
        n_failed += int((frame["metric"] == "cell_failure").sum())
        frames.append(frame)
    if not frames:
        print("no shard files found")
        return 1
    merged = pd.concat(frames, ignore_index=True)
    missing = declared - seen
    unknown = seen - declared
    if missing:
        raise SystemExit(f"{len(missing)} manifest cells produced no rows")
    if unknown:
        raise SystemExit(f"{len(unknown)} rows are absent from the manifest")
    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_rows(merged.to_dict("records"), MERGED_PATH)
    print(f"wrote {MERGED_PATH}")
    print(f"  cells: {len(seen)} of {len(declared)}; failed cells: {n_failed}")
    audit = {
        "manifest_checksum": manifest["manifest_checksum"],
        "n_cells_declared": len(declared),
        "n_cells_observed": len(seen),
        "n_failed_cells": n_failed,
        "duplicate_keys": 0,
    }
    (MERGED_PATH.parent / "merge_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("freeze")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=8)
    run_parser.add_argument("--track", choices=("c", "d", "e"), default="")
    run_parser.add_argument("--dgp", default="")
    run_parser.add_argument("--limit", type=int, default=0)

    subparsers.add_parser("merge")

    arguments = parser.parse_args()
    if arguments.command == "freeze":
        return freeze(arguments)
    if arguments.command == "run":
        return run(arguments)
    return merge(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
