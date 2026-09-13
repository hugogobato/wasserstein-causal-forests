#!/usr/bin/env python3
"""Launcher for the isolated WCF sensitivity and assignment-DGP study.

    python research/run_wcf_sensitivity.py freeze [--blocks ...]
    python research/run_wcf_sensitivity.py pilot --dgp IC1 --seed 9999 [--pairs ...]
    python research/run_wcf_sensitivity.py run --workers N [--blocks ...] [--limit N]
    python research/run_wcf_sensitivity.py merge [--partial]
    python research/run_wcf_sensitivity.py status

Everything lives under `results/wcf_sensitivity/`; the historical Phase 6 and
6.5 manifests and result directories are never touched. Thread-pinning
variables are set before NumPy is imported, exactly as in
`research/run_phase6.py`. Cells checkpoint after every completion, so an
interrupted shard loses at most the cell in flight, and a failed cell is
retried only after its failure rows are archived for evidence.
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
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The Causal-DRF driver requires the authors' causal-clean package from the
# project-local library; same contract as the Phase 6.5 launcher.
_CAUSAL_LIB = ROOT / "results" / "Rlib" / "causal_drf"
if (_CAUSAL_LIB / "drf").is_dir():
    os.environ.setdefault("WCF_CAUSAL_DRF_R_LIB", str(_CAUSAL_LIB))

from wasserstein_causal_forests.g3.manifest import Cell  # noqa: E402
from wasserstein_causal_forests.g3.wcf_sensitivity import (  # noqa: E402
    INCOME_DGPS,
    INCOME_GRID,
    REQUIRED_LAW_METRICS,
    WCF_BLOCKS,
    WCF_PRIMARY_BLOCKS,
    WCF_SENSITIVITY_CONTRACT_ID,
    apply_method_registry,
    build_wcf_method_registry,
    build_wcf_sensitivity_cells,
    build_wcf_sensitivity_manifest,
)

RESULTS_DIRECTORY = ROOT / "results" / "wcf_sensitivity"
MANIFEST_PATH = RESULTS_DIRECTORY / "manifest.json"
SHARDS_DIRECTORY = RESULTS_DIRECTORY / "shards"
FAILURES_DIRECTORY = RESULTS_DIRECTORY / "failures"
LOGS_DIRECTORY = RESULTS_DIRECTORY / "logs"
MERGED_DIRECTORY = RESULTS_DIRECTORY / "merged"
MERGED_PATH = MERGED_DIRECTORY / "wcf_sensitivity_results.parquet"
CACHE_DIRECTORY = ROOT / "results" / "rcpp_cache"
CONFIG_PATH = ROOT / "configs" / "simulation_wcf_sensitivity.yaml"
COST_PILOT_PATH = RESULTS_DIRECTORY / "cost_pilot.json"

_SENSITIVITY_BLOCKS = frozenset(
    {"sym", "sym_null", "align", "align_null", "propensity", "flex_rf", "confirm"}
)
_INCOME_BLOCKS = frozenset({"income_onefactor", "income_baselines", "income_factorial"})
_INCOME_PAIRS = ((5, 10), (25, 10), (49, 10), (25, 5), (25, 25))
_REQUIRED_FIELDS = ("manifest_checksum", "estimator_source_hash", "contract_id")


def _set_stage(stage: str) -> None:
    """Redirect every path to `results/wcf_sensitivity/<stage>/`.

    The primary stage keeps its manifest checksum, so the Colab overflow
    sidecars stay verifiable. Additional blocks (null companions, factorial)
    freeze their own manifest in a sibling directory instead of changing the
    primary manifest under the running study.
    """

    global RESULTS_DIRECTORY, MANIFEST_PATH, SHARDS_DIRECTORY
    global FAILURES_DIRECTORY, LOGS_DIRECTORY, MERGED_DIRECTORY, MERGED_PATH
    global CONFIG_PATH, COST_PILOT_PATH

    stage = stage.strip().strip("/")
    if not stage:
        return
    RESULTS_DIRECTORY = ROOT / "results" / "wcf_sensitivity" / stage
    MANIFEST_PATH = RESULTS_DIRECTORY / "manifest.json"
    SHARDS_DIRECTORY = RESULTS_DIRECTORY / "shards"
    FAILURES_DIRECTORY = RESULTS_DIRECTORY / "failures"
    LOGS_DIRECTORY = RESULTS_DIRECTORY / "logs"
    MERGED_DIRECTORY = RESULTS_DIRECTORY / "merged"
    MERGED_PATH = MERGED_DIRECTORY / "wcf_sensitivity_results.parquet"
    CONFIG_PATH = ROOT / "configs" / f"simulation_wcf_sensitivity_{stage}.yaml"
    COST_PILOT_PATH = RESULTS_DIRECTORY / "cost_pilot.json"


def _import_runner():
    from wasserstein_causal_forests.g3 import runner

    return runner


def _register_sensitivity(blocks: tuple[str, ...], *, quiet: bool = False) -> None:
    """Register the companion DGP and method modules when they are available.

    Blocks that only need the historical income regimes and the Phase 6
    `cwdb_dr` adapter keep working before the companion modules land; blocks
    that depend on them fail here with a message naming the missing module.
    """

    names = set(blocks)
    needs_dgps = bool(names & _SENSITIVITY_BLOCKS)
    needs_methods = bool(names - _INCOME_BLOCKS)
    for module_name, register_name, required in (
        ("sensitivity_dgps", "register_sensitivity_dgps", needs_dgps),
        ("sensitivity_methods", "register_sensitivity_methods", needs_methods),
    ):
        try:
            module = __import__(
                f"wasserstein_causal_forests.g3.{module_name}", fromlist=[register_name]
            )
            getattr(module, register_name)()
        except ImportError as error:
            if required:
                raise SystemExit(
                    f"{module_name} is required for blocks {sorted(names)} but is "
                    f"unavailable: {error}"
                ) from error
            if not quiet:
                print(f"note: {module_name} not registered ({error})")


def _selected_blocks(text: str) -> tuple[str, ...]:
    if not text:
        return WCF_PRIMARY_BLOCKS
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in WCF_BLOCKS]
    if unknown:
        raise SystemExit(
            f"unknown blocks {unknown}; choose from {list(WCF_BLOCKS)}"
        )
    return names


def _write_rows(rows: list[dict], path: Path) -> None:
    _import_runner().write_rows(rows, path)


def _read_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pandas().to_dict("records")


def _verify_sidecar(path: Path, manifest: dict) -> None:
    sidecar = path.with_suffix(".meta.json")
    if not sidecar.exists():
        raise SystemExit(
            f"{path.name} has no sidecar {sidecar.name}; refusing to use it"
        )
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = {
        "manifest_checksum": manifest["manifest_checksum"],
        "estimator_source_hash": manifest.get("estimator_source_hash"),
        "contract_id": WCF_SENSITIVITY_CONTRACT_ID,
    }
    for field in _REQUIRED_FIELDS:
        if metadata.get(field) != expected[field]:
            raise SystemExit(
                f"refusing to append to {path.name}: sidecar {field} is "
                f"{metadata.get(field)!r}, current manifest has {expected[field]!r}"
            )


def _write_sidecar(path: Path, checksum: str, source_hash: str, contract_id: str) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "manifest_checksum": checksum,
            "estimator_source_hash": source_hash,
            "contract_id": contract_id,
        }
        for field in _REQUIRED_FIELDS:
            if existing.get(field) != expected[field]:
                raise SystemExit(
                    f"refusing to append to shard sidecar {path.name}: {field} is "
                    f"{existing.get(field)!r}, current run has {expected[field]!r}"
                )
    path.write_text(
        json.dumps(
            {
                "manifest_checksum": checksum,
                "estimator_source_hash": source_hash,
                "contract_id": contract_id,
                "updated_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _successful_keys(selected_keys: set[str], manifest: dict) -> set[str]:
    """Cells whose rows exist and contain no `cell_failure` row."""

    successful: set[str] = set()
    for path in sorted(SHARDS_DIRECTORY.glob("shard_*.parquet")):
        _verify_sidecar(path, manifest)
        by_cell: dict[str, list[dict]] = {}
        for row in _read_rows(path):
            by_cell.setdefault(row["cell_key"], []).append(row)
        for key, rows in by_cell.items():
            if key not in selected_keys:
                continue
            if not any(row.get("metric") == "cell_failure" for row in rows):
                successful.add(key)
    return successful


def _archive_and_purge_failures(retry_keys: set[str]) -> None:
    """Move the failure rows of cells about to be retried out of the shards."""

    moved: list[dict] = []
    for path in sorted(SHARDS_DIRECTORY.glob("shard_*.parquet")):
        by_cell: dict[str, list[dict]] = {}
        for row in _read_rows(path):
            by_cell.setdefault(row["cell_key"], []).append(row)
        retained: list[dict] = []
        for key, rows in by_cell.items():
            failed = any(row.get("metric") == "cell_failure" for row in rows)
            if failed and key in retry_keys:
                moved.extend(rows)
            else:
                retained.extend(rows)
        if len(retained) == sum(len(rows) for rows in by_cell.values()):
            continue
        if retained:
            temporary = path.with_suffix(".parquet.tmp")
            _write_rows(retained, temporary)
            os.replace(temporary, path)
        else:
            path.unlink()
    if not moved:
        return
    FAILURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    attempt = FAILURES_DIRECTORY / f"attempt_{stamp}.parquet"
    counter = 1
    while attempt.exists():
        attempt = FAILURES_DIRECTORY / f"attempt_{stamp}_{counter:02d}.parquet"
        counter += 1
    _write_rows(moved, attempt)
    print(f"archived {len(moved)} failure rows to {attempt}")


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


def _worker(payload: tuple) -> dict:
    (
        cell_dicts, index, contract_id, checksum, source_hash, registry, blocks,
    ) = payload
    runner = _import_runner()
    runner.pin_to_one_thread()
    apply_method_registry({"method_registry": registry})
    _register_sensitivity(tuple(blocks), quiet=True)
    from wasserstein_causal_forests.g3 import common_grid

    cells = [
        Cell(**{k: v for k, v in item.items() if k not in {"cell_key", "test_seed"}})
        for item in cell_dicts
    ]
    suffix = f"{index:03d}"
    shard_path = SHARDS_DIRECTORY / f"shard_{suffix}.parquet"
    _write_sidecar(
        SHARDS_DIRECTORY / f"shard_{suffix}.meta.json",
        checksum, source_hash, contract_id,
    )
    rows = _read_rows(shard_path) if shard_path.exists() else []
    log_path = LOGS_DIRECTORY / f"shard_{suffix}.jsonl"
    n_failed = 0
    started = time.perf_counter()
    for cell in cells:
        cell_rows = runner.run_cell(
            cell,
            cache_directory=CACHE_DIRECTORY,
            manifest_contract_id=contract_id,
            extra_evaluators=(common_grid.evaluate_common_grid,),
        )
        rows.extend(cell_rows)
        failed = cell_rows[0]["status"] == "failed"
        n_failed += int(failed)
        temporary = shard_path.with_suffix(".parquet.tmp")
        runner.write_rows(rows, temporary)
        os.replace(temporary, shard_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "cell_key": cell.key,
                        "grid": cell.grid,
                        "dgp": cell.dgp,
                        "n_train": cell.n_train,
                        "n_grid": cell.n_grid,
                        "n_particles": cell.n_particles,
                        "method": cell.method,
                        "seed": cell.seed,
                        "status": "failed" if failed else "ok",
                        "n_rows": len(cell_rows),
                        "wall_seconds": round(cell_rows[0].get("wall_seconds", 0.0), 3),
                        "finished_at": time.time(),
                    }
                )
                + "\n"
            )
    return {
        "suffix": suffix,
        "n_cells": len(cells),
        "n_failed": n_failed,
        "n_rows": len(rows),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "output": str(shard_path),
    }


def freeze(arguments: argparse.Namespace) -> int:
    blocks = _selected_blocks(arguments.blocks)
    _import_runner()
    _register_sensitivity(blocks)
    document = build_wcf_sensitivity_manifest(blocks)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(document, indent=2), encoding="utf-8")
    try:
        import yaml

        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
    except ImportError:
        print("PyYAML not installed; skipped the YAML rendering")
    print(f"wrote {MANIFEST_PATH}")
    print(f"  cells:    {document['n_cells']}")
    print(f"  checksum: {document['manifest_checksum']}")
    for block, count in document["block_counts"].items():
        print(f"  {block:<20} {count}")
    return 0


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit("manifest not frozen; run `freeze` first")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _common_grid_evaluators() -> tuple:
    try:
        from wasserstein_causal_forests.g3 import common_grid
    except ImportError as error:
        raise SystemExit(
            f"the common-grid evaluator is required to run cells but is "
            f"unavailable: {error}"
        ) from error
    evaluator = getattr(common_grid, "evaluate_common_grid", None)
    if evaluator is None:
        raise SystemExit("common_grid.evaluate_common_grid is missing")
    return (evaluator,)


def run(arguments: argparse.Namespace) -> int:
    document = _load_manifest()
    manifest_blocks = tuple(document.get("blocks", ()))
    blocks = (
        _selected_blocks(arguments.blocks) if arguments.blocks else manifest_blocks
    )
    missing_blocks = [name for name in blocks if name not in manifest_blocks]
    if missing_blocks:
        raise SystemExit(
            f"blocks {missing_blocks} are not in the frozen manifest "
            f"{list(manifest_blocks)}; re-freeze to add them"
        )
    apply_method_registry(document)
    _register_sensitivity(blocks)
    _common_grid_evaluators()

    selected = [dict(item) for item in document["cells"]]
    if arguments.blocks:
        wanted = {cell.key for cell in build_wcf_sensitivity_cells(blocks)}
        selected = [item for item in selected if item["cell_key"] in wanted]
    if getattr(arguments, "methods", ""):
        wanted_methods = {
            name.strip() for name in arguments.methods.split(",") if name.strip()
        }
        unknown = wanted_methods - set(document["method_registry"])
        if unknown:
            raise SystemExit(
                f"methods {sorted(unknown)} are not in the frozen registry"
            )
        selected = [
            item for item in selected if item["method"] in wanted_methods
        ]
    if getattr(arguments, "keys_file", ""):
        keys_path = Path(arguments.keys_file)
        if not keys_path.exists():
            raise SystemExit(f"keys file {keys_path} does not exist")
        wanted_keys = set(json.loads(keys_path.read_text(encoding="utf-8")))
        declared_keys = {item["cell_key"] for item in document["cells"]}
        unknown = wanted_keys - declared_keys
        if unknown:
            raise SystemExit(
                f"{len(unknown)} keys in {keys_path} are absent from the manifest"
            )
        selected = [
            item for item in selected if item["cell_key"] in wanted_keys
        ]
    selected_keys = {item["cell_key"] for item in selected}

    SHARDS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    successful = _successful_keys(selected_keys, document)
    retry_keys = selected_keys - successful
    _archive_and_purge_failures(retry_keys)

    remaining = [item for item in selected if item["cell_key"] not in successful]
    if arguments.limit:
        remaining = remaining[: arguments.limit]
    if not remaining:
        print("nothing to run")
        return 0

    workers = arguments.workers
    shards = _shard_by_replication(remaining, workers)
    active = [shard for shard in shards if shard]
    print(
        f"running {len(remaining)} cells across {len(active)} workers "
        f"({min(len(s) for s in active)}-{max(len(s) for s in active)} each)",
        flush=True,
    )
    payloads = [
        (
            shard,
            index,
            WCF_SENSITIVITY_CONTRACT_ID,
            document["manifest_checksum"],
            document.get("estimator_source_hash", ""),
            document["method_registry"],
            list(manifest_blocks),
        )
        for index, shard in enumerate(shards)
        if shard
    ]
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for summary in pool.map(_worker, payloads):
            summaries.append(summary)
            print(
                f"shard {summary['suffix']} done: {summary['n_cells']} cells, "
                f"{summary['n_failed']} failed, {summary['wall_seconds']}s",
                flush=True,
            )
    total_failed = sum(summary["n_failed"] for summary in summaries)
    print(f"\n{sum(s['n_cells'] for s in summaries)} cells, {total_failed} failed")
    return 0


def _law_metric_gaps(merged, registry: dict) -> list[dict]:
    """Missing REQUIRED_LAW_METRICS per observed (method, dgp, seed, n_train)."""

    law_methods = {
        name for name, entry in registry.items() if entry.get("produces_law")
    }
    subset = merged[
        merged["method"].isin(law_methods) & (merged["status"] == "ok")
    ]
    gaps: list[dict] = []
    for (method, dgp, seed, n_train), group in subset.groupby(
        ["method", "dgp", "seed", "n_train"]
    ):
        present = set(group["metric"])
        missing = [metric for metric in REQUIRED_LAW_METRICS if metric not in present]
        if missing:
            gaps.append(
                {
                    "method": method,
                    "dgp": dgp,
                    "seed": int(seed),
                    "n_train": int(n_train),
                    "missing": missing,
                }
            )
    return gaps


def merge(arguments: argparse.Namespace) -> int:
    """Union every shard into one parquet, refusing duplicates or gaps."""

    import pandas as pd
    import pyarrow.parquet as pq

    manifest = _load_manifest()
    declared = {item["cell_key"]: item for item in manifest["cells"]}
    shard_paths = sorted(SHARDS_DIRECTORY.glob("shard_*.parquet"))
    if not shard_paths:
        print("no shard files found")
        return 1

    frames = []
    seen: set[str] = set()
    n_failed_cells = 0
    test_seed_violations: list[dict] = []
    for path in shard_paths:
        _verify_sidecar(path, manifest)
        frame = pq.read_table(path).to_pandas()
        duplicates = seen & set(frame["cell_key"])
        if duplicates:
            raise SystemExit(
                f"duplicate cell keys across shards: {sorted(duplicates)[:4]}"
            )
        seen |= set(frame["cell_key"])
        n_failed_cells += int(
            frame.loc[frame["metric"] == "cell_failure", "cell_key"].nunique()
        )
        for key, group in frame.groupby(
            ["grid", "dgp", "n_train", "n_grid", "n_particles", "seed"]
        ):
            expected = int(manifest["test_seed_offset"]) + int(key[5])
            bad = group[group["test_seed"] != expected]
            if len(bad):
                test_seed_violations.append(
                    {
                        "cell_key": str(bad.iloc[0]["cell_key"]),
                        "expected": expected,
                        "observed": int(bad.iloc[0]["test_seed"]),
                    }
                )
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    unknown = seen - set(declared)
    missing = set(declared) - seen
    if unknown:
        raise SystemExit(
            f"{len(unknown)} rows are absent from the manifest: {sorted(unknown)[:4]}"
        )
    if missing and not arguments.partial:
        raise SystemExit(
            f"{len(missing)} manifest cells produced no rows; use --partial to "
            f"merge an incomplete study"
        )
    if test_seed_violations:
        raise SystemExit(f"test-seed violations: {test_seed_violations[:4]}")
    law_gaps = _law_metric_gaps(merged, manifest["method_registry"])
    if law_gaps:
        raise SystemExit(f"law-metric gaps: {law_gaps[:4]}")

    MERGED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_rows(merged.to_dict("records"), MERGED_PATH)
    audit = {
        "status": "PASS",
        "contract_id": WCF_SENSITIVITY_CONTRACT_ID,
        "manifest_checksum": manifest["manifest_checksum"],
        "estimator_source_hash": manifest.get("estimator_source_hash"),
        "partial": bool(arguments.partial),
        "n_shards": len(shard_paths),
        "n_cells_declared": len(declared),
        "n_cells_observed": len(seen),
        "n_cells_missing": len(missing),
        "missing_cells": sorted(missing)[:50],
        "n_failed_cells": n_failed_cells,
        "n_duplicate_keys": 0,
        "test_seed_violations": test_seed_violations[:50],
        "law_metric_gaps": law_gaps[:50],
    }
    (MERGED_DIRECTORY / "merge_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"wrote {MERGED_PATH}")
    print(
        f"  cells: {len(seen)} of {len(declared)}; failed cells: {n_failed_cells}; "
        f"partial: {bool(arguments.partial)}"
    )
    if missing:
        print(f"  missing cells tolerated by --partial: {len(missing)}")
    if n_failed_cells:
        print(f"  failed cells recorded in {MERGED_DIRECTORY / 'merge_audit.json'}")
    return 0


def status(_: argparse.Namespace) -> int:
    """Per-block declared, done, failed, and pending cell counts."""

    document = _load_manifest()
    declared_keys = {item["cell_key"] for item in document["cells"]}
    done: set[str] = set()
    failed: set[str] = set()
    if SHARDS_DIRECTORY.exists():
        for path in sorted(SHARDS_DIRECTORY.glob("shard_*.parquet")):
            by_cell: dict[str, list[dict]] = {}
            for row in _read_rows(path):
                by_cell.setdefault(row["cell_key"], []).append(row)
            for key, rows in by_cell.items():
                if key not in declared_keys:
                    continue
                if any(row.get("metric") == "cell_failure" for row in rows):
                    failed.add(key)
                else:
                    done.add(key)
    print(f"{'block':<20} {'declared':>8} {'done':>8} {'failed':>8} {'pending':>8}")
    for block in document.get("blocks", ()):
        keys = {cell.key for cell in build_wcf_sensitivity_cells([block])}
        n_declared = len(keys)
        n_done = len(keys & done)
        n_failed = len(keys & failed)
        n_pending = n_declared - n_done - n_failed
        print(
            f"{block:<20} {n_declared:>8} {n_done:>8} {n_failed:>8} {n_pending:>8}"
        )
    print(f"{'TOTAL':<20} {len(declared_keys):>8} {len(done):>8} "
          f"{len(failed):>8} {len(declared_keys) - len(done) - len(failed):>8}")
    return 0


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    if not text:
        return list(_INCOME_PAIRS)
    pairs = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        grid, separator, particles = token.lower().partition("x")
        if not separator:
            raise SystemExit(f"pair {token!r} is not of the form KxM")
        pairs.append((int(grid), int(particles)))
    return pairs


def _available_memory_mb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


_PILOT_CHILD = """\
import json
import os
import resource
import sys

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "R_NUM_THREADS",
):
    os.environ[_variable] = "1"

sys.path.insert(0, {source!r})

from pathlib import Path
from wasserstein_causal_forests.g3.manifest import Cell
from wasserstein_causal_forests.g3 import common_grid, runner
from wasserstein_causal_forests.g3.wcf_sensitivity import apply_method_registry

task = json.loads({payload!r})
runner.pin_to_one_thread()
apply_method_registry({{"method_registry": task["registry"]}})
cell = Cell(**task["cell"])
rows = runner.run_cell(
    cell,
    cache_directory=None if task["cache"] is None else Path(task["cache"]),
    manifest_contract_id=task["contract_id"],
    extra_evaluators=(common_grid.evaluate_common_grid,),
)
failed = bool(rows and rows[0].get("status") == "failed")
print("STATUS=" + ("failed" if failed else "ok"))
print("PEAK_RSS_MB=" + format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, ".3f"))
print("N_ROWS=" + str(len(rows)))
if failed:
    print("FAILURE_REASON=" + str(rows[0].get("failure_reason", ""))[:300])
"""


def _child_values(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def pilot(arguments: argparse.Namespace) -> int:
    """One-cell-per-method cost probe in fresh subprocesses; no shard writes."""

    _import_runner()
    if arguments.dgp not in INCOME_DGPS:
        raise SystemExit(
            f"--dgp must be one of {list(INCOME_DGPS)}, not {arguments.dgp!r}"
        )
    if MANIFEST_PATH.exists():
        document = _load_manifest()
        registry = document["method_registry"]
        contract_id = document.get("manifest_contract_id", WCF_SENSITIVITY_CONTRACT_ID)
    else:
        registry = build_wcf_method_registry()
        contract_id = WCF_SENSITIVITY_CONTRACT_ID
    pairs = _parse_pairs(arguments.pairs)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    for n_grid, n_particles in pairs:
        for method in ("cwdb_dr", "causal_drf"):
            if method not in registry:
                raise SystemExit(f"method {method!r} is missing from the registry")
            tasks.append(
                {
                    "registry": registry,
                    "contract_id": contract_id,
                    "cache": str(CACHE_DIRECTORY),
                    "cell": {
                        "grid": INCOME_GRID,
                        "dgp": arguments.dgp,
                        "n_train": 1000,
                        "n_grid": n_grid,
                        "n_particles": n_particles,
                        "method": method,
                        "seed": arguments.seed,
                    },
                }
            )

    def _run(task: dict) -> dict:
        script = _PILOT_CHILD.format(
            source=str(ROOT / "src"), payload=json.dumps(task)
        )
        cell = task["cell"]
        before = _available_memory_mb()
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        wall = time.perf_counter() - started
        after = _available_memory_mb()
        values = _child_values(completed.stdout)
        peak = values.get("PEAK_RSS_MB")
        peak_value = float(peak) if peak is not None else None
        status = values.get("STATUS", "failed" if completed.returncode else "ok")
        reason = values.get("FAILURE_REASON", "")
        if completed.returncode and not reason:
            reason = (
                completed.stderr.strip().splitlines()[-1]
                if completed.stderr.strip()
                else ""
            )
        record = {
            "dgp": arguments.dgp,
            "seed": arguments.seed,
            "method": cell["method"],
            "n_train": cell["n_train"],
            "n_grid": cell["n_grid"],
            "n_particles": cell["n_particles"],
            "wall_seconds": round(wall, 3),
            "peak_rss_mb": peak_value,
            "status": status,
            "failure_reason": reason,
            "returncode": completed.returncode,
            "mem_available_before_mb": before,
            "mem_available_after_mb": after,
        }
        peak_text = "nan" if peak_value is None else f"{peak_value:.0f}MB"
        print(
            f"  {record['method']:<10} K={record['n_grid']:<3} "
            f"M={record['n_particles']:<3} {record['status']:<6} "
            f"{record['wall_seconds']:7.2f}s peak={peak_text}",
            flush=True,
        )
        return record

    from concurrent.futures import ThreadPoolExecutor

    concurrency = max(1, int(getattr(arguments, "concurrency", 1) or 1))
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            runs = list(pool.map(_run, tasks))
    else:
        runs = [_run(task) for task in tasks]

    available = _available_memory_mb()
    peaks = [run["peak_rss_mb"] for run in runs if run["peak_rss_mb"]]
    max_peak = max(peaks) if peaks else 0.0
    cpu_count = os.cpu_count()
    if available is not None and max_peak > 0:
        recommended = min(6, max(1, int(0.7 * available / max_peak)))
    else:
        recommended = 1
    payload = {
        "contract_id": WCF_SENSITIVITY_CONTRACT_ID,
        "dgp": arguments.dgp,
        "seed": arguments.seed,
        "pairs": [list(pair) for pair in pairs],
        "runs": runs,
        "cpu_count": cpu_count,
        "available_ram_mb": available,
        "max_peak_worker_mb": max_peak,
        "safety_fraction": 0.7,
        "max_workers_cap": 6,
        "recommended_workers": recommended,
    }
    COST_PILOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_PILOT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {COST_PILOT_PATH}")
    print(f"  CPU count:                 {cpu_count}")
    print(f"  available RAM after pilot: {available}")
    print(f"  max peak worker RSS:       {max_peak}")
    print(f"  recommended workers:       {recommended}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="",
        help=(
            "optional stage directory under results/wcf_sensitivity/; empty is "
            "the primary frozen study. Use before the subcommand, e.g. "
            "`run_wcf_sensitivity.py --stage nulls freeze --blocks sym_null,align_null`"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--blocks", default="")

    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument("--dgp", default="IC1")
    pilot_parser.add_argument("--seed", type=int, default=9999)
    pilot_parser.add_argument("--pairs", default="")
    pilot_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="fresh subprocesses run at once; each stays single-threaded",
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workers", type=int, required=True)
    run_parser.add_argument("--blocks", default="")
    run_parser.add_argument(
        "--methods",
        default="",
        help="comma-separated method subset; empty runs every method in the block",
    )
    run_parser.add_argument(
        "--keys-file",
        default="",
        help="JSON list of cell keys to run; restricts the selected blocks",
    )
    run_parser.add_argument("--limit", type=int, default=0)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--partial", action="store_true")

    subparsers.add_parser("status")

    arguments = parser.parse_args()
    _set_stage(arguments.stage)
    if arguments.command == "freeze":
        return freeze(arguments)
    if arguments.command == "pilot":
        return pilot(arguments)
    if arguments.command == "run":
        return run(arguments)
    if arguments.command == "merge":
        return merge(arguments)
    return status(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
