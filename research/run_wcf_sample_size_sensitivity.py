#!/usr/bin/env python3
"""Run the WCF sample-size sensitivity experiment.

The experiment reuses the paper's selected designs and estimator registry but
stores its own manifest and results under ``results/wcf_sample_size_sensitivity``.
The source DGP identifiers are kept in the manifest, together with their paper
labels, so that the experiment can be audited against the manuscript without
duplicating or modifying the DGP implementation.

Typical use from the repository root is::

    rtk python3 research/run_wcf_sample_size_sensitivity.py freeze
    rtk python3 research/run_wcf_sample_size_sensitivity.py run --workers 4
    rtk python3 research/run_wcf_sample_size_sensitivity.py merge
    rtk python3 research/run_wcf_sample_size_sensitivity.py summarize

The default sizes are 125, 250, 500, and 1000 observations. The last two
overlap the principal study, while 125 and 250 probe settings in which the
treated and control arms contain substantially fewer observations. Every
design uses ten paired Monte Carlo seeds, ``K=25``, ``M=10``, and an independent
test sample of 1000 units. Use ``--n-values`` and ``--seeds`` at ``freeze``
time to create a clearly distinct manifest for another design.

Each worker is single-threaded and checkpoints after every cell. Existing
results are never overwritten silently: a manifest is immutable once a run
has started, and a sidecar records the manifest checksum, estimator source
hash, and contract identifier for every shard. Failed cells remain explicit
rows and are retried only on a subsequent run with the same manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Numerical libraries must be pinned before importing NumPy through the WCF
# package. This also keeps a four-worker local run from oversubscribing BLAS.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "R_NUM_THREADS",
):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3 import runner  # noqa: E402
from wasserstein_causal_forests.g3.manifest import (  # noqa: E402
    BOOSTING_BUDGET,
    ESTIMAND_CONTRACT_ID,
    EVALUATION_MANIFEST,
    N_TEST,
    TEST_SEED_OFFSET,
    TRAINING_FUNCTIONALS,
    Cell,
)
from wasserstein_causal_forests.g3.wcf_sensitivity import (  # noqa: E402
    WCF_SENSITIVITY_CONTRACT_ID,
    apply_method_registry,
    build_wcf_method_registry,
    estimator_source_hash,
    software_revision,
)

CONTRACT_ID = "WCF-SAMPLE-SIZE-v1"
ALL_DGP_CONTRACT_ID = "WCF-SAMPLE-SIZE-ALL-v1"
GRID_ID = "wcf_sample_size_v1_logit_f3"
RESULTS_DIRECTORY = ROOT / "results" / "wcf_sample_size_sensitivity"
MANIFEST_PATH = RESULTS_DIRECTORY / "manifest.json"
SHARDS_DIRECTORY = RESULTS_DIRECTORY / "shards"
FAILURES_DIRECTORY = RESULTS_DIRECTORY / "failures"
LOGS_DIRECTORY = RESULTS_DIRECTORY / "logs"
MERGED_DIRECTORY = RESULTS_DIRECTORY / "merged"
MERGED_PATH = MERGED_DIRECTORY / "wcf_sample_size_results.parquet"
CACHE_DIRECTORY = ROOT / "results" / "rcpp_cache"

# The paper's labels and the source identifiers used by the frozen DGP
# implementation. The family-specific method roster is part of the manifest:
# ordinary WCF is never instantiated on a structural-zero design.
PAPER_DGPS_MAIN: dict[str, str] = {
    "LS1": "IC1",
    "LS3": "IC3",
    "S3": "D5",
    "S4": "D6",
    "S5": "D7",
}
PAPER_DGPS_ALL: dict[str, str] = {
    "LS0": "IC0",
    "LS1": "IC1",
    "LS2": "IC2",
    "LS3": "IC3",
    "S1": "D0",
    "S2": "D2",
    "S3": "D5",
    "S4": "D6",
    "S5": "D7",
    "S6": "D8",
    "Z0": "ZI0",
    "Z1": "ZI1",
    "Z2": "ZI2",
    "Z3": "ZI3",
}
PAPER_DGP_FAMILY: dict[str, str] = {
    **{label: "LS" for label in ("LS0", "LS1", "LS2", "LS3")},
    **{label: "S" for label in ("S1", "S2", "S3", "S4", "S5", "S6")},
    **{label: "Z" for label in ("Z0", "Z1", "Z2", "Z3")},
}
DEFAULT_N_VALUES: tuple[int, ...] = (125, 250, 500, 1000)
ORDINARY_METHODS: tuple[str, ...] = ("cwdb_dr", "causal_drf", "drf")
ZERO_METHODS: tuple[str, ...] = ("cwdb_zipt", "causal_drf", "drf")
DEFAULT_METHODS = ORDINARY_METHODS
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))
DEFAULT_N_GRID = 25
DEFAULT_N_PARTICLES = 10
_SIDECAR_FIELDS = ("manifest_checksum", "estimator_source_hash", "contract_id")


def _methods_for_family(family: str) -> tuple[str, ...]:
    if family == "Z":
        return ZERO_METHODS
    if family in {"LS", "S"}:
        return ORDINARY_METHODS
    raise ValueError(f"unknown DGP family {family!r}")


def _registry_for_methods(
    methods: tuple[str, ...], source_hash: str | None = None
) -> dict[str, dict[str, Any]]:
    """Resolve ordinary and two-part method entries for a manifest."""

    registry = build_wcf_method_registry(source_hash)
    if "cwdb_zipt" in methods:
        from wasserstein_causal_forests.g3.phase65 import PHASE65_METHOD_REGISTRY

        registry["cwdb_zipt"] = copy.deepcopy(PHASE65_METHOD_REGISTRY["cwdb_zipt"])
    missing = sorted(set(methods) - set(registry))
    if missing:
        raise SystemExit(
            f"methods {missing} are unavailable; resolved registry has "
            f"{sorted(registry)}"
        )
    return registry


def _set_stage(stage: str) -> None:
    """Redirect this experiment to a named sibling result directory."""

    global RESULTS_DIRECTORY, MANIFEST_PATH, SHARDS_DIRECTORY
    global FAILURES_DIRECTORY, LOGS_DIRECTORY, MERGED_DIRECTORY, MERGED_PATH
    stage = stage.strip().strip("/")
    if not stage:
        return
    RESULTS_DIRECTORY = ROOT / "results" / "wcf_sample_size_sensitivity" / stage
    MANIFEST_PATH = RESULTS_DIRECTORY / "manifest.json"
    SHARDS_DIRECTORY = RESULTS_DIRECTORY / "shards"
    FAILURES_DIRECTORY = RESULTS_DIRECTORY / "failures"
    LOGS_DIRECTORY = RESULTS_DIRECTORY / "logs"
    MERGED_DIRECTORY = RESULTS_DIRECTORY / "merged"
    MERGED_PATH = MERGED_DIRECTORY / "wcf_sample_size_results.parquet"


def _parse_ints(text: str, *, name: str, positive: bool = True) -> tuple[int, ...]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as error:
            raise SystemExit(f"{name} contains a non-integer value {token!r}") from error
        if positive and value < 1:
            raise SystemExit(f"{name} values must be positive, got {value}")
        values.append(value)
    if not values:
        raise SystemExit(f"{name} must contain at least one integer")
    if len(set(values)) != len(values):
        raise SystemExit(f"{name} contains duplicates: {values}")
    return tuple(values)


def _parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(token.strip() for token in text.split(",") if token.strip())
    if not methods:
        raise SystemExit("--methods must contain at least one method")
    if len(set(methods)) != len(methods):
        raise SystemExit(f"--methods contains duplicates: {methods}")
    return methods


def _cells(
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    paper_dgps: dict[str, str],
    methods_by_family: dict[str, tuple[str, ...]],
    *,
    n_grid: int = DEFAULT_N_GRID,
    n_particles: int = DEFAULT_N_PARTICLES,
) -> list[Cell]:
    return [
        Cell(GRID_ID, source_dgp, n_train, n_grid, n_particles, method, seed)
        for paper_dgp, source_dgp in paper_dgps.items()
        for n_train in n_values
        for method in methods_by_family[PAPER_DGP_FAMILY[paper_dgp]]
        for seed in seeds
    ]


def _validate_design(
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    paper_dgps: dict[str, str],
    methods_by_family: dict[str, tuple[str, ...]],
    n_grid: int,
    n_particles: int,
) -> None:
    if n_grid < 2:
        raise SystemExit("--n-grid must be at least 2")
    if n_particles < 1:
        raise SystemExit("--n-particles must be positive")
    if len(seeds) < 3:
        raise SystemExit("at least three replications are required for a paired SE")
    methods = tuple(
        method
        for family in ("LS", "S", "Z")
        for method in methods_by_family.get(family, ())
    )
    registry = _registry_for_methods(methods)
    missing = sorted(set(methods) - set(registry))
    if missing:
        raise SystemExit(
            f"methods {missing} are not in the frozen WCF registry; "
            f"available methods are {sorted(registry)}"
        )
    if any(n < 3 for n in n_values):
        raise SystemExit("sample sizes below 3 are not supported")
    for family, family_methods in methods_by_family.items():
        if family in {"LS", "S"} and set(family_methods) - set(ORDINARY_METHODS):
            raise SystemExit(
                f"ordinary family {family} cannot use non-ordinary methods: "
                f"{sorted(set(family_methods) - set(ORDINARY_METHODS))}"
            )
        if family == "Z" and set(family_methods) != set(ZERO_METHODS):
            raise SystemExit(
                f"structural-zero family must use exactly {ZERO_METHODS}, "
                f"got {family_methods}"
            )
    cells = _cells(
        n_values,
        seeds,
        paper_dgps,
        methods_by_family,
        n_grid=n_grid,
        n_particles=n_particles,
    )
    keys = [cell.key for cell in cells]
    if len(keys) != len(set(keys)):
        raise SystemExit("cell-key collision in sample-size design")
    allowed_sources = set(paper_dgps.values())
    for label, source in paper_dgps.items():
        family = PAPER_DGP_FAMILY[label]
        expected = set(methods_by_family[family])
        observed = {cell.method for cell in cells if cell.dgp == source}
        if observed != expected:
            raise SystemExit(
                f"method roster mismatch for {label}: expected {sorted(expected)}, "
                f"observed {sorted(observed)}"
            )
    if any(cell.dgp not in allowed_sources for cell in cells):
        raise SystemExit("cell roster contains a DGP outside the selected manifest")


def build_manifest(
    *,
    n_values: tuple[int, ...] = DEFAULT_N_VALUES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    roster: str = "main",
    n_grid: int = DEFAULT_N_GRID,
    n_particles: int = DEFAULT_N_PARTICLES,
) -> dict[str, Any]:
    """Create the complete, content-addressed sample-size manifest."""

    if roster == "main":
        paper_dgps = PAPER_DGPS_MAIN
        contract_id = CONTRACT_ID
        methods_by_family = {"LS": methods, "S": methods}
    elif roster == "all":
        paper_dgps = PAPER_DGPS_ALL
        contract_id = ALL_DGP_CONTRACT_ID
        methods_by_family = {"LS": ORDINARY_METHODS, "S": ORDINARY_METHODS, "Z": ZERO_METHODS}
    else:
        raise SystemExit("roster must be 'main' or 'all'")
    _validate_design(
        n_values,
        seeds,
        paper_dgps,
        methods_by_family,
        n_grid,
        n_particles,
    )
    methods_in_manifest = tuple(
        dict.fromkeys(
            method
            for family in ("LS", "S", "Z")
            for method in methods_by_family.get(family, ())
        )
    )
    cells = _cells(
        n_values,
        seeds,
        paper_dgps,
        methods_by_family,
        n_grid=n_grid,
        n_particles=n_particles,
    )
    source_hash = estimator_source_hash()
    registry = _registry_for_methods(methods_in_manifest, source_hash)
    document: dict[str, Any] = {
        "manifest_contract_id": contract_id,
        "parent_sensitivity_contract_id": WCF_SENSITIVITY_CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "grid_id": GRID_ID,
        "roster": roster,
        "paper_dgp_map": copy.deepcopy(paper_dgps),
        "paper_dgp_family": {
            label: PAPER_DGP_FAMILY[label] for label in paper_dgps
        },
        "method_roster_by_family": {
            family: list(method_roster)
            for family, method_roster in methods_by_family.items()
        },
        "dgp_order": list(paper_dgps),
        "n_values": list(n_values),
        "replication_seeds": list(seeds),
        "n_replications": len(seeds),
        "methods": list(methods_in_manifest),
        "n_grid": n_grid,
        "n_particles": n_particles,
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "boosting_budget": copy.deepcopy(BOOSTING_BUDGET),
        "evaluation_manifest": {
            **EVALUATION_MANIFEST,
            "functionals": list(EVALUATION_MANIFEST["functionals"]),
        },
        "method_registry": {
            name: copy.deepcopy(registry[name]) for name in methods_in_manifest
        },
        "software_revision": software_revision(),
        "estimator_source_hash": source_hash,
        "cells": [cell.to_dict() for cell in cells],
    }
    document["n_cells"] = len(cells)
    document["manifest_checksum"] = hashlib.sha256(
        json.dumps(document["cells"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return document


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_yaml_if_available(path: Path, document: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError:
        print("PyYAML unavailable; skipped YAML manifest rendering")
        return
    yaml_path = path.with_suffix(".yaml")
    if yaml_path.exists():
        raise SystemExit(
            f"refusing to overwrite existing YAML manifest {yaml_path}; "
            "remove it only after preserving the old experiment"
        )
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def freeze(arguments: argparse.Namespace) -> int:
    document = build_manifest(
        n_values=_parse_ints(arguments.n_values, name="--n-values"),
        seeds=_parse_ints(arguments.seeds, name="--seeds", positive=False),
        methods=_parse_methods(arguments.methods),
        roster=arguments.roster,
        n_grid=arguments.n_grid,
        n_particles=arguments.n_particles,
    )
    if MANIFEST_PATH.exists() and not arguments.force:
        old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if old.get("manifest_checksum") == document["manifest_checksum"]:
            print(f"manifest already frozen at {MANIFEST_PATH}")
            print(f"  checksum: {document['manifest_checksum']}")
            return 0
        raise SystemExit(
            f"{MANIFEST_PATH} already contains another experiment; use a new "
            "results directory or pass --force only after preserving it"
        )
    if MANIFEST_PATH.exists() and arguments.force:
        existing_shards = list(SHARDS_DIRECTORY.glob("*") if SHARDS_DIRECTORY.exists() else [])
        existing_merged = list(MERGED_DIRECTORY.glob("*") if MERGED_DIRECTORY.exists() else [])
        if existing_shards or existing_merged:
            raise SystemExit(
                "refusing --force because result files already exist under "
                f"{RESULTS_DIRECTORY}; preserve them and choose a new directory"
            )
    _write_json(MANIFEST_PATH, document)
    _write_yaml_if_available(MANIFEST_PATH, document)
    print(f"wrote {MANIFEST_PATH}")
    print(f"  cells:        {document['n_cells']}")
    print(f"  replications: {document['n_replications']}")
    print(f"  sizes:        {document['n_values']}")
    print(f"  methods:      {document['methods']}")
    print(f"  checksum:     {document['manifest_checksum']}")
    return 0


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"manifest not found at {MANIFEST_PATH}; run freeze first"
        )
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    required = (
        "manifest_contract_id",
        "manifest_checksum",
        "estimator_source_hash",
        "method_registry",
        "cells",
    )
    missing = [field for field in required if field not in document]
    if missing:
        raise SystemExit(f"manifest is missing required fields: {missing}")
    if document["manifest_contract_id"] not in {CONTRACT_ID, ALL_DGP_CONTRACT_ID}:
        raise SystemExit(
            f"expected {CONTRACT_ID}, got {document['manifest_contract_id']!r}"
        )
    # Manifests produced before the all-DGP extension remain readable. New
    # manifests always carry the explicit family and method roster fields.
    paper_dgp_map = document.setdefault("paper_dgp_map", copy.deepcopy(PAPER_DGPS_MAIN))
    paper_dgp_family = document.setdefault(
        "paper_dgp_family",
        {
            label: PAPER_DGP_FAMILY[label]
            for label in paper_dgp_map
        },
    )
    method_roster = document.setdefault(
        "method_roster_by_family",
        {"LS": list(ORDINARY_METHODS), "S": list(ORDINARY_METHODS)},
    )
    source_to_label = {source: label for label, source in paper_dgp_map.items()}
    if len(source_to_label) != len(paper_dgp_map):
        raise SystemExit("paper DGP map contains duplicate source identifiers")
    declared = [item["cell_key"] for item in document["cells"]]
    if len(declared) != len(set(declared)):
        raise SystemExit("manifest contains duplicate cell keys")
    expected_checksum = hashlib.sha256(
        json.dumps(document["cells"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    if document["manifest_checksum"] != expected_checksum:
        raise SystemExit(
            "manifest checksum does not match its cell list; refusing to run"
        )
    for item in document["cells"]:
        cell = Cell(
            **{
                key: value
                for key, value in item.items()
                if key not in {"cell_key", "test_seed"}
            }
        )
        if item.get("cell_key") != cell.key or item.get("test_seed") != cell.test_seed:
            raise SystemExit(f"manifest cell identity mismatch for {item}")
        label = source_to_label.get(item["dgp"])
        if label is None:
            raise SystemExit(f"manifest cell uses an unmapped DGP {item['dgp']!r}")
        family = paper_dgp_family.get(label)
        expected = set(method_roster.get(family, ()))
        if item["method"] not in expected:
            raise SystemExit(
                f"invalid method {item['method']!r} for {label} ({family}); "
                f"expected one of {sorted(expected)}"
            )
        if family == "Z" and item["method"] == "cwdb_dr":
            raise SystemExit(
                f"ordinary-WCF cell found for structural-zero DGP {label}: {item}"
            )
        if family == "Z" and not item["dgp"].startswith("ZI"):
            raise SystemExit(f"Z family must use a ZI source DGP, got {item['dgp']!r}")
    return document


def _read_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    runner.write_rows(rows, path)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _verify_sidecar(path: Path, document: dict[str, Any]) -> None:
    sidecar = _sidecar_path(path)
    if not sidecar.exists():
        raise SystemExit(f"missing sidecar for {path}: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = {
        "manifest_checksum": document["manifest_checksum"],
        "estimator_source_hash": document["estimator_source_hash"],
        "contract_id": document["manifest_contract_id"],
    }
    for field in _SIDECAR_FIELDS:
        if payload.get(field) != expected[field]:
            raise SystemExit(
                f"sidecar mismatch in {sidecar}: {field}={payload.get(field)!r}, "
                f"expected {expected[field]!r}"
            )


def _write_sidecar(path: Path, document: dict[str, Any]) -> None:
    sidecar = _sidecar_path(path)
    expected = {
        "manifest_checksum": document["manifest_checksum"],
        "estimator_source_hash": document["estimator_source_hash"],
        "contract_id": document["manifest_contract_id"],
    }
    if sidecar.exists():
        old = json.loads(sidecar.read_text(encoding="utf-8"))
        for field in _SIDECAR_FIELDS:
            if old.get(field) != expected[field]:
                raise SystemExit(f"cannot reuse incompatible sidecar {sidecar}")
    else:
        _write_json(sidecar, {**expected, "updated_at": time.time()})


def _successful_keys(document: dict[str, Any]) -> set[str]:
    successful: set[str] = set()
    declared = {item["cell_key"] for item in document["cells"]}
    if not SHARDS_DIRECTORY.exists():
        return successful
    for path in sorted(SHARDS_DIRECTORY.glob("shard_*.parquet")):
        _verify_sidecar(path, document)
        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _read_rows(path):
            by_cell[row["cell_key"]].append(row)
        for key, rows in by_cell.items():
            if key in declared and not any(
                row.get("metric") == "cell_failure" for row in rows
            ):
                successful.add(key)
    return successful


def _archive_failed_rows(document: dict[str, Any], retry_keys: set[str]) -> None:
    """Move old failure rows away before retrying their cells."""

    moved: list[dict[str, Any]] = []
    if not SHARDS_DIRECTORY.exists():
        return
    for path in sorted(SHARDS_DIRECTORY.glob("shard_*.parquet")):
        rows = _read_rows(path)
        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_cell[row["cell_key"]].append(row)
        retained: list[dict[str, Any]] = []
        for key, group in by_cell.items():
            failed = any(row.get("metric") == "cell_failure" for row in group)
            if failed and key in retry_keys:
                moved.extend(group)
            else:
                retained.extend(group)
        if len(retained) == len(rows):
            continue
        if retained:
            temporary = path.with_suffix(".parquet.tmp")
            _write_rows(retained, temporary)
            os.replace(temporary, path)
        else:
            # This deletion only removes a shard that contains no successful
            # result rows. The failure rows have just been archived below.
            path.unlink()
            sidecar = _sidecar_path(path)
            if sidecar.exists():
                sidecar.unlink()
    if moved:
        FAILURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = FAILURES_DIRECTORY / f"attempt_{stamp}.parquet"
        counter = 1
        while archive.exists():
            archive = FAILURES_DIRECTORY / f"attempt_{stamp}_{counter:02d}.parquet"
            counter += 1
        _write_rows(moved, archive)
        _write_json(
            archive.with_suffix(".meta.json"),
            {
                "manifest_checksum": document["manifest_checksum"],
                "estimator_source_hash": document["estimator_source_hash"],
                "contract_id": document["manifest_contract_id"],
                "n_rows": len(moved),
                "archived_at": time.time(),
            },
        )
        print(f"archived {len(moved)} failure rows to {archive}")


def _shard_by_replication(cells: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        key = (
            cell["grid"],
            cell["dgp"],
            cell["n_train"],
            cell["n_grid"],
            cell["n_particles"],
            cell["seed"],
        )
        groups[key].append(cell)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(max(1, workers))]
    # Each group includes all methods for one DGP, n, and seed. Keeping it
    # intact preserves test-truth caching and makes every method comparison
    # seed-paired within one worker.
    for index, key in enumerate(sorted(groups)):
        shards[index % len(shards)].extend(groups[key])
    return shards


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    cell_dicts, index, document = payload
    runner.pin_to_one_thread()
    apply_method_registry({"method_registry": document["method_registry"]})
    shard_path = SHARDS_DIRECTORY / f"shard_{index:03d}.parquet"
    log_path = LOGS_DIRECTORY / f"shard_{index:03d}.jsonl"
    _write_sidecar(shard_path, document)
    rows = _read_rows(shard_path) if shard_path.exists() else []
    started = time.perf_counter()
    n_failed = 0
    for item in cell_dicts:
        cell = Cell(
            **{
                key: value
                for key, value in item.items()
                if key not in {"cell_key", "test_seed"}
            }
        )
        cell_rows = runner.run_cell(
            cell,
            cache_directory=CACHE_DIRECTORY,
            manifest_contract_id=document["manifest_contract_id"],
        )
        rows.extend(cell_rows)
        failed = bool(cell_rows and cell_rows[0].get("status") == "failed")
        n_failed += int(failed)
        temporary = shard_path.with_suffix(".parquet.tmp")
        _write_rows(rows, temporary)
        os.replace(temporary, shard_path)
        LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "cell_key": cell.key,
                        **asdict(cell),
                        "status": "failed" if failed else "ok",
                        "n_rows": len(cell_rows),
                        "wall_seconds": round(
                            float(cell_rows[0].get("wall_seconds", 0.0)), 3
                        ),
                        "finished_at": time.time(),
                    }
                )
                + "\n"
            )
    return {
        "shard": index,
        "n_cells": len(cell_dicts),
        "n_failed": n_failed,
        "n_rows": len(rows),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "output": str(shard_path),
    }


def run(arguments: argparse.Namespace) -> int:
    document = _load_manifest()
    selected = [dict(item) for item in document["cells"]]
    paper_dgps = document["paper_dgp_map"]
    if arguments.dgps:
        wanted_labels = {
            token.strip() for token in arguments.dgps.split(",") if token.strip()
        }
        unknown = wanted_labels - set(paper_dgps)
        if unknown:
            raise SystemExit(
                f"unknown paper DGP labels {sorted(unknown)}; choose from "
                f"{sorted(paper_dgps)}"
            )
        wanted_source = {paper_dgps[label] for label in wanted_labels}
        selected = [item for item in selected if item["dgp"] in wanted_source]
    if arguments.n_values:
        wanted_n = set(_parse_ints(arguments.n_values, name="--n-values"))
        selected = [item for item in selected if item["n_train"] in wanted_n]
    if arguments.methods:
        wanted_methods = set(_parse_methods(arguments.methods))
        unknown = wanted_methods - set(document["method_registry"])
        if unknown:
            raise SystemExit(f"methods {sorted(unknown)} are absent from the manifest")
        selected = [item for item in selected if item["method"] in wanted_methods]
    if not selected:
        raise SystemExit("no cells match the requested run filters")

    selected_keys = {item["cell_key"] for item in selected}
    SHARDS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    successful = _successful_keys(document) & selected_keys
    _archive_failed_rows(document, selected_keys - successful)
    remaining = [item for item in selected if item["cell_key"] not in successful]
    if arguments.limit:
        remaining = remaining[: arguments.limit]
    if not remaining:
        print("nothing to run")
        return 0

    worker_count = max(1, arguments.workers)
    shards = _shard_by_replication(remaining, worker_count)
    payloads = [
        (shard, index, document)
        for index, shard in enumerate(shards)
        if shard
    ]
    print(
        f"running {len(remaining)} cells across {len(payloads)} workers; "
        f"single-threaded workers={worker_count}",
        flush=True,
    )
    summaries: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for summary in pool.map(_worker, payloads):
            summaries.append(summary)
            print(
                f"shard {summary['shard']:03d} done: {summary['n_cells']} cells, "
                f"{summary['n_failed']} failed, {summary['wall_seconds']}s",
                flush=True,
            )
    print(
        f"completed {sum(item['n_cells'] for item in summaries)} cells, "
        f"{sum(item['n_failed'] for item in summaries)} failed"
    )
    return 0


def merge(arguments: argparse.Namespace) -> int:
    import pandas as pd

    document = _load_manifest()
    declared = {item["cell_key"] for item in document["cells"]}
    paths = sorted(SHARDS_DIRECTORY.glob("shard_*.parquet"))
    if not paths:
        print("no shard files found")
        return 1
    frames = []
    seen: set[str] = set()
    n_failed = 0
    for path in paths:
        _verify_sidecar(path, document)
        frame = pd.DataFrame(_read_rows(path))
        if frame.empty:
            continue
        keys = set(frame["cell_key"])
        duplicates = seen & keys
        if duplicates:
            raise SystemExit(f"duplicate cell keys across shards: {sorted(duplicates)[:5]}")
        seen |= keys
        n_failed += int(
            frame.loc[frame["metric"] == "cell_failure", "cell_key"].nunique()
        )
        frames.append(frame)
    if not frames:
        raise SystemExit("shards contain no result rows")
    unknown = seen - declared
    missing = declared - seen
    if unknown:
        raise SystemExit(f"shards contain {len(unknown)} unknown cell keys")
    if missing and not arguments.partial:
        raise SystemExit(
            f"{len(missing)} manifest cells are missing; use --partial only "
            "for an explicitly incomplete exploratory merge"
        )
    merged = pd.concat(frames, ignore_index=True)
    MERGED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_rows(merged.to_dict("records"), MERGED_PATH)
    audit = {
        "status": "PASS" if not missing else "PARTIAL",
        "contract_id": document["manifest_contract_id"],
        "manifest_checksum": document["manifest_checksum"],
        "estimator_source_hash": document["estimator_source_hash"],
        "n_shards": len(paths),
        "n_cells_declared": len(declared),
        "n_cells_observed": len(seen),
        "n_cells_missing": len(missing),
        "missing_cells": sorted(missing),
        "n_failed_cells": n_failed,
        "n_duplicate_keys": 0,
    }
    _write_json(MERGED_DIRECTORY / "merge_audit.json", audit)
    print(f"wrote {MERGED_PATH}")
    print(
        f"cells: {len(seen)} of {len(declared)}; failed: {n_failed}; "
        f"partial: {bool(missing)}"
    )
    return 0


def summarize(_: argparse.Namespace) -> int:
    """Aggregate every metric by paper DGP, n, and method.

    The raw cell-level parquet remains the primary artifact. This summary is a
    convenience table: means are over successful Monte Carlo replications and
    ``mc_se`` is the standard error of that mean. It preserves metric and
    target identifiers, rather than averaging incompatible targets together.
    """

    import pandas as pd

    document = _load_manifest()
    if not MERGED_PATH.exists():
        raise SystemExit(f"merged results not found at {MERGED_PATH}; run merge first")
    frame = pd.DataFrame(_read_rows(MERGED_PATH))
    frame = frame[(frame["status"] == "ok") & frame["value"].notna()].copy()
    if frame.empty:
        raise SystemExit("merged results contain no successful metric rows")
    reverse = {
        source: label for label, source in document["paper_dgp_map"].items()
    }
    frame["paper_dgp"] = frame["dgp"].map(reverse).fillna(frame["dgp"])
    group_columns = [
        "paper_dgp",
        "dgp",
        "n_train",
        "n_grid",
        "n_particles",
        "method",
        "metric",
        "target_id",
    ]
    summary = (
        frame.groupby(group_columns, dropna=False)["value"]
        .agg(mean="mean", mc_sd="std", n_replications="count")
        .reset_index()
    )
    summary["mc_se"] = summary["mc_sd"] / summary["n_replications"].pow(0.5)
    summary = summary.sort_values(group_columns).reset_index(drop=True)
    MERGED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = MERGED_DIRECTORY / "sample_size_summary.csv"
    summary.to_csv(csv_path, index=False)
    json_path = MERGED_DIRECTORY / "sample_size_summary.json"
    json_path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"rows: {len(summary)}")
    return 0


def status(_: argparse.Namespace) -> int:
    document = _load_manifest()
    declared = {item["cell_key"] for item in document["cells"]}
    done: set[str] = set()
    failed: set[str] = set()
    for path in sorted(SHARDS_DIRECTORY.glob("shard_*.parquet")):
        _verify_sidecar(path, document)
        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _read_rows(path):
            by_cell[row["cell_key"]].append(row)
        for key, rows in by_cell.items():
            if key not in declared:
                continue
            if any(row.get("metric") == "cell_failure" for row in rows):
                failed.add(key)
            else:
                done.add(key)
    print(f"declared: {len(declared)}")
    print(f"done:     {len(done)}")
    print(f"failed:   {len(failed)}")
    print(f"pending:  {len(declared - done - failed)}")
    return 0


def _default_workers() -> int:
    """Conservative default for the 20-thread workstation.

    Four single-threaded processes leave headroom for other work and keep the
    memory multiplier bounded. Users can increase this after a pilot, but the
    default is intentionally not the full logical CPU count.
    """

    return max(1, min(4, (os.cpu_count() or 4) // 4))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="",
        help=(
            "optional sibling directory under results/wcf_sample_size_sensitivity; "
            "use this for a deliberately separate smoke or follow-up manifest"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze", help="write an immutable manifest")
    freeze_parser.add_argument(
        "--roster",
        choices=("main", "all"),
        default="main",
        help="main five-design roster or all LS/S/Z manuscript designs",
    )
    freeze_parser.add_argument(
        "--n-values", default=",".join(str(value) for value in DEFAULT_N_VALUES)
    )
    freeze_parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )
    freeze_parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    freeze_parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    freeze_parser.add_argument("--n-particles", type=int, default=DEFAULT_N_PARTICLES)
    freeze_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="run pending manifest cells")
    run_parser.add_argument("--workers", type=int, default=_default_workers())
    run_parser.add_argument("--dgps", default="", help="paper labels, e.g. LS1,S4")
    run_parser.add_argument("--n-values", default="", help="subset of frozen sizes")
    run_parser.add_argument("--methods", default="", help="subset of frozen methods")
    run_parser.add_argument("--limit", type=int, default=0)

    merge_parser = subparsers.add_parser("merge", help="merge shards")
    merge_parser.add_argument("--partial", action="store_true")

    subparsers.add_parser("summarize", help="write mean and Monte Carlo SE tables")
    subparsers.add_parser("status", help="show manifest progress")

    arguments = parser.parse_args(argv)
    _set_stage(arguments.stage)
    if arguments.command == "freeze":
        return freeze(arguments)
    if arguments.command == "run":
        return run(arguments)
    if arguments.command == "merge":
        return merge(arguments)
    if arguments.command == "summarize":
        return summarize(arguments)
    return status(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
