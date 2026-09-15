#!/usr/bin/env python3
"""Run the symmetric 50-replication WCF confirmatory evaluation.

This is a new experiment, separate from every historical tournament and the
sample-size study.  It evaluates all 14 paper DGPs at n=1000 with 50 paired
seeds.  Every method receives both plug-in law-functional evaluation and a
strict three-fold AIPW evaluation.  LS1/LS3 TCATE uses X4, LS2 uses X3, and
the null LS0 design retains X1 as a stability check.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "R_NUM_THREADS",
):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3 import runner  # noqa: E402
from wasserstein_causal_forests.g3.confirmatory_evaluation import (  # noqa: E402
    ACTIVE_MODERATOR_COLUMN,
    CONFIRMATORY_PROTOCOL_ID,
    NUISANCE_FOLDS,
    evaluate_confirmatory,
)
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
    apply_method_registry,
    build_wcf_method_registry,
    software_revision,
)
CONTRACT_ID = "WCF-CONFIRMATORY-R50-v1"
GRID_ID = "wcf_confirmatory_r50_v1"
N_TRAIN = 1000
N_GRID = 25
N_PARTICLES = 10
SEEDS = tuple(range(50))
RESULTS_DIRECTORY = ROOT / "results" / "wcf_confirmatory_r50"
MANIFEST_PATH = RESULTS_DIRECTORY / "manifest.json"
SHARDS_DIRECTORY = RESULTS_DIRECTORY / "shards"
LOGS_DIRECTORY = RESULTS_DIRECTORY / "logs"
MERGED_DIRECTORY = RESULTS_DIRECTORY / "merged"
MERGED_PATH = MERGED_DIRECTORY / "wcf_confirmatory_results.parquet"
CACHE_DIRECTORY = RESULTS_DIRECTORY / "cache"

ERROR_METRIC_SUFFIXES = ("_abs_error", "_rmse")
PAPER_DGPS_ALL = {
    "LS0": "IC0", "LS1": "IC1", "LS2": "IC2", "LS3": "IC3",
    "S1": "D0", "S2": "D2", "S3": "D5", "S4": "D6",
    "S5": "D7", "S6": "D8",
    "Z0": "ZI0", "Z1": "ZI1", "Z2": "ZI2", "Z3": "ZI3",
}
PAPER_DGP_FAMILY = {
    **{label: "LS" for label in ("LS0", "LS1", "LS2", "LS3")},
    **{label: "S" for label in ("S1", "S2", "S3", "S4", "S5", "S6")},
    **{label: "Z" for label in ("Z0", "Z1", "Z2", "Z3")},
}
ORDINARY_METHODS = ("cwdb_dr", "causal_drf", "drf")
ZERO_METHODS = ("cwdb_zipt", "causal_drf", "drf")


def methods_for_label(label: str) -> tuple[str, ...]:
    return ZERO_METHODS if PAPER_DGP_FAMILY[label] == "Z" else ORDINARY_METHODS


def cells() -> list[Cell]:
    return [
        Cell(GRID_ID, source, N_TRAIN, N_GRID, N_PARTICLES, method, seed)
        for label, source in PAPER_DGPS_ALL.items()
        for seed in SEEDS
        for method in methods_for_label(label)
    ]


def source_hash() -> str:
    """Hash every source file that can affect a confirmatory result."""

    paths = list((ROOT / "src" / "wasserstein_causal_forests").rglob("*.py"))
    for directory in (ROOT / "research" / "baselines", ROOT / "code" / "drfinference-main"):
        if directory.exists():
            paths.extend(
                path for path in directory.rglob("*")
                if path.is_file()
                and "results" not in path.relative_to(directory).parts
                and "__pycache__" not in path.parts
                and not path.name.endswith(":Zone.Identifier")
            )
    paths.extend(
        path for path in (
            ROOT / "research" / "run_wcf_confirmatory.py",
            ROOT / "requirements-colab.txt",
        ) if path.exists()
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def method_registry(estimator_hash: str) -> dict[str, dict[str, Any]]:
    registry = build_wcf_method_registry(estimator_hash)
    # Dense-grid calibration is not part of this experiment.  Omitting it
    # avoids computing and storing a second WCF-only functional layer.
    registry["cwdb_dr"]["parameters"]["common_grid_levels"] = ""
    from wasserstein_causal_forests.g3.phase65 import PHASE65_METHOD_REGISTRY
    registry["cwdb_zipt"] = copy.deepcopy(PHASE65_METHOD_REGISTRY["cwdb_zipt"])
    return {
        name: copy.deepcopy(registry[name])
        for name in (*ORDINARY_METHODS, *ZERO_METHODS)
    }


def build_manifest() -> dict[str, Any]:
    estimator_hash = source_hash()
    roster = cells()
    document: dict[str, Any] = {
        "manifest_contract_id": CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "evaluation_protocol_id": CONFIRMATORY_PROTOCOL_ID,
        "grid_id": GRID_ID,
        "paper_dgp_map": copy.deepcopy(PAPER_DGPS_ALL),
        "paper_dgp_family": copy.deepcopy(PAPER_DGP_FAMILY),
        "dgp_order": list(PAPER_DGPS_ALL),
        "method_roster_by_family": {
            "LS": list(ORDINARY_METHODS),
            "S": list(ORDINARY_METHODS),
            "Z": list(ZERO_METHODS),
        },
        "n_train": N_TRAIN,
        "n_grid": N_GRID,
        "n_particles": N_PARTICLES,
        "replication_seeds": list(SEEDS),
        "n_replications": len(SEEDS),
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "nuisance_folds": NUISANCE_FOLDS,
        "propensity_model": "five-fold cross-fitted logistic regression",
        "propensity_clip": [0.02, 0.98],
        "moderator_edges": [-0.5, 0.0, 0.5],
        "active_moderator_columns_zero_based": copy.deepcopy(ACTIVE_MODERATOR_COLUMN),
        "default_moderator_column_zero_based": 0,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "reported_functionals": list(
            __import__(
                "wasserstein_causal_forests.pta_bcf.targets",
                fromlist=["GRID_FUNCTIONALS"],
            ).GRID_FUNCTIONALS
        ),
        "boosting_budget": copy.deepcopy(BOOSTING_BUDGET),
        "evaluation_manifest": copy.deepcopy(EVALUATION_MANIFEST),
        "method_registry": method_registry(estimator_hash),
        "software_revision": software_revision(),
        "estimator_source_hash": estimator_hash,
        "cells": [cell.to_dict() for cell in roster],
    }
    document["n_cells"] = len(roster)
    checksum_payload = copy.deepcopy(document)
    document["manifest_checksum"] = hashlib.sha256(
        json.dumps(checksum_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return document


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    runner.write_rows(rows, path)


def freeze(_: argparse.Namespace) -> int:
    document = build_manifest()
    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if old == document:
            print(f"manifest already frozen at {MANIFEST_PATH}")
            return 0
        raise SystemExit(
            f"refusing to overwrite distinct manifest at {MANIFEST_PATH}"
        )
    _write_json(MANIFEST_PATH, document)
    print(f"wrote {MANIFEST_PATH}")
    print(f"cells: {document['n_cells']}; paired replications: {14 * len(SEEDS)}")
    print(f"checksum: {document['manifest_checksum']}")
    return 0


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"missing manifest: {MANIFEST_PATH}")
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if document.get("manifest_contract_id") != CONTRACT_ID:
        raise SystemExit("confirmatory manifest contract mismatch")
    payload = {key: value for key, value in document.items() if key != "manifest_checksum"}
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if checksum != document.get("manifest_checksum"):
        raise SystemExit("confirmatory manifest checksum mismatch")
    keys = [item["cell_key"] for item in document["cells"]]
    if len(keys) != len(set(keys)) or len(keys) != 2100:
        raise SystemExit("confirmatory manifest cell roster is invalid")
    return document


def _common(cell: Cell) -> dict[str, Any]:
    return {
        **asdict(cell),
        "cell_key": cell.key,
        "test_seed": cell.test_seed,
        "manifest_contract_id": CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "evaluation_manifest_id": EVALUATION_MANIFEST["manifest_id"],
        "evaluation_protocol_id": CONFIRMATORY_PROTOCOL_ID,
        "method_role": runner.METHOD_REGISTRY[cell.method]["role"],
        "n_test": N_TEST,
    }


def run_cell(cell: Cell, *, cache_directory: Path | None = None) -> list[dict[str, Any]]:
    started = time.perf_counter()
    common = _common(cell)
    try:
        dgp = runner.build_dgp(cell.dgp, cell.n_grid)
        train = dgp.sample(cell.n_train, seed=cell.seed)
        test = dgp.sample(N_TEST, seed=cell.test_seed)
        output = runner.build_adapter(cell, cache_directory).fit_predict(
            train, test.X, dgp, TRAINING_FUNCTIONALS, seed=cell.seed
        )
        rows = evaluate_confirmatory(
            cell,
            output,
            train,
            dgp,
            test.X,
            runner.evaluation_manifest(cell.n_grid),
            cache_directory=cache_directory,
        )
        rows.append({
            "metric": "process_peak_ram",
            "target_id": "NONE_OPERATIONAL",
            "arm": None,
            "detail": "absolute process high-water mark, megabytes",
            "value": runner.peak_ram_mb(),
            "status": "ok",
            "failure_reason": "",
        })
    except Exception as error:  # noqa: BLE001
        return [{
            **common,
            "metric": "cell_failure",
            "target_id": "NONE_OPERATIONAL",
            "arm": None,
            "detail": traceback.format_exc(limit=5)[-1600:],
            "value": None,
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}"[:500],
            "wall_seconds": time.perf_counter() - started,
        }]
    wall = time.perf_counter() - started
    return [{**common, **row, "wall_seconds": wall} for row in rows]


def _replication_groups(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[(item["dgp"], item["n_train"], item["seed"])].append(item)
    return [groups[key] for key in sorted(groups)]


def _worker(payload: tuple[int, list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    index, items, document = payload
    runner.pin_to_one_thread()
    apply_method_registry({"method_registry": document["method_registry"]})
    path = SHARDS_DIRECTORY / f"shard_{index:03d}.parquet"
    log_path = LOGS_DIRECTORY / f"shard_{index:03d}.jsonl"
    rows = _read_rows(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["cell_key"]].append(row)
    done = {
        key for key, values in grouped.items()
        if not any(row.get("status") == "failed" for row in values)
    }
    failed = set(grouped) - done
    if failed:
        rows = [row for row in rows if row["cell_key"] not in failed]
    started = time.perf_counter()
    for item in items:
        if item["cell_key"] in done:
            continue
        cell = Cell(**{k: v for k, v in item.items() if k not in {"cell_key", "test_seed"}})
        cell_rows = run_cell(cell, cache_directory=CACHE_DIRECTORY)
        rows.extend(cell_rows)
        temporary = path.with_suffix(".tmp")
        _write_rows(rows, temporary)
        os.replace(temporary, path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "cell_key": cell.key,
                "status": cell_rows[0]["status"],
                "n_rows": len(cell_rows),
                "wall_seconds": cell_rows[0]["wall_seconds"],
                "finished_at": time.time(),
            }) + "\n")
    return {"shard": index, "cells": len(items), "seconds": time.perf_counter() - started}


def run(arguments: argparse.Namespace) -> int:
    document = load_manifest()
    apply_method_registry({"method_registry": document["method_registry"]})
    SHARDS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    items = document["cells"][: arguments.limit or None]
    groups = _replication_groups(items)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(arguments.workers)]
    for index, group in enumerate(groups):
        buckets[index % arguments.workers].extend(group)
    payloads = [(i, bucket, document) for i, bucket in enumerate(buckets) if bucket]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for result in pool.map(_worker, payloads):
            print(json.dumps(result), flush=True)
    return 0


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def merge(arguments: argparse.Namespace) -> int:
    import pandas as pd
    document = load_manifest()
    declared = {item["cell_key"] for item in document["cells"]}
    frames = []
    seen: set[str] = set()
    for path in sorted(SHARDS_DIRECTORY.glob("shard_*.parquet")):
        frame = pd.DataFrame(_read_rows(path))
        keys = set(frame["cell_key"])
        if seen & keys:
            raise SystemExit(f"duplicate cells across shards: {sorted(seen & keys)[:3]}")
        seen |= keys
        frames.append(frame)
    missing = declared - seen
    unknown = seen - declared
    if unknown or (missing and not arguments.partial):
        raise SystemExit(f"merge coverage failure: missing={len(missing)}, unknown={len(unknown)}")
    merged = pd.concat(frames, ignore_index=True)
    MERGED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_rows(merged.to_dict("records"), MERGED_PATH)
    audit = {
        "status": "PARTIAL" if missing else "PASS",
        "manifest_checksum": document["manifest_checksum"],
        "estimator_source_hash": document["estimator_source_hash"],
        "n_cells_declared": len(declared),
        "n_cells_observed": len(seen),
        "n_cells_missing": len(missing),
        "n_failed_cells": int(merged.loc[merged.metric == "cell_failure", "cell_key"].nunique()),
        "n_duplicate_cells": 0,
    }
    _write_json(MERGED_DIRECTORY / "merge_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


def _rmse_and_se(values):
    import numpy as np
    squared = np.square(np.asarray(values, dtype=float))
    mean_square = float(np.mean(squared))
    rmse = float(np.sqrt(mean_square))
    if len(squared) < 2 or rmse == 0.0:
        return rmse, 0.0
    se_mean_square = float(np.std(squared, ddof=1) / np.sqrt(len(squared)))
    return rmse, se_mean_square / (2.0 * rmse)


def summarize(_: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd
    document = load_manifest()
    frame = pd.DataFrame(_read_rows(MERGED_PATH))
    frame = frame[(frame.status == "ok") & frame.value.notna()].copy()
    reverse = {source: label for label, source in document["paper_dgp_map"].items()}
    frame["paper_dgp"] = frame.dgp.map(reverse)
    keys = ["paper_dgp", "dgp", "method", "metric", "target_id", "detail"]
    records = []
    for key, group in frame.groupby(keys, dropna=False):
        values = group.value.to_numpy(float)
        if str(key[3]).endswith(ERROR_METRIC_SUFFIXES):
            estimate, mc_se = _rmse_and_se(values)
            aggregation = "root_mean_square_over_replications"
        else:
            estimate = float(np.mean(values))
            mc_se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            aggregation = "mean_over_replications"
        records.append({**dict(zip(keys, key)), "estimate": estimate, "mc_se": mc_se,
                        "n_replications": len(values), "aggregation": aggregation})
    summary = pd.DataFrame(records).sort_values(keys)
    summary.to_csv(MERGED_DIRECTORY / "confirmatory_summary.csv", index=False)

    paired = []
    error = frame[frame.metric.str.endswith(ERROR_METRIC_SUFFIXES)].copy()
    pair_keys = ["paper_dgp", "dgp", "metric", "target_id", "detail"]
    for key, group in error.groupby(pair_keys, dropna=False):
        wide = group.pivot(index="seed", columns="method", values="value")
        methods = sorted(wide.columns)
        for i, left in enumerate(methods):
            for right in methods[i + 1:]:
                block = wide[[left, right]].dropna()
                if len(block) < 2:
                    continue
                x = np.square(block[left].to_numpy(float))
                y = np.square(block[right].to_numpy(float))
                rx, ry = float(np.sqrt(x.mean())), float(np.sqrt(y.mean()))
                gradient = np.array([0.0 if rx == 0 else 1 / (2 * rx),
                                     0.0 if ry == 0 else -1 / (2 * ry)])
                covariance = np.cov(np.column_stack([x, y]), rowvar=False, ddof=1) / len(block)
                se = float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))
                paired.append({**dict(zip(pair_keys, key)), "method_left": left,
                               "method_right": right, "rmse_difference": rx - ry,
                               "paired_mc_se": se, "ci95_low": rx - ry - 1.96 * se,
                               "ci95_high": rx - ry + 1.96 * se,
                               "n_pairs": len(block)})
    pd.DataFrame(paired).to_csv(MERGED_DIRECTORY / "paired_rmse_differences.csv", index=False)
    print(f"wrote {len(summary)} summary rows and {len(paired)} paired comparisons")
    return 0


def status(_: argparse.Namespace) -> int:
    document = load_manifest()
    declared = {item["cell_key"] for item in document["cells"]}
    seen: set[str] = set()
    failed: set[str] = set()
    for path in SHARDS_DIRECTORY.glob("shard_*.parquet"):
        for row in _read_rows(path):
            seen.add(row["cell_key"])
            if row.get("status") == "failed":
                failed.add(row["cell_key"])
    print(f"complete={len(seen - failed)}/{len(declared)} failed={len(failed)} missing={len(declared - seen)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=2)
    run_parser.add_argument("--limit", type=int, default=0)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--partial", action="store_true")
    sub.add_parser("summarize")
    sub.add_parser("status")
    args = parser.parse_args()
    return {"freeze": freeze, "run": run, "merge": merge,
            "summarize": summarize, "status": status}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
