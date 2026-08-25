"""WP3-B2: validate shard outputs, reconcile them against the manifest, merge.

The audit is the deliverable as much as the merged table is. Every manifest cell
must appear exactly once across the shards, as a success or as a failure, and a
cell that failed stays in the merged table as a failure row. Dropping failed or
schema-incompatible rows would turn a method's fragility into an absence of
evidence, so the merge refuses rather than cleans: a duplicate key, an unknown
key, or a missing cell makes the audit `FAIL` and the caller decides.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

#: Columns every result row must carry. Reproducibility requires the claim's
#: cell coordinates, its contract identifiers, and its operational cost.
REQUIRED_COLUMNS = (
    "grid",
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "method",
    "seed",
    "cell_key",
    "test_seed",
    "manifest_contract_id",
    "estimand_contract_id",
    "evaluation_manifest_id",
    "method_role",
    "n_test",
    "metric",
    "target_id",
    "arm",
    "value",
    "status",
    "failure_reason",
    "wall_seconds",
)

VALID_STATUSES = {"ok", "not_applicable", "failed"}


def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _write_table(rows: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        target = path.with_suffix(".jsonl")
        target.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        return hashlib.sha256(target.read_bytes()).hexdigest()
    table = pa.table({name: [row.get(name) for row in rows] for name in columns})
    pq.write_table(table, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_results(
    shard_directory: Path, manifest_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Merge every shard, reconcile against the manifest, and write the audit."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["cell_key"]: item for item in manifest["cells"]}

    shard_files = sorted(
        [p for p in shard_directory.glob("shard_*.parquet")]
        + [p for p in shard_directory.glob("shard_*.jsonl")]
    )
    rows: list[dict[str, Any]] = []
    shard_checksums: dict[str, str] = {}
    for path in shard_files:
        shard_checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend(_read_table(path))

    problems: list[str] = []

    missing_columns = sorted(
        {column for column in REQUIRED_COLUMNS
         for row in rows[:1] if column not in row}
    )
    if missing_columns:
        problems.append(f"rows are missing required columns: {missing_columns}")

    bad_status = sorted({str(row.get("status")) for row in rows} - VALID_STATUSES)
    if bad_status:
        problems.append(f"unrecognised status values: {bad_status}")

    # Each cell contributes one block of rows; a key appearing in two shards is
    # a dispatch fault, not something to deduplicate away.
    seen_by_shard: dict[str, set[str]] = {}
    for path in shard_files:
        keys = {row["cell_key"] for row in _read_table(path)}
        for name, other in seen_by_shard.items():
            overlap = keys & other
            if overlap:
                problems.append(
                    f"{len(overlap)} cell keys appear in both {name} and {path.name}"
                )
        seen_by_shard[path.name] = keys

    observed = Counter(row["cell_key"] for row in rows)
    unknown = sorted(set(observed) - set(expected))
    if unknown:
        problems.append(f"{len(unknown)} result keys are not in the manifest")
    missing = sorted(set(expected) - set(observed))
    if missing:
        problems.append(f"{len(missing)} manifest cells produced no rows")

    for key, item in expected.items():
        for row in rows:
            if row["cell_key"] == key:
                for field in ("grid", "dgp", "n_train", "n_grid", "n_particles",
                              "method", "seed"):
                    if row[field] != item[field]:
                        problems.append(
                            f"cell {key} row disagrees with the manifest on {field}"
                        )
                break

    failed_cells = sorted(
        {row["cell_key"] for row in rows if row["status"] == "failed"}
    )
    failure_reasons = {
        row["cell_key"]: row["failure_reason"]
        for row in rows
        if row["status"] == "failed"
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    checksum = _write_table(rows, output_directory / "main_results.parquet") if rows else ""

    audit = {
        "manifest_contract_id": manifest["manifest_contract_id"],
        "manifest_checksum": manifest["manifest_checksum"],
        "n_manifest_cells": len(expected),
        "n_observed_cells": len(observed),
        "n_duplicate_keys": sum(1 for count in observed.values() if count == 0),
        "n_rows": len(rows),
        "n_failed_cells": len(failed_cells),
        "failed_cells": failed_cells[:200],
        "failure_reasons": dict(list(failure_reasons.items())[:50]),
        "missing_cells": missing[:200],
        "unknown_cells": unknown[:200],
        "shard_files": shard_checksums,
        "merged_checksum": checksum,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    (output_directory / "merge_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit
