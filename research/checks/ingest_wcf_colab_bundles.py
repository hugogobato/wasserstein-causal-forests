#!/usr/bin/env python3
"""Ingest Colab WCF shard bundles into the primary shard directory.

The Colab notebooks pack one `wcf_sensitivity_shard.zip` per shard with
`wcf_sensitivity_parquet.parquet` and `wcf_sensitivity_parquet.meta.json` at the
archive root. This tool unzips every bundle found under
`results/wcf_sensitivity/colab_bundles/`, renames the pair to
`shard_colab_<NN>.parquet` / `shard_colab_<NN>.meta.json`, refuses sidecars that
do not match the frozen manifest and source hash, and reports cell keys that
already exist in the local shards so the merge stays duplicate-free.

Usage:

    python3 research/checks/ingest_wcf_colab_bundles.py [--dry-run]

A duplicate is not an error: when both the local runner and Colab produced the
same cell under the same manifest and source hash, the local copy is kept and
the duplicate rows are dropped from the Colab shard, with every drop recorded in
`results/wcf_sensitivity/colab_bundles/dedupe_report.json`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "wcf_sensitivity"
REQUIRED_FIELDS = ("manifest_checksum", "estimator_source_hash", "contract_id")


def _stage_paths(stage: str) -> dict[str, Path]:
    base = RESULTS / stage if stage else RESULTS
    return {
        "base": base,
        "shards": base / "shards",
        "bundles": base / "colab_bundles",
        "manifest": base / "manifest.json",
    }


def _read_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pandas().to_dict("records")


def _write_rows(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise SystemExit(f"refusing to write an empty shard {path}")
    columns = sorted({key for row in rows for key in row})
    table = pa.table({name: [row.get(name) for row in rows] for name in columns})
    pq.write_table(table, path)


def _bundle_index(path: Path) -> int | None:
    """Shard index from a `...shard_<NN>.zip` name, or None when absent."""

    stem = path.stem
    for token in reversed(stem.replace("-", "_").split("_")):
        if token.isdigit():
            return int(token)
    return None


def _local_cell_keys(shards: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(shards.glob("shard_*.parquet")):
        for row in _read_rows(path):
            keys.add(row["cell_key"])
    return keys


def _extract_bundle(bundle: Path, staging: Path) -> tuple[Path, Path, dict]:
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(staging)
    parquet = next(staging.rglob("wcf_sensitivity_parquet.parquet"), None)
    sidecar = next(staging.rglob("wcf_sensitivity_parquet.meta.json"), None)
    if parquet is None or sidecar is None:
        raise SystemExit(
            f"{bundle.name}: expected wcf_sensitivity_parquet.parquet and "
            "wcf_sensitivity_parquet.meta.json inside the zip"
        )
    config_path = next(staging.rglob("shard_config.json"), None)
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path is not None
        else {}
    )
    return parquet, sidecar, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage",
        default="",
        help="stage directory under results/wcf_sensitivity/, e.g. confirm",
    )
    arguments = parser.parse_args()

    paths = _stage_paths(arguments.stage)
    shards_directory = paths["shards"]
    bundles_directory = paths["bundles"]
    manifest_path = paths["manifest"]
    if not manifest_path.exists():
        raise SystemExit(f"manifest {manifest_path} is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "manifest_checksum": manifest["manifest_checksum"],
        "estimator_source_hash": manifest.get("estimator_source_hash"),
        "contract_id": manifest.get(
            "manifest_contract_id", manifest.get("contract_id")
        ),
    }
    declared = {item["cell_key"] for item in manifest["cells"]}
    bundles = sorted(bundles_directory.glob("*.zip"))
    if not bundles:
        print(f"no bundles found under {bundles_directory}")
        return 1

    shards_directory.mkdir(parents=True, exist_ok=True)
    local_keys = _local_cell_keys(shards_directory)
    report: dict[str, object] = {
        "bundles": [],
        "duplicate_cells": [],
        "missing_from_bundles": [],
        "sidecar_mismatches": [],
    }
    seen_bundle: dict[str, str] = {}

    for bundle in bundles:
        with tempfile.TemporaryDirectory(prefix="wcf_colab_") as temporary:
            parquet, sidecar, config = _extract_bundle(bundle, Path(temporary))
            index = config.get("shard_index")
            if index is None:
                index = _bundle_index(bundle)
            if index is None:
                safe = "".join(
                    character if character.isalnum() else "_"
                    for character in bundle.stem
                )
                name = f"shard_colab_{safe}"
            else:
                name = f"shard_colab_{int(index):02d}"
                if name in seen_bundle:
                    report["bundles"].append(
                        {
                            "bundle": bundle.name,
                            "target": name,
                            "status": "duplicate bundle skipped",
                        }
                    )
                    print(f"SKIP {bundle.name}: shard {index} already ingested")
                    continue
                seen_bundle[name] = bundle.name
            entry: dict[str, object] = {"bundle": bundle.name, "target": name}
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            mismatch = {
                field: {
                    "bundle": metadata.get(field),
                    "manifest": expected[field],
                }
                for field in REQUIRED_FIELDS
                if metadata.get(field) != expected[field]
            }
            if mismatch:
                report["sidecar_mismatches"].append(
                    {"bundle": bundle.name, "fields": mismatch}
                )
                entry["sidecar"] = "mismatch"
                report["bundles"].append(entry)
                print(f"SKIP {bundle.name}: sidecar mismatch {mismatch}")
                continue
            rows = _read_rows(parquet)
            keys = {row["cell_key"] for row in rows}
            outside = keys - declared
            if outside:
                raise SystemExit(
                    f"{bundle.name}: {len(outside)} cell keys are absent from the "
                    f"frozen manifest, e.g. {sorted(outside)[:3]}"
                )
            duplicate_keys = keys & local_keys
            if duplicate_keys:
                rows = [row for row in rows if row["cell_key"] not in duplicate_keys]
                report["duplicate_cells"].extend(sorted(duplicate_keys))
                entry["duplicates_dropped"] = sorted(duplicate_keys)
            if not arguments.dry_run and rows:
                _write_rows(rows, shards_directory / f"{name}.parquet")
                shutil.copyfile(sidecar, shards_directory / f"{name}.meta.json")
            local_keys |= {row["cell_key"] for row in rows}
            entry.update(
                {
                    "cells": len(keys),
                    "rows": len(rows),
                    "methods": sorted({row["method"] for row in rows}),
                }
            )
            report["bundles"].append(entry)
            print(
                f"OK   {bundle.name} -> {name}.parquet"
                + (f" (dropped {len(duplicate_keys)} duplicate cells)" if duplicate_keys else "")
            )

    all_keys = (
        _local_cell_keys(shards_directory) if not arguments.dry_run else local_keys
    )
    report["missing_from_bundles"] = sorted(declared - all_keys)
    report["n_declared"] = len(declared)
    report["n_present"] = len(all_keys & declared)
    (bundles_directory / "dedupe_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"declared {len(declared)}; present in shards {len(all_keys & declared)}; "
        f"missing {len(report['missing_from_bundles'])}"
    )
    if report["missing_from_bundles"]:
        print("missing keys written to dedupe_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
