#!/usr/bin/env python3
"""Validate and import the confirmatory Colab result bundles."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.run_wcf_confirmatory import (  # noqa: E402
    MANIFEST_PATH,
    RESULTS_DIRECTORY,
    SHARDS_DIRECTORY,
    load_manifest,
)
from research.checks.wcf_confirmatory_make_colab_notebooks import (  # noqa: E402
    PAPER_COMMIT,
    PAPER_REPOSITORY,
    PAPER_SETUP_PATH,
    PAPER_SETUP_SHA256,
    PAPER_STUDY_PATH,
    PAPER_STUDY_SHA256,
)
from research.checks.wcf_sensitivity_make_colab_notebooks import (  # noqa: E402
    CAUSAL_CLEAN_COMMIT,
)

REQUIRED = {
    "confirmatory_results.parquet",
    "execution_log.jsonl",
    "manifest_slice.json",
    "completion.json",
    "sha256_inventory.json",
}
ROW_COORDINATES = (
    "grid", "dgp", "n_train", "n_grid", "n_particles", "method", "seed",
    "test_seed", "manifest_contract_id", "evaluation_protocol_id",
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_new_or_identical(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"refusing to overwrite nonidentical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def validate_rows(frame: pd.DataFrame, slice_doc: dict, archive_name: str) -> None:
    declared = {item["cell_key"]: item for item in slice_doc["cells"]}
    observed = set(frame.cell_key.astype(str))
    if observed != set(declared):
        raise SystemExit(
            f"{archive_name}: result coverage mismatch, "
            f"missing={len(set(declared)-observed)}, extra={len(observed-set(declared))}"
        )
    identity = ["cell_key", "metric", "target_id", "arm", "detail"]
    successful = frame[frame.status != "failed"]
    if successful.duplicated(identity).any():
        raise SystemExit(f"{archive_name}: duplicate successful metric identities")
    for key, group in frame.groupby("cell_key"):
        item = declared[str(key)]
        for coordinate in ROW_COORDINATES:
            expected = (
                item.get(coordinate)
                if coordinate in item
                else slice_doc.get(coordinate)
            )
            if coordinate == "manifest_contract_id":
                expected = slice_doc["manifest_contract_id"]
            if coordinate == "evaluation_protocol_id":
                expected = slice_doc["evaluation_protocol_id"]
            values = set(group[coordinate].dropna().tolist())
            if values != {expected}:
                raise SystemExit(
                    f"{archive_name}: {key} has invalid {coordinate}: "
                    f"{values} versus {expected!r}"
                )
        failures = group[group.status == "failed"]
        if not failures.empty and len(group) != 1:
            raise SystemExit(f"{archive_name}: failed cell {key} has partial metric rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument(
        "--generation-record",
        default="colab/wcf_confirmatory_r50_26_shards/notebook_generation.json",
    )
    args = parser.parse_args()
    source = Path(args.source_dir)
    if not source.is_absolute():
        source = ROOT / source
    generation_path = Path(args.generation_record)
    if not generation_path.is_absolute():
        generation_path = ROOT / generation_path
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    manifest = load_manifest()
    if generation["manifest_checksum"] != manifest["manifest_checksum"]:
        raise SystemExit("generation record and frozen manifest disagree")

    archives = sorted(source.glob("*.zip"))
    if not archives:
        raise SystemExit(f"no ZIP bundles in {source}")
    accepted: dict[int, dict] = {}
    global_keys: set[str] = set()
    declared_global = {item["cell_key"] for item in manifest["cells"]}
    for path in archives:
        raw = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = set(archive.namelist())
            if missing := REQUIRED - names:
                raise SystemExit(f"{path.name}: missing {sorted(missing)}")
            payloads = {name: archive.read(name) for name in REQUIRED}
        completion = json.loads(payloads["completion.json"])
        slice_doc = json.loads(payloads["manifest_slice.json"])
        inventory = json.loads(payloads["sha256_inventory.json"])
        for name, expected_hash in inventory.items():
            if name not in payloads:
                raise SystemExit(f"{path.name}: inventory names absent file {name}")
            if sha256(payloads[name]) != expected_hash:
                raise SystemExit(f"{path.name}: hash mismatch for {name}")
        for field, expected in (
            ("manifest_checksum", manifest["manifest_checksum"]),
            ("estimator_source_hash", manifest["estimator_source_hash"]),
            ("evaluation_protocol_id", manifest["evaluation_protocol_id"]),
            ("source_archive_sha256", generation["source_archive_sha256"]),
            ("causal_drf_paper_repository", PAPER_REPOSITORY),
            ("causal_drf_paper_commit", PAPER_COMMIT),
            ("causal_drf_paper_files", {
                PAPER_SETUP_PATH: PAPER_SETUP_SHA256,
                PAPER_STUDY_PATH: PAPER_STUDY_SHA256,
            }),
            ("causal_clean_drf_commit", CAUSAL_CLEAN_COMMIT),
        ):
            if completion.get(field) != expected:
                raise SystemExit(f"{path.name}: invalid completion field {field}")
        if slice_doc["manifest_checksum"] != manifest["manifest_checksum"]:
            raise SystemExit(f"{path.name}: invalid manifest slice")
        frame = pd.read_parquet(io.BytesIO(payloads["confirmatory_results.parquet"]))
        validate_rows(frame, slice_doc, path.name)
        index = int(completion["shard_index"])
        if int(completion["shard_total"]) != generation["n_shards"]:
            raise SystemExit(f"{path.name}: shard-total mismatch")
        keys = {item["cell_key"] for item in slice_doc["cells"]}
        if keys & global_keys:
            raise SystemExit(f"{path.name}: overlaps a previously accepted shard")
        if index in accepted:
            raise SystemExit(f"duplicate shard index {index}")
        accepted[index] = {"path": path, "raw": raw, "payloads": payloads,
                           "frame": frame, "keys": keys}
        global_keys |= keys

    expected_indices = set(range(generation["n_shards"]))
    if set(accepted) != expected_indices and not args.partial:
        raise SystemExit(
            f"shard coverage mismatch: missing={sorted(expected_indices-set(accepted))}"
        )
    if global_keys - declared_global:
        raise SystemExit("bundles contain undeclared cell keys")
    if global_keys != declared_global and not args.partial:
        raise SystemExit(f"global cell coverage missing {len(declared_global-global_keys)} cells")

    raw_directory = RESULTS_DIRECTORY / "colab_bundles"
    SHARDS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for index, record in accepted.items():
        raw_target = raw_directory / record["path"].name
        write_new_or_identical(raw_target, record["raw"])
        parquet_target = SHARDS_DIRECTORY / f"shard_colab_{index:02d}.parquet"
        write_new_or_identical(
            parquet_target, record["payloads"]["confirmatory_results.parquet"]
        )
        write_new_or_identical(
            parquet_target.with_suffix(".meta.json"),
            json.dumps({
                "manifest_checksum": manifest["manifest_checksum"],
                "estimator_source_hash": manifest["estimator_source_hash"],
                "evaluation_protocol_id": manifest["evaluation_protocol_id"],
                "source_archive_sha256": generation["source_archive_sha256"],
                "source_bundle": record["path"].name,
                "source_bundle_sha256": sha256(record["raw"]),
            }, indent=2).encode("utf-8") + b"\n",
        )
    audit = {
        "status": "PARTIAL" if global_keys != declared_global else "PASS",
        "source_directory": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "manifest_checksum": manifest["manifest_checksum"],
        "source_archive_sha256": generation["source_archive_sha256"],
        "archives_imported": len(accepted),
        "cells_observed": len(global_keys),
        "cells_declared": len(declared_global),
        "missing_shards": sorted(expected_indices - set(accepted)),
        "failed_cells": int(sum(
            record["frame"].loc[record["frame"].status == "failed", "cell_key"].nunique()
            for record in accepted.values()
        )),
    }
    write_new_or_identical(
        RESULTS_DIRECTORY / "colab_import_audit.json",
        json.dumps(audit, indent=2).encode("utf-8") + b"\n",
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
