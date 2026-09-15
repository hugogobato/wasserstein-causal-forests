#!/usr/bin/env python3
"""Validate and import WCF sample-size Colab ZIPs into a runner stage."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "execution_log.jsonl",
    "wcf_sample_size_parquet.meta.json",
    "wcf_sample_size_parquet.parquet",
    "shard_config.json",
    "manifest_slice.json",
}


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"refusing to overwrite nonidentical file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir")
    parser.add_argument("--stage", default="all_dgps")
    args = parser.parse_args()

    source = Path(args.source_dir)
    if not source.is_absolute():
        source = ROOT / source
    stage = ROOT / "results" / "wcf_sample_size_sensitivity" / args.stage
    manifest_path = stage / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing frozen manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = {cell["cell_key"] for cell in manifest["cells"]}

    archives = sorted(source.glob("*.zip"))
    if not archives:
        raise SystemExit(f"no ZIP archives found in {source}")
    accepted: dict[int, dict] = {}
    duplicate_archives: list[dict] = []
    all_slice_keys: set[str] = set()
    status_counts: Counter[str] = Counter()

    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            members = set(bundle.namelist())
            missing = REQUIRED - members
            if missing:
                raise SystemExit(f"{archive.name} is missing {sorted(missing)}")
            config = json.loads(bundle.read("shard_config.json"))
            sidecar = json.loads(bundle.read("wcf_sample_size_parquet.meta.json"))
            slice_document = json.loads(bundle.read("manifest_slice.json"))
            parquet_bytes = bundle.read("wcf_sample_size_parquet.parquet")
            log_bytes = bundle.read("execution_log.jsonl")

        for field, expected in (
            ("manifest_checksum", manifest["manifest_checksum"]),
            ("manifest_contract_id", manifest["manifest_contract_id"]),
        ):
            if config[field] != expected:
                raise SystemExit(f"{archive.name}: invalid {field}")
        if sidecar["manifest_checksum"] != manifest["manifest_checksum"]:
            raise SystemExit(f"{archive.name}: sidecar checksum mismatch")
        if sidecar["estimator_source_hash"] != manifest["estimator_source_hash"]:
            raise SystemExit(f"{archive.name}: estimator source hash mismatch")
        if sidecar["contract_id"] != manifest["manifest_contract_id"]:
            raise SystemExit(f"{archive.name}: sidecar contract mismatch")
        if slice_document["manifest_checksum"] != manifest["manifest_checksum"]:
            raise SystemExit(f"{archive.name}: manifest-slice checksum mismatch")

        index = int(config["shard_index"])
        slice_keys = {cell["cell_key"] for cell in slice_document["cells"]}
        frame = pd.read_parquet(io.BytesIO(parquet_bytes))
        result_keys = set(frame["cell_key"].dropna().astype(str))
        if slice_keys != result_keys:
            raise SystemExit(
                f"{archive.name}: result keys differ from its manifest slice "
                f"(missing={len(slice_keys-result_keys)}, extra={len(result_keys-slice_keys)})"
            )
        if not slice_keys <= declared:
            raise SystemExit(f"{archive.name}: contains cells absent from frozen manifest")
        record = {
            "archive": archive.name,
            "index": index,
            "slice_keys": slice_keys,
            "parquet": parquet_bytes,
            "sidecar": bundle_sidecar_bytes(sidecar),
            "log": log_bytes,
            "rows": len(frame),
        }
        if index in accepted:
            prior = accepted[index]
            if prior["slice_keys"] != slice_keys or prior["parquet"] != parquet_bytes:
                raise SystemExit(
                    f"conflicting archives for shard index {index}: "
                    f"{prior['archive']} and {archive.name}"
                )
            duplicate_archives.append(
                {"index": index, "kept": prior["archive"], "duplicate": archive.name}
            )
            continue
        accepted[index] = record
        status_counts.update(frame["status"].fillna("missing").astype(str))
        overlap = all_slice_keys & slice_keys
        if overlap:
            raise SystemExit(f"cell overlap across shard indices: {len(overlap)} cells")
        all_slice_keys.update(slice_keys)

    # Trust the shard_total declared by every archive rather than a filename.
    shard_totals = set()
    for path in archives:
        with zipfile.ZipFile(path) as bundle:
            shard_totals.add(json.loads(bundle.read("shard_config.json"))["shard_total"])
    if len(shard_totals) != 1:
        raise SystemExit(f"inconsistent shard totals: {sorted(shard_totals)}")
    shard_total = int(next(iter(shard_totals)))
    expected_indices = set(range(shard_total))
    if set(accepted) != expected_indices:
        raise SystemExit(
            f"shard-index coverage failure: missing={sorted(expected_indices-set(accepted))}, "
            f"extra={sorted(set(accepted)-expected_indices)}"
        )
    if all_slice_keys != declared:
        raise SystemExit(
            f"global cell coverage failure: missing={len(declared-all_slice_keys)}, "
            f"extra={len(all_slice_keys-declared)}"
        )

    for index, record in sorted(accepted.items()):
        _write_new_or_identical(stage / "shards" / f"shard_colab_{index:02d}.parquet", record["parquet"])
        _write_new_or_identical(stage / "shards" / f"shard_colab_{index:02d}.meta.json", record["sidecar"])
        _write_new_or_identical(stage / "logs" / f"shard_colab_{index:02d}.jsonl", record["log"])

    audit = {
        "source_directory": str(source.relative_to(ROOT)),
        "stage": args.stage,
        "manifest_checksum": manifest["manifest_checksum"],
        "archives_seen": len(archives),
        "shards_imported": len(accepted),
        "cells_declared": len(declared),
        "cells_covered": len(all_slice_keys),
        "duplicate_archives": duplicate_archives,
        "row_status_counts": dict(sorted(status_counts.items())),
        "archive_by_shard": {
            str(index): record["archive"] for index, record in sorted(accepted.items())
        },
    }
    audit_path = stage / "colab_import_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


def bundle_sidecar_bytes(sidecar: dict) -> bytes:
    return (json.dumps(sidecar, indent=2) + "\n").encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
