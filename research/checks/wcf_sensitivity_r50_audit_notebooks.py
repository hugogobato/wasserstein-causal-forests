#!/usr/bin/env python3
"""Audit the r50 sensitivity Colab notebooks against the frozen manifest.

Checks that every notebook parses, carries the frozen contract identifiers,
and that the union of the embedded manifest slices covers every declared cell
exactly once.

Usage::

    python3 research/checks/wcf_sensitivity_r50_audit_notebooks.py \
        --shards-dir colab/wcf_sensitivity_r50_120_shards
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.run_wcf_sensitivity_r50 import load_manifest  # noqa: E402

SLICE_PATTERN = re.compile(
    r"MANIFEST_SLICE = json\.loads\('''(.*?)'''\)", re.DOTALL
)
INDEX_PATTERN = re.compile(r"SHARD_INDEX = (\d+)")
SOURCE_SHA_PATTERN = re.compile(r"SOURCE_ARCHIVE_SHA256 = '([0-9a-f]{64})'")


def audit(shards_dir: Path, generation_record: Path | None) -> dict:
    manifest = load_manifest()
    declared = {item["cell_key"] for item in manifest["cells"]}
    seen: dict[str, int] = {}
    notebooks = sorted(shards_dir.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"no notebooks in {shards_dir}")
    expected_sha = None
    if generation_record and generation_record.exists():
        record = json.loads(generation_record.read_text(encoding="utf-8"))
        expected_sha = record.get("source_archive_sha256")
    for path in notebooks:
        file_index = int(re.search(r"_shard_(\d+)\.ipynb$", path.name).group(1))
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if len(notebook["cells"]) != 11:
            raise SystemExit(f"{path.name}: expected 11 cells")
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        text = "\n".join(sources)
        match = SLICE_PATTERN.search(text)
        if not match:
            raise SystemExit(f"{path.name}: missing manifest slice")
        slice_document = json.loads(match.group(1))
        shard_index = int(INDEX_PATTERN.search(text).group(1))
        if shard_index != file_index:
            raise SystemExit(f"{path.name}: shard index {shard_index} != {file_index}")
        if slice_document["manifest_checksum"] != manifest["manifest_checksum"]:
            raise SystemExit(f"{path.name}: manifest checksum mismatch")
        if slice_document["estimator_source_hash"] != manifest["estimator_source_hash"]:
            raise SystemExit(f"{path.name}: estimator source hash mismatch")
        if slice_document["manifest_contract_id"] != manifest["manifest_contract_id"]:
            raise SystemExit(f"{path.name}: manifest contract mismatch")
        if slice_document["evaluation_protocol_id"] != manifest["evaluation_protocol_id"]:
            raise SystemExit(f"{path.name}: evaluation protocol mismatch")
        source_sha = SOURCE_SHA_PATTERN.search(text)
        if not source_sha:
            raise SystemExit(f"{path.name}: missing source archive hash")
        if expected_sha and source_sha.group(1) != expected_sha:
            raise SystemExit(f"{path.name}: source archive hash mismatch")
        for item in slice_document["cells"]:
            key = item["cell_key"]
            if key in seen:
                raise SystemExit(f"{path.name}: duplicate cell {key}")
            if key not in declared:
                raise SystemExit(f"{path.name}: unknown cell {key}")
            seen[key] = shard_index
    missing = declared - set(seen)
    if missing:
        raise SystemExit(f"missing {len(missing)} cells, e.g. {sorted(missing)[:3]}")
    return {
        "status": "PASS",
        "notebooks": len(notebooks),
        "cells_declared": len(declared),
        "cells_covered": len(seen),
        "n_duplicates": 0,
        "manifest_checksum": manifest["manifest_checksum"],
        "estimator_source_hash": manifest["estimator_source_hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-dir", default=str(ROOT / "colab" / "wcf_sensitivity_r50_120_shards")
    )
    parser.add_argument("--generation-record", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    shards_dir = Path(args.shards_dir)
    if not shards_dir.is_absolute():
        shards_dir = ROOT / shards_dir
    record = (
        Path(args.generation_record)
        if args.generation_record
        else shards_dir / "notebook_generation.json"
    )
    audit_result = audit(shards_dir, record)
    if not args.no_write:
        (shards_dir / "notebook_audit.json").write_text(
            json.dumps(audit_result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(audit_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
