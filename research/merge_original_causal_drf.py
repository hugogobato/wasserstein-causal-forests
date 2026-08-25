#!/usr/bin/env python3
"""Audit and merge the causal-only rerun made with the paper implementation."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.merge import merge_results  # noqa: E402


def main() -> int:
    source_manifest = ROOT / "results/manifests/main_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    cells = [cell for cell in manifest["cells"] if cell["method"] == "causal_drf"]
    rerun_manifest = dict(manifest)
    rerun_manifest["cells"] = cells
    rerun_manifest["n_cells"] = len(cells)
    rerun_manifest["manifest_checksum"] = hashlib.sha256(
        json.dumps(cells, sort_keys=True).encode("utf-8")
    ).hexdigest()

    manifest_path = ROOT / "results/manifests/original_causal_drf_manifest.json"
    manifest_path.write_text(json.dumps(rerun_manifest, indent=2), encoding="utf-8")

    shard_directory = ROOT / "results/original_causal_drf_shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    for shard in sorted((ROOT / "results/main").glob("shard_original_causal_drf_*.parquet")):
        shutil.copy2(shard, shard_directory / shard.name)

    output_directory = ROOT / "results/merged_original_causal_drf"
    audit = merge_results(shard_directory, manifest_path, output_directory)
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
