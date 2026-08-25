#!/usr/bin/env python3
"""Audit and merge the DRF-only rerun made with the supplied paper code."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.manifest import Cell  # noqa: E402
from wasserstein_causal_forests.g3.merge import merge_results  # noqa: E402


def main() -> int:
    source_manifest = ROOT / "results/manifests/main_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    # The paper DRF is a Section 11 add-on, not a member of the frozen G3
    # method roster. Clone the old W-DRF-T coordinates so the add-on has the
    # identical cells, seeds, test designs, and contract fields without
    # changing the historical manifest or gate count.
    cells = []
    for cell in manifest["cells"]:
        if cell["method"] != "wdrft" or cell["grid"] not in {"main", "smallk"}:
            continue
        cells.append(Cell(
            cell["grid"], cell["dgp"], cell["n_train"], cell["n_grid"],
            cell["n_particles"], "drf", cell["seed"]
        ).to_dict())
    rerun_manifest = dict(manifest)
    rerun_manifest["cells"] = cells
    rerun_manifest["n_cells"] = len(cells)
    rerun_manifest["method_registry"] = {
        **manifest["method_registry"],
        "drf": {
            "role": "baseline",
            "adapter": "forest",
            "produces_law": True,
            "parameters": {"method": "drf"},
        },
    }
    rerun_manifest["manifest_checksum"] = hashlib.sha256(
        json.dumps(cells, sort_keys=True).encode("utf-8")
    ).hexdigest()

    manifest_path = ROOT / "results/manifests/original_drf_manifest.json"
    manifest_path.write_text(json.dumps(rerun_manifest, indent=2), encoding="utf-8")

    shard_directory = ROOT / "results/original_drf_shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    for shard in sorted((ROOT / "results/main").glob("shard_original_drf_*.parquet")):
        shutil.copy2(shard, shard_directory / shard.name)

    output_directory = ROOT / "results/merged_original_drf"
    audit = merge_results(shard_directory, manifest_path, output_directory)
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
