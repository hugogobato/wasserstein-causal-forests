#!/usr/bin/env python3
"""Combine every WCF sensitivity stage into one audit-checked parquet.

Stages: primary (880 cells), null companions (360), factorial corners (80),
fresh-seed confirmation (160), and the random-forest flexible propensity run
(40). Cells are distinct across stages by construction; this script refuses to
write a combined file if any cell key appears twice, and records the per-stage
cell counts in a sidecar audit next to the parquet.

Usage:

    python3 research/checks/wcf_sensitivity_combine.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "wcf_sensitivity"
OUTPUT = RESULTS / "merged" / "wcf_sensitivity_combined.parquet"
AUDIT = RESULTS / "merged" / "combined_audit.json"

STAGES = {
    "primary": RESULTS / "merged" / "wcf_sensitivity_results.parquet",
    "nulls": RESULTS / "nulls" / "merged" / "wcf_sensitivity_results.parquet",
    "factorial": RESULTS / "factorial" / "merged" / "wcf_sensitivity_results.parquet",
    "confirm": RESULTS / "confirm" / "merged" / "wcf_sensitivity_results.parquet",
    "flex_rf": RESULTS / "flex_rf" / "merged" / "wcf_sensitivity_results.parquet",
}


def main() -> int:
    frames = []
    audit: dict[str, object] = {"stages": {}, "duplicate_cells": [], "n_rows": 0}
    seen: set[str] = set()
    for stage, path in STAGES.items():
        if not path.exists():
            raise SystemExit(f"stage {stage!r} merged parquet is missing: {path}")
        frame = pd.read_parquet(path)
        keys = set(frame["cell_key"])
        duplicates = keys & seen
        if duplicates:
            raise SystemExit(
                f"{stage}: {len(duplicates)} cell keys overlap earlier stages"
            )
        seen |= keys
        frames.append(frame)
        audit["stages"][stage] = {
            "path": str(path.relative_to(ROOT)),
            "n_cells": len(keys),
            "n_rows": int(len(frame)),
            "methods": sorted(frame["method"].unique()),
            "grids": sorted(frame["grid"].unique()),
        }
        print(f"{stage:10s} {len(keys):5d} cells, {len(frame):6d} rows")
    combined = pd.concat(frames, ignore_index=True)
    audit["n_cells"] = len(seen)
    audit["n_rows"] = int(len(combined))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT, index=False)
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(seen)} cells, {len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
