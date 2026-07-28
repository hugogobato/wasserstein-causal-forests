"""Merge and validate independent simulation-shard JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.evaluation import validate_result_rows


CellKey = tuple[str, int, str, int]
ResultKey = tuple[str, int, str, int, str, str]


def _manifest_version(row: dict) -> str:
    parts = str(row["evaluation_manifest_id"]).split("-")
    if len(parts) < 2 or not parts[1].startswith("v"):
        raise ValueError(
            f"cannot read an evaluation contract version from {row['evaluation_manifest_id']!r}"
        )
    return parts[1]


def load_rows(paths: list[str | Path]) -> list[dict]:
    """Load shard files and reject duplicate cells, rows, or mixed contracts."""
    merged: list[dict] = []
    seen_cells: set[CellKey] = set()
    seen_results: set[ResultKey] = set()
    versions: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list")
        validate_result_rows(rows)
        for row in rows:
            # v2 rows standardized worst_standardized_error by a
            # realization-dependent empirical scale and v3 rows use the frozen
            # scale, so the two are not comparable on that metric.  Merging them
            # would silently average incompatible numbers.
            versions.add(_manifest_version(row))
            if len(versions) > 1:
                raise ValueError(
                    "inputs mix evaluation contract versions "
                    f"{sorted(versions)}; merge each pilot separately"
                )
            cell = (
                row["dgp_id"], row["n_regions"],
                row["observation_regime"], row["seed"],
            )
            result = cell + (row["method"], row["metric"])
            if result in seen_results:
                raise ValueError(f"duplicate result row across inputs: {result}")
            seen_results.add(result)
            seen_cells.add(cell)
            merged.append(row)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSON files produced by runner.py")
    parser.add_argument("--out", required=True, help="merged JSON output path")
    args = parser.parse_args()

    rows = load_rows(args.inputs)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, default=str))
    cells = {
        (row["dgp_id"], row["n_regions"], row["observation_regime"], row["seed"])
        for row in rows
    }
    methods = {row["method"] for row in rows}
    print(f"Merged {len(args.inputs)} files, {len(cells)} cells, {len(rows)} rows")
    print(f"Contract: {_manifest_version(rows[0])}, {len(methods)} methods")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
