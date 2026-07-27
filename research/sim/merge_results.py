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


def load_rows(paths: list[str | Path]) -> list[dict]:
    """Load shard files and reject duplicate cells or result rows."""
    merged: list[dict] = []
    seen_cells: set[CellKey] = set()
    seen_results: set[ResultKey] = set()
    for raw_path in paths:
        path = Path(raw_path)
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list")
        validate_result_rows(rows)
        for row in rows:
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
    print(f"Merged {len(args.inputs)} files, {len(cells)} cells, {len(rows)} rows")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
