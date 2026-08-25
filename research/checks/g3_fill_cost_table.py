#!/usr/bin/env python3
"""Insert the measured cost table into the preregistration.

Run after `g3_cost_pilot.py`, before freezing the manifest:

    python research/checks/g3_fill_cost_table.py

Replaces the `<!-- COST_TABLE -->` marker in
`research/simulation_preregistration.md` with the measured per-cell wall time
and peak memory, and appends the projected total. Keeping this mechanical means
the document's cost basis is the pilot's output rather than a transcription.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.manifest import enumerate_cells  # noqa: E402

PILOT = ROOT / "results" / "manifests" / "cost_pilot.json"
DOCUMENT = ROOT / "research" / "simulation_preregistration.md"
MARKER = "<!-- COST_TABLE -->"
WORKERS = 6


def main() -> int:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    measurements = pilot["measurements"]

    lines = [
        "| Method | $n$ | $K$ | $M$ | Wall seconds per cell | Peak RSS delta (MB) | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in measurements:
        peak = "n/a" if row["peak_ram_mb"] is None else f"{row['peak_ram_mb']:.0f}"
        lines.append(
            f"| `{row['method']}` | {row['n_train']} | {row['n_grid']} | "
            f"{row['n_particles']} | {row['wall_seconds']:.1f} | {peak} | "
            f"{row['status']} |"
        )

    by_shape = {
        (r["method"], r["n_train"], r["n_grid"], r["n_particles"]): r["wall_seconds"]
        for r in measurements if r["status"] == "ok"
    }
    by_method: dict[str, list[float]] = {}
    for r in measurements:
        if r["status"] == "ok":
            by_method.setdefault(r["method"], []).append(r["wall_seconds"])

    per_method_total: dict[str, float] = {}
    unmeasured = 0
    for cell in enumerate_cells():
        shape = (cell.method, cell.n_train, cell.n_grid, cell.n_particles)
        if shape in by_shape:
            cost = by_shape[shape]
        elif cell.method in by_method:
            cost = max(by_method[cell.method])
            unmeasured += 1
        else:
            continue
        per_method_total[cell.method] = per_method_total.get(cell.method, 0.0) + cost

    total = sum(per_method_total.values())
    lines.append("")
    lines.append("Projected total, costing every manifest cell at its own measured")
    lines.append("shape where one exists and at that method's worst measured shape")
    lines.append(f"otherwise ({unmeasured} cells fall in the second case):")
    lines.append("")
    lines.append("| Method | Cells | Projected CPU hours |")
    lines.append("|---|---|---|")
    counts: dict[str, int] = {}
    for cell in enumerate_cells():
        counts[cell.method] = counts.get(cell.method, 0) + 1
    for method in sorted(per_method_total, key=lambda m: -per_method_total[m]):
        lines.append(
            f"| `{method}` | {counts.get(method, 0)} | "
            f"{per_method_total[method] / 3600.0:.1f} |"
        )
    lines.append(f"| **total** | {sum(counts.values())} | **{total / 3600.0:.1f}** |")
    lines.append("")
    lines.append(
        f"At {WORKERS} workers that is about {total / 3600.0 / WORKERS:.1f} wall "
        "hours. The projection is conservative in two ways: it costs every cell "
        "on regime D6, whose mixture outer law gives the quadrature truth twice "
        "the nodes of any other regime, and it charges every cell a full oracle "
        "evaluation, whereas the dispatcher keeps a replication's methods in one "
        "worker so they share one cached oracle."
    )

    text = DOCUMENT.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit(f"marker {MARKER} not found in {DOCUMENT}")
    DOCUMENT.write_text(text.replace(MARKER, "\n".join(lines)), encoding="utf-8")
    print(f"filled the cost table in {DOCUMENT}")
    print(f"projected {total / 3600.0:.1f} CPU hours, "
          f"{total / 3600.0 / WORKERS:.1f} wall hours at {WORKERS} workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
