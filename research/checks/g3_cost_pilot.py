#!/usr/bin/env python3
"""WP3-A cost pilot: one seed per method per representative cell.

Run from the repository root:

    python research/checks/g3_cost_pilot.py

Measures wall time and peak RAM for every method at the sample sizes and grid
resolutions the frozen grids use, so the manifest's total cost estimate rests on
measurement rather than extrapolation. Writes
`results/manifests/cost_pilot.json`.

This runs before the preregistration is frozen and touches no decisive seed: it
uses seed 9999, which the manifest does not enumerate, so no pilot cell can be
confused with a tournament cell.
"""

from __future__ import annotations

import os

# Pin every numerical library to one thread BEFORE any of them is imported.
# OpenMP reads these at initialisation, so setting them after `import numpy` or
# `import stochtree` is silently ineffective: the pilot originally did that and
# spent 20 minutes on a cell that takes 32 seconds single-threaded, because
# eight OpenMP threads spin-waited on work too small to divide.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
    os.environ[_variable] = "1"

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3 import r_bridge  # noqa: E402
from wasserstein_causal_forests.g3.manifest import Cell, build_grids  # noqa: E402
from wasserstein_causal_forests.g3.runner import run_cell  # noqa: E402

PILOT_SEED = 9999
CACHE = ROOT / "results" / "rcpp_cache"
OUTPUT = ROOT / "results" / "manifests" / "cost_pilot.json"
#: One representative regime. D6 is the most expensive truth (its outer law is
#: a mixture, so the quadrature rule has twice the nodes), which keeps the
#: estimate conservative.
PILOT_DGP = "D6"


def pilot_cells() -> list[Cell]:
    """One cell per (grid, method, n, K, M) shape, at a seed outside the manifest."""

    seen: set[tuple] = set()
    cells: list[Cell] = []
    for grid in build_grids():
        for method in grid.methods:
            for n_train in grid.n_train:
                for n_grid in grid.n_grid:
                    for n_particles in grid.n_particles:
                        shape = (method, n_train, n_grid, n_particles)
                        if shape in seen:
                            continue
                        seen.add(shape)
                        cells.append(
                            Cell(
                                grid=grid.grid,
                                dgp=PILOT_DGP,
                                n_train=n_train,
                                n_grid=n_grid,
                                n_particles=n_particles,
                                method=method,
                                seed=PILOT_SEED,
                            )
                        )
    return cells


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print("warming the Rcpp cache before any timing", flush=True)
    r_bridge.warm_rcpp_cache(CACHE)

    measurements: list[dict[str, object]] = []
    started = time.time()
    for cell in pilot_cells():
        rows = run_cell(cell, cache_directory=CACHE)
        head = rows[0]
        wall = float(head["wall_seconds"])
        status = head["status"] if head["metric"] == "cell_failure" else "ok"
        peak = next(
            (row["value"] for row in rows if row["metric"] == "peak_ram"), None
        )
        measurements.append(
            {
                "grid": cell.grid,
                "method": cell.method,
                "n_train": cell.n_train,
                "n_grid": cell.n_grid,
                "n_particles": cell.n_particles,
                "wall_seconds": round(wall, 2),
                "peak_ram_mb": None if peak is None else round(float(peak), 1),
                "status": status,
                "failure_reason": head.get("failure_reason", ""),
            }
        )
        print(
            f"{cell.method:16s} n={cell.n_train:5d} K={cell.n_grid:3d} "
            f"M={cell.n_particles:3d}  {wall:8.1f}s  {status}"
            + (f"  {head.get('failure_reason', '')[:120]}" if status != "ok" else ""),
            flush=True,
        )

    document = {
        "check": "g3_cost_pilot",
        "pilot_seed": PILOT_SEED,
        "pilot_dgp": PILOT_DGP,
        "elapsed_seconds": round(time.time() - started, 1),
        "measurements": measurements,
    }
    OUTPUT.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"\nwrote {OUTPUT}", flush=True)
    return 0 if all(m["status"] == "ok" for m in measurements) else 1


if __name__ == "__main__":
    raise SystemExit(main())
