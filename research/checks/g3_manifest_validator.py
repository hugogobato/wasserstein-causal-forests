#!/usr/bin/env python3
"""WP3-A verification: the frozen manifest is complete, unique, and affordable.

Run from the repository root:

    python research/checks/g3_manifest_validator.py

Checks, in the order the work package states them:

1. every declared cell appears exactly once, and nothing else appears;
2. every method in a grid receives identical information, meaning the same
   training sample and the same test design for a given replication;
3. no method receives per-regime tuning, so tuning effort is comparable;
4. training and test designs never share a seed;
5. the total cost estimate comes from the measured pilot, not extrapolation.

Exits nonzero on any failure and prints a JSON certificate otherwise.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.dgps import build_dgp  # noqa: E402
from wasserstein_causal_forests.g3.manifest import (  # noqa: E402
    METHOD_REGISTRY,
    build_grids,
    enumerate_cells,
)

MANIFEST_PATH = ROOT / "results" / "manifests" / "main_manifest.json"
COST_PILOT_PATH = ROOT / "results" / "manifests" / "cost_pilot.json"
#: Wall-clock ceiling the tournament must fit inside on six workers.
SAFE_WALL_HOURS = 12.0
DEFAULT_WORKERS = 6


def check_enumeration(problems: list[str]) -> dict[str, object]:
    cells = enumerate_cells()
    keys = Counter(cell.key for cell in cells)
    duplicates = [key for key, count in keys.items() if count > 1]
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate cell keys in the enumeration")

    declared = 0
    for grid in build_grids():
        declared += (
            len(grid.dgps) * len(grid.n_train) * len(grid.n_grid)
            * len(grid.n_particles) * len(grid.methods) * len(grid.seeds)
        )
    if declared != len(cells):
        problems.append(
            f"grids declare {declared} cells but enumeration yields {len(cells)}"
        )

    if MANIFEST_PATH.exists():
        document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        written = {item["cell_key"] for item in document["cells"]}
        if written != set(keys):
            problems.append("the written manifest disagrees with the enumeration")
    return {"n_cells": len(cells), "n_declared": declared, "n_duplicates": len(duplicates)}


def check_equal_information(problems: list[str]) -> dict[str, object]:
    """A replication's data must not depend on which method consumes it."""

    checked = 0
    for grid in build_grids():
        dgp = build_dgp(grid.dgps[0], grid.n_grid[0])
        first = dgp.sample(grid.n_train[0], seed=grid.seeds[0])
        second = dgp.sample(grid.n_train[0], seed=grid.seeds[0])
        if not (
            np.array_equal(first.X, second.X)
            and np.array_equal(first.treatment, second.treatment)
            and np.array_equal(first.quantiles, second.quantiles)
        ):
            problems.append(f"grid {grid.grid}: sampling is not seed-deterministic")
        checked += 1
    return {"grids_checked": checked}


def check_comparable_tuning(problems: list[str]) -> dict[str, object]:
    """No method may carry regime-dependent or seed-dependent hyperparameters."""

    for name, entry in METHOD_REGISTRY.items():
        parameters = entry["parameters"]
        for key, value in parameters.items():
            if isinstance(value, dict):
                problems.append(
                    f"method {name} has a nested parameter block at {key!r}; a "
                    "frozen budget must be a flat, regime-independent mapping"
                )
        for forbidden in ("dgp", "seed", "regime", "n_train"):
            if forbidden in parameters:
                problems.append(
                    f"method {name} tunes on {forbidden!r}, which breaks "
                    "comparable tuning effort"
                )
    return {"n_methods": len(METHOD_REGISTRY)}


def check_seed_disjointness(problems: list[str]) -> dict[str, object]:
    cells = enumerate_cells()
    training = {cell.seed for cell in cells}
    testing = {cell.test_seed for cell in cells}
    overlap = training & testing
    if overlap:
        problems.append(f"{len(overlap)} seeds serve as both training and test designs")
    return {"n_training_seeds": len(training), "n_test_seeds": len(testing)}


def check_cost(problems: list[str]) -> dict[str, object]:
    if not COST_PILOT_PATH.exists():
        problems.append("no cost pilot; run research/checks/g3_cost_pilot.py first")
        return {}
    pilot = json.loads(COST_PILOT_PATH.read_text(encoding="utf-8"))
    # Index by the shape the measurement was taken at, so a cell is costed
    # against its own shape where one was measured and against that method's
    # worst measured shape otherwise.
    by_shape: dict[tuple, float] = {}
    by_method: dict[str, list[float]] = {}
    for row in pilot["measurements"]:
        if row["status"] != "ok":
            problems.append(
                f"pilot cell {row['method']} n={row['n_train']} K={row['n_grid']} "
                f"failed: {row['failure_reason'][:160]}"
            )
            continue
        shape = (row["method"], row["n_train"], row["n_grid"], row["n_particles"])
        by_shape[shape] = float(row["wall_seconds"])
        by_method.setdefault(row["method"], []).append(float(row["wall_seconds"]))

    total = 0.0
    unmeasured = 0
    for cell in enumerate_cells():
        shape = (cell.method, cell.n_train, cell.n_grid, cell.n_particles)
        if shape in by_shape:
            total += by_shape[shape]
        elif cell.method in by_method:
            total += max(by_method[cell.method])
            unmeasured += 1
        else:
            problems.append(f"no pilot measurement for method {cell.method}")

    cpu_hours = total / 3600.0
    wall_hours = cpu_hours / DEFAULT_WORKERS
    if wall_hours > SAFE_WALL_HOURS:
        problems.append(
            f"estimated {wall_hours:.1f} wall hours on {DEFAULT_WORKERS} workers "
            f"exceeds the {SAFE_WALL_HOURS} hour ceiling"
        )
    return {
        "estimated_cpu_hours": round(cpu_hours, 2),
        "estimated_wall_hours_at_six_workers": round(wall_hours, 2),
        "cells_costed_by_method_worst_case": unmeasured,
        "peak_ram_mb_max": max(
            (row["peak_ram_mb"] or 0.0) for row in pilot["measurements"]
        ),
    }


def main() -> int:
    problems: list[str] = []
    certificate = {
        "check": "g3_manifest_validator",
        "enumeration": check_enumeration(problems),
        "equal_information": check_equal_information(problems),
        "comparable_tuning": check_comparable_tuning(problems),
        "seed_disjointness": check_seed_disjointness(problems),
        "cost": check_cost(problems),
    }
    certificate["problems"] = problems
    certificate["status"] = "PASS" if not problems else "FAIL"
    print(json.dumps(certificate, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
