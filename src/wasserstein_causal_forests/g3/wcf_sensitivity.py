"""Isolated manifest for the WCF sensitivity and assignment-DGP study.

The study in `report/sensitivity_and_dgp_experiments.md` is deliberately kept
out of the historical tournament manifests: the cells here are built directly
from `Cell` objects, stored under `results/wcf_sensitivity/`, and never touch
`results/manifests/`. The DGP and method identifiers are referenced as strings
only, so this module imports without the companion
`g3.sensitivity_dgps`/`g3.sensitivity_methods` modules; the launcher registers
those lazily and the runner resolves them at fit time.

Three distinctions matter for what a cell means:

* `Cell.key` hashes only the seven cell coordinates. Every estimator setting
  that is not in the key (fold count, classifier factory, clipping interval)
  lives in the serialized method registry, which `apply_method_registry`
  installs into `METHOD_REGISTRY` in every worker before any fit.
* The primary `cwdb_dr` specification is three-fold cross-fitting with the
  logistic propensity factory, matching the principal income specification.
  The flexible and oracle variants are distinct method names, so their rows can
  never be confused with the principal ones.
* The factorial block reuses the one-factor coordinates it overlaps and adds
  exactly eighty new IC1/IC3 cells; deduplication is by `Cell.key`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .dgps import MODERATOR_EDGES
from .manifest import (
    BOOSTING_BUDGET,
    ESTIMAND_CONTRACT_ID,
    EVALUATION_MANIFEST,
    METHOD_REGISTRY,
    N_TEST,
    TEST_SEED_OFFSET,
    TRAINING_FUNCTIONALS,
    Cell,
)

WCF_SENSITIVITY_CONTRACT_ID = "WCF-SENSITIVITY-v1"

INCOME_GRID = "wcf_sensitivity_v1_logit_f3"
SYM_GRID = "wcf_sensitivity_v1_sym"
SYM_NULL_GRID = "wcf_sensitivity_v1_sym_null"
ALIGN_GRID = "wcf_sensitivity_v1_align"
ALIGN_NULL_GRID = "wcf_sensitivity_v1_align_null"
FLEX_GRID = "wcf_sensitivity_v1_flex"

INCOME_DGPS: tuple[str, ...] = ("IC0", "IC1", "IC2", "IC3")
SYM_DGPS: tuple[str, ...] = ("SYM-RANDOM", "SYM-LIN", "SYM-NL", "SYM-MU")
ALIGN_DGPS: tuple[str, ...] = ("SYM-ALIGN", "SYM-IRREL")
PROPENSITY_DGPS: tuple[str, ...] = ("SYM-NL", "SYM-MU")

N_TRAIN_INCOME = 1000
SYM_N_TRAIN: tuple[int, ...] = (500, 1000)
SEEDS: tuple[int, ...] = tuple(range(10))

PRIMARY_INCOME_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 10), (25, 10), (49, 10), (25, 5), (25, 25),
)
FACTORIAL_INCOME_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (k, m) for k in (5, 25, 49) for m in (5, 10, 25)
)

PROPENSITY_CLIP = (0.02, 0.98)
COMMON_GRID_LEVELS = "COMMON199+INTERIOR"
CONTRAST_CANDIDATES: tuple[float, ...] = (0.0, 50.0, 500.0)
SELECTION_FOLDS = 3

#: Metrics the merge audit requires for every law-producing method over each
#: (dgp, seed, n_train) group. The common-grid metrics are deliberately absent:
#: they belong to a separate evaluator and a separate audit.
REQUIRED_LAW_METRICS: tuple[str, ...] = (
    "mean_quantile_rmse",
    "kernel_law_error",
    "tate_functional_rmse",
    "tcate_functional_rmse",
    "reference_effect_rmse",
    "reference_tcate_rmse",
)

_ESTIMATOR_SOURCE_FILES: tuple[str, ...] = (
    "src/wasserstein_causal_forests/cwdb/dr_calibration.py",
    "src/wasserstein_causal_forests/g3/phase6_methods.py",
    "src/wasserstein_causal_forests/g3/sensitivity_methods.py",
    "src/wasserstein_causal_forests/g3/sensitivity_dgps.py",
    "src/wasserstein_causal_forests/g3/common_grid.py",
)


def income_onefactor() -> list[Cell]:
    """Grid resolution at M = 10 plus particle resolution at K = 25."""

    return [
        Cell(INCOME_GRID, dgp, N_TRAIN_INCOME, k, m, "cwdb_dr", seed)
        for dgp in INCOME_DGPS
        for k, m in PRIMARY_INCOME_PAIRS
        for seed in SEEDS
    ]


def income_baselines() -> list[Cell]:
    """Forest baselines at the three K values with M held at 10."""

    return [
        Cell(INCOME_GRID, dgp, N_TRAIN_INCOME, k, 10, method, seed)
        for dgp in INCOME_DGPS
        for k in (5, 25, 49)
        for method in ("causal_drf", "drf")
        for seed in SEEDS
    ]


def income_factorial() -> list[Cell]:
    """The 3x3 interaction grid on the two scientifically decisive regimes.

    The block has 180 cells, 100 of which repeat `income_onefactor`
    coordinates and are removed by key deduplication when both are requested.
    """

    return [
        Cell(INCOME_GRID, dgp, N_TRAIN_INCOME, k, m, "cwdb_dr", seed)
        for dgp in ("IC1", "IC3")
        for k, m in FACTORIAL_INCOME_PAIRS
        for seed in SEEDS
    ]


def _sym_block(dgp_ids: Sequence[str], grid: str) -> list[Cell]:
    return [
        Cell(grid, dgp, n_train, 25, 10, method, seed)
        for dgp in dgp_ids
        for n_train in SYM_N_TRAIN
        for method in ("cwdb_dr", "causal_drf", "drf")
        for seed in SEEDS
    ]


def sym() -> list[Cell]:
    """Symmetric-law regimes under random, linear, nonlinear, and mu assignment."""

    return _sym_block(SYM_DGPS, SYM_GRID)


def sym_null() -> list[Cell]:
    """The `tau(x) = 0` placebo companions of the symmetric block."""

    return _sym_block(tuple(f"{dgp}-NULL" for dgp in SYM_DGPS), SYM_NULL_GRID)


def _align_block(dgp_ids: Sequence[str], grid: str) -> list[Cell]:
    return [
        Cell(grid, dgp, n_train, 25, 10, method, seed)
        for dgp in dgp_ids
        for n_train in SYM_N_TRAIN
        for method in ("cwdb_dr", "causal_drf", "drf")
        for seed in SEEDS
    ]


def align() -> list[Cell]:
    """Aligned versus irrelevant assignment on the same outcome surfaces."""

    return _align_block(ALIGN_DGPS, ALIGN_GRID)


def align_null() -> list[Cell]:
    """The `tau(x) = 0` placebo companions of the alignment block."""

    return _align_block(tuple(f"{dgp}-NULL" for dgp in ALIGN_DGPS), ALIGN_NULL_GRID)


def propensity() -> list[Cell]:
    """Logistic pairing rows live in `sym`; this block swaps the classifier."""

    return [
        Cell(FLEX_GRID, dgp, n_train, 25, 10, method, seed)
        for dgp in PROPENSITY_DGPS
        for n_train in SYM_N_TRAIN
        for method in ("cwdb_dr_flex", "cwdb_dr_oracle")
        for seed in SEEDS
    ]


WCF_BLOCK_BUILDERS: dict[str, Callable[[], list[Cell]]] = {
    "income_onefactor": income_onefactor,
    "income_baselines": income_baselines,
    "income_factorial": income_factorial,
    "sym": sym,
    "sym_null": sym_null,
    "align": align,
    "align_null": align_null,
    "propensity": propensity,
}

WCF_BLOCKS: tuple[str, ...] = tuple(WCF_BLOCK_BUILDERS)

WCF_PRIMARY_BLOCKS: tuple[str, ...] = (
    "income_onefactor",
    "income_baselines",
    "sym",
    "align",
    "propensity",
)


def resolve_blocks(blocks: Iterable[str] | str | None = None) -> tuple[str, ...]:
    """Canonical block names, defaulting to the primary (non-null) roster."""

    if blocks is None:
        return WCF_PRIMARY_BLOCKS
    if isinstance(blocks, str):
        blocks = (blocks,)
    names = tuple(blocks)
    unknown = [name for name in names if name not in WCF_BLOCK_BUILDERS]
    if unknown:
        raise ValueError(
            f"unknown WCF sensitivity blocks {unknown}; expected any of "
            f"{list(WCF_BLOCKS)}"
        )
    return names


def build_wcf_sensitivity_cells(
    blocks: Iterable[str] | str | None = None,
) -> list[Cell]:
    """Every requested block's cells, deduplicated by `Cell.key`.

    Overlap between `income_onefactor` and `income_factorial` is expected and
    silently removed. A key collision between two non-identical cells cannot
    occur while `Cell.key` hashes every coordinate, so if one appears the
    manifest is corrupt and construction stops.
    """

    names = resolve_blocks(blocks)
    cells: list[Cell] = []
    seen: dict[str, Cell] = {}
    for name in names:
        for cell in WCF_BLOCK_BUILDERS[name]():
            existing = seen.get(cell.key)
            if existing is not None:
                if existing != cell:
                    raise ValueError(
                        f"cell key collision across grids {existing.grid!r} "
                        f"and {cell.grid!r}: {existing} vs {cell}"
                    )
                continue
            seen[cell.key] = cell
            cells.append(cell)
    return cells


def _repository_root() -> Path:
    """Best-effort repository root, used only for provenance metadata."""

    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate
    return current.parents[3]


def estimator_source_files() -> tuple[Path, ...]:
    """Existing estimator sources that define the WCF numerics."""

    root = _repository_root()
    return tuple(
        path for name in _ESTIMATOR_SOURCE_FILES if (path := root / name).is_file()
    )


def estimator_source_hash() -> str:
    """SHA-256 over the existing estimator sources, path and bytes interleaved.

    Missing sources are skipped rather than hashed as empty, because a module
    that has not landed yet must not make two different code states look equal
    once it does.
    """

    root = _repository_root()
    digest = hashlib.sha256()
    for path in estimator_source_files():
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def software_revision() -> dict[str, Any]:
    """HEAD revision and working-tree dirtiness, best effort."""

    root = _repository_root()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return {"revision": revision, "dirty": bool(porcelain.strip()), "error": None}
    except Exception as error:  # noqa: BLE001 - provenance must not block freeze
        return {
            "revision": None,
            "dirty": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _duplicate_method_configurations(
    cells: Sequence[Cell], registry: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Assert every (method, grid) maps to exactly one registry configuration.

    `apply_method_registry` installs entries by method name alone, so two
    different estimator settings under one method name across two grids would
    silently collapse. The check records the mapping that was actually frozen.
    """

    configurations: dict[str, str] = {}
    grids_by_method: dict[str, set[str]] = {}
    for cell in cells:
        if cell.method not in registry:
            raise ValueError(f"method {cell.method!r} is missing from the registry")
        signature = json.dumps(registry[cell.method], sort_keys=True)
        configurations[f"{cell.method}|{cell.grid}"] = signature
        grids_by_method.setdefault(cell.method, set()).add(cell.grid)
    duplicates = {
        method: sorted(grids)
        for method, grids in grids_by_method.items()
        if len({configurations[f"{method}|{grid}"] for grid in grids}) > 1
    }
    if duplicates:
        raise ValueError(
            f"methods with conflicting configurations across grids: {duplicates}"
        )
    return {
        "status": "PASS",
        "n_configurations": len(configurations),
        "n_duplicates": 0,
        "methods": {method: sorted(grids) for method, grids in grids_by_method.items()},
    }


def build_wcf_method_registry(
    source_hash: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The resolved method registry frozen beside the cells.

    The three C-WDB variants are distinct method names because `Cell.key` does
    not hash registry parameters. The forest baselines are copied from the
    frozen registry unchanged.
    """

    source_hash = estimator_source_hash() if source_hash is None else source_hash

    def cwdb(name: str) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "role": "variant",
            "adapter": "cwdb_dr",
            "produces_law": True,
            "cross_fitted": True,
            "parameters": {
                "contrast_candidates": list(CONTRAST_CANDIDATES),
                "n_folds": SELECTION_FOLDS,
                "common_grid_levels": COMMON_GRID_LEVELS,
            },
            "propensity_clip": list(PROPENSITY_CLIP),
            "boosting_budget": copy.deepcopy(BOOSTING_BUDGET),
            "estimator_source_hash": source_hash,
        }
        if name == "cwdb_dr":
            entry["parameters"]["propensity_factory"] = "logistic"
        elif name == "cwdb_dr_flex":
            entry["parameters"]["propensity_factory"] = "hist_gradient_boosting"
        elif name == "cwdb_dr_oracle":
            entry["role"] = "diagnostic"
            entry["parameters"]["oracle_propensity"] = True
        else:
            raise ValueError(f"unknown C-WDB sensitivity method {name!r}")
        return entry

    return {
        "cwdb_dr": cwdb("cwdb_dr"),
        "cwdb_dr_flex": cwdb("cwdb_dr_flex"),
        "cwdb_dr_oracle": cwdb("cwdb_dr_oracle"),
        "causal_drf": copy.deepcopy(METHOD_REGISTRY["causal_drf"]),
        "drf": copy.deepcopy(METHOD_REGISTRY["drf"]),
    }


def build_wcf_sensitivity_manifest(
    blocks: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """The frozen sensitivity document for the requested blocks."""

    names = resolve_blocks(blocks)
    cells = build_wcf_sensitivity_cells(names)
    source_hash = estimator_source_hash()
    registry = build_wcf_method_registry(source_hash)
    document: dict[str, Any] = {
        "manifest_contract_id": WCF_SENSITIVITY_CONTRACT_ID,
        "estimand_contract_id": ESTIMAND_CONTRACT_ID,
        "moderator_edges": list(MODERATOR_EDGES),
        "blocks": list(names),
        "n_cells": len(cells),
        "n_test": N_TEST,
        "test_seed_offset": TEST_SEED_OFFSET,
        "training_functionals": list(TRAINING_FUNCTIONALS),
        "boosting_budget": copy.deepcopy(BOOSTING_BUDGET),
        "evaluation_manifest": {
            **EVALUATION_MANIFEST,
            "functionals": list(EVALUATION_MANIFEST["functionals"]),
        },
        "propensity_clip": list(PROPENSITY_CLIP),
        "method_registry": registry,
        "block_counts": {
            name: len(WCF_BLOCK_BUILDERS[name]()) for name in names
        },
        "software_revision": software_revision(),
        "estimator_source_hash": source_hash,
        "cells": [cell.to_dict() for cell in cells],
    }
    document["duplicate_method_configurations"] = _duplicate_method_configurations(
        cells, registry
    )
    document["manifest_checksum"] = hashlib.sha256(
        json.dumps(document["cells"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return document


def apply_method_registry(document: dict[str, Any]) -> None:
    """Install a serialized registry into the process-wide `METHOD_REGISTRY`.

    Workers call this after loading the frozen manifest, because `Cell.key`
    does not hash estimator parameters: without it a worker would silently fit
    the historical `cwdb_dr` settings instead of the frozen three-fold
    specification.
    """

    for name, entry in document["method_registry"].items():
        METHOD_REGISTRY[name] = copy.deepcopy(entry)
