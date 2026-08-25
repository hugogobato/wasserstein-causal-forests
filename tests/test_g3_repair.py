"""The repair track must not be able to disturb the frozen tournament."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wasserstein_causal_forests.g3.analysis import compute_gate_flags
from wasserstein_causal_forests.g3.manifest import (
    FROZEN_G3_METHODS,
    METHOD_REGISTRY,
    build_manifest,
    enumerate_cells,
)
from wasserstein_causal_forests.g3.repair import (
    REPAIR_METHOD_REGISTRY,
    REPAIR_METHODS,
    build_repair_grids,
    build_repair_manifest,
    enumerate_repair_cells,
)
from wasserstein_causal_forests.g3.runner import build_adapter

ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = ROOT / "results" / "manifests" / "main_manifest.json"


def test_registering_repair_methods_leaves_the_frozen_cells_alone() -> None:
    """A registry entry is not a cell; only a GridSpec makes cells."""

    cells = enumerate_cells()
    assert len(cells) == 4110
    assert {cell.method for cell in cells} == set(FROZEN_G3_METHODS)
    assert not {cell.method for cell in cells} & set(REPAIR_METHODS)


@pytest.mark.skipif(
    not FROZEN_MANIFEST.exists(), reason="the frozen manifest is not checked in"
)
def test_frozen_manifest_checksum_survives_the_repair_registry() -> None:
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    assert build_manifest()["manifest_checksum"] == frozen["manifest_checksum"]


def test_repair_cells_never_collide_with_frozen_cells() -> None:
    frozen_keys = {cell.key for cell in enumerate_cells()}
    repair_keys = {cell.key for cell in enumerate_repair_cells()}
    assert not frozen_keys & repair_keys
    assert len(repair_keys) == len(enumerate_repair_cells())


def test_repair_cells_reuse_the_frozen_coordinates_and_seeds() -> None:
    """A repair row must pair seed by seed with the frozen rows it faces."""

    frozen_main = [
        cell for cell in enumerate_cells()
        if cell.grid == "main" and cell.dgp == "D2" and cell.method == "cwdb_v1"
    ]
    repair_main = [
        cell for cell in enumerate_repair_cells(("D2",), grids=("main",))
        if cell.method == "cwdb_r1_ridge"
    ]
    coordinates = lambda cells: sorted(  # noqa: E731
        (c.n_train, c.n_grid, c.n_particles, c.seed) for c in cells
    )
    assert coordinates(frozen_main) == coordinates(repair_main)


def test_repair_manifest_restricts_to_the_requested_regimes_and_methods() -> None:
    document = build_repair_manifest(("D2",), ("cwdb_r1_ridge",), ("main",))
    cells = document["cells"]
    assert {cell["dgp"] for cell in cells} == {"D2"}
    assert {cell["method"] for cell in cells} == {"cwdb_r1_ridge"}
    assert {cell["grid"] for cell in cells} == {"main"}
    assert document["manifest_contract_id"] == "G3-REPAIR-v1"
    assert document["parent_manifest_contract_id"] == "G3-MAIN-v1"


def test_repair_manifest_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown repair methods"):
        build_repair_grids(methods=("not_a_method",))


def test_every_repair_method_builds_an_adapter() -> None:
    for method in REPAIR_METHODS:
        assert method in METHOD_REGISTRY
        cell = enumerate_repair_cells(("D2",), (method,), ("main",))[0]
        adapter = build_adapter(cell, None)
        assert adapter.produces_law


def test_the_cross_fitted_variant_is_the_only_one_that_scans_strengths() -> None:
    scanning = [
        name for name, entry in REPAIR_METHOD_REGISTRY.items()
        if "contrast_candidates" in entry["parameters"]
    ]
    assert scanning == ["cwdb_r3_cvridge"]


def test_repair_variants_keep_the_frozen_architecture_and_repulsion() -> None:
    """Only the contrast rule may differ; the claim rests on the rest."""

    for entry in REPAIR_METHOD_REGISTRY.values():
        assert entry["parameters"]["architecture"] == "v1"
        assert entry["parameters"]["sharing"] == "partial"
        assert entry["produces_law"] is True


def test_a_staged_run_writes_shards_that_cannot_clobber_the_earlier_stage() -> None:
    from wasserstein_causal_forests.g3.cli import _completed_keys, shard_suffix

    stage_one = {shard_suffix(index) for index in range(10)}
    stage_two = {shard_suffix(index, "s2") for index in range(10)}
    assert not stage_one & stage_two
    # The resume scan must still see both stages' logs.
    assert all(name.startswith(("0", "1", "s2")) for name in stage_one | stage_two)
    assert callable(_completed_keys)


def test_resume_scan_reads_every_stage_of_a_track(tmp_path) -> None:
    from wasserstein_causal_forests.g3.cli import _completed_keys

    log = tmp_path / "repair_execution_log.jsonl"
    (tmp_path / "repair_execution_log_000.jsonl").write_text(
        json.dumps({"cell_key": "aaa"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "repair_execution_log_s2_000.jsonl").write_text(
        json.dumps({"cell_key": "bbb"}) + "\n", encoding="utf-8"
    )
    # A different track's log lives beside it and must not be picked up.
    (tmp_path / "execution_log_000.jsonl").write_text(
        json.dumps({"cell_key": "ccc"}) + "\n", encoding="utf-8"
    )
    assert _completed_keys(log) == {"aaa", "bbb"}


def test_rule_one_baseline_ignores_methods_outside_the_frozen_roster() -> None:
    """A repair variant must not lower the bar it is judged against."""

    def row(method: str, seed: int, value: float) -> dict:
        return {
            "grid": "main", "dgp": "D2", "n_train": 500, "n_grid": 25,
            "n_particles": 10, "method": method, "seed": seed,
            "cell_key": f"{method}-{seed}", "metric": "mean_quantile_rmse",
            "target_id": "MEANQ-A-K", "arm": None, "value": value, "status": "ok",
        }

    rows = [row("cwdb_v1", s, 0.20) for s in range(5)]
    rows += [row("causal_drf", s, 0.10) for s in range(5)]
    baseline_only = compute_gate_flags(rows)["rule_1_correctness"]
    rows += [row("cwdb_r1_ridge", s, 0.01) for s in range(5)]
    with_repair = compute_gate_flags(rows)["rule_1_correctness"]
    assert with_repair["d2_best_baseline"] == baseline_only["d2_best_baseline"] == 0.10
    assert compute_gate_flags(rows, claimant="cwdb_r1_ridge")[
        "rule_1_correctness"
    ]["d2_best_baseline"] == 0.10
