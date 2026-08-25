"""Manifest enumeration, merge reconciliation, and gate-flag logic.

These are the parts of the tournament that decide what counts as evidence, so
they are pinned on synthetic rows where the right answer is known. The merge is
checked on the failure modes WP3-B2 names explicitly: duplicates, unknown keys,
and missing cells must make the audit fail rather than be cleaned away.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wasserstein_causal_forests.g3.analysis import (
    compute_gate_flags,
    failure_rates,
    paired_comparison,
)
from wasserstein_causal_forests.g3.manifest import (
    DECISION_SE_MULTIPLE,
    METHOD_REGISTRY,
    Cell,
    build_grids,
    build_manifest,
    enumerate_cells,
)
from wasserstein_causal_forests.g3.merge import merge_results


def test_every_cell_is_enumerated_exactly_once() -> None:
    cells = enumerate_cells()
    keys = [cell.key for cell in cells]
    assert len(set(keys)) == len(keys)

    declared = sum(
        len(g.dgps) * len(g.n_train) * len(g.n_grid)
        * len(g.n_particles) * len(g.methods) * len(g.seeds)
        for g in build_grids()
    )
    assert declared == len(cells)


def test_cell_keys_depend_on_every_coordinate() -> None:
    base = Cell("main", "D1", 500, 25, 10, "cwdb_v1", 0)
    for field, value in (
        ("grid", "smallk"), ("dgp", "D2"), ("n_train", 1000), ("n_grid", 5),
        ("n_particles", 25), ("method", "cwdb_v0"), ("seed", 1),
    ):
        other = Cell(**{**base.__dict__, field: value})
        assert other.key != base.key, f"key ignores {field}"


def test_training_and_test_seeds_never_collide() -> None:
    cells = enumerate_cells()
    assert not ({c.seed for c in cells} & {c.test_seed for c in cells})


def test_every_grid_method_is_registered() -> None:
    for grid in build_grids():
        for method in grid.methods:
            assert method in METHOD_REGISTRY


def test_pta_forced_is_confined_to_the_small_grid() -> None:
    """PTA-F's cost accelerates in D = K + J + 1, so it runs only at D = 8."""

    for grid in build_grids():
        if "pta_f" in grid.methods:
            assert grid.n_grid == (5,), (
                "PTA-F must stay at K = 5; its dense covariance makes D = 28 "
                "unaffordable"
            )


def test_manifest_checksum_tracks_the_cells() -> None:
    first = build_manifest()
    second = build_manifest()
    assert first["manifest_checksum"] == second["manifest_checksum"]
    assert first["n_cells"] == len(first["cells"])
    assert "gate_rules" in first and "primary_law_metric" in first


def synthetic_rows(
    claimant_values: dict[int, float], comparator_values: dict[int, float],
    *, metric: str = "kernel_law_error", dgp: str = "D1",
) -> list[dict]:
    rows = []
    for method, values in (("cwdb_v1", claimant_values), ("causal_drf", comparator_values)):
        for seed, value in values.items():
            rows.append({
                "grid": "main", "dgp": dgp, "n_train": 500, "n_grid": 25,
                "n_particles": 10, "method": method, "seed": seed,
                "cell_key": f"{method}-{dgp}-{seed}", "metric": metric,
                "target_id": "LAW-A-K", "arm": 0, "value": value,
                "status": "ok", "failure_reason": "", "wall_seconds": 1.0,
            })
    return rows


def test_paired_comparison_uses_the_within_seed_difference() -> None:
    """A constant offset with large between-seed spread must still be detected."""

    rng = np.random.default_rng(0)
    level = rng.uniform(1.0, 20.0, size=20)
    claimant = {s: float(level[s]) for s in range(20)}
    comparator = {s: float(level[s] + 0.5) for s in range(20)}

    result = paired_comparison(
        synthetic_rows(claimant, comparator), "kernel_law_error",
        comparator="causal_drf", grid="main", dgp="D1",
    )
    assert result is not None
    assert result["paired_mean_difference"] == pytest.approx(-0.5)
    assert result["paired_standard_error"] == pytest.approx(0.0, abs=1e-12)
    assert result["seed_win_fraction"] == 1.0
    # The unpaired spread is far larger than the effect, so only pairing sees it.
    assert np.std(list(claimant.values())) > 1.0


def test_paired_comparison_declines_a_noise_difference() -> None:
    rng = np.random.default_rng(3)
    claimant = {s: float(rng.normal(1.0, 1.0)) for s in range(20)}
    comparator = {s: float(rng.normal(1.0, 1.0)) for s in range(20)}
    result = paired_comparison(
        synthetic_rows(claimant, comparator), "kernel_law_error",
        comparator="causal_drf", grid="main", dgp="D1",
    )
    assert result is not None
    assert not result["claimant_wins"]
    assert abs(result["paired_mean_difference"]) < (
        DECISION_SE_MULTIPLE * result["paired_standard_error"]
    )


def test_mode_coverage_sign_is_inverted() -> None:
    """Higher coverage is better, so a higher claimant value must read as a win."""

    claimant = {s: 1.0 for s in range(10)}
    comparator = {s: 0.5 for s in range(10)}
    result = paired_comparison(
        synthetic_rows(claimant, comparator, metric="mode_coverage", dgp="D6"),
        "mode_coverage", comparator="causal_drf", grid="main", dgp="D6",
    )
    assert result is not None
    assert result["paired_mean_difference"] < 0.0
    assert result["claimant_wins"]


def test_paired_comparison_needs_enough_seeds() -> None:
    assert paired_comparison(
        synthetic_rows({0: 1.0, 1: 1.0}, {0: 2.0, 1: 2.0}),
        "kernel_law_error", comparator="causal_drf", grid="main", dgp="D1",
    ) is None


def test_gate_verdict_is_not_go_without_evidence() -> None:
    """An empty tournament must not pass: absence of a loss is not a win."""

    flags = compute_gate_flags([])
    assert flags["summary"]["verdict"] == "NOT-GO"
    assert not flags["rule_2_law_advantage"]["passed"]
    assert not flags["rule_3_transfer"]["passed"]


def test_rule_four_only_counts_targets_rule_three_won() -> None:
    """Beating PTA-S where Causal-DRF won is not the claimed transfer."""

    rows = []
    for method, value in (
        ("cwdb_v1", 1.0), ("causal_drf", 0.5), ("pta_s", 5.0)
    ):
        for seed in range(10):
            rows.append({
                "grid": "main", "dgp": "D5", "n_train": 500, "n_grid": 25,
                "n_particles": 10, "method": method, "seed": seed,
                "cell_key": f"{method}-{seed}", "metric": "reference_tcate_rmse",
                "target_id": "REF-TCATE-K", "arm": None, "value": value,
                "status": "ok", "failure_reason": "", "wall_seconds": 1.0,
            })
    flags = compute_gate_flags(rows)
    # C-WDB loses to Causal-DRF here, so rule 3 fails and rule 4 has nothing to
    # evaluate even though C-WDB beats PTA-S by a wide margin.
    assert not flags["rule_3_transfer"]["passed"]
    assert not flags["rule_4_beats_direct_learner"]["passed"]


def test_failure_rates_count_cells_not_rows() -> None:
    rows = [
        {"method": "m", "cell_key": "a", "status": "ok"},
        {"method": "m", "cell_key": "a", "status": "ok"},
        {"method": "m", "cell_key": "b", "status": "failed"},
    ]
    assert failure_rates(rows)["m"] == {
        "n_cells": 2, "n_failed": 1, "failure_rate": 0.5
    }


def merge_fixture(tmp_path: Path, keys: list[str], manifest_keys: list[str]):
    shard_directory = tmp_path / "main"
    shard_directory.mkdir()
    rows = [
        {
            "grid": "main", "dgp": "D1", "n_train": 500, "n_grid": 25,
            "n_particles": 10, "method": "cwdb_v1", "seed": index,
            "cell_key": key, "test_seed": 900000 + index,
            "manifest_contract_id": "G3-MAIN-v1",
            "estimand_contract_id": "G0-WP0-A-v1",
            "evaluation_manifest_id": "G3-EVAL-v1", "method_role": "claimant",
            "n_test": 1000, "metric": "mean_quantile_rmse",
            "target_id": "MEANQ-A-K", "arm": None, "value": 0.5,
            "status": "ok", "failure_reason": "", "wall_seconds": 1.0,
        }
        for index, key in enumerate(keys)
    ]
    (shard_directory / "shard_000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "manifest_contract_id": "G3-MAIN-v1",
            "manifest_checksum": "deadbeef",
            "cells": [
                {
                    "grid": "main", "dgp": "D1", "n_train": 500, "n_grid": 25,
                    "n_particles": 10, "method": "cwdb_v1", "seed": index,
                    "cell_key": key,
                }
                for index, key in enumerate(manifest_keys)
            ],
        }),
        encoding="utf-8",
    )
    return shard_directory, manifest


def test_merge_passes_when_every_cell_is_present(tmp_path: Path) -> None:
    shards, manifest = merge_fixture(tmp_path, ["k0", "k1"], ["k0", "k1"])
    audit = merge_results(shards, manifest, tmp_path / "merged")
    assert audit["status"] == "PASS", audit["problems"]
    assert audit["n_observed_cells"] == 2


def test_merge_reports_a_missing_cell(tmp_path: Path) -> None:
    shards, manifest = merge_fixture(tmp_path, ["k0"], ["k0", "k1"])
    audit = merge_results(shards, manifest, tmp_path / "merged")
    assert audit["status"] == "FAIL"
    assert "k1" in audit["missing_cells"]


def test_merge_reports_an_unknown_cell(tmp_path: Path) -> None:
    shards, manifest = merge_fixture(tmp_path, ["k0", "kX"], ["k0"])
    audit = merge_results(shards, manifest, tmp_path / "merged")
    assert audit["status"] == "FAIL"
    assert "kX" in audit["unknown_cells"]


def test_merge_reports_a_cell_split_across_two_shards(tmp_path: Path) -> None:
    shards, manifest = merge_fixture(tmp_path, ["k0", "k1"], ["k0", "k1"])
    duplicate = (shards / "shard_000.jsonl").read_text(encoding="utf-8")
    (shards / "shard_001.jsonl").write_text(duplicate, encoding="utf-8")
    audit = merge_results(shards, manifest, tmp_path / "merged")
    assert audit["status"] == "FAIL"
    assert any("appear in both" in problem for problem in audit["problems"])


def test_law_methods_evaluate_functionals_outside_their_training_manifest() -> None:
    """The D7 transfer claim must not be disabled by the adapter layer.

    A method holding a conditional law can integrate any declared grid
    functional against its own atoms, including one first named at evaluation
    time. An earlier version passed only the training manifest through, so every
    law method reported `not_applicable` on the unseen functionals and the
    transfer test silently measured nothing.
    """

    from wasserstein_causal_forests.g3.dgps import build_dgp
    from wasserstein_causal_forests.g3.evaluation import (
        EvaluationManifest,
        evaluate,
    )
    from wasserstein_causal_forests.g3.methods import CWDBAdapter
    from wasserstein_causal_forests.pta_bcf.targets import GRID_FUNCTIONALS

    trained_on = ("grid_mean", "grid_sd")
    unseen = {"grid_skewness", "grid_upper_tail_mean"}
    assert unseen < set(GRID_FUNCTIONALS)
    assert not (unseen & set(trained_on))

    dgp = build_dgp("D7", 5)
    train = dgp.sample(120, seed=0)
    test = dgp.sample(80, seed=999)
    output = CWDBAdapter(n_particles=4, n_estimators=4).fit_predict(
        train, test.X, dgp, trained_on, seed=0
    )

    assert unseen <= set(output.supported_functionals)
    for name in unseen:
        assert name in output.functionals

    rows = evaluate(
        output,
        dgp,
        test.X,
        EvaluationManifest(
            "probe", trained_on, tail_level_index=4, tail_threshold=1.5,
            mode_radius=1.0, mode_mass_floor=0.15, n_law_rows=40,
        ),
    )
    for name in unseen:
        reported = [
            r for r in rows
            if r["target_id"] == f"TCATE-K-{name}" and r["status"] == "ok"
        ]
        assert reported, f"law method reported nothing for unseen {name}"


def test_mean_only_methods_still_decline_unseen_functionals() -> None:
    """PTA's inability to transfer is the finding, so it must be preserved."""

    from wasserstein_causal_forests.g3.methods import _output_from_target_means
    from wasserstein_causal_forests.pta_bcf.targets import uniform_grid_manifest

    manifest = uniform_grid_manifest(
        5, functionals=("grid_mean", "grid_sd"), reference_quantiles=np.zeros(5)
    )
    output = _output_from_target_means(
        manifest,
        {arm: np.zeros((3, manifest.dimension)) for arm in (0, 1)},
        ("grid_mean", "grid_sd"),
        fit_seconds=0.0, predict_seconds=0.0, peak_ram_mb=0.0,
    )
    assert output.law is None
    assert "grid_skewness" not in output.supported_functionals
