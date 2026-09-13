"""Manifest construction for the isolated WCF sensitivity study.

The block builders reference DGP and method names as strings only, so these
tests run before the companion `g3.sensitivity_dgps` and
`g3.sensitivity_methods` modules land. Importing `wcf_sensitivity` is itself
part of the contract: the module must not need those companions.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from wasserstein_causal_forests.g3 import runner
from wasserstein_causal_forests.g3.manifest import METHOD_REGISTRY, Cell
from wasserstein_causal_forests.g3.wcf_sensitivity import (
    REQUIRED_LAW_METRICS,
    WCF_BLOCKS,
    WCF_PRIMARY_BLOCKS,
    WCF_SENSITIVITY_CONTRACT_ID,
    apply_method_registry,
    build_wcf_method_registry,
    build_wcf_sensitivity_cells,
    build_wcf_sensitivity_manifest,
)

EXACT_BLOCK_COUNTS = {
    "income_onefactor": 200,
    "income_baselines": 240,
    "income_factorial": 180,
    "sym": 240,
    "sym_null": 240,
    "align": 120,
    "align_null": 120,
    "propensity": 80,
}


def test_block_roster_is_complete() -> None:
    assert set(WCF_BLOCKS) == set(EXACT_BLOCK_COUNTS)
    assert len(WCF_BLOCKS) == len(set(WCF_BLOCKS))


@pytest.mark.parametrize("block,count", sorted(EXACT_BLOCK_COUNTS.items()))
def test_exact_block_counts(block: str, count: int) -> None:
    cells = build_wcf_sensitivity_cells([block])
    assert len(cells) == count
    assert len({cell.key for cell in cells}) == count


def test_factorial_adds_exactly_eighty_new_cells() -> None:
    onefactor = {cell.key for cell in build_wcf_sensitivity_cells(["income_onefactor"])}
    factorial = build_wcf_sensitivity_cells(["income_factorial"])
    assert len(factorial) == 180
    assert len({cell.key for cell in factorial} - onefactor) == 80
    combined = build_wcf_sensitivity_cells(["income_onefactor", "income_factorial"])
    assert len(combined) == 280
    assert len({cell.key for cell in combined}) == len(combined)


def test_primary_default_count_and_no_duplicate_keys() -> None:
    cells = build_wcf_sensitivity_cells()
    assert len(cells) == 880
    assert len({cell.key for cell in cells}) == 880
    assert sum(EXACT_BLOCK_COUNTS[block] for block in WCF_PRIMARY_BLOCKS) == 880
    primary_grids = {cell.grid for cell in cells}
    assert "wcf_sensitivity_v1_sym_null" not in primary_grids
    assert "wcf_sensitivity_v1_align_null" not in primary_grids


def test_every_cell_method_resolves_in_the_registry() -> None:
    registry = build_wcf_method_registry()
    for block in WCF_BLOCKS:
        for cell in build_wcf_sensitivity_cells([block]):
            assert cell.method in registry, f"{cell.method} missing for {block}"


def test_registry_copies_the_frozen_forest_entries() -> None:
    registry = build_wcf_method_registry()
    assert registry["causal_drf"] == METHOD_REGISTRY["causal_drf"]
    assert registry["drf"] == METHOD_REGISTRY["drf"]


def test_registry_freezes_three_folds_and_the_classifier_factories() -> None:
    registry = build_wcf_method_registry()
    for name in ("cwdb_dr", "cwdb_dr_flex", "cwdb_dr_oracle"):
        entry = registry[name]
        assert entry["adapter"] == "cwdb_dr"
        assert entry["parameters"]["n_folds"] == 3
        assert entry["parameters"]["contrast_candidates"] == [0.0, 50.0, 500.0]
        assert entry["parameters"]["common_grid_levels"] == "COMMON199+INTERIOR"
        assert entry["propensity_clip"] == [0.02, 0.98]
        assert entry["estimator_source_hash"]
    assert (
        registry["cwdb_dr"]["parameters"]["propensity_factory"] == "logistic"
    )
    assert (
        registry["cwdb_dr_flex"]["parameters"]["propensity_factory"]
        == "hist_gradient_boosting"
    )
    assert registry["cwdb_dr_oracle"]["parameters"]["oracle_propensity"] is True
    assert registry["cwdb_dr_oracle"]["role"] == "diagnostic"
    assert registry["cwdb_dr"]["produces_law"] is True
    assert registry["cwdb_dr_flex"]["produces_law"] is True
    assert registry["cwdb_dr_oracle"]["produces_law"] is True
    assert REQUIRED_LAW_METRICS == (
        "mean_quantile_rmse",
        "kernel_law_error",
        "tate_functional_rmse",
        "tcate_functional_rmse",
        "reference_effect_rmse",
        "reference_tcate_rmse",
    )


def test_manifest_checksum_is_stable_and_counts_are_consistent() -> None:
    first = build_wcf_sensitivity_manifest()
    second = build_wcf_sensitivity_manifest()
    assert first["manifest_contract_id"] == WCF_SENSITIVITY_CONTRACT_ID
    assert first["manifest_checksum"] == second["manifest_checksum"]
    assert first["n_cells"] == len(first["cells"]) == 880
    assert second["n_cells"] == 880
    assert first["n_test"] == 1000
    assert first["test_seed_offset"] == 900_000
    assert first["propensity_clip"] == [0.02, 0.98]
    assert first["block_counts"] == {
        "income_onefactor": 200,
        "income_baselines": 240,
        "sym": 240,
        "align": 120,
        "propensity": 80,
    }
    assert first["duplicate_method_configurations"]["status"] == "PASS"
    assert len(first["estimator_source_hash"]) == 64
    assert "revision" in first["software_revision"]


def test_manifest_declares_only_requested_blocks() -> None:
    document = build_wcf_sensitivity_manifest(["income_baselines"])
    assert document["n_cells"] == 240
    assert document["blocks"] == ["income_baselines"]
    assert document["block_counts"] == {"income_baselines": 240}


def test_null_blocks_require_an_explicit_request() -> None:
    primary = {cell.key for cell in build_wcf_sensitivity_cells()}
    nulls = build_wcf_sensitivity_cells(["sym_null", "align_null"])
    assert len(nulls) == 360
    assert not (primary & {cell.key for cell in nulls})
    assert "sym_null" not in WCF_PRIMARY_BLOCKS
    assert "align_null" not in WCF_PRIMARY_BLOCKS


def test_unknown_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown WCF sensitivity blocks"):
        build_wcf_sensitivity_cells(["not-a-block"])


def test_apply_method_registry_writes_the_entries() -> None:
    document = build_wcf_sensitivity_manifest(
        ["propensity", "income_onefactor"]
    )
    snapshot = copy.deepcopy(METHOD_REGISTRY)
    try:
        apply_method_registry(document)
        for name, entry in document["method_registry"].items():
            assert METHOD_REGISTRY[name] == entry
        assert METHOD_REGISTRY["cwdb_dr_flex"]["parameters"]["propensity_factory"] == (
            "hist_gradient_boosting"
        )
        assert METHOD_REGISTRY["cwdb_dr_oracle"]["role"] == "diagnostic"
    finally:
        METHOD_REGISTRY.clear()
        METHOD_REGISTRY.update(snapshot)


class _FakeDGP:
    def sample(self, n: int, *, seed: int) -> SimpleNamespace:
        return SimpleNamespace(
            X=np.zeros((n, 2)),
            treatment=np.zeros(n),
            quantiles=np.zeros((n, 2)),
        )


class _FakeAdapter:
    def fit_predict(self, train, X_test, dgp, functionals, *, seed):
        return SimpleNamespace(anything=True)


def _patch_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "build_dgp", lambda dgp_id, n_grid: _FakeDGP())
    monkeypatch.setattr(runner, "build_adapter", lambda cell, cache: _FakeAdapter())
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda *args, **kwargs: [
            {
                "metric": "native_metric",
                "target_id": "T",
                "arm": None,
                "detail": "",
                "value": 1.0,
                "status": "ok",
                "failure_reason": "",
            }
        ],
    )


def test_run_cell_runs_extra_evaluators_and_records_process_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runner(monkeypatch)
    calls: list[tuple[str, int]] = []

    def evaluator(cell, output, dgp, X_test):
        calls.append((cell.method, X_test.shape[0]))
        return [
            {
                "metric": "extra_metric",
                "target_id": "COMMON199",
                "arm": None,
                "detail": "",
                "value": 2.0,
                "status": "ok",
                "failure_reason": "",
            }
        ]

    rows = runner.run_cell(
        Cell("main", "D1", 10, 5, 5, "cwdb_v1", 0),
        extra_evaluators=(evaluator,),
    )
    metrics = [row["metric"] for row in rows]
    assert "native_metric" in metrics
    assert "extra_metric" in metrics
    assert "process_peak_ram" in metrics
    assert calls == [("cwdb_v1", 1000)]
    peak = next(row for row in rows if row["metric"] == "process_peak_ram")
    assert peak["target_id"] == "NONE_OPERATIONAL"
    assert peak["arm"] is None
    assert peak["status"] == "ok"
    assert peak["value"] > 0.0


def test_extra_evaluator_failure_marks_the_cell_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runner(monkeypatch)

    def evaluator(cell, output, dgp, X_test):
        raise RuntimeError("common-grid boom")

    rows = runner.run_cell(
        Cell("main", "D1", 10, 5, 5, "cwdb_v1", 0),
        extra_evaluators=(evaluator,),
    )
    assert rows[0]["metric"] == "cell_failure"
    assert rows[0]["status"] == "failed"
    assert "common-grid boom" in rows[0]["failure_reason"]

