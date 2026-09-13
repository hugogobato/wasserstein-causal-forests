"""Fast tests for the WCF sensitivity analysis layer.

Every fixture is a synthetic frame with three seeds and, for law metrics, two
arms per cell. No estimator is fitted and no simulation cell is executed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wasserstein_causal_forests.g3.wcf_sensitivity_analysis import (
    alignment_balance,
    cell_metrics,
    decide_k_m,
    failure_summary,
    load_results,
    native_vs_common,
    one_factor_k,
    one_factor_m,
    paired_difference,
    plot_sensitivity,
    replication_summary,
    write_tables,
)

INCOME_GRID = "wcf_sensitivity_v1_logit_f3"
SYM_GRID = "wcf_sensitivity_v1_sym"
SEEDS = (0, 1, 2)


def _row(**overrides):
    row = {
        "grid": INCOME_GRID,
        "dgp": "IC1",
        "n_train": 1000,
        "n_grid": 25,
        "n_particles": 10,
        "method": "cwdb_dr",
        "seed": 0,
        "test_seed": 900_000,
        "metric": "",
        "target_id": "",
        "arm": None,
        "status": "ok",
        "value": 0.0,
        "wall_seconds": 1.0,
    }
    row.update(overrides)
    row["cell_key"] = (
        f"{row['grid']}|{row['dgp']}|{row['n_train']}|{row['n_grid']}|"
        f"{row['n_particles']}|{row['method']}|{row['seed']}"
    )
    return row


def _cell_rows(values, *, dgp="IC1", metric="reference_tcate_rmse",
               target="REF-TCATE-K", n_grid=25, n_particles=10, **extra):
    return [
        _row(
            dgp=dgp,
            metric=metric,
            target_id=target,
            n_grid=n_grid,
            n_particles=n_particles,
            seed=seed,
            value=value,
            **extra,
        )
        for seed, value in enumerate(values)
    ]


def test_cell_metrics_averages_arms_and_keeps_functionals_separate():
    rows = []
    for seed, (arm_zero, arm_one) in enumerate(
        [(1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]
    ):
        rows.append(
            _row(seed=seed, metric="kernel_law_error", target_id="LAW-A-K",
                 arm=0, value=arm_zero)
        )
        rows.append(
            _row(seed=seed, metric="kernel_law_error", target_id="LAW-A-K",
                 arm=1, value=arm_one)
        )
        rows.append(
            _row(seed=seed, metric="tate_functional_rmse",
                 target_id="TATE-K-grid_mean", value=0.5 + seed)
        )
        rows.append(
            _row(seed=seed, metric="tate_functional_rmse",
                 target_id="TATE-K-grid_sd", value=1.5 + seed)
        )
    cells = cell_metrics(pd.DataFrame(rows))

    law = cells[cells["metric"] == "kernel_law_error"].sort_values("seed")
    assert len(law) == 3
    assert np.allclose(law["value"], [2.0, 3.0, 4.0])
    assert set(law["n_arm_rows"]) == {2}

    functional = cells[cells["metric"] == "tate_functional_rmse"]
    assert len(functional) == 6
    for seed in SEEDS:
        block = functional[functional["seed"] == seed].sort_values("target_id")
        assert list(block["target_id"]) == ["TATE-K-grid_mean", "TATE-K-grid_sd"]
        assert np.allclose(block["value"], [0.5 + seed, 1.5 + seed])


def test_cell_metrics_filters_failures_and_failure_summary_counts_them():
    rows = [
        _row(metric="cell_failure", status="failed", value=None,
             target_id="NONE_OPERATIONAL", failure_reason="boom"),
        _row(metric="reference_tcate_rmse", target_id="REF-TCATE-K", value=1.0),
        _row(metric="reference_tcate_rmse", target_id="REF-TCATE-K",
             status="not_applicable", value=None),
    ]
    frame = pd.DataFrame(rows)
    cells = cell_metrics(frame)
    assert list(cells["metric"]) == ["reference_tcate_rmse"]
    failures = failure_summary(frame)
    assert len(failures) == 1
    assert int(failures["n_failed_cells"].iloc[0]) == 1
    assert "boom" in failures["failure_reasons"].iloc[0]


def test_replication_summary_matches_hand_computation():
    values = [1.0, 2.0, 3.0]
    frame = pd.DataFrame(
        _cell_rows(values, metric="reference_tcate_rmse", target="REF-TCATE-K")
    )
    cells = cell_metrics(frame)
    summary = replication_summary(cells)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert int(row["n_seeds"]) == 3
    assert row["mean"] == pytest.approx(2.0)
    assert row["sd"] == pytest.approx(1.0)
    assert row["mc_se"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert row["median"] == pytest.approx(2.0)


def test_paired_difference_exact_three_seed_example():
    a = pd.DataFrame(
        _cell_rows([1.0, 2.0, 3.0], metric="m", target="T")
    )
    b = pd.DataFrame(
        _cell_rows([0.5, 1.0, 2.5], metric="m", target="T")
    )
    result = paired_difference(a, b)
    assert result["n_pairs"] == 3
    assert result["paired_delta"] == pytest.approx(2.0 / 3.0)
    assert result["paired_sd"] == pytest.approx(np.sqrt(1.0 / 12.0))
    assert result["paired_mc_se"] == pytest.approx(1.0 / 6.0)
    assert result["p_value"] is not None


def test_paired_difference_never_mixes_target_ids():
    a = pd.DataFrame(
        _cell_rows([1.0], metric="reference_tcate_rmse", target="REF-TCATE-K")
    )
    b = pd.DataFrame(
        _cell_rows(
            [2.0], metric="reference_tcate_rmse", target="REF-TCATE-COMMON199"
        )
    )
    assert paired_difference(a, b)["n_pairs"] == 0


def _decision_table(*, primary, kernel, alternatives):
    rows = []
    for dgp in ("IC1", "IC3"):
        rows.extend(
            _cell_rows(primary[dgp], dgp=dgp, metric="reference_tcate_rmse",
                       target="REF-TCATE-K")
        )
        rows.extend(
            _cell_rows(kernel[dgp], dgp=dgp, metric="kernel_law_error",
                       target="LAW-A-K")
        )
        for (n_grid, n_particles), builder in alternatives.items():
            rows.extend(
                _cell_rows(builder(dgp, primary[dgp]), dgp=dgp,
                           metric="reference_tcate_rmse", target="REF-TCATE-K",
                           n_grid=n_grid, n_particles=n_particles)
            )
            rows.extend(
                _cell_rows(builder(dgp, kernel[dgp]), dgp=dgp,
                           metric="kernel_law_error", target="LAW-A-K",
                           n_grid=n_grid, n_particles=n_particles)
            )
    return pd.DataFrame(rows)


def test_decide_k_m_retains_primary_when_within_tolerance_and_no_improvement():
    primary = {"IC1": [1.0, 1.2, 0.8], "IC3": [2.0, 2.2, 1.8]}
    kernel = {"IC1": [3.0, 3.3, 2.7], "IC3": [4.0, 4.4, 3.6]}

    def worse(_dgp, values):
        return [value * 1.01 for value in values]

    alternatives = {
        (5, 10): worse,
        (49, 10): worse,
        (25, 5): worse,
        (25, 25): worse,
    }
    decision = decide_k_m(_decision_table(
        primary=primary, kernel=kernel, alternatives=alternatives
    ))
    assert decision["evaluable"] is True
    assert decision["condition_a"]["passed"] is True
    assert decision["condition_b"]["any_improves"] is False
    assert decision["retain_primary"] is True
    assert decision["rule_failed"] is False
    assert decision["recommended_pair"] is None


def test_decide_k_m_fails_and_recommends_the_constructed_improving_pair():
    primary = {"IC1": [1.0, 1.2, 0.8], "IC3": [2.0, 2.2, 1.8]}
    kernel = {"IC1": [3.0, 3.3, 2.7], "IC3": [4.0, 4.4, 3.6]}
    improvements = {
        "reference_tcate_rmse": [0.4, 0.5, 0.6],
        "kernel_law_error": [0.2, 0.3, 0.4],
    }

    def worse(_dgp, values):
        return [value * 1.01 for value in values]

    def improving(metric_key):
        def builder(_dgp, values):
            return list(np.asarray(values) - np.asarray(improvements[metric_key]))

        return builder

    rows = []
    for dgp in ("IC1", "IC3"):
        rows.extend(_cell_rows(primary[dgp], dgp=dgp,
                               metric="reference_tcate_rmse", target="REF-TCATE-K"))
        rows.extend(_cell_rows(kernel[dgp], dgp=dgp,
                               metric="kernel_law_error", target="LAW-A-K"))
        for n_grid, n_particles in ((5, 10), (25, 5), (25, 25)):
            rows.extend(_cell_rows(worse(dgp, primary[dgp]), dgp=dgp,
                                   metric="reference_tcate_rmse",
                                   target="REF-TCATE-K", n_grid=n_grid,
                                   n_particles=n_particles))
            rows.extend(_cell_rows(worse(dgp, kernel[dgp]), dgp=dgp,
                                   metric="kernel_law_error", target="LAW-A-K",
                                   n_grid=n_grid, n_particles=n_particles))
        rows.extend(_cell_rows(improving("reference_tcate_rmse")(dgp, primary[dgp]),
                               dgp=dgp, metric="reference_tcate_rmse",
                               target="REF-TCATE-K", n_grid=49, n_particles=10))
        rows.extend(_cell_rows(improving("kernel_law_error")(dgp, kernel[dgp]),
                               dgp=dgp, metric="kernel_law_error",
                               target="LAW-A-K", n_grid=49, n_particles=10))
    decision = decide_k_m(pd.DataFrame(rows))
    assert decision["condition_b"]["any_improves"] is True
    assert decision["retain_primary"] is False
    assert decision["rule_failed"] is True
    assert decision["recommended_pair"] == [49, 10]


def test_decide_k_m_reports_selected_pair_when_a_passes_but_b_holds():
    primary = {"IC1": [1.0, 1.05, 0.95], "IC3": [2.0, 2.1, 1.9]}
    kernel = {"IC1": [3.0, 3.15, 2.85], "IC3": [4.0, 4.2, 3.8]}

    def worse(_dgp, values):
        return [value * 1.01 for value in values]

    def slightly_better(_dgp, values):
        return [value * 0.97 for value in values]

    alternatives = {
        (5, 10): worse,
        (49, 10): slightly_better,
        (25, 5): worse,
        (25, 25): worse,
    }
    decision = decide_k_m(_decision_table(
        primary=primary, kernel=kernel, alternatives=alternatives
    ))
    assert decision["condition_a"]["passed"] is True
    assert decision["condition_b"]["any_improves"] is True
    assert decision["retain_primary"] is False
    assert decision["recommended_pair"] == [49, 10]


def test_native_vs_common_keeps_target_ids_distinct():
    rows = []
    for seed in SEEDS:
        rows.append(
            _row(grid=SYM_GRID, dgp="SYM-NL", n_train=500, seed=seed,
                 metric="reference_tcate_rmse", target_id="REF-TCATE-K",
                 value=1.0 + seed)
        )
        rows.append(
            _row(grid=SYM_GRID, dgp="SYM-NL", n_train=500, seed=seed,
                 metric="reference_tcate_rmse", target_id="REF-TCATE-COMMON199",
                 value=0.5 + seed)
        )
        rows.append(
            _row(grid=SYM_GRID, dgp="SYM-NL", n_train=500, seed=seed,
                 metric="reference_tcate_rmse", target_id="REF-TCATE-INTERIOR",
                 value=0.25 + seed)
        )
        rows.append(
            _row(grid=SYM_GRID, dgp="SYM-NL", n_train=500, seed=seed,
                 metric="kernel_law_error", target_id="LAW-A-K",
                 value=2.0 + seed)
        )
        rows.append(
            _row(grid=SYM_GRID, dgp="SYM-NL", n_train=500, seed=seed,
                 metric="kernel_law_error", target_id="LAW-A-COMMON199",
                 value=1.25 + seed)
        )
        rows.append(
            _row(grid=SYM_GRID, dgp="SYM-NL", n_train=500, n_grid=49, seed=seed,
                 metric="reference_tcate_rmse", target_id="REF-TCATE-K",
                 value=3.0 + seed)
        )
        rows.append(
            _row(grid=SYM_GRID, dgp="SYM-NL", n_train=500, n_grid=49, seed=seed,
                 metric="reference_tcate_rmse", target_id="REF-TCATE-COMMON199",
                 value=2.0 + seed)
        )
    table = native_vs_common(pd.DataFrame(rows))
    assert set(table["native_target_id"]) == {"REF-TCATE-K", "LAW-A-K"}
    assert set(table["common_target_id"]) == {
        "REF-TCATE-COMMON199",
        "REF-TCATE-INTERIOR",
        "LAW-A-COMMON199",
    }
    assert (table["native_target_id"] != table["common_target_id"]).all()
    common = table[
        (table["metric"] == "reference_tcate_rmse")
        & (table["common_target_id"] == "REF-TCATE-COMMON199")
    ]
    assert sorted(common["n_grid"]) == [25, 49]
    assert set(common["n_pairs"]) == {3}
    assert common[common["n_grid"] == 25]["paired_delta"].iloc[0] == pytest.approx(0.5)
    assert common[common["n_grid"] == 49]["paired_delta"].iloc[0] == pytest.approx(1.0)
    interior = table[
        (table["metric"] == "reference_tcate_rmse")
        & (table["common_target_id"] == "REF-TCATE-INTERIOR")
    ].iloc[0]
    assert interior["paired_delta"] == pytest.approx(0.75)


def test_load_results_raises_clear_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="wcf_sensitivity"):
        load_results(tmp_path / "missing.parquet")


def test_alignment_balance_reads_design_checks(tmp_path):
    document = {
        "seed": 1,
        "n_rows": 10,
        "alignment_marginal_max_abs_diff": 0.01,
        "checks_passed": True,
        "regimes": {
            "SYM-ALIGN": {
                "description": "aligned",
                "propensity_mean": 0.45,
                "propensity_min": 0.1,
                "propensity_max": 0.9,
                "treated_fraction": 0.46,
                "largest_abs_smd": {"feature": "sin_pi_x1", "value": 0.2},
            },
            "SYM-IRREL": {
                "description": "irrelevant",
                "propensity_mean": 0.45,
                "propensity_min": 0.1,
                "propensity_max": 0.9,
                "treated_fraction": 0.44,
                "largest_abs_smd": {"feature": "x6", "value": 0.03},
            },
        },
    }
    path = tmp_path / "design_checks.json"
    path.write_text(__import__("json").dumps(document), encoding="utf-8")
    balance = alignment_balance(path)
    assert list(balance["dgp"]) == ["SYM-ALIGN", "SYM-IRREL"]
    assert balance["largest_abs_smd_value"].iloc[0] == pytest.approx(0.2)
    assert balance["alignment_marginal_max_abs_diff"].iloc[0] == pytest.approx(0.01)


def test_write_tables_and_plot_sensitivity(tmp_path):
    primary = {"IC1": [1.0, 1.1, 0.9], "IC3": [2.0, 2.1, 1.9]}
    kernel = {"IC1": [3.0, 3.1, 2.9], "IC3": [4.0, 4.1, 3.9]}

    def worse(_dgp, values):
        return [value * 1.01 for value in values]

    alternatives = {
        (5, 10): worse,
        (49, 10): worse,
        (25, 5): worse,
        (25, 25): worse,
    }
    frame = _decision_table(primary=primary, kernel=kernel, alternatives=alternatives)
    tables = {
        "one_factor_k": one_factor_k(frame),
        "one_factor_m": one_factor_m(frame),
        "native_vs_common": native_vs_common(frame),
    }
    paths = write_tables(tables, tmp_path)
    assert (tmp_path / "one_factor_k.csv").is_file()
    assert (tmp_path / "one_factor_m.csv").is_file()
    assert len(paths) == 3
    pytest.importorskip("matplotlib")
    figures = plot_sensitivity(tables, tmp_path)
    assert (tmp_path / "sensitivity_curves.pdf").is_file()
    assert (tmp_path / "paired_differences.pdf").is_file()
    assert "sensitivity_curves.pdf" in [path.name for path in figures]
