#!/usr/bin/env python3
"""Supplementary tables for the final K/M decision document.

Writes three CSVs under `results/wcf_sensitivity/analysis/`:

* `factorial_ic1_ic3.csv`: the 3x3 (K, M) interaction tables for IC1 and IC3 on
  the native reference-TCATE and the common-grid law error.
* `null_companions.csv`: the placebo check on the null companion regimes. The
  true contrasts are zero, so every entry is an absolute false-effect error.
* `runtime_summary.csv`: measured wall seconds per method and grid shape.

Usage:

    python3 research/checks/wcf_sensitivity_tables_for_decision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.wcf_sensitivity import INCOME_GRID  # noqa: E402
from wasserstein_causal_forests.g3.wcf_sensitivity_analysis import cell_metrics  # noqa: E402

ANALYSIS = ROOT / "results" / "wcf_sensitivity" / "analysis"
COMBINED = (
    ROOT / "results" / "wcf_sensitivity" / "merged" / "wcf_sensitivity_combined.parquet"
)


def main() -> int:
    cells = cell_metrics(pd.read_parquet(COMBINED))

    income = cells[
        (cells["method"] == "cwdb_dr") & (cells["grid"] == INCOME_GRID)
    ]
    factorial_rows = []
    for dgp in ("IC1", "IC3"):
        for metric, target in (
            ("reference_tcate_rmse", "REF-TCATE-K"),
            ("kernel_law_error", "LAW-A-COMMON199"),
            ("reference_effect_rmse", "REF-ATE-K"),
        ):
            subset = income[
                (income["dgp"] == dgp) & (income["target_id"] == target)
            ]
            pivot = subset.pivot_table(
                index="n_grid", columns="n_particles", values="value", aggfunc="mean"
            )
            for n_grid, row in pivot.iterrows():
                for n_particles, value in row.items():
                    factorial_rows.append(
                        {
                            "dgp": dgp,
                            "metric": metric,
                            "target_id": target,
                            "n_grid": int(n_grid),
                            "n_particles": int(n_particles),
                            "mean": round(float(value), 6),
                        }
                    )
    factorial = pd.DataFrame(factorial_rows).sort_values(
        ["dgp", "target_id", "n_grid", "n_particles"]
    )
    factorial.to_csv(ANALYSIS / "factorial_ic1_ic3.csv", index=False)
    print("factorial IC1/IC3 (REF-TCATE-K):")
    print(
        factorial[factorial["target_id"] == "REF-TCATE-K"]
        .pivot_table(
            index=["dgp", "n_grid"], columns="n_particles", values="mean"
        )
        .round(5)
        .to_string()
    )

    nulls = cells[cells["dgp"].str.endswith("-NULL")]
    null_rows = []
    for target in ("TATE-K-grid_mean", "REF-ATE-K", "REF-TCATE-K", "TCATE-K-grid_mean"):
        subset = nulls[nulls["target_id"] == target]
        if subset.empty:
            continue
        summary = (
            subset.groupby(["method", "n_train"])["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["target_id"] = target
        summary["mc_se"] = summary["std"] / summary["count"] ** 0.5
        null_rows.append(summary)
    null_table = pd.concat(null_rows, ignore_index=True)
    null_table.to_csv(ANALYSIS / "null_companions.csv", index=False)
    print("\nnull companion placebo errors (mean over seeds, lower is better):")
    print(
        null_table.pivot_table(
            index=["target_id", "method"], columns="n_train", values="mean"
        )
        .round(4)
        .to_string()
    )

    runtime = (
        cells.groupby(["method", "n_train", "n_grid", "n_particles"])["wall_seconds"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values(["method", "n_train", "n_grid", "n_particles"])
    )
    runtime.to_csv(ANALYSIS / "runtime_summary.csv", index=False)
    print("\nruntime summary written to runtime_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
