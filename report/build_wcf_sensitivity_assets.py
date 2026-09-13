#!/usr/bin/env python3
"""Build the LaTeX table assets for the WCF sensitivity study.

Inputs (read only, under ``results/wcf_sensitivity/analysis``)::

    cell_metrics.csv, one_factor_k.csv, one_factor_m.csv, confirm_paired.csv,
    sym_methods.csv, null_companions.csv, runtime_summary.csv

Outputs (overwritten, under ``report/tables_generated``)::

    wcf_sensitivity_km.tex, wcf_sensitivity_resolution.tex,
    wcf_sensitivity_assignment.tex, wcf_sensitivity_propensity.tex,
    wcf_sensitivity_factorial.tex, wcf_sensitivity_placebo.tex,
    wcf_sensitivity_cost.tex

Every value is a seed mean, a Monte Carlo standard error, or a seed-paired
summary computed from the frozen CSVs; nothing is fitted and no numeric result
is hardcoded.  Run from the repository root with::

    python3 report/build_wcf_sensitivity_assets.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "wcf_sensitivity" / "analysis"
DEFAULT_OUT = ROOT / "report" / "tables_generated"

PRIMARY_GRID = "wcf_sensitivity_v1_logit_f3"
CONFIRM_GRID = "wcf_sensitivity_v1_confirm"
SYM_GRID = "wcf_sensitivity_v1_sym"
FLEX_GRID = "wcf_sensitivity_v1_flex"
FLEX_RF_GRID = "wcf_sensitivity_v1_flex_rf"
CWDB = "cwdb_dr"
METHOD_HEAD = {"cwdb_dr": "WCF", "causal_drf": "Causal-DRF", "drf": "DRF"}
WCF_N = 1000
SEEDS_PRIMARY = tuple(range(10))
SEEDS_CONFIRM = tuple(range(20, 40))
IC_DGPS = ("IC0", "IC1", "IC2", "IC3")
SYM_DGPS = ("SYM-RANDOM", "SYM-LIN", "SYM-NL", "SYM-MU")
K_VALUES = (5, 25, 49)
M_VALUES = (5, 10, 25)
REF_TCATE_NATIVE = "REF-TCATE-K"


def _read_csv(path: Path, label: str, required: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing {label} input: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    return frame


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot summarise an empty cell")
    mean = sum(numbers) / len(numbers)
    if len(numbers) > 1:
        variance = sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
        se = math.sqrt(variance / len(numbers))
    else:
        se = 0.0
    return mean, se


def _fmt(mean: float, se: float) -> str:
    if not (math.isfinite(mean) and math.isfinite(se)):
        raise ValueError(f"non-finite summary: mean={mean}, se={se}")
    return f"{mean:.4f}({se:.4f})"


def _fmt_pct(pct: float, se: float) -> str:
    if not (math.isfinite(pct) and math.isfinite(se)):
        raise ValueError(f"non-finite paired summary: pct={pct}, se={se}")
    return f"{pct:.1f}({se:.1f})"


def _latex_escape(value: object) -> str:
    text = str(value)
    return text.replace("&", r"\&").replace("_", r"\_")


def _summary_dict(
    frame: pd.DataFrame, keys: Sequence[str], label: str
) -> dict[tuple, tuple[float, float]]:
    entries: dict[tuple, tuple[float, float]] = {}
    for key, group in frame.groupby(list(keys), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        if group["seed"].duplicated().any():
            raise ValueError(f"{label} has several rows for one seed at {key_tuple}")
        entries[key_tuple] = _mean_se(group["value"].tolist())
    return entries


def _require_keys(entries: Mapping, expected: Sequence[tuple], label: str) -> None:
    missing = [key for key in expected if key not in entries]
    if missing:
        raise ValueError(f"{label} is missing expected cells: {missing[:5]}")


def _cell(entries: Mapping, key: tuple, formatter=_fmt) -> str:
    if key not in entries:
        return "--"
    mean, se = entries[key]
    return formatter(mean, se)


def _assert_close(
    observed: tuple[float, float],
    expected: tuple[float, float],
    context: str,
    tol: float = 1e-6,
) -> None:
    if any(abs(a - b) > tol for a, b in zip(observed, expected)):
        raise ValueError(
            f"{context}: summaries disagree, observed={observed}, expected={expected}"
        )


def _panel(title: str, span: int) -> str:
    return rf"\multicolumn{{{span}}}{{l}}{{\textit{{{title}}}}} \\"


def _wrap_table(
    *,
    label: str,
    caption: str,
    columns: str,
    head_lines: Sequence[str],
    body_lines: Sequence[str],
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        *head_lines,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _one_factor_native(
    frame: pd.DataFrame, coordinate: str, label: str
) -> dict[tuple, tuple[float, float]]:
    selected = frame[
        (frame["n_train"] == WCF_N)
        & (frame["target_id"] == REF_TCATE_NATIVE)
        & (frame["metric"] == "reference_tcate_rmse")
    ]
    entries: dict[tuple, tuple[float, float]] = {}
    for _, row in selected.iterrows():
        key = (str(row["dgp"]), int(row[coordinate]))
        if key in entries:
            raise ValueError(f"{label} has duplicate rows for {key}")
        entries[key] = (float(row["mean"]), float(row["mc_se"]))
    return entries


def _paired_pct(
    frame: pd.DataFrame,
    dgp: str,
    metric: str,
    target: str,
    seeds: Sequence[int],
) -> tuple[float, float]:
    """Return the paired percentage difference of K=49 over K=25 at M=10."""
    selected = frame[
        (frame["dgp"] == dgp)
        & (frame["metric"] == metric)
        & (frame["target_id"] == target)
    ]
    seed_list = list(seeds)
    low = selected[(selected["n_grid"] == 25) & (selected["n_particles"] == 10)]
    high = selected[(selected["n_grid"] == 49) & (selected["n_particles"] == 10)]
    low = low.set_index("seed")["value"]
    high = high.set_index("seed")["value"]
    for series, label in ((low, "K=25"), (high, "K=49")):
        if series.index.duplicated().any():
            raise ValueError(f"duplicate seeds in {label} rows for {dgp}/{target}")
    missing = set(seed_list).difference(low.index).union(
        set(seed_list).difference(high.index)
    )
    if missing:
        raise ValueError(f"missing seeds for {dgp}/{target}: {sorted(missing)}")
    low = low.loc[seed_list]
    high = high.loc[seed_list]
    delta = [float(high[seed]) - float(low[seed]) for seed in seed_list]
    mean_low = sum(float(low[seed]) for seed in seed_list) / len(seed_list)
    if mean_low <= 0:
        raise ValueError(f"non-positive baseline for {dgp}/{target}")
    mean_delta, se_delta = _mean_se(delta)
    return 100.0 * mean_delta / mean_low, 100.0 * se_delta / mean_low


def _confirm_paired_pct(
    frame: pd.DataFrame, dgp: str, target: str
) -> tuple[float, float]:
    selected = frame[
        (frame["dgp"] == dgp)
        & (frame["target_id"] == target)
        & (frame["method_a"] == CWDB)
        & (frame["method_b"] == CWDB)
        & (frame["n_train"] == WCF_N)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one confirmation row for {dgp}/{target}")
    row = selected.iloc[0]
    if str(row["coords_a"]) != "K25M10" or str(row["coords_b"]) != "K49M10":
        raise ValueError(f"unexpected confirmation coordinates for {dgp}/{target}")
    mean_a = float(row["mean_a"])
    if mean_a <= 0:
        raise ValueError(f"non-positive confirmation baseline for {dgp}/{target}")
    return float(row["relative_pct"]), 100.0 * float(row["mc_se"]) / mean_a


def _build_km_table(
    cell_metrics: pd.DataFrame,
    one_factor_k: pd.DataFrame,
    one_factor_m: pd.DataFrame,
) -> str:
    targets = {
        "native": REF_TCATE_NATIVE,
        "law_common": "LAW-A-COMMON199",
        "ref_common": "REF-TCATE-COMMON199",
        "law_interior": "LAW-A-INTERIOR",
        "ref_interior": "REF-TCATE-INTERIOR",
    }
    selected = cell_metrics[
        (cell_metrics["grid"] == PRIMARY_GRID)
        & (cell_metrics["method"] == CWDB)
        & (cell_metrics["n_train"] == WCF_N)
        & cell_metrics["seed"].isin(SEEDS_PRIMARY)
        & cell_metrics["dgp"].isin(IC_DGPS)
        & cell_metrics["target_id"].isin(targets.values())
    ]
    summary = _summary_dict(
        selected, ("dgp", "n_grid", "n_particles", "target_id"), "K/M table"
    )
    native_k = _one_factor_native(one_factor_k, "n_grid", "one-factor K")
    native_m = _one_factor_native(one_factor_m, "n_particles", "one-factor M")

    expected = [(dgp, k, 10, targets["native"]) for dgp in IC_DGPS for k in K_VALUES]
    expected += [(dgp, 25, m, targets["native"]) for dgp in IC_DGPS for m in M_VALUES]
    _require_keys(summary, expected, "K/M table")
    for dgp in IC_DGPS:
        for k in K_VALUES:
            _assert_close(
                native_k[(dgp, k)],
                summary[(dgp, k, 10, targets["native"])],
                f"K sensitivity for {dgp} at K={k}",
            )
        for m in M_VALUES:
            _assert_close(
                native_m[(dgp, m)],
                summary[(dgp, 25, m, targets["native"])],
                f"M sensitivity for {dgp} at M={m}",
            )

    head = [
        _panel(r"Panel A: $K$ sensitivity at $M=10$", 7),
        r"DGP & $K$ & Ref. TCATE native & Law common & Ref. TCATE common"
        r" & Law interior & Ref. TCATE interior \\",
    ]
    body: list[str] = []
    for dgp in IC_DGPS:
        for k in K_VALUES:
            values = [
                _cell(native_k, (dgp, k)),
                _cell(summary, (dgp, k, 10, targets["law_common"])),
                _cell(summary, (dgp, k, 10, targets["ref_common"])),
                _cell(summary, (dgp, k, 10, targets["law_interior"])),
                _cell(summary, (dgp, k, 10, targets["ref_interior"])),
            ]
            body.append(f"{dgp} & {k} & " + " & ".join(values) + r" \\")
    body.extend(
        [
            r"\midrule",
            _panel(r"Panel B: $M$ sensitivity at $K=25$", 7),
            r"DGP & $M$ & Ref. TCATE native & Law common & Ref. TCATE common"
            r" & Law interior & Ref. TCATE interior \\",
            r"\midrule",
        ]
    )
    for dgp in IC_DGPS:
        for m in M_VALUES:
            values = [
                _cell(native_m, (dgp, m)),
                _cell(summary, (dgp, 25, m, targets["law_common"])),
                _cell(summary, (dgp, 25, m, targets["ref_common"])),
                _cell(summary, (dgp, 25, m, targets["law_interior"])),
                _cell(summary, (dgp, 25, m, targets["ref_interior"])),
            ]
            body.append(f"{dgp} & {m} & " + " & ".join(values) + r" \\")

    caption = (
        r"Sensitivity of WCF to the quantile grid size $K$ and the particle budget "
        r"$M$ at $n=1000$ for designs IC0 to IC3. Panel A varies $K$ at $M=10$; "
        r"Panel B varies $M$ at $K=25$. Entries are mean(SE) over ten replications. "
        r"The native reference TCATE is evaluated on each cell's own $K$-grid, while "
        r"the common and interior columns are evaluated on fixed 199-level targets. "
        r"The primary setting $(K,M)=(25,10)$ appears in both panels."
    )
    return _wrap_table(
        label="tab:sensitivity-km",
        caption=caption,
        columns="llrrrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_resolution_table(
    cell_metrics: pd.DataFrame, confirm_paired: pd.DataFrame
) -> str:
    columns = (
        ("Full-range Law", "kernel_law_error", "LAW-A-COMMON199", True),
        ("Interior Law", "kernel_law_error", "LAW-A-INTERIOR", False),
        ("Full-range Ref. TCATE", "reference_tcate_rmse", "REF-TCATE-COMMON199", True),
        ("Interior Ref. TCATE", "reference_tcate_rmse", "REF-TCATE-INTERIOR", False),
        ("Ref.-ATE common", "reference_effect_rmse", "REF-ATE-COMMON199", True),
    )
    primary = cell_metrics[
        (cell_metrics["grid"] == PRIMARY_GRID)
        & (cell_metrics["method"] == CWDB)
        & (cell_metrics["n_train"] == WCF_N)
    ]
    confirm = cell_metrics[
        (cell_metrics["grid"] == CONFIRM_GRID)
        & (cell_metrics["method"] == CWDB)
        & (cell_metrics["n_train"] == WCF_N)
    ]

    # The shipped confirmation summary omits the interior targets, so those two
    # columns are paired directly from the frozen confirmation cell metrics.
    # The three shared columns must agree with both sources.
    for dgp in IC_DGPS:
        for _, metric, target, from_confirm in columns:
            if not from_confirm:
                continue
            _assert_close(
                _confirm_paired_pct(confirm_paired, dgp, target),
                _paired_pct(confirm, dgp, metric, target, SEEDS_CONFIRM),
                f"confirmation cross-check for {dgp}/{target}",
                tol=0.06,
            )

    head = [
        r"DGP & Full-range Law & Interior Law & Full-range Ref. TCATE"
        r" & Interior Ref. TCATE & Ref.-ATE common \\",
    ]
    body: list[str] = []
    for title, frame, seeds, use_confirm in (
        ("Seeds 0 to 9 (primary run)", primary, SEEDS_PRIMARY, False),
        ("Seeds 20 to 39 (fresh-seed confirmation)", confirm, SEEDS_CONFIRM, True),
    ):
        body.append(_panel(title, 6))
        body.append(r"\midrule")
        for dgp in IC_DGPS:
            values = []
            for _, metric, target, _ in columns:
                # Single rounding from the frozen cell metrics for both seed
                # blocks; the shipped confirm_paired.csv is only a cross-check.
                statistic = _paired_pct(frame, dgp, metric, target, seeds)
                values.append(_fmt_pct(*statistic))
            body.append(f"{dgp} & " + " & ".join(values) + r" \\")

    caption = (
        r"Seed-paired resolution differences at $n=1000$ for designs IC0 to IC3, "
        r"computed as $K=49$ minus $K=25$ at $M=10$. Entries are percentage changes, "
        r"so a negative value means the larger grid has the smaller error; the value "
        r"in parentheses is the Monte Carlo standard error of the paired percentage "
        r"difference in percentage points. The first block uses seeds 0 to 9 from the "
        r"primary run and the second block uses the fresh-seed confirmation at seeds "
        r"20 to 39."
    )
    return _wrap_table(
        label="tab:sensitivity-resolution",
        caption=caption,
        columns="lrrrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_assignment_table(sym_methods: pd.DataFrame) -> str:
    selected = sym_methods[
        (sym_methods["target_id"] == REF_TCATE_NATIVE)
        & (sym_methods["metric"] == "reference_tcate_rmse")
        & (sym_methods["n_grid"] == 25)
        & (sym_methods["n_particles"] == 10)
        & sym_methods["method"].isin(METHOD_HEAD)
        & sym_methods["comparator"].isna()
        & sym_methods["dgp"].isin(SYM_DGPS)
    ]
    entries: dict[tuple, tuple[float, float]] = {}
    for _, row in selected.iterrows():
        key = (str(row["dgp"]), int(row["n_train"]), str(row["method"]))
        if key in entries:
            raise ValueError(f"assignment table has duplicate rows for {key}")
        entries[key] = (float(row["mean"]), float(row["mc_se"]))
    _require_keys(
        entries,
        [
            (dgp, n_train, method)
            for n_train in (500, WCF_N)
            for dgp in SYM_DGPS
            for method in METHOD_HEAD
        ],
        "assignment table",
    )

    head = [
        r"DGP & WCF & Causal-DRF & DRF \\",
    ]
    body: list[str] = []
    for panel_label, n_train in (("Panel A", 500), ("Panel B", WCF_N)):
        body.append(_panel(rf"{panel_label}: $n={n_train}$", 4))
        body.append(r"\midrule")
        for dgp in SYM_DGPS:
            values = [
                _cell(entries, (dgp, n_train, method)) for method in METHOD_HEAD
            ]
            body.append(f"{_latex_escape(dgp)} & " + " & ".join(values) + r" \\")

    caption = (
        r"Symmetric and nonlinear assignment designs at $n=500$ and $n=1000$. "
        r"Entries are native reference-TCATE errors, mean(SE) over ten replications; "
        r"lower is better. SYM-RANDOM is a constant-propensity calibration control, "
        r"SYM-LIN uses a linear assignment index, and SYM-NL and SYM-MU use nonlinear "
        r"assignment with the same symmetric outcome laws."
    )
    return _wrap_table(
        label="tab:sensitivity-assignment",
        caption=caption,
        columns="lrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_propensity_table(cell_metrics: pd.DataFrame) -> str:
    columns = (
        ("Logistic", SYM_GRID, "cwdb_dr"),
        ("Random forest", FLEX_RF_GRID, "cwdb_dr_flex_rf"),
        ("Gradient boosting", FLEX_GRID, "cwdb_dr_flex"),
        ("Oracle", FLEX_GRID, "cwdb_dr_oracle"),
    )
    selected = cell_metrics[
        (cell_metrics["target_id"] == REF_TCATE_NATIVE)
        & (cell_metrics["metric"] == "reference_tcate_rmse")
        & (cell_metrics["n_grid"] == 25)
        & (cell_metrics["n_particles"] == 10)
        & (cell_metrics["n_train"].isin((500, WCF_N)))
        & cell_metrics["seed"].isin(SEEDS_PRIMARY)
        & cell_metrics["dgp"].isin(("SYM-NL", "SYM-MU"))
    ]
    summaries: dict[tuple[str, str], dict[tuple, tuple[float, float]]] = {}
    for name, grid, method in columns:
        sub = selected[(selected["grid"] == grid) & (selected["method"] == method)]
        summaries[(grid, method)] = _summary_dict(
            sub, ("dgp", "n_train"), f"propensity {name}"
        )
        _require_keys(
            summaries[(grid, method)],
            [(dgp, n_train) for n_train in (500, WCF_N) for dgp in ("SYM-NL", "SYM-MU")],
            f"propensity {name}",
        )

    head = [
        r"DGP & Logistic & Random forest & Gradient boosting & Oracle \\",
    ]
    body: list[str] = []
    for panel_label, n_train in (("Panel A", 500), ("Panel B", WCF_N)):
        body.append(_panel(rf"{panel_label}: $n={n_train}$", 5))
        body.append(r"\midrule")
        for dgp in ("SYM-NL", "SYM-MU"):
            values = [
                _cell(summaries[(grid, method)], (dgp, n_train))
                for _, grid, method in columns
            ]
            body.append(f"{_latex_escape(dgp)} & " + " & ".join(values) + r" \\")

    caption = (
        r"Propensity-model sensitivity on the SYM-NL and SYM-MU designs at $n=500$ "
        r"and $n=1000$. Entries are native reference-TCATE errors, mean(SE) over ten "
        r"replications; lower is better. The columns replace the default logistic "
        r"propensity with a random-forest factory, a gradient-boosting factory, and "
        r"the oracle propensity, which is a diagnostic and not a feasible competitor. "
        r"The outcome learner, grid, and particle budget are held fixed."
    )
    return _wrap_table(
        label="tab:sensitivity-propensity",
        caption=caption,
        columns="lrrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_factorial_table(cell_metrics: pd.DataFrame) -> str:
    selected = cell_metrics[
        (cell_metrics["grid"] == PRIMARY_GRID)
        & (cell_metrics["method"] == CWDB)
        & (cell_metrics["n_train"] == WCF_N)
        & cell_metrics["seed"].isin(SEEDS_PRIMARY)
        & (cell_metrics["target_id"] == REF_TCATE_NATIVE)
        & cell_metrics["dgp"].isin(("IC1", "IC3"))
    ]
    summary = _summary_dict(
        selected, ("dgp", "n_grid", "n_particles"), "factorial table"
    )
    _require_keys(
        summary,
        [
            (dgp, k, m)
            for dgp in ("IC1", "IC3")
            for k in K_VALUES
            for m in M_VALUES
        ],
        "factorial table",
    )

    head = [
        r"$K$ & $M=5$ & $M=10$ & $M=25$ \\",
    ]
    body: list[str] = []
    for panel_label, dgp in (("Panel A", "IC1"), ("Panel B", "IC3")):
        body.append(_panel(rf"{panel_label}: {dgp}", 4))
        body.append(r"\midrule")
        for k in K_VALUES:
            values = [_cell(summary, (dgp, k, m)) for m in M_VALUES]
            body.append(f"{k} & " + " & ".join(values) + r" \\")

    caption = (
        r"Interaction grid for the quantile grid size $K$ and the particle budget "
        r"$M$ on IC1 and IC3 at $n=1000$ with seeds 0 to 9. Entries are native "
        r"reference-TCATE errors, mean(SE) over ten replications; lower is better, "
        r"rows are $K$, and columns are $M$."
    )
    return _wrap_table(
        label="tab:sensitivity-factorial",
        caption=caption,
        columns="lrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_placebo_table(null_companions: pd.DataFrame) -> str:
    rows = (
        ("TATE grid mean", "TATE-K-grid_mean"),
        ("TCATE grid mean", "TCATE-K-grid_mean"),
        ("Ref. TATE", "REF-ATE-K"),
        ("Ref. TCATE", "REF-TCATE-K"),
    )
    entries: dict[tuple, tuple[float, float]] = {}
    for _, row in null_companions.iterrows():
        key = (str(row["target_id"]), int(row["n_train"]), str(row["method"]))
        if key in entries:
            raise ValueError(f"placebo table has duplicate rows for {key}")
        entries[key] = (float(row["mean"]), float(row["mc_se"]))
    _require_keys(
        entries,
        [
            (target, n_train, method)
            for _, target in rows
            for n_train in (500, WCF_N)
            for method in METHOD_HEAD
        ],
        "placebo table",
    )

    head = [
        r"Target & WCF & Causal-DRF & DRF \\",
    ]
    body: list[str] = []
    for panel_label, n_train in (("Panel A", 500), ("Panel B", WCF_N)):
        body.append(_panel(rf"{panel_label}: $n={n_train}$", 4))
        body.append(r"\midrule")
        for label, target in rows:
            values = [_cell(entries, (target, n_train, method)) for method in METHOD_HEAD]
            body.append(f"{_latex_escape(label)} & " + " & ".join(values) + r" \\")

    caption = (
        r"Null companion designs with exactly zero treatment effects, so every "
        r"entry is an absolute false-effect error. Entries are mean(SE) pooled over "
        r"the six null companion designs and ten replications; lower is better. The "
        r"first two rows use the grid-mean functional, and the last two rows use the "
        r"reference-distance marginal and conditional targets."
    )
    return _wrap_table(
        label="tab:sensitivity-placebo",
        caption=caption,
        columns="lrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_cost_table(runtime_summary: pd.DataFrame) -> str:
    pairs = (
        (5, 5),
        (5, 10),
        (5, 25),
        (25, 5),
        (25, 10),
        (25, 25),
        (49, 5),
        (49, 10),
        (49, 25),
    )
    selected = runtime_summary[
        (runtime_summary["method"] == CWDB)
        & (runtime_summary["n_train"] == WCF_N)
    ].set_index(["n_grid", "n_particles"])
    if selected.index.duplicated().any():
        raise ValueError("runtime summary has duplicate method coordinates")
    if (25, 10) not in selected.index:
        raise ValueError("runtime summary lacks the reference coordinate (25,10)")
    reference = float(selected.loc[(25, 10), "median"])
    if reference <= 0:
        raise ValueError("non-positive reference runtime")

    head = [
        r"$K$ & $M$ & Median wall seconds & Relative to $(25,10)$ \\",
    ]
    body: list[str] = []
    for pair in pairs:
        if pair not in selected.index:
            continue
        median = float(selected.loc[pair, "median"])
        body.append(f"{pair[0]} & {pair[1]} & {median:.4f} & {median / reference:.4f}" + r" \\")

    caption = (
        r"Wall-clock cost of WCF at $n=1000$. Medians are wall seconds per cell "
        r"under seven concurrent single-threaded workers; the final column is the "
        r"ratio to the $(K,M)=(25,10)$ median. Pairs absent from the runtime summary "
        r"are omitted."
    )
    return _wrap_table(
        label="tab:sensitivity-cost",
        caption=caption,
        columns="llrr",
        head_lines=head,
        body_lines=body,
    )


def build(output_dir: Path) -> Mapping[str, Path]:
    cell_metrics = _read_csv(
        ANALYSIS / "cell_metrics.csv",
        "cell_metrics",
        (
            "grid",
            "dgp",
            "n_train",
            "n_grid",
            "n_particles",
            "method",
            "seed",
            "metric",
            "target_id",
            "value",
        ),
    )
    one_factor_k = _read_csv(
        ANALYSIS / "one_factor_k.csv",
        "one_factor_k",
        ("dgp", "n_train", "n_grid", "n_particles", "metric", "target_id", "mean", "mc_se"),
    )
    one_factor_m = _read_csv(
        ANALYSIS / "one_factor_m.csv",
        "one_factor_m",
        ("dgp", "n_train", "n_grid", "n_particles", "metric", "target_id", "mean", "mc_se"),
    )
    confirm_paired = _read_csv(
        ANALYSIS / "confirm_paired.csv",
        "confirm_paired",
        (
            "dgp",
            "n_train",
            "metric",
            "target_id",
            "method_a",
            "coords_a",
            "method_b",
            "coords_b",
            "mean_a",
            "mc_se",
            "relative_pct",
        ),
    )
    sym_methods = _read_csv(
        ANALYSIS / "sym_methods.csv",
        "sym_methods",
        (
            "dgp",
            "n_train",
            "n_grid",
            "n_particles",
            "method",
            "comparator",
            "metric",
            "target_id",
            "mean",
            "mc_se",
        ),
    )
    null_companions = _read_csv(
        ANALYSIS / "null_companions.csv",
        "null_companions",
        ("method", "n_train", "mean", "mc_se", "target_id"),
    )
    runtime_summary = _read_csv(
        ANALYSIS / "runtime_summary.csv",
        "runtime_summary",
        ("method", "n_train", "n_grid", "n_particles", "median"),
    )

    outputs = {
        "wcf_sensitivity_km": _build_km_table(
            cell_metrics, one_factor_k, one_factor_m
        ),
        "wcf_sensitivity_resolution": _build_resolution_table(
            cell_metrics, confirm_paired
        ),
        "wcf_sensitivity_assignment": _build_assignment_table(sym_methods),
        "wcf_sensitivity_propensity": _build_propensity_table(cell_metrics),
        "wcf_sensitivity_factorial": _build_factorial_table(cell_metrics),
        "wcf_sensitivity_placebo": _build_placebo_table(null_companions),
        "wcf_sensitivity_cost": _build_cost_table(runtime_summary),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, text in outputs.items():
        path = output_dir / f"{name}.tex"
        path.write_text(text, encoding="utf-8")
        written[name] = path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for generated LaTeX assets",
    )
    args = parser.parse_args()
    outputs = build(args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
