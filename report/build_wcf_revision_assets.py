#!/usr/bin/env python3
"""Build the evidence ledger and compact tables used by the WCF revision.

The script deliberately reads the frozen merged result files and performs no
model fitting.  It keeps the final ``cwdb_dr`` method under the paper label
``WCF`` and aggregates only after arm-specific and functional rows have been
collapsed within a seed.  Standard errors therefore describe variation over
the simulation seeds rather than variation over arms or functionals.

Run from the repository root with::

    rtk python3 report/build_wcf_revision_assets.py

The generated files are written below ``report/tables_generated``.  They are
review artifacts and do not modify the manuscript source.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "report" / "tables_generated"

PHASE6 = ROOT / "results" / "merged_phase6" / "phase6_results.parquet"
ORIG_DRF = ROOT / "results" / "merged_original_drf" / "main_results.parquet"
ORIG_CDRF = ROOT / "results" / "merged_original_causal_drf" / "main_results.parquet"
PHASE65 = ROOT / "results" / "merged_phase65" / "phase65_results.parquet"

SEEDS = tuple(range(10))

COMMON_METRICS = (
    "mean_quantile_rmse",
    "kernel_law_error",
    "tate_functional_rmse",
    "tcate_functional_rmse",
    "reference_effect_rmse",
    "reference_tcate_rmse",
)
ZI_METRICS = COMMON_METRICS + ("zero_mass_abs_error", "mass_contrast_rmse")

FUNCTION_TARGETS = {
    "tate_functional_rmse": (
        "TATE-K-grid_mean",
        "TATE-K-grid_sd",
        "TATE-K-grid_skewness",
        "TATE-K-grid_upper_tail_mean",
    ),
    "tcate_functional_rmse": (
        "TCATE-K-grid_mean",
        "TCATE-K-grid_sd",
        "TCATE-K-grid_skewness",
        "TCATE-K-grid_upper_tail_mean",
    ),
}

METRIC_HEAD = {
    "mean_quantile_rmse": "MeanQ",
    "kernel_law_error": "Law",
    "tate_functional_rmse": "TATE",
    "tcate_functional_rmse": "TCATE",
    "reference_effect_rmse": "Ref. TATE",
    "reference_tcate_rmse": "Ref. TCATE",
    "zero_mass_abs_error": "Zero mass",
    "mass_contrast_rmse": "Mass contrast",
}

METHOD_HEAD = {
    "cwdb_dr": "WCF",
    "cwdb_zipt": "Two-part WCF",
    "causal_drf": "Causal-DRF",
    "drf": "DRF",
}

SUITE_DGPS = {
    "income": ("IC0", "IC1", "IC2", "IC3"),
    "abstract": ("D0", "D2", "D5", "D6", "D7", "D8"),
    "zi": ("ZI0", "ZI1", "ZI2", "ZI3"),
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing {label} result file: {path}")
    frame = pd.read_parquet(path)
    required = {
        "cell_key",
        "dgp",
        "grid",
        "method",
        "metric",
        "n_train",
        "seed",
        "status",
        "target_id",
        "value",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    # A raw metric row is identified by these fields.  Duplicate rows would
    # silently change a seed-level average, so fail before any aggregation.
    row_key = ["cell_key", "metric", "target_id", "arm", "detail"]
    duplicates = frame.duplicated(row_key, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, row_key].head(3).to_dict("records")
        raise ValueError(f"duplicate raw result rows in {label}: {sample}")
    return frame


def _subset(
    frame: pd.DataFrame,
    *,
    grid: str,
    methods: Sequence[str],
    dgps: Sequence[str],
    n_train: int,
    seeds: Iterable[int] = SEEDS,
) -> pd.DataFrame:
    out = frame[
        (frame["grid"] == grid)
        & frame["method"].isin(methods)
        & frame["dgp"].isin(dgps)
        & (frame["n_train"] == n_train)
        & (frame["n_grid"] == 25)
        & (frame["n_particles"] == 10)
        & frame["seed"].isin(tuple(seeds))
        & (frame["status"] == "ok")
    ].copy()
    if out.empty:
        raise ValueError(
            f"empty result subset grid={grid!r}, methods={methods}, "
            f"dgps={dgps}, n_train={n_train}"
        )
    return out


def _cell_meta(frame: pd.DataFrame, cell_keys: Sequence[str]) -> pd.DataFrame:
    cols = ["cell_key", "method", "dgp", "n_train", "seed"]
    meta = frame[frame["cell_key"].isin(cell_keys)][cols].drop_duplicates()
    counts = meta.groupby("cell_key", sort=False).size()
    if (counts != 1).any():
        raise ValueError("cell_key maps to more than one method/dgp/n/seed")
    return meta


def _metric_cells(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one value per cell and seed for a requested metric.

    Arm rows and functional rows are averaged inside the cell first.  The
    returned frame is then suitable for a seed-level mean and standard error.
    """
    q = frame[frame["metric"] == metric].copy()
    if q.empty:
        raise ValueError(f"metric {metric!r} absent from selected result subset")
    if q["value"].isna().any():
        raise ValueError(f"metric {metric!r} contains missing values")

    if metric in FUNCTION_TARGETS:
        expected = set(FUNCTION_TARGETS[metric])
        found = set(q["target_id"].dropna().astype(str))
        if found != expected:
            raise ValueError(
                f"{metric} target IDs differ from the expected four: "
                f"found={sorted(found)}, expected={sorted(expected)}"
            )
        counts = q.groupby("cell_key", sort=False)["target_id"].nunique()
        if (counts != len(expected)).any():
            raise ValueError(f"{metric} does not have four functionals per cell")
        values = q.groupby("cell_key", sort=False)["value"].mean().rename("value")
        target_aggregation = "mean over four functionals within seed"
    elif metric in ("kernel_law_error", "zero_mass_abs_error"):
        # Both metrics are reported for the two arms.  The arm average is
        # taken before calculating uncertainty over seeds.
        counts = q.groupby("cell_key", sort=False)["arm"].nunique(dropna=True)
        if (counts != 2).any():
            raise ValueError(f"{metric} does not have two arm rows per cell")
        values = q.groupby("cell_key", sort=False)["value"].mean().rename("value")
        target_aggregation = "mean over two arms within seed"
    else:
        counts = q.groupby("cell_key", sort=False).size()
        if (counts != 1).any():
            raise ValueError(f"{metric} has multiple rows per cell unexpectedly")
        values = q.set_index("cell_key")["value"].rename("value")
        target_aggregation = "single reported value within seed"

    meta = _cell_meta(q, values.index.tolist()).set_index("cell_key")
    out = meta.join(values, how="inner").reset_index()
    out["metric"] = metric
    out["target_aggregation"] = target_aggregation
    return out


def _summary_rows(
    frame: pd.DataFrame,
    *,
    suite: str,
    source: str,
    grid: str,
    methods: Sequence[str],
    dgps: Sequence[str],
    n_train: int,
    metrics: Sequence[str],
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    selected = _subset(
        frame,
        grid=grid,
        methods=methods,
        dgps=dgps,
        n_train=n_train,
        seeds=seeds,
    )
    rows: list[dict] = []
    expected = set(seeds)
    for metric in metrics:
        cells = _metric_cells(selected, metric)
        expected_groups = {(dgp, method) for dgp in dgps for method in methods}
        actual_groups = set(zip(cells.dgp, cells.method))
        if actual_groups != expected_groups:
            raise ValueError(f"incomplete method/design coverage for {metric}")
        for (dgp, method, n), group in cells.groupby(
            ["dgp", "method", "n_train"], sort=False
        ):
            observed = set(group["seed"].astype(int))
            if observed != expected or len(group) != len(expected):
                raise ValueError(
                    f"incomplete seed set for {suite}/{dgp}/{method}/{metric}: "
                    f"observed={sorted(observed)}, expected={sorted(expected)}"
                )
            values = group["value"].to_numpy(dtype=float)
            mean = float(values.mean())
            se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            rows.append(
                {
                    "suite": suite,
                    "source": source,
                    "grid": grid,
                    "dgp": dgp,
                    "n_train": int(n),
                    "method": method,
                    "metric": metric,
                    "target_aggregation": group["target_aggregation"].iloc[0],
                    "n_seeds": len(values),
                    "seed_min": min(observed),
                    "seed_max": max(observed),
                    "mean": mean,
                    "se": se,
                    "mean_se": _fmt(mean, se),
                }
            )
    return pd.DataFrame(rows)


def _functional_rows(
    frame: pd.DataFrame,
    *,
    suite: str,
    source: str,
    grid: str,
    methods: Sequence[str],
    dgps: Sequence[str],
    n_train: int,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    selected = _subset(
        frame,
        grid=grid,
        methods=methods,
        dgps=dgps,
        n_train=n_train,
        seeds=seeds,
    )
    rows: list[dict] = []
    for metric, targets in FUNCTION_TARGETS.items():
        q = selected[selected["metric"] == metric].copy()
        for (dgp, method, target), group in q.groupby(
            ["dgp", "method", "target_id"], sort=False
        ):
            if target not in targets:
                raise ValueError(f"unexpected target {target!r} in {metric}")
            raw_counts = group.groupby("cell_key", sort=False).size()
            if (raw_counts != 1).any():
                raise ValueError(
                    f"{metric} has multiple rows for a functional cell in "
                    f"{suite}/{dgp}/{method}/{target}"
                )
            cells = group.groupby("cell_key", sort=False)["value"].mean()
            # This check catches an accidental metadata collision without
            # relying on row order.
            meta = _cell_meta(group, cells.index.tolist()).set_index("cell_key")
            seed_values = meta.join(cells.rename("value"), how="inner")
            observed = set(seed_values["seed"].astype(int))
            if observed != set(seeds) or len(seed_values) != len(seeds):
                raise ValueError(
                    f"incomplete functional seed set for {suite}/{dgp}/{method}/{target}"
                )
            values = seed_values["value"].to_numpy(dtype=float)
            mean = float(values.mean())
            se = float(values.std(ddof=1) / math.sqrt(len(values)))
            rows.append(
                {
                    "suite": suite,
                    "source": source,
                    "grid": grid,
                    "dgp": dgp,
                    "n_train": int(n_train),
                    "method": method,
                    "metric": metric,
                    "target_id": target,
                    "n_seeds": len(values),
                    "seed_min": min(observed),
                    "seed_max": max(observed),
                    "mean": mean,
                    "se": se,
                    "mean_se": _fmt(mean, se),
                }
            )
    return pd.DataFrame(rows)


def _fmt(mean: float, se: float) -> str:
    return f"{mean:.4f}({se:.4f})"


def _latex_escape(value: object) -> str:
    text = str(value)
    return text.replace("&", r"\&").replace("_", r"\_")


def _table(
    summary: pd.DataFrame,
    *,
    label: str,
    caption: str,
    metrics: Sequence[str],
    dgp_order: Sequence[str],
    method_order: Sequence[str],
) -> str:
    pivot = summary.pivot_table(
        index=["dgp", "method"],
        columns="metric",
        values="mean_se",
        aggfunc="first",
    )
    columns = "ll" + "r" * len(metrics)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "DGP & Method & " + " & ".join(METRIC_HEAD[m] for m in metrics) + r" \\",
        r"\midrule",
    ]
    for dgp in dgp_order:
        for method in method_order:
            key = (dgp, method)
            if key not in pivot.index:
                continue
            row = pivot.loc[key]
            vals = [row.get(metric, "") for metric in metrics]
            lines.append(
                _latex_escape(dgp)
                + " & "
                + METHOD_HEAD.get(method, _latex_escape(method))
                + " & "
                + " & ".join(_latex_escape(v) for v in vals)
                + r" \\",
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _functional_table(
    detail: pd.DataFrame,
    *,
    label: str,
    caption: str,
    dgp_order: Sequence[str],
    method_order: Sequence[str],
    n_train: int,
) -> str:
    q = detail[detail["n_train"] == n_train].copy()
    index = q.set_index(["dgp", "method", "target_id"])
    target_order = FUNCTION_TARGETS["tate_functional_rmse"]
    rows = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{lllrr}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        r"DGP & Method & Functional & TATE & TCATE \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"DGP & Method & Functional & TATE & TCATE \\",
        r"\midrule",
        r"\endhead",
    ]
    for dgp in dgp_order:
        for method in method_order:
            for target in target_order:
                tate_key = (dgp, method, target)
                tcate_target = target.replace("TATE-", "TCATE-", 1)
                tcate_key = (dgp, method, tcate_target)
                if tate_key not in index.index or tcate_key not in index.index:
                    continue
                tate = index.loc[tate_key, "mean_se"]
                tcate = index.loc[tcate_key, "mean_se"]
                name = target.removeprefix("TATE-K-grid_").replace("_", " ")
                rows.append(
                    f"{_latex_escape(dgp)} & {METHOD_HEAD.get(method, method)} & "
                    f"{_latex_escape(name)} & {_latex_escape(tate)} & "
                    f"{_latex_escape(tcate)} " + r"\\"
                )
    rows.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(rows)


def _save_csv(path: Path, frames: Sequence[pd.DataFrame]) -> None:
    ledger = pd.concat(frames, ignore_index=True, sort=False)
    # The summary key must be unique.  This catches accidental mixing of two
    # result sources or duplicate functional rows before publication.
    key = [
        "suite",
        "source",
        "grid",
        "dgp",
        "n_train",
        "method",
        "metric",
    ]
    if ledger.duplicated(key).any():
        raise ValueError("duplicate summary cells in evidence ledger")
    ledger = ledger.sort_values(
        ["suite", "n_train", "dgp", "method", "metric"], kind="stable"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(path, index=False, float_format="%.10g")


def build(output_dir: Path) -> Mapping[str, Path]:
    phase6 = _read(PHASE6, "phase6")
    original_drf = _read(ORIG_DRF, "original DRF")
    original_cdrf = _read(ORIG_CDRF, "original Causal-DRF")
    phase65 = _read(PHASE65, "phase65")

    # Income: final WCF and the two forest baselines are all reported at both
    # sample sizes from the matched phase6 run.  The original merged files
    # are reserved for the D-suite baselines below.
    income_frames = []
    income_detail = []
    for n in (500, 1000):
        income_frames.extend(
            [
                _summary_rows(
                    phase6,
                    suite="income",
                    source="merged_phase6",
                    grid="income",
                    methods=["cwdb_dr"],
                    dgps=SUITE_DGPS["income"],
                    n_train=n,
                    metrics=COMMON_METRICS,
                ),
                _summary_rows(
                    phase6,
                    suite="income",
                    source="merged_phase6",
                    grid="income",
                    methods=["causal_drf"],
                    dgps=SUITE_DGPS["income"],
                    n_train=n,
                    metrics=COMMON_METRICS,
                ),
                _summary_rows(
                    phase6,
                    suite="income",
                    source="merged_phase6",
                    grid="income",
                    methods=["drf"],
                    dgps=SUITE_DGPS["income"],
                    n_train=n,
                    metrics=COMMON_METRICS,
                ),
            ]
        )
        income_detail.extend(
            [
                _functional_rows(
                    phase6,
                    suite="income",
                    source="merged_phase6",
                    grid="income",
                    methods=["cwdb_dr"],
                    dgps=SUITE_DGPS["income"],
                    n_train=n,
                ),
                _functional_rows(
                    phase6,
                    suite="income",
                    source="merged_phase6",
                    grid="income",
                    methods=["causal_drf"],
                    dgps=SUITE_DGPS["income"],
                    n_train=n,
                ),
                _functional_rows(
                    phase6,
                    suite="income",
                    source="merged_phase6",
                    grid="income",
                    methods=["drf"],
                    dgps=SUITE_DGPS["income"],
                    n_train=n,
                ),
            ]
        )

    # Main nonlinear/symmetric assignment designs at n=1000.  The design
    # comparison uses the final WCF rows from phase6 and matched-ten-seed
    # baseline rows from the original merged files.
    abstract_frames = [
        _summary_rows(
            phase6,
            suite="abstract",
            source="merged_phase6",
            grid="main",
            methods=["cwdb_dr"],
            dgps=SUITE_DGPS["abstract"],
            n_train=1000,
            metrics=COMMON_METRICS,
        ),
        _summary_rows(
            original_cdrf,
            suite="abstract",
            source="merged_original_causal_drf",
            grid="main",
            methods=["causal_drf"],
            dgps=SUITE_DGPS["abstract"],
            n_train=1000,
            metrics=COMMON_METRICS,
        ),
        _summary_rows(
            original_drf,
            suite="abstract",
            source="merged_original_drf",
            grid="main",
            methods=["drf"],
            dgps=SUITE_DGPS["abstract"],
            n_train=1000,
            metrics=COMMON_METRICS,
        ),
    ]
    abstract_detail = [
        _functional_rows(
            phase6,
            suite="abstract",
            source="merged_phase6",
            grid="main",
            methods=["cwdb_dr"],
            dgps=SUITE_DGPS["abstract"],
            n_train=1000,
        ),
        _functional_rows(
            original_cdrf,
            suite="abstract",
            source="merged_original_causal_drf",
            grid="main",
            methods=["causal_drf"],
            dgps=SUITE_DGPS["abstract"],
            n_train=1000,
        ),
        _functional_rows(
            original_drf,
            suite="abstract",
            source="merged_original_drf",
            grid="main",
            methods=["drf"],
            dgps=SUITE_DGPS["abstract"],
            n_train=1000,
        ),
    ]

    # The zero-inflated comparison intentionally excludes ordinary WCF rows:
    # WCF-ZIPT is the optional two-part extension and is compared with the two
    # established forest baselines at the common n=1000 setting.
    zi_frames = [
        _summary_rows(
            phase65,
            suite="zi",
            source="merged_phase65",
            grid="e_zi",
            methods=["cwdb_zipt", "causal_drf", "drf"],
            dgps=SUITE_DGPS["zi"],
            n_train=1000,
            metrics=ZI_METRICS,
        )
    ]
    all_summary = income_frames + abstract_frames + zi_frames
    all_detail = income_detail + abstract_detail

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = output_dir / "wcf_revision_evidence.csv"
    detail_csv = output_dir / "wcf_revision_functional_detail.csv"
    tables = output_dir / "wcf_revision_tables.tex"
    _save_csv(evidence, all_summary)
    detail = pd.concat(all_detail, ignore_index=True, sort=False)
    detail_key = ["suite", "source", "grid", "dgp", "n_train", "method", "metric", "target_id"]
    if detail.duplicated(detail_key).any():
        raise ValueError("duplicate functional cells in detail ledger")
    detail.sort_values(["suite", "n_train", "dgp", "method", "metric", "target_id"], kind="stable").to_csv(
        detail_csv, index=False, float_format="%.10g"
    )

    summary = pd.concat(all_summary, ignore_index=True, sort=False)
    text = [
        "% Generated by report/build_wcf_revision_assets.py; do not edit by hand.",
        "% Each entry is mean(SE) over ten simulation seeds.",
        _table(
            summary[(summary.suite == "income") & (summary.n_train == 500)],
            label="tab:revision-income-500",
            caption="Location and shape-effect designs at $n=500$. Entries are mean(SE) over ten replications. MeanQ is quantile-vector RMSE; Law is squared MMD. TATE columns give mean absolute error and TCATE columns give mean bin RMSE. Functional errors are averaged within each replication.",
            metrics=COMMON_METRICS,
            dgp_order=SUITE_DGPS["income"],
            method_order=["cwdb_dr", "causal_drf", "drf"],
        ),
        _table(
            summary[(summary.suite == "income") & (summary.n_train == 1000)],
            label="tab:revision-income-1000",
            caption="Location and shape-effect designs at $n=1000$. Entries are mean(SE) over ten replications, using the same metrics as Table~\\ref{tab:revision-income-500}.",
            metrics=COMMON_METRICS,
            dgp_order=SUITE_DGPS["income"],
            method_order=["cwdb_dr", "causal_drf", "drf"],
        ),
        _table(
            summary[summary.suite == "abstract"],
            label="tab:revision-abstract",
            caption="Symmetric and nonlinear designs at $n=1000$. Entries are mean(SE) over ten replications, using the same metrics as Table~\\ref{tab:revision-income-500}.",
            metrics=COMMON_METRICS,
            dgp_order=SUITE_DGPS["abstract"],
            method_order=["cwdb_dr", "causal_drf", "drf"],
        ),
        _table(
            summary[summary.suite == "zi"],
            label="tab:revision-zi",
            caption="Structural-zero outcomes at $n=1000$. Entries are mean(SE) over ten replications. Zero mass and mass contrast are RMSEs; Law is squared MMD. Two-part WCF estimates the mixture without AIPW functional calibration.",
            metrics=("zero_mass_abs_error", "mass_contrast_rmse", "kernel_law_error"),
            dgp_order=SUITE_DGPS["zi"],
            method_order=["cwdb_zipt", "causal_drf", "drf"],
        ),
        _functional_table(
            detail[detail.suite == "income"],
            label="tab:revision-functional-detail",
            caption=r"Functional TATE and TCATE errors for the income designs at $n=1000$. The four functional targets are shown separately to complement the within-seed averages in Tables~\ref{tab:revision-income-1000} and~\ref{tab:revision-abstract}.",
            dgp_order=SUITE_DGPS["income"],
            method_order=["cwdb_dr", "causal_drf", "drf"],
            n_train=1000,
        ),
        _functional_table(
            detail[detail.suite == "abstract"],
            label="tab:revision-functional-abstract",
            caption="Functional TATE and TCATE errors for the abstract designs at $n=1000$, shown separately by functional target.",
            dgp_order=SUITE_DGPS["abstract"],
            method_order=["cwdb_dr", "causal_drf", "drf"],
            n_train=1000,
        ),
    ]
    tables.write_text("\n".join(text), encoding="utf-8")
    for name, block in zip(
        ("income_500", "income_1000", "abstract", "zi"), text[2:6], strict=True
    ):
        (output_dir / f"wcf_revision_{name}.tex").write_text(block, encoding="utf-8")
    (output_dir / "wcf_revision_functionals.tex").write_text(
        "\n".join(text[6:]), encoding="utf-8"
    )
    return {"evidence": evidence, "detail": detail_csv, "tables": tables}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for generated CSV and LaTeX assets",
    )
    args = parser.parse_args()
    outputs = build(args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
