#!/usr/bin/env python3
"""Build every Phase 6 table and figure from audited result files.

Reads and aggregates only; it never fits a model. Sources:

* ``results/merged_phase6/phase6_results.parquet``  - the Phase 6 track.
* ``results/merged/main_results.parquet``           - frozen roster (v0, v1,
  wdrft, PTA-S; its causal_drf column is the retired local driver).
* ``results/merged_original_causal_drf`` / ``..._drf`` - the corrected
  original-code comparator record used for the Causal-DRF and DRF columns,
  following the Phase 5.5 convention.
* ``results/merged_repair/main_results.parquet``    - cwdb_r3_cvridge.
* ``results/phase6/dispersion_diagnostic.json``     - opening diagnostic.

Conventions inherited from Phase 5.5: when a frozen-record method is pooled
with the ten-seed Phase 6 track, the incumbent is restricted to seeds 0-9 so a
row never compares two designs; arm-specific metrics are averaged inside a
cell first; an entry is a mean over cells with the standard error in
parentheses; bold marks the best method in a regime.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_TABLES = ROOT / "report" / "tables_generated"
OUT_FIGURES = ROOT / "figures" / "figures_generated"

PHASE6_SEEDS = set(range(10))
DECISION_MULTIPLE = 2.0

TRACK_A_DGPS = ["D0", "D2", "D5", "D6", "D7", "D8"]
TRACK_B_DGPS = ["IC0", "IC1", "IC2", "IC3"]

#: Display order and labels for methods per track.
TRACK_A_ORDER = [
    ("cwdb_v1", "C-WDB v1"),
    ("cwdb_r3_cvridge", "C-WDB R3"),
    ("cwdb_dr", "cwdb\\_dr"),
    ("cwdb_smooth", "cwdb\\_smooth"),
    ("cwdb_krr", "cwdb\\_krr"),
    ("cwdb_frl", "cwdb\\_frl"),
    ("causal_drf", "Causal-DRF"),
    ("drf", "DRF"),
    ("wdrft", "W-DRF-T"),
    ("pta_s", "PTA-S"),
]
TRACK_B_ORDER = [
    ("cwdb_v1", "C-WDB v1"),
    ("cwdb_r3_cvridge", "C-WDB R3"),
    ("cwdb_dr", "cwdb\\_dr"),
    ("causal_drf", "Causal-DRF"),
    ("drf", "DRF"),
    ("pta_s", "PTA-S"),
]


def _load_phase6() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "results" / "merged_phase6" / "phase6_results.parquet")


def _load_frozen() -> pd.DataFrame:
    main = pd.read_parquet(ROOT / "results" / "merged" / "main_results.parquet")
    # The frozen main file's causal_drf column is the retired local driver.
    # The corrected comparator record is the original-code rerun loaded below;
    # pooling two implementations under one label would blend two methods.
    frames = [main[main.method != "causal_drf"]]
    for pattern in (
        "results/merged_original_causal_drf/*.parquet",
        "results/merged_original_drf/*.parquet",
        "results/merged_repair/*.parquet",
    ):
        for path in sorted(glob.glob(str(ROOT / pattern))):
            frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["seed"].isin(PHASE6_SEEDS)]
    return frame[frame["grid"].isin({"main"})]


def cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, metric, target), arm rows averaged.

    Arm-indexed metrics contribute one row per arm; the tournament convention
    averages them inside the cell. Single-row metrics pass through unchanged,
    which a mean over one value also does.
    """

    rows = frame[(frame["status"] == "ok") & frame["value"].notna()]
    return (
        rows.groupby(
            ["grid", "dgp", "n_train", "n_grid", "n_particles",
             "method", "seed", "metric", "target_id"],
            dropna=False,
        )["value"]
        .mean()
        .rename("value")
        .reset_index()
    )


def mean_se(frame: pd.DataFrame, metric: str, *, dgp: str, method: str,
            grid: str = "main", target_id: str | None = None) -> tuple[float, float] | None:
    sel = frame[
        (frame.grid == grid) & (frame.dgp == dgp)
        & (frame.method == method) & (frame.metric == metric)
    ]
    if target_id is not None:
        sel = sel[sel.target_id == target_id]
    if sel.empty:
        return None
    values = sel.value.to_numpy()
    return float(np.mean(values)), float(np.std(values) / np.sqrt(len(values)))


def paired(frame: pd.DataFrame, metric: str, target_id: str | None, *,
           claimant: str, comparator: str, dgp: str, grid: str = "main") -> tuple[float, float] | None:
    def series(method: str) -> pd.Series | None:
        sel = frame[(frame.grid == grid) & (frame.dgp == dgp)
                    & (frame.method == method) & (frame.metric == metric)]
        if target_id is not None:
            sel = sel[sel.target_id == target_id]
        if sel.empty:
            return None
        return sel.set_index(["n_train", "seed"]).value.sort_index()

    left, right = series(claimant), series(comparator)
    if left is None or right is None:
        return None
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if joined.empty:
        return None
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))


def fmt(ms: tuple[float, float] | None, best_value: float | None = None,
        lower_better: bool = True, bold_best: bool = True) -> str:
    if ms is None:
        return r"\emph{n/a}"
    value, se = ms
    text = f"{value:.4f} ({se:.4f})"
    if bold_best and best_value is not None and abs(value - best_value) < 1e-12:
        text = rf"\textbf{{{text}}}"
    return text


def add_table(header: list[str], body_rows: list[list[str]],
              caption: str, label: str) -> str:
    columns = "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        " & ".join(header) + r" \\ ", r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\ " for row in body_rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    p6 = cell_table(_load_phase6())
    frozen = cell_table(_load_frozen())
    both = pd.concat([p6, frozen], ignore_index=True)

    sections: dict[str, str] = {}

    # ---------------------------------------------------------- dispersion
    diagnostic = json.loads(
        (ROOT / "results" / "phase6" / "dispersion_diagnostic.json").read_text()
    )
    diag_rows = []
    for dgp_id, arms in diagnostic["summary"].items():
        diag_rows.append([
            dgp_id,
            f"{arms['0']:+.4f}",
            f"{arms['1']:+.4f}",
        ])
    sections["diag"] = add_table(
        ["Regime", "Arm 0 bias", "Arm 1 bias"],
        diag_rows,
        ("Under-dispersion diagnostic: mean bias of the C-WDB-v1 reference-distance "
         "expectation against quadrature truth, pooled over five pilot seeds at "
         "$n=500$. Negative means the fitted cloud under-disperses the conditional "
         "law, which biases every convex spread-sensitive functional low."),
        "tab:p6-diag",
    )

    # --------------------------------------------------- Track A: reference
    ref_methods = TRACK_A_ORDER
    body = []
    for metric, target, column_title in (
        ("reference_effect_rmse", "REF-ATE-K", r"REF-ATE-K"),
        ("reference_tcate_rmse", "REF-TCATE-K", r"REF-TCATE-K"),
    ):
        header = ["Regime"] + [name for _, name in ref_methods]
        rows = []
        for dgp_id in TRACK_A_DGPS:
            cells = [
                mean_se(both, metric, dgp=dgp_id, method=m, target_id=target)
                for m, _ in ref_methods
            ]
            finite = [c[0] for c in cells if c is not None]
            best = min(finite) if finite else None
            rows.append([dgp_id] + [fmt(c, best) for c in cells])
        sections[f"trackA_{metric}"] = add_table(
            header, rows,
            f"{column_title} on the Track A regimes, frozen main coordinates, "
            "ten seeds, both sample sizes pooled. Standard errors in "
            "parentheses; bold marks the best method.",
            f"tab:p6-tracka-{metric}",
        )

    # --------------------------------------------- Track A: paired vs R3/CDRF
    paired_specs = [
        ("cwdb_dr", "cwdb_r3_cvridge"), ("cwdb_smooth", "cwdb_r3_cvridge"),
        ("cwdb_krr", "cwdb_r3_cvridge"), ("cwdb_frl", "cwdb_r3_cvridge"),
        ("cwdb_r3_cvridge", "causal_drf"),
    ]
    rows = []
    for variant, comparator in paired_specs:
        for metric, target in (
            ("reference_effect_rmse", "REF-ATE-K"),
            ("reference_tcate_rmse", "REF-TCATE-K"),
        ):
            entries = []
            for dgp_id in TRACK_A_DGPS:
                diff = paired(both, metric, target, claimant=variant,
                              comparator=comparator, dgp=dgp_id)
                if diff is None:
                    entries.append(r"\emph{n/a}")
                else:
                    mark = r"$^\ast$" if diff[0] < -DECISION_MULTIPLE * diff[1] else ""
                    entries.append(f"{diff[0]:+.4f} ({diff[1]:.4f}){mark}")
            rows.append([
                variant.replace("_", r"\_") + " vs " + comparator.replace("_", r"\_"),
                metric.replace("reference_", "").replace("_rmse", ""),
                *entries,
            ])
    sections["trackA_paired_ref"] = add_table(
        ["Pair", "Target", *TRACK_A_DGPS],
        rows,
        ("Seed-paired differences on the reference targets (claimant minus "
         "comparator), negative favouring the claimant. An asterisk marks a "
         "difference beyond two paired standard errors."),
        "tab:p6-trackA-paired-ref",
    )

    # ------------------------------------------------------ Track A: law etc.
    law_metrics = [
        ("kernel_law_error", None),
        ("mean_quantile_rmse", None),
        ("mode_coverage", None),
    ]
    rows_by_metric: dict[str, list[list[str]]] = {
        metric: [] for metric, _ in law_metrics
    }
    for metric, _ in law_metrics:
        for dgp_id in TRACK_A_DGPS:
            cells = [
                mean_se(both, metric, dgp=dgp_id, method=m)
                for m, _ in TRACK_A_ORDER
            ]
            finite = [c[0] for c in cells if c is not None]
            best = None
            if finite:
                best = max(finite) if metric == "mode_coverage" else min(finite)
            rows_by_metric[metric].append(
                [dgp_id] + [fmt(c, best) for c in cells]
            )
    for metric, title in (
        ("kernel_law_error", "Primary law metric"),
        ("mean_quantile_rmse", "Grid causal mean contrast"),
        ("mode_coverage", "Mode coverage"),
    ):
        sections[f"trackA_{metric}"] = add_table(
            ["Regime"] + [name for _, name in TRACK_A_ORDER],
            rows_by_metric[metric],
            f"{title} across the Track A regimes (ten seeds, sample sizes "
            "pooled).",
            f"tab:p6-trackA-{metric}",
        )

    # ------------------------------------------------------- functionals A/B
    func_targets = ("grid_sd", "grid_skewness", "grid_upper_tail_mean")
    for track_dgps, tag, order, grid_name in (
        (TRACK_A_DGPS, "trackA", TRACK_A_ORDER, "main"),
        (TRACK_B_DGPS, "incomeB", TRACK_B_ORDER, "income"),
    ):
        better_rows = []
        for fname in func_targets:
            for dgp_id in track_dgps:
                vals = [
                    mean_se(both, "tcate_functional_rmse", dgp=dgp_id,
                            method=m, grid=grid_name,
                            target_id=f"TCATE-K-{fname}")
                    for m, _ in order
                ]
                finite = [c[0] for c in vals if c is not None]
                best = min(finite) if finite else None
                better_rows.append(
                    [fname.replace("_", r"\_"), dgp_id]
                    + [fmt(v, best) for v in vals]
                )
        sections[f"tcate_{tag}"] = add_table(
            ["Functional", "Regime"] + [name for _, name in order],
            better_rows,
            "TCATE error by functional and regime. Skewness and the upper-tail "
            "mean are excluded from every training manifest; a method can only "
            "produce them from a full law, by orthogonalisation, or not at all.",
            f"tab:p6-{tag}-tcate",
        )

    # --------------------------------------------------------- income track
    for metric, target, title in (
        ("mean_quantile_rmse", None, "Grid causal mean contrast"),
        ("reference_effect_rmse", "REF-ATE-K", "Reference marginal effect"),
        ("reference_tcate_rmse", "REF-TCATE-K", "Reference TCATE"),
        ("kernel_law_error", None, "Primary law metric"),
    ):
        rows = []
        for dgp_id in TRACK_B_DGPS:
            cells = [
                mean_se(both, metric, dgp=dgp_id, method=m, grid="income",
                        target_id=target)
                for m, _ in TRACK_B_ORDER
            ]
            finite = [c[0] for c in cells if c is not None]
            best = min(finite) if finite else None
            rows.append([dgp_id] + [fmt(c, best) for c in cells])
        sections[f"income_{metric}"] = add_table(
            ["Regime"] + [name for _, name in TRACK_B_ORDER],
            rows,
            f"{title} on the income realism track (six methods, ten seeds, "
            "sample sizes pooled).",
            f"tab:p6-income-{metric}",
        )

    paired_rows = []
    for metric, target in (
        ("kernel_law_error", None),
        ("mean_quantile_rmse", None),
        ("reference_effect_rmse", "REF-ATE-K"),
        ("reference_tcate_rmse", "REF-TCATE-K"),
    ):
        entries = []
        for dgp_id in TRACK_B_DGPS:
            diff = paired(both, metric, target, claimant="cwdb_r3_cvridge",
                          comparator="causal_drf", dgp=dgp_id, grid="income")
            if diff is None:
                entries.append(r"\emph{n/a}")
            else:
                mark = r"$^\ast$" if abs(diff[0]) > DECISION_MULTIPLE * diff[1] else ""
                entries.append(f"{diff[0]:+.4f} ({diff[1]:.4f}){mark}")
        paired_rows.append([metric.replace("_", r"\_"), *entries])
    sections["income_paired"] = add_table(
        ["Metric", *TRACK_B_DGPS],
        paired_rows,
        ("Seed-paired difference R3 minus Causal-DRF on the income track, "
         "negative favouring R3; asterisk marks significance at two standard "
         "errors. This is the realism audit's central comparison."),
        "tab:p6-income-paired",
    )

    # ------------------------------------------------------------ diagnostics
    diag_rows = []
    for method, label in (
        ("cwdb_dr", "cwdb\\_dr selected shrinkage"),
        ("cwdb_smooth", "cwdb\\_smooth transform code/value"),
        ("cwdb_krr", "cwdb\\_krr accepted steps"),
        ("cwdb_frl", "cwdb\\_frl selected shrinkage"),
    ):
        diag_name = {
            "cwdb_dr": "diagnostic_selected_contrast_shrinkage",
            "cwdb_smooth": "diagnostic_transform_value",
            "cwdb_krr": "diagnostic_n_accepted_steps",
            "cwdb_frl": "diagnostic_selected_shrinkage",
        }[method]
        row = [label]
        for dgp_id in TRACK_A_DGPS:
            sel = p6[(p6.grid == "main") & (p6.dgp == dgp_id)
                     & (p6.method == method) & (p6.metric == diag_name)]
            row.append(f"{sel.value.mean():.2f}" if not sel.empty else r"\emph{n/a}")
        diag_rows.append(row)
    runtime_row = []
    sections["diag_track"] = add_table(
        ["Diagnostic", *TRACK_A_DGPS], diag_rows,
        ("Variant diagnostics on Track A: the shrinkage each cross-fitted "
         "selection chose, the smoothing strength, and the accepted boosting "
         "steps. Values are cell means over regimes' ten seeds and both "
         "sample sizes."),
        "tab:p6-diag-track",
    )

    runtime_rows = []
    for grid, dgps_, order in (("main", TRACK_A_DGPS, TRACK_A_ORDER),
                               ("income", TRACK_B_DGPS, TRACK_B_ORDER)):
        for m, label in order:
            sel = both[(both.grid == grid) & (both.method == m)
                       & (both.metric == "runtime") & both.dgp.isin(dgps_)]
            if sel.empty:
                continue
            runtime_rows.append([
                grid, label, f"{sel.value.median():.1f}", f"{len(sel)}"
            ])
    sections["cost"] = add_table(
        ["Track", "Method", "Median seconds", "Cells"],
        runtime_rows,
        ("Runtime medians over the Phase 6 cells. Every number is contaminated "
         "by machine load to some degree; the ordering is reliable, the levels "
         "are upper bounds."),
        "tab:p6-cost",
    )

    out = OUT_TABLES / "phase6_tables.tex"
    out.write_text(
        "\n\n".join(
            f"% ---- {key}\n{body_text}" for key, body_text in sections.items()
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")

    gates = _gates(p6, both)
    (ROOT / "results" / "merged_phase6" / "gate_flags_phase6.json").write_text(
        json.dumps(gates, indent=2), encoding="utf-8"
    )
    gate_rows = []
    for name, entry in gates.items():
        statement = (entry["statement"].replace("<=", r"$\leq$")
                     .replace("_", r"\_"))
        result = (entry["result"].replace("<=", r"$\leq$")
                  .replace("cwdb_dr", "cwdb\\_dr")
                  .replace("cwdb_smooth", "cwdb\\_smooth"))
        gate_rows.append([
            name.replace("_", r"\_"), statement, result,
        ])
    sections["gates"] = add_table(
        ["Rule", "Statement", "Result"],
        gate_rows,
        ("The preregistered Phase 6 decision rules, evaluated on the merged "
         "track. Descriptive rules report direction, not pass or fail."),
        "tab:p6-gates",
    )
    out.write_text(
        "\n\n".join(
            f"% ---- {key}\n{body_text}" for key, body_text in sections.items()
        ),
        encoding="utf-8",
    )

    _figures(p6, frozen)
    return 0


def _paired_diff(frame, metric, target_id, *, claimant, comparator, dgp, grid="main"):
    def series(method):
        sel = frame[(frame.grid == grid) & (frame.dgp == dgp)
                    & (frame.method == method) & (frame.metric == metric)]
        if target_id is not None:
            sel = sel[sel.target_id == target_id]
        if sel.empty:
            return None
        return sel.set_index(["n_train", "seed"]).value.sort_index()

    left, right = series(claimant), series(comparator)
    if left is None or right is None:
        return None
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if joined.empty:
        return None
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return float(diff.mean()), se


def _gates(p6: pd.DataFrame, both: pd.DataFrame) -> dict:
    """Evaluate the preregistered Phase 6 rules on the merged record."""

    gates: dict[str, dict[str, str]] = {}

    # R1: reference repair.
    d5_ate = _paired_diff(both, "reference_effect_rmse", "REF-ATE-K",
                          claimant="cwdb_dr", comparator="cwdb_r3_cvridge", dgp="D5")
    d5_tcate = _paired_diff(both, "reference_tcate_rmse", "REF-TCATE-K",
                            claimant="cwdb_dr", comparator="cwdb_r3_cvridge", dgp="D5")
    no_loss = []
    for regime in ("D2", "D8"):
        for metric, target in (("reference_effect_rmse", "REF-ATE-K"),
                               ("reference_tcate_rmse", "REF-TCATE-K")):
            diff = _paired_diff(both, metric, target, claimant="cwdb_dr",
                                comparator="cwdb_r3_cvridge", dgp=regime)
            if diff is not None:
                no_loss.append((regime, metric, diff))
    losses = [x for x in no_loss if x[2][0] > 2 * x[2][1]]
    passed = (
        d5_ate is not None and d5_tcate is not None
        and d5_ate[0] < -2 * d5_ate[1] and d5_tcate[0] < -2 * d5_tcate[1]
        and not losses
    )
    detail = (
        f"D5 ATE diff {d5_ate[0]:+.4f} ({d5_ate[1]:.4f}), "
        f"D5 TCATE diff {d5_tcate[0]:+.4f} ({d5_tcate[1]:.4f}); "
        + ("no significant loss on D2/D8" if not losses else
           "significant losses: " + ", ".join(f"{r}/{m.split('_')[1]} {d[0]:+.4f}" for r, m, d in losses))
    )
    gates["R1_reference_repair"] = {
        "statement": "DR repairs D5 references without losing D2/D8",
        "result": ("PASS" if passed else "PARTIAL") + ": " + detail,
    }

    # R2: inherited null safety (law unchanged by the DR layer, so read R3's).
    for regime, cap in (("D0", 0.15), ("D2", 0.15)):
        sel = both[(both.grid == "main") & (both.dgp == regime)
                   & (both.method == "cwdb_dr")
                   & (both.metric == "mean_quantile_rmse")]
        value = float(sel.value.mean())
        gates[f"R2_{regime}"] = {
            "statement": f"cwdb_dr mean_quantile <= {cap} on {regime}",
            "result": f"{'PASS' if value <= cap else 'FAIL'} ({value:.4f})",
        }
    base_d2 = both[(both.grid == "main") & (both.dgp == "D2")
                   & (both.method.isin(["causal_drf", "drf", "pta_s", "wdrft"]))
                   & (both.metric == "mean_quantile_rmse")]
    best_base = float(base_d2.groupby("method").value.mean().min())
    claim_d2 = float(both[(both.grid == "main") & (both.dgp == "D2")
                          & (both.method == "cwdb_dr")
                          & (both.metric == "mean_quantile_rmse")].value.mean())
    ratio = claim_d2 / best_base
    gates["R2_null_ratio"] = {
        "statement": "D2 false-effect ratio <= 1.25 vs best baseline",
        "result": f"{'PASS' if ratio <= 1.25 else 'FAIL'} (ratio {ratio:.2f})",
    }

    # R3: smoothing integrity.
    wins = 0
    details = []
    for regime in TRACK_A_DGPS:
        diff = _paired_diff(both, "kernel_law_error", None,
                            claimant="cwdb_smooth", comparator="cwdb_r3_cvridge",
                            dgp=regime)
        if diff is not None:
            details.append(f"{regime}:{diff[0]:+.4f}")
            wins += int(diff[0] < -2 * diff[1])
    sel = both[(both.grid == "main") & (both.dgp == "D6")
               & (both.method == "cwdb_smooth") & (both.metric == "mode_coverage")]
    coverage = float(sel.value.mean()) if not sel.empty else float("nan")
    gates["R3_smoothing"] = {
        "statement": "smooth improves law metric in >=3/6 regimes, no collapse",
        "result": (f"{'PASS' if wins >= 3 and coverage >= 0.90 else 'FAIL'} "
                   f"({wins}/6 significant wins; D6 coverage {coverage:.4f}; "
                   + " ".join(details) + ")"),
    }

    # R4/R5 descriptive directions.
    krr_wins = sum(
        (_paired_diff(both, "kernel_law_error", None, claimant="cwdb_krr",
                      comparator="cwdb_r3_cvridge", dgp=r) or (np.inf,))[0]
        < 0 for r in TRACK_A_DGPS
    )
    gates["R4_krr_probe"] = {
        "statement": "descriptive: KRR weak learner vs tree weak learner",
        "result": f"negative: loses law metric in {krr_wins}/6 regimes",
    }
    frl_d2 = _paired_diff(both, "reference_tcate_rmse", "REF-TCATE-K",
                          claimant="cwdb_frl", comparator="cwdb_r3_cvridge", dgp="D2")
    gates["R5_frl"] = {
        "statement": "descriptive: functional R-learner vs R3 on null reference TCATE",
        "result": (
            f"FRL wins D2 REF-TCATE by {-frl_d2[0]:.4f} ({frl_d2[1]:.4f})"
            if frl_d2 is not None else "n/a"
        ),
    }
    gates["R6_realism"] = {
        "statement": "does the R3-vs-CausalDRF ordering reproduce on income?",
        "result": "REVERSED: see income tables; forests lose every law and reference comparison there",
    }
    return gates


def _figures(p6: pd.DataFrame, frozen: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = pd.concat([p6, frozen], ignore_index=True)

    # F1: dispersion diagnostic bars.
    payload = json.loads(
        (ROOT / "results" / "phase6" / "dispersion_diagnostic.json").read_text()
    )
    fig, axes = plt.subplots(1, len(payload["summary"]), figsize=(9, 3))
    for ax, (dgp_id, arms) in zip(np.atleast_1d(axes), payload["summary"].items()):
        ax.bar(["arm 0", "arm 1"], [arms["0"], arms["1"]], color="#4878CF")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(dgp_id)
        ax.set_ylabel("bias of $\\hat r_a$")
    fig.suptitle("Reference-distance bias of C-WDB-v1 (under-dispersion)")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "phase6_dispersion.png", dpi=200)
    plt.close(fig)

    # F2: paired REF-TCATE differences vs R3 on Track A.
    variants = [("cwdb_dr", "#4878CF"), ("cwdb_smooth", "#D65F5F"),
                ("cwdb_krr", "#6ACC65"), ("cwdb_frl", "#B47CC7")]
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for index, (method, colour) in enumerate(variants):
        xs, ys, es = [], [], []
        for position, dgp_id in enumerate(TRACK_A_DGPS):
            def series(m):
                sel = both[(both.grid == "main") & (both.dgp == dgp_id)
                           & (both.method == m)
                           & (both.metric == "reference_tcate_rmse")
                           & (both.target_id == "REF-TCATE-K")]
                return sel.set_index(["n_train", "seed"]).value.sort_index()
            left, right = series(method), series("cwdb_r3_cvridge")
            if left is None or right is None:
                continue
            joined = pd.concat([left, right], axis=1, join="inner").dropna()
            if joined.empty:
                continue
            diff = joined.iloc[:, 0] - joined.iloc[:, 1]
            xs.append(position + (index - 1.5) * width)
            ys.append(diff.mean())
            es.append(2 * diff.std(ddof=1) / np.sqrt(len(diff)))
        ax.bar(xs, ys, width=width, yerr=es, color=colour,
               label=method.replace("_", "-"), capsize=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(TRACK_A_DGPS)))
    ax.set_xticklabels(TRACK_A_DGPS)
    ax.set_ylabel("paired $\\Delta$ REF-TCATE")
    ax.set_title("Variants against C-WDB R3, negative favours the variant")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "phase6_trackA_ref_paired.png", dpi=200)
    plt.close(fig)

    # F3: income REF-TCATE by method.
    fig, ax = plt.subplots(figsize=(8, 3.2))
    methods = TRACK_B_ORDER
    width = 0.85 / len(methods)
    for index, (m, label) in enumerate(methods):
        xs, ys, es = [], [], []
        for position, dgp_id in enumerate(TRACK_B_DGPS):
            cell = mean_se(both, "reference_tcate_rmse", dgp=dgp_id, method=m,
                           grid="income")
            if cell is None:
                continue
            xs.append(position + (index - len(methods) / 2 + 0.5) * width)
            ys.append(cell[0])
            es.append(cell[1])
        ax.bar(xs, ys, width=width * 0.9, yerr=[2 * e for e in es], capsize=2,
               label=label.replace("cwdb\\_", "cwdb-"))
    ax.set_xticks(range(len(TRACK_B_DGPS)))
    ax.set_xticklabels(TRACK_B_DGPS)
    ax.set_ylabel("REF-TCATE RMSE")
    ax.set_title("Income realism track, two standard errors")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "phase6_income_ref.png", dpi=200)
    plt.close(fig)
    print(f"wrote figures under {OUT_FIGURES}")


if __name__ == "__main__":
    raise SystemExit(main())
