"""Generate the Phase 5.5 Stage 2 tables for the technical report.

Stage 2 runs the only surviving Phase 5.5 claimant, `cwdb_mutau`, on the whole
frozen main grid at the incumbents' own twenty seeds. This file reads the
audited result files and aggregates them; it never fits a model, exactly as
`build_report_assets.py` and `build_phase55_assets.py` do.

Three conventions differ from `build_phase55_assets.py`, and each follows the
Stage 2 preregistration rather than a choice made here:

* twenty seeds, not ten, because every regime now carries the incumbents' own
  replication count and no comparator has to be cut down to meet the claimant;
* all ten regimes, so the D3 sharing check, the D5 law-separation check, the D6
  multimodality check and D9 are present for the first time for this method;
* runtime comes from the Stage 2 rows alone. `mutau.py` was made roughly five
  times faster after Stage 1 ran, with no accuracy metric moving, so a pooled
  cost figure would describe no implementation that ever existed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_report_assets as base

ROOT = Path(__file__).resolve().parents[1]
TABLES = Path(__file__).resolve().parent / "tables_generated"
PAYLOAD = ROOT / "results" / "merged_phase55_stage2" / "stage2_analysis_payload.json"

ALL_DGPS = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
CLAIMANT = "cwdb_mutau"

base.LABELS.update({"cwdb_mutau": r"C-WDB mutau ($\mu/\tau$ shared tree)"})
base.TEX_LABELS.update({"cwdb_mutau": "mutau"})

#: Every method producing the common contrast target `MEANQ-A-K`.
CONTRAST_ORDER = ["cwdb_mutau", "cwdb_r3_cvridge", "cwdb_v1", "pta_s",
                  "causal_drf", "drf"]
#: Every method producing a conditional law.
LAW_ORDER = ["cwdb_mutau", "cwdb_r3_cvridge", "cwdb_v1", "wdrft",
             "causal_drf", "drf"]

FUNCTIONALS = ["grid_mean", "grid_sd", "grid_skewness", "grid_upper_tail_mean"]


def load() -> pd.DataFrame:
    """Every track a Stage 2 comparison draws on, at twenty seeds.

    The claimant's rows are the union of its Stage 1 rows with the Stage 2
    table. Their cell keys are disjoint by construction (Stage 2 enumerates only
    what Stage 1 did not run), so the union is a plain concatenation, and the
    two stages were verified not to differ on any accuracy metric before it was
    taken.
    """

    frames = []
    for name, track in (
        ("merged", "frozen"),
        ("merged_repair", "repair"),
        ("merged_original_causal_drf", "original_causal_drf"),
        ("merged_original_drf", "original_drf"),
        ("merged_phase55", "phase55"),
        ("merged_phase55_stage2", "phase55_stage2"),
    ):
        frame = pd.read_parquet(ROOT / "results" / name / "main_results.parquet")
        frame["track"] = track
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    # The original-code reruns supersede the retired local drivers of the same
    # name. This must be done by source track, not by contract identifier: both
    # carry `G3-MAIN-v1`, so filtering on the identifier would delete the
    # comparator rather than the incumbent.
    data = data[
        ~(
            data.method.isin({"causal_drf", "drf"})
            & data.track.isin({"frozen", "repair"})
        )
    ]
    # Only the claimant continues from Phase 5.5; rmean and xmean rows would
    # otherwise appear in a table whose caption says they are absent.
    data = data[~(data.method.isin({"cwdb_rmean", "cwdb_xmean"}))]
    return data.reset_index(drop=True)


def payload() -> dict:
    return json.loads(PAYLOAD.read_text(encoding="utf-8"))


# --------------------------------------------------------------------- tables


def regime_table(data, metric, methods, caption, label, *, target=None,
                 digits=4, note="") -> str:
    return base.regime_table(
        data, metric, "main", methods, caption, label,
        target=target, digits=digits, dgps=ALL_DGPS, note=note,
    )


def functional_table(data, metric, prefix, methods, caption, label) -> str:
    """Four functionals by ten regimes, one block per functional."""

    header = (
        " & ".join(["Functional", "Regime"] + [base.TEX_LABELS[m] for m in methods])
        + r" \\"
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}ll" + "r" * len(methods) + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for name in FUNCTIONALS:
        for index, dgp in enumerate(ALL_DGPS):
            cells, raw = [], []
            for method in methods:
                mean, error, _ = base.summary(
                    data, metric, "main", dgp, method, f"{prefix}-{name}"
                )
                raw.append(mean)
                cells.append(base.fmt(mean, error))
            finite = [v for v in raw if np.isfinite(v)]
            if finite:
                best = min(finite)
                for position, value in enumerate(raw):
                    if np.isfinite(value) and value == best:
                        cells[position] = r"\textbf{" + cells[position] + "}"
            label_cell = name.replace("_", r"\_") if index == 0 else ""
            lines.append(" & ".join([label_cell, dgp] + cells) + r" \\")
        if name != FUNCTIONALS[-1]:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def paired_table(data, comparators, caption, label, *,
                 metric="mean_quantile_rmse") -> str:
    header = (
        " & ".join(
            ["Regime"]
            + [f"\\multicolumn{{3}}{{c}}{{vs {base.TEX_LABELS[c]}}}"
               for c in comparators]
        )
        + r" \\"
    )
    sub = " & ".join([" "] + ["diff.", "SE", "$|$diff$|/$SE"] * len(comparators)) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "rrr" * len(comparators) + r"@{}}",
        r"\toprule",
        header,
        sub,
        r"\midrule",
    ]
    for dgp in ALL_DGPS:
        cells: list[str] = []
        for comparator in comparators:
            result = base.paired(data, metric, "main", dgp, CLAIMANT, comparator)
            if result is None:
                cells += ["n/a", "", ""]
                continue
            mean, error, _ = result
            ratio = abs(mean) / error if error > 0 else float("inf")
            marker = ""
            if mean < -2 * error:
                marker = r"$^{\ast}$"
            elif mean > 2 * error:
                marker = r"$^{\dagger}$"
            cells += [f"{mean:+.4f}{marker}", f"{error:.4f}", f"{ratio:.1f}"]
        lines.append(" & ".join([dgp] + cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.96\textwidth}{\scriptsize Negative favours "
        r"\texttt{mutau}. $\ast$ marks a claimant win and $\dagger$ a comparator "
        r"win, both at more than two paired standard errors, the frozen decision "
        r"multiple. The third column of each block expresses the same difference "
        r"in standard errors, so a reader can see how far past the threshold a "
        r"result sits rather than only that it crossed.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def gate_table() -> str:
    """The preregistered rules and clauses, with the verdict each returned."""

    doc = payload()
    rule1 = doc["rule_1_correctness"]
    rule2 = doc["rule_2_law_advantage"]
    rule5 = doc["rule_5_no_collapse"]
    rule6 = doc["rule_6_cost"]
    clause = doc["accuracy_clause"]
    degraded = doc["degradation_clause"]["violated"]
    clause_regimes = sorted({win["dgp"] for win in clause["wins"]})
    transfer_wins = sum(
        1
        for regime in doc["functional_surface"].values()
        for metric in regime.values()
        if (entry := metric.get("causal_drf")) and entry["verdict"] == "claimant"
    ) + sum(
        1
        for regime in doc["contrast_surface"].values()
        for name, metric in regime.items()
        if name.startswith("reference_")
        and (entry := metric.get("causal_drf"))
        and entry["verdict"] == "claimant"
    )

    def mark(passed: bool) -> str:
        return r"\textbf{PASS}" if passed else r"\textbf{FAIL}"

    rows = [
        (
            "Rule 1, correctness and nulls",
            rf"D0 {rule1['500']['d0_mean']:.4f} and {rule1['1000']['d0_mean']:.4f} "
            rf"under 0.15; D2 ratio {rule1['500']['d2_ratio']:.2f} and "
            rf"{rule1['1000']['d2_ratio']:.2f} under 1.25",
            mark(rule1["500"]["pass"] and rule1["1000"]["pass"]),
        ),
        (
            "Rule 2, law advantage over Causal-DRF",
            rf"{len(rule2['wins_vs_causal_drf'])} regimes of "
            rf"{rule2['required']} required: "
            + ", ".join(rule2["wins_vs_causal_drf"]),
            mark(len(rule2["wins_vs_causal_drf"]) >= rule2["required"]),
        ),
        (
            "Rules 3 and 4, transfer and direct learner",
            rf"{transfer_wins} functional or reference targets won against "
            rf"Causal-DRF; {len(clause['wins'])} accuracy wins against PTA-S "
            rf"across {len(clause_regimes)} regimes",
            mark(bool(clause["wins"])),
        ),
        (
            "Rule 5, no particle collapse",
            rf"effective support {rule5['effective_support_median']:.1f} of 10 "
            rf"against a floor of 6; D6 mode coverage "
            rf"{rule5['d6_mode_coverage']:.4f}",
            mark(rule5["pass"]),
        ),
        (
            "Rule 6, cost",
            rf"median {rule6['claimant_median_runtime_s']:.1f}\,s against "
            rf"{rule6['reference_median_runtime_s']:.2f}\,s, ratio "
            rf"{rule6['ratio']:.1f} against a cap of {rule6['cap']:.0f}",
            mark(rule6["pass"]),
        ),
        (
            "Accuracy clause, at least one win over PTA-S",
            rf"{len(clause['wins'])} wins on targets both methods estimate, in "
            + ", ".join(clause_regimes),
            mark(clause["pass"]),
        ),
        (
            "Degradation clause, D3 sharing and D7 shape",
            (
                ", ".join(degraded) + " crosses the multiple against R3"
                if degraded else "no regime crosses the multiple"
            ),
            mark(not degraded),
        ),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Phase 5.5 Stage 2 against the rules frozen in "
        r"\texttt{research/simulation\_preregistration\_phase55\_stage2.md} "
        r"before the first Stage 2 cell was executed. Claimant \texttt{cwdb\_mutau}, "
        r"ten regimes, twenty seeds.}",
        r"\label{tab:p55s2-gate}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}p{0.30\textwidth}p{0.50\textwidth}l@{}}",
        r"\toprule",
        r"Rule & Result & Verdict \\",
        r"\midrule",
    ]
    for name, result, status in rows:
        lines.append(f"{name} & {result} & {status}" + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.96\textwidth}{\scriptsize Rules 1 through 6 "
        r"are the inherited G3 gate rules. The last two rows are the clauses the "
        r"phase document froze specifically for a Stage 2 full-law candidate. "
        r"The degradation clause was given its numerical definition (a paired "
        r"\texttt{kernel\_law\_error} loss to R3 crossing two standard errors on "
        r"D3 or D7) in the preregistration, before the result existed.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def degradation_table() -> str:
    doc = payload()["degradation_clause"]["comparisons"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{The degradation clause, per sample size. Paired "
        r"\texttt{kernel\_law\_error} difference, \texttt{mutau} minus C-WDB R3, "
        r"on the two regimes the phase document names.}",
        r"\label{tab:p55s2-degradation}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llrrrl@{}}",
        r"\toprule",
        r"Regime & $n$ & diff. & SE & $|$diff$|/$SE & Verdict \\",
        r"\midrule",
    ]
    for dgp in ("D3", "D7"):
        for n in ("500", "1000"):
            entry = doc[dgp][n]
            if entry is None:
                lines.append(f"{dgp} & {n} & n/a & & & absent" + r" \\")
                continue
            word = {
                "claimant": r"\texttt{mutau} better",
                "comparator": r"\textbf{R3 better}",
                "tie": "tie",
            }[entry["verdict"]]
            lines.append(
                f"{dgp} & {n} & {entry['paired_mean_difference']:+.5f} & "
                f"{entry['paired_standard_error']:.5f} & "
                f"{abs(entry['se_ratio']):.1f} & {word}" + r" \\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.9\textwidth}{\scriptsize D3 is the "
        r"separate-head-favourable regime, which is what the $\mu/\tau$ "
        r"reparameterisation puts at risk because it carries the prognostic and "
        r"contrast fields in one shared tree. D7 is the pure-shape-transfer "
        r"regime. Positive favours R3.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def diagnostics_table(data: pd.DataFrame) -> str:
    metrics = [
        ("diagnostic_effective_support", "Effective support (of $M = 10$)", 2),
        ("diagnostic_selected_contrast_shrinkage", "Selected contrast strength", 1),
        ("diagnostic_n_boosting_steps", "Boosting steps", 0),
        ("diagnostic_train_risk", "Training risk", 4),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mechanism diagnostics for \texttt{cwdb\_mutau} across all ten "
        r"regimes, pooled over sample sizes and twenty seeds.}",
        r"\label{tab:p55s2-diagnostics}",
        r"\scriptsize",
        r"\begin{tabular}{@{}l" + "r" * len(metrics) + r"@{}}",
        r"\toprule",
        " & ".join(["Regime"] + [name for _, name, _ in metrics]) + r" \\",
        r"\midrule",
    ]
    for dgp in ALL_DGPS:
        cells = []
        for metric, _, digits in metrics:
            mean, _, _ = base.summary(data, metric, "main", dgp, CLAIMANT)
            cells.append("n/a" if not np.isfinite(mean) else f"{mean:.{digits}f}")
        lines.append(" & ".join([dgp] + cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.9\textwidth}{\scriptsize The rule 5 floor is "
        r"$0.6M = 6$ particles. The selected contrast strength is chosen per cell "
        r"by held-out risk over the frozen candidate set $\{0, 50, 500\}$, so a "
        r"regime mean between the candidates means the selection varies across "
        r"cells rather than that an intermediate value was ever used.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def cost_table(data: pd.DataFrame) -> str:
    doc = payload()["rule_6_cost"]
    stage2 = pd.read_parquet(
        ROOT / "results" / "merged_phase55_stage2" / "main_results.parquet"
    )
    runtime = data[(data.metric == "runtime") & (data.status == "ok")
                   & (data.grid == "main")]
    rows = []
    for method in ["causal_drf", "drf", "cwdb_v1", "pta_s", "cwdb_r3_cvridge"]:
        block = runtime[runtime.method == method].value
        if block.empty:
            continue
        rows.append((base.TEX_LABELS[method], block.median(), block.max()))
    claimant_block = stage2[(stage2.metric == "runtime") & (stage2.status == "ok")].value
    rows.append((base.TEX_LABELS[CLAIMANT], claimant_block.median(),
                 claimant_block.max()))
    rows.sort(key=lambda row: row[1])
    reference = doc["reference_median_runtime_s"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Runtime on the main grid. The \texttt{mutau} row is measured "
        r"on the Stage 2 rows alone; every other row is that method's own frozen "
        r"record.}",
        r"\label{tab:p55s2-cost}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Method & Median (s) & Max (s) & Ratio to Causal-DRF \\",
        r"\midrule",
    ]
    for name, median, maximum in rows:
        lines.append(
            f"{name} & {median:.2f} & {maximum:.1f} & {median / reference:.1f}"
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.92\textwidth}{\scriptsize The rule 6 cap is "
        rf"60. The denominator is the original-code Causal-DRF median of "
        rf"{reference:.2f}\,s, declared in the preregistration before the run "
        r"because the project contains two Causal-DRF records and the verdict "
        r"depends on which is used. Every figure here is an upper bound: the "
        r"Stage 2 cells ran sixteen at a time on one laptop, and the same "
        r"implementation measured on an idle machine completes a cell in about "
        r"a fifth of the time. Runtime in this project is dominated by machine "
        r"load and is not a clean measure of algorithmic cost.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = load()
    TABLES.mkdir(parents=True, exist_ok=True)
    blocks = [
        gate_table(),
        regime_table(
            data, "mean_quantile_rmse", CONTRAST_ORDER,
            "Stage 2 grid causal mean: \\texttt{mean\\_quantile\\_rmse} against "
            "\\texttt{MEANQ-A-K}, every regime, twenty seeds. Mean over cells "
            "with the standard error in parentheses; bold marks the best method "
            "in a regime.",
            "tab:p55s2-meanq",
        ),
        regime_table(
            data, "barycenter_rmse", CONTRAST_ORDER,
            "Stage 2 arm-level barycenter error. A different quantity from the "
            "contrast and never pooled with it.",
            "tab:p55s2-bary",
        ),
        regime_table(
            data, "reference_effect_rmse", CONTRAST_ORDER,
            "Stage 2 Wasserstein reference effect, \\texttt{REF-ATE-K}.",
            "tab:p55s2-refate",
        ),
        regime_table(
            data, "reference_tcate_rmse", CONTRAST_ORDER,
            "Stage 2 conditional Wasserstein reference effect, "
            "\\texttt{REF-TCATE-K}.",
            "tab:p55s2-reftcate",
        ),
        regime_table(
            data, "kernel_law_error", LAW_ORDER,
            "Stage 2 primary law metric, \\texttt{kernel\\_law\\_error}. Only "
            "methods that produce a conditional law appear.",
            "tab:p55s2-kernel",
        ),
        regime_table(
            data, "arm_energy_risk", LAW_ORDER,
            "Stage 2 excess energy risk by arm, averaged inside the cell.",
            "tab:p55s2-energy",
        ),
        regime_table(
            data, "mode_coverage", LAW_ORDER,
            "Stage 2 mode coverage. D6 is the multimodal regime and is the only "
            "row that carries information; a uniform 1.0000 elsewhere records "
            "that the single-mode regimes stayed single-mode.",
            "tab:p55s2-mode",
        ),
        regime_table(
            data, "tail_calibration", LAW_ORDER,
            "Stage 2 upper-tail calibration.",
            "tab:p55s2-tail",
        ),
        functional_table(
            data, "tcate_functional_rmse", "TCATE-K",
            ["cwdb_mutau", "cwdb_r3_cvridge", "pta_s", "causal_drf", "drf"],
            "Stage 2 conditional functional targets, \\texttt{TCATE-K}. A blank "
            "cell is a target the method cannot supply by contract.",
            "tab:p55s2-tcate",
        ),
        functional_table(
            data, "tate_functional_rmse", "TATE-K",
            ["cwdb_mutau", "cwdb_r3_cvridge", "pta_s", "causal_drf", "drf"],
            "Stage 2 marginal functional targets, \\texttt{TATE-K}.",
            "tab:p55s2-tate",
        ),
        paired_table(
            data, ["cwdb_r3_cvridge", "cwdb_v1", "causal_drf"],
            "Stage 2 seed-paired differences on the primary law metric, "
            "\\texttt{kernel\\_law\\_error}.",
            "tab:p55s2-paired-law",
            metric="kernel_law_error",
        ),
        paired_table(
            data, ["cwdb_r3_cvridge", "pta_s", "causal_drf"],
            "Stage 2 seed-paired differences on the grid causal mean, "
            "\\texttt{mean\\_quantile\\_rmse}.",
            "tab:p55s2-paired-meanq",
        ),
        degradation_table(),
        diagnostics_table(data),
        cost_table(data),
    ]
    destination = TABLES / "phase55_stage2_tables.tex"
    destination.write_text("\n".join(blocks), encoding="utf-8")
    print(f"wrote {destination} ({len(blocks)} tables)")


if __name__ == "__main__":
    main()
