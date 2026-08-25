"""Generate the complete Phase 5.5 result tables for the technical report.

Companion to `build_report_assets.py`, and deliberately a separate file: the
Phase 5.5 variants ran on their own manifest (`G3-PHASE55-v1`), which covers
four of the ten regimes at ten seeds, so their numbers cannot be dropped into
the frozen tournament tables without silently changing what a column means.

Two conventions are enforced here and both differ from the main report:

* every method, incumbent included, is restricted to seeds 0-9, because the
  Phase 5.5 manifest declared ten. Pooling a twenty-seed incumbent mean against
  a ten-seed claimant mean in the same row would compare two different designs.
* the regimes are D0, D2, D7, D8 only. The Phase 5.5 screen never ran D1, D3,
  D4, D5, D6, or D9, and an empty cell is printed as `n/a` rather than left to
  look like a loss.

Nothing here fits a model. It reads audited parquet and writes LaTeX.

Run with `python3 report/build_phase55_assets.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_report_assets as base

ROOT = Path(__file__).resolve().parents[1]
TABLES = Path(__file__).resolve().parent / "tables_generated"

#: The Phase 5.5 manifest's own coordinates.
P55_DGPS = ["D0", "D2", "D7", "D8"]
IMB_DGPS = ["D2-imb", "D7-imb", "D8-imb"]
P55_SEEDS = 10

NEW_LABELS = {
    "cwdb_rmean": "C-WDB rmean (vector R-learner)",
    "cwdb_mutau": r"C-WDB mutau ($\mu/\tau$ shared tree)",
    "cwdb_xmean": "C-WDB xmean (vector X-learner)",
}
NEW_TEX = {
    "cwdb_rmean": "rmean",
    "cwdb_mutau": "mutau",
    "cwdb_xmean": "xmean",
}
base.LABELS.update(NEW_LABELS)
base.TEX_LABELS.update(NEW_TEX)

#: Contrast roster: every method that produces `MEANQ-A-K`.
CONTRAST_ORDER = [
    "cwdb_rmean",
    "cwdb_mutau",
    "cwdb_xmean",
    "cwdb_r3_cvridge",
    "pta_s",
    "causal_drf",
    "drf",
]
#: Law roster: only the methods that produce a conditional law.
LAW_ORDER = ["cwdb_mutau", "cwdb_r3_cvridge", "causal_drf", "drf"]

#: Inherited G3 rule 1 thresholds, unchanged for the screen.
D0_CAP = 0.15
D2_CAP = 0.15
D2_RATIO_CAP = 1.25
FROZEN_BASELINES = {"pta_s", "cwdb_v0", "cwdb_v1", "wdrft", "causal_drf"}


def load() -> pd.DataFrame:
    """Every track the Phase 5.5 comparisons draw on, restricted to ten seeds."""

    frames = []
    for name, track in (
        ("merged", "frozen"),
        ("merged_repair", "repair"),
        ("merged_original_causal_drf", "original_causal_drf"),
        ("merged_original_drf", "original_drf"),
        ("merged_phase55", "phase55"),
    ):
        frame = pd.read_parquet(ROOT / "results" / name / "main_results.parquet")
        frame["track"] = track
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    # the original-code reruns supersede the local drivers of the same name
    data = data[
        ~(
            data.method.isin({"causal_drf", "drf"})
            & data.track.isin({"frozen", "repair"})
        )
    ]
    return data[data.seed < P55_SEEDS].reset_index(drop=True)


# --------------------------------------------------------------------- tables


def regime_table(
    data: pd.DataFrame,
    metric: str,
    methods: list[str],
    caption: str,
    label: str,
    *,
    grid: str = "main",
    dgps: list[str] | None = None,
    target: str | None = None,
    digits: int = 4,
    note: str = "",
) -> str:
    return base.regime_table(
        data, metric, grid, methods, caption, label,
        target=target, digits=digits, dgps=dgps or P55_DGPS, note=note,
    )


def functional_table(
    data: pd.DataFrame,
    metric: str,
    prefix: str,
    methods: list[str],
    caption: str,
    label: str,
) -> str:
    """Four functionals by four regimes, one block per functional."""

    names = ["grid_mean", "grid_sd", "grid_skewness", "grid_upper_tail_mean"]
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
    for name in names:
        for index, dgp in enumerate(P55_DGPS):
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
            first = f"\\texttt{{{name.replace('_', chr(92) + '_')}}}" if index == 0 else ""
            lines.append(" & ".join([first, dgp] + cells) + r" \\")
        if name != names[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def paired_table(
    data: pd.DataFrame,
    claimant: str,
    comparators: list[str],
    caption: str,
    label: str,
    *,
    metric: str = "mean_quantile_rmse",
    grid: str = "main",
    dgps: list[str] | None = None,
) -> str:
    """Seed-paired differences, the only statistic a verdict may rest on."""

    dgps = dgps or P55_DGPS
    header = (
        " & ".join(
            ["Regime"]
            + [f"\\multicolumn{{3}}{{c}}{{vs {base.TEX_LABELS[c]}}}" for c in comparators]
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
    for dgp in dgps:
        cells: list[str] = []
        for comparator in comparators:
            result = base.paired(data, metric, grid, dgp, claimant, comparator)
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
        r"\par\vspace{2pt}\parbox{0.96\textwidth}{\scriptsize Negative favours the "
        r"claimant. $\ast$ marks a claimant win and $\dagger$ a comparator win, both "
        r"at more than two paired standard errors, which is the frozen decision "
        r"multiple. The third column of each block is the same quantity expressed "
        r"in standard errors, so a reader can see how far past the threshold a "
        r"result sits rather than only that it crossed.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def rule1_table(data: pd.DataFrame) -> str:
    """The inherited correctness screen, per method and per sample size.

    One convention, applied identically to every method: the D2 false-effect
    ratio divides a single $n$'s D2 mean by the best frozen baseline at that
    same $n$. Pooling the baseline across sample sizes, or mixing a per-$n$
    numerator with a pooled denominator, is not the convention.
    """

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Inherited rule 1, evaluated per sample size on the ten Phase "
        r"5.5 seeds. The D0 cap is 0.15, the D2 cap is 0.15, and the D2 "
        r"false-effect ratio cap is 1.25 times the best frozen baseline at the "
        r"same $n$.}",
        r"\label{tab:p55-rule1}",
        r"\small",
        r"\begin{tabular}{@{}llrrrrl@{}}",
        r"\toprule",
        r"Method & $n$ & D0 & D2 & baseline & ratio & Verdict \\",
        r"\midrule",
    ]
    methods = CONTRAST_ORDER
    for method in methods:
        for n_train in (500, 1000):
            block = data[
                (data.grid == "main")
                & (data.method == method)
                & (data.n_train == n_train)
                & (data.metric == "mean_quantile_rmse")
                & (data.status == "ok")
            ]
            d0 = block[block.dgp == "D0"].value
            d2 = block[block.dgp == "D2"].value
            if d0.empty or d2.empty:
                continue
            frozen = data[
                (data.grid == "main")
                & (data.dgp == "D2")
                & (data.n_train == n_train)
                & (data.metric == "mean_quantile_rmse")
                & (data.status == "ok")
                & (data.method.isin(FROZEN_BASELINES))
            ]
            baseline = frozen.groupby("method").value.mean().min()
            d0_mean, d2_mean = float(d0.mean()), float(d2.mean())
            ratio = d2_mean / baseline
            passes = (
                d0_mean <= D0_CAP and d2_mean <= D2_CAP and ratio <= D2_RATIO_CAP
            )
            verdict = r"\textsc{pass}" if passes else r"\textbf{FAIL}"
            lines.append(
                f"{base.TEX_LABELS[method]} & {n_train} & {d0_mean:.4f} & "
                f"{d2_mean:.4f} & {baseline:.4f} & {ratio:.2f} & {verdict} \\\\"
            )
        lines.append(r"\addlinespace")
    lines = lines[:-1]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.9\textwidth}{\scriptsize The baseline column "
        r"is the smallest D2 mean among PTA-S, C-WDB-v0, C-WDB-v1, W-DRF-T and "
        r"Causal-DRF at that $n$, on the same ten seeds. Causal-DRF and DRF are "
        r"included as rows for reference; they are incumbents, not claimants, and "
        r"rule 1 was never a gate on them.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def diagnostics_table(data: pd.DataFrame) -> str:
    """The mechanism diagnostics each variant emits, which no other method has."""

    rows: list[tuple[str, str, str]] = [
        ("cwdb_rmean", "diagnostic_selected_contrast_shrinkage", "selected ridge $\\lambda$"),
        ("cwdb_mutau", "diagnostic_selected_contrast_shrinkage", "selected ridge $\\lambda$"),
        ("cwdb_mutau", "diagnostic_n_boosting_steps", "boosting steps used"),
        ("cwdb_rmean", "diagnostic_n_boosting_steps", "boosting steps used"),
        ("cwdb_mutau", "diagnostic_effective_support", "effective particle support"),
        ("cwdb_xmean", "diagnostic_ehat_mean", "$\\bar{\\hat e}$"),
        ("cwdb_xmean", "diagnostic_ehat_sd", "sd of $\\hat e$"),
        ("cwdb_rmean", "diagnostic_train_risk", "training risk"),
        ("cwdb_mutau", "diagnostic_train_risk", "training risk"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mechanism diagnostics recorded by the Phase 5.5 variants, main "
        r"grid, mean over the twenty cells of a regime with the standard error in "
        r"parentheses. These are not comparisons: no incumbent emits them.}",
        r"\label{tab:p55-diagnostics}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll" + "r" * len(P55_DGPS) + r"@{}}",
        r"\toprule",
        " & ".join(["Method", "Diagnostic"] + P55_DGPS) + r" \\",
        r"\midrule",
    ]
    for method, metric, pretty in rows:
        cells = []
        any_finite = False
        for dgp in P55_DGPS:
            mean, error, _ = base.summary(data, metric, "main", dgp, method)
            any_finite |= bool(np.isfinite(mean))
            cells.append(base.fmt(mean, error, digits=3))
        if not any_finite:
            continue
        lines.append(
            " & ".join([base.TEX_LABELS[method], pretty] + cells) + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.94\textwidth}{\scriptsize The effective "
        r"support is the number of the $M=10$ particles carrying non-negligible "
        r"weight, so 10 is the ceiling and the floor for rule 5 is $0.6M=6$. "
        r"The selected $\lambda$ is chosen per cell from $\{0, 50, 500\}$ by "
        r"cross-fitted selection, so its mean is a per-regime average of a "
        r"discrete choice and not itself a tuned value.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def imbalance_table(data: pd.DataFrame) -> str:
    """The extension grid that exists only to test the X-learner's premise."""

    methods = ["cwdb_rmean", "cwdb_xmean"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{The \texttt{imbalance} extension grid, $n=500$, ten seeds. This "
        r"grid was added for WP5.5-D: the vector X-learner's claimed advantage is "
        r"specific to arm imbalance, so a result outside this grid could not "
        r"support it. The last two columns are the seed-paired difference of "
        r"\texttt{xmean} against \texttt{rmean} on \texttt{mean\_quantile\_rmse}.}",
        r"\label{tab:p55-imbalance}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\texttt{mean\_quantile\_rmse}} & "
        r"\multicolumn{2}{c}{\texttt{barycenter\_rmse}} & "
        r"\multicolumn{2}{c}{paired diff.} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Regime & rmean & xmean & rmean & xmean & diff. & SE \\",
        r"\midrule",
    ]
    for dgp in IMB_DGPS:
        cells = []
        for metric in ("mean_quantile_rmse", "barycenter_rmse"):
            for method in methods:
                mean, error, _ = base.summary(data, metric, "imbalance", dgp, method)
                cells.append(base.fmt(mean, error))
        result = base.paired(
            data, "mean_quantile_rmse", "imbalance", dgp, "cwdb_xmean", "cwdb_rmean"
        )
        if result is None:
            cells += ["n/a", ""]
        else:
            mean, error, _ = result
            marker = r"$^{\ast}$" if mean < -2 * error else (
                r"$^{\dagger}$" if mean > 2 * error else ""
            )
            cells += [f"{mean:+.4f}{marker}", f"{error:.4f}"]
        lines.append(" & ".join([dgp] + cells) + r" \\")
    ehat = []
    for dgp in IMB_DGPS:
        mean, _, _ = base.summary(
            data, "diagnostic_ehat_mean", "imbalance", dgp, "cwdb_xmean"
        )
        ehat.append(f"{dgp} {mean:.3f}")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.9\textwidth}{\scriptsize Negative favours "
        r"\texttt{xmean}; $\dagger$ marks an \texttt{rmean} win beyond two paired "
        r"standard errors. The imbalance actually realised, as the mean fitted "
        r"propensity: " + ", ".join(ehat) + r".}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def cost_table(data: pd.DataFrame) -> str:
    methods = [
        "cwdb_rmean",
        "cwdb_mutau",
        "cwdb_xmean",
        "cwdb_r3_cvridge",
        "pta_s",
        "causal_drf",
        "drf",
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Operational cost on the Phase 5.5 coordinates: every "
        r"\texttt{main} grid cell of D0, D2, D7 and D8 at ten seeds. Two cost "
        r"ratios are reported against the inherited cap of 60, because the rule-6 "
        r"verdict depends on which Causal-DRF is the denominator and the project "
        r"contains two.}",
        r"\label{tab:p55-cost}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Method & Cells & Median (s) & Mean (s) & Max (s) & Ratio$_{\text{orig}}$ "
        r"& Ratio$_{\text{local}}$ & Median RAM (MB) \\",
        r"\midrule",
    ]
    scope = data[
        (data.grid == "main")
        & (data.dgp.isin(P55_DGPS))
        & (data.status == "ok")
    ]
    runtime = scope[scope.metric == "runtime"]
    memory = scope[scope.metric == "peak_ram"]
    reference = runtime[runtime.method == "causal_drf"].value.median()
    # The retired local Causal-DRF driver, kept only as the denominator the
    # G3.5 memo quoted, so the two cost statements can be reconciled on paper.
    local = pd.read_parquet(ROOT / "results/merged/main_results.parquet")
    local = local[
        (local.method == "causal_drf")
        & (local.metric == "runtime")
        & (local.status == "ok")
        & (local.grid == "main")
    ]
    local_reference = local.value.median()
    for method in methods:
        block = runtime[runtime.method == method].value
        if block.empty:
            continue
        ram = memory[memory.method == method].value
        lines.append(
            f"{base.LABELS[method]} & {len(block)} & {block.median():.2f} & "
            f"{block.mean():.2f} & {block.max():.1f} & "
            f"{block.median() / reference:.1f} & "
            f"{block.median() / local_reference:.1f} & "
            f"{(ram.median() if len(ram) else float('nan')):.0f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.94\textwidth}{\scriptsize "
        r"Ratio$_{\text{orig}}$ divides by the original-code Causal-DRF rerun on "
        rf"exactly these cells (median {reference:.2f}\,s), which is the "
        r"comparator record the rest of this report uses. "
        r"Ratio$_{\text{local}}$ divides by the retired local Causal-DRF driver "
        r"pooled over its own \texttt{main} grid, all ten regimes (median "
        rf"{local_reference:.2f}\,s), which is "
        r"the denominator the G3.5 screen memo quoted. Every Phase 5.5 cell ran "
        r"on a loaded machine alongside other work, so these are upper bounds on "
        r"the variants' cost rather than clean measurements; the ratios are "
        r"reported anyway because the cap is a declared gate and a favourable "
        r"rerun cannot be assumed.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def budget_probe_table() -> str:
    """The post-decision diagnostic that names the mechanism behind the D0 wall.

    Read from the probe's own JSON rather than recomputed, for the same reason
    the rest of this file reads parquet: a table in a report must be the
    artefact an audit can re-open, not a second implementation of it.
    """

    import json

    path = ROOT / "results" / "merged_phase55" / "xmean_budget_probe.json"
    if not path.exists():
        return ""
    probe = json.loads(path.read_text(encoding="utf-8"))
    budgets = [str(b) for b in probe["budgets"]]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Post-decision diagnostic for \texttt{cwdb\_xmean}: "
        r"\texttt{mean\_quantile\_rmse} as the frozen three-step effect budget is "
        r"relaxed, five seeds. The frozen budget is the first column. No column "
        r"is simultaneously effect-recovering and null-safe, and on the imbalance "
        r"grid the best available column is still the frozen one and still loses "
        r"to \texttt{rmean}.}",
        r"\label{tab:p55-budget}",
        r"\small",
        r"\begin{tabular}{@{}l" + "r" * (len(budgets) + 1) + r"@{}}",
        r"\toprule",
        " & ".join(
            ["Cell"] + [f"$B={b}$" for b in budgets] + [r"\texttt{rmean}"]
        )
        + r" \\",
        r"\midrule",
    ]
    for key, block in probe["main"].items():
        cells = [f"{block[b]:.4f}" for b in budgets] + ["n/a"]
        lines.append(" & ".join([f"\\texttt{{{key}}}"] + cells) + r" \\")
    lines.append(r"\midrule")
    for key, block in probe["imbalance"].items():
        cells = [f"{block[b]:.4f}" for b in budgets] + [f"{block['rmean']:.4f}"]
        lines.append(
            " & ".join([f"\\texttt{{{key}/n500}}"] + cells) + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.9\textwidth}{\scriptsize $B$ is "
        r"\texttt{n\_estimators} in the effect regression; the learning rate "
        r"stays at 0.12, so the frozen budget carries total boosting weight 0.36 "
        r"against the R-loss contrast's 2.4. Generated by "
        r"\texttt{research/checks/phase55\_xmean\_budget\_probe.py}. This probe "
        r"explains a verdict already reached and no threshold depends on it.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = load()
    written: list[str] = []

    written.append(
        regime_table(
            data, "mean_quantile_rmse", CONTRAST_ORDER,
            "Grid causal mean error \\texttt{mean\\_quantile\\_rmse} against "
            "\\texttt{MEANQ-A-K}, main grid, mean over 20 cells "
            "($n\\in\\{500,1000\\}$, ten seeds) with the standard error in "
            "parentheses. This is the primary contrast metric and the one every "
            "Phase 5.5 variant was screened on.",
            "tab:p55-meanq",
        )
    )
    written.append(
        regime_table(
            data, "barycenter_rmse", CONTRAST_ORDER,
            "Arm-level barycenter error \\texttt{barycenter\\_rmse}, main grid. "
            "This is an arm-level quantity, never pooled with the contrast of "
            "Table~\\ref{tab:p55-meanq}.",
            "tab:p55-bary",
        )
    )
    written.append(
        regime_table(
            data, "kernel_law_error", LAW_ORDER,
            "Declared primary law metric \\texttt{kernel\\_law\\_error} (squared "
            "MMD to the true conditional law), main grid. Only \\texttt{mutau} "
            "produces a law among the Phase 5.5 variants; \\texttt{rmean} and "
            "\\texttt{xmean} target the contrast only and are absent by "
            "construction, not by omission.",
            "tab:p55-kernel",
        )
    )
    written.append(
        regime_table(
            data, "arm_energy_risk", LAW_ORDER,
            "Excess energy risk over the true conditional law "
            "\\texttt{arm\\_energy\\_risk}, main grid. This is the loss the law "
            "methods actually optimise.",
            "tab:p55-energy",
        )
    )
    written.append(
        regime_table(
            data, "mode_coverage", LAW_ORDER,
            "\\texttt{mode\\_coverage}, the fraction of true outer modes carrying "
            "predicted mass, main grid. Higher is better. D6, the only multimodal "
            "regime, is not in the Phase 5.5 manifest, so every row here is a "
            "degenerate single-mode check.",
            "tab:p55-mode",
        )
    )
    written.append(
        regime_table(
            data, "tail_calibration", LAW_ORDER,
            "\\texttt{tail\\_calibration}, absolute error of a predeclared "
            "threshold event under the grid law, main grid.",
            "tab:p55-tail",
        )
    )
    written.append(
        regime_table(
            data, "reference_tcate_rmse", CONTRAST_ORDER,
            "Conditional Wasserstein reference effect \\texttt{REF-TCATE-K} with "
            "$\\nu_\\star$ the standard normal, main grid.",
            "tab:p55-reftcate",
        )
    )
    written.append(
        regime_table(
            data, "reference_effect_rmse", CONTRAST_ORDER,
            "Marginal Wasserstein reference effect \\texttt{REF-ATE-K}, main grid.",
            "tab:p55-refate",
        )
    )
    written.append(
        functional_table(
            data, "tcate_functional_rmse", "TCATE-K", LAW_ORDER,
            "Conditional functional treatment effects "
            "\\texttt{tcate\\_functional\\_rmse}, main grid. D7 is the transfer "
            "regime the law claim rests on.",
            "tab:p55-tcate",
        )
    )
    written.append(
        functional_table(
            data, "tate_functional_rmse", "TATE-K", LAW_ORDER,
            "Marginal functional treatment effects "
            "\\texttt{tate\\_functional\\_rmse}, main grid.",
            "tab:p55-tate",
        )
    )
    written.append(rule1_table(data))
    written.append(
        paired_table(
            data, "cwdb_rmean", ["pta_s", "cwdb_r3_cvridge", "causal_drf"],
            "Seed-paired \\texttt{mean\\_quantile\\_rmse} differences for "
            "\\texttt{cwdb\\_rmean}. The D8 column is the primary confounding "
            "test and the D0 column is the one the screen memo initially omitted.",
            "tab:p55-paired-rmean",
        )
    )
    written.append(
        paired_table(
            data, "cwdb_mutau", ["cwdb_r3_cvridge", "pta_s", "causal_drf"],
            "Seed-paired \\texttt{mean\\_quantile\\_rmse} differences for "
            "\\texttt{cwdb\\_mutau}. The R3 block is the informative one: "
            "\\texttt{mutau} is a reparameterisation of the same shared tree, so a "
            "difference against R3 is a difference the $\\mu/\\tau$ leaf rule "
            "caused.",
            "tab:p55-paired-mutau",
        )
    )
    written.append(
        paired_table(
            data, "cwdb_xmean", ["cwdb_rmean", "pta_s", "cwdb_r3_cvridge"],
            "Seed-paired \\texttt{mean\\_quantile\\_rmse} differences for "
            "\\texttt{cwdb\\_xmean} on the main grid. The imbalance grid, which is "
            "where its premise lives, is Table~\\ref{tab:p55-imbalance}.",
            "tab:p55-paired-xmean",
        )
    )
    written.append(
        paired_table(
            data, "cwdb_mutau", ["cwdb_r3_cvridge", "causal_drf"],
            "Seed-paired \\texttt{kernel\\_law\\_error} differences for "
            "\\texttt{cwdb\\_mutau}, the law metric.",
            "tab:p55-paired-mutau-law",
            metric="kernel_law_error",
        )
    )
    written.append(imbalance_table(data))
    probe = budget_probe_table()
    if probe:
        written.append(probe)
    written.append(diagnostics_table(data))
    written.append(cost_table(data))

    (TABLES / "phase55_tables.tex").write_text("\n".join(written), encoding="utf-8")
    print(f"wrote {TABLES / 'phase55_tables.tex'} ({len(written)} tables)")


if __name__ == "__main__":
    main()
