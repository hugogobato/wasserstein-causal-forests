"""Generate the complete result tables and figures for the technical report.

Reads the frozen and repair merged result sets plus the original-code
Causal-DRF and separate-DRF reruns, and emits one LaTeX table per metric family plus the report figures. The
aggregation convention matches `research/checks/g3_gate_flags.py`: arm-specific
rows are averaged within a cell, cells are keyed by (grid, dgp, n, K, M, method,
seed), and a regime mean pools every cell of that regime in the named grid.

Nothing here recomputes a model. It reads frozen parquet and writes derived
artefacts, so it can be rerun at any time and must reproduce byte-identical
numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = Path(__file__).resolve().parent / "tables_generated"
FIGURES = Path(__file__).resolve().parent / "figures_generated"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

DGPS = [f"D{i}" for i in range(10)]

#: Fixed categorical hue order, validated for CVD separation and lightness.
#: Colour follows the method, never its rank, and is never cycled: a figure that
#: would need a seventh series is split into facets instead.
COLOURS = {
    "cwdb_r3_cvridge": "#0072B2",
    "causal_drf": "#D55E00",
    "drf": "#009E73",
    "cwdb_v1": "#009E73",
    "wdrft": "#E69F00",
    "pta_s": "#CC79A7",
    "cwdb_v0": "#56B4E9",
    "cwdb_r2_threshold3": "#009E73",
    "cwdb_r1_ridge": "#E69F00",
    "cwdb_v1_pooledinit": "#CC79A7",
    "sqw2_booster": "#D55E00",
    "cwdb_v1_noshrink": "#E69F00",
    "pta_f": "#56B4E9",
}
MARKERS = {
    "cwdb_r3_cvridge": "o",
    "causal_drf": "s",
    "drf": "^",
    "cwdb_v1": "^",
    "wdrft": "D",
    "pta_s": "v",
    "cwdb_v0": "P",
    "cwdb_r2_threshold3": "^",
    "cwdb_r1_ridge": "D",
    "cwdb_v1_pooledinit": "v",
    "sqw2_booster": "s",
    "cwdb_v1_noshrink": "D",
    "pta_f": "P",
}
LABELS = {
    "cwdb_r3_cvridge": "C-WDB R3 (cv-ridge)",
    "cwdb_r2_threshold3": "C-WDB R2$'$ ($c{=}3$)",
    "cwdb_r2_threshold": "C-WDB R2 ($c{=}1$)",
    "cwdb_r1_ridge": "C-WDB R1 (ridge 50)",
    "cwdb_v1_pooledinit": "C-WDB v1 pooled-init",
    "cwdb_v1": "C-WDB v1",
    "cwdb_v0": "C-WDB v0",
    "cwdb_v1_noshrink": "C-WDB v1 no-shrink",
    "sqw2_booster": "squared-$W_2$ booster",
    "causal_drf": "Causal-DRF",
    "drf": "DRF (separate, paper benchmark)",
    "wdrft": "W-DRF-T",
    "pta_s": "PTA-S",
    "pta_f": "PTA-F",
}
#: Short column headers for the wide complete-results tables. The mapping is
#: printed once in the report, so no information is lost by using them.
TEX_LABELS = {
    "cwdb_r3_cvridge": "R3",
    "cwdb_r2_threshold3": "R2$'$",
    "cwdb_r2_threshold": "R2",
    "cwdb_r1_ridge": "R1",
    "cwdb_v1_pooledinit": "v1-pi",
    "cwdb_v1": "v1",
    "cwdb_v0": "v0",
    "cwdb_v1_noshrink": "v1-ns",
    "sqw2_booster": "sq-$W_2$",
    "causal_drf": "C-DRF",
    "drf": "DRF",
    "wdrft": "W-DRF-T",
    "pta_s": "PTA-S",
    "pta_f": "PTA-F",
}

MAIN_ORDER = [
    "cwdb_r3_cvridge",
    "causal_drf",
    "drf",
]
LAW_ORDER = [
    "cwdb_r3_cvridge",
    "causal_drf",
    "drf",
]

HIGHER_IS_BETTER = {"mode_coverage"}


def load() -> pd.DataFrame:
    frozen = pd.read_parquet(ROOT / "results/merged/main_results.parquet")
    repair = pd.read_parquet(ROOT / "results/merged_repair/main_results.parquet")
    frozen["track"] = "frozen"
    repair["track"] = "repair"
    data = pd.concat([frozen, repair], ignore_index=True)
    original_path = ROOT / "results/merged_original_causal_drf/main_results.parquet"
    if original_path.exists():
        original = pd.read_parquet(original_path)
        data = data[data.method != "causal_drf"]
        original["track"] = "original_causal_drf"
        data = pd.concat([data, original], ignore_index=True)
    paper_drf_path = ROOT / "results/merged_original_drf/main_results.parquet"
    if paper_drf_path.exists():
        paper_drf = pd.read_parquet(paper_drf_path)
        data = data[data.method != "drf"]
        paper_drf["track"] = "original_drf"
        data = pd.concat([data, paper_drf], ignore_index=True)
    return data


def cell_series(
    data: pd.DataFrame,
    metric: str,
    grid: str,
    dgp: str,
    method: str,
    target: str | None = None,
    n_particles: int | None = None,
) -> pd.Series:
    """Seed-indexed cell values, arm rows averaged, for one regime and method."""

    mask = (
        (data.grid == grid)
        & (data.dgp == dgp)
        & (data.method == method)
        & (data.metric == metric)
        & (data.status == "ok")
        & data.value.notna()
    )
    if target is not None:
        mask &= data.target_id == target
    if n_particles is not None:
        mask &= data.n_particles == n_particles
    block = data[mask]
    if block.empty:
        return pd.Series(dtype=float)
    return block.groupby(["n_train", "seed"]).value.mean()


def summary(
    data: pd.DataFrame,
    metric: str,
    grid: str,
    dgp: str,
    method: str,
    target: str | None = None,
    n_particles: int | None = None,
) -> tuple[float, float, int]:
    values = cell_series(data, metric, grid, dgp, method, target, n_particles)
    if values.empty:
        return float("nan"), float("nan"), 0
    array = values.to_numpy()
    error = (
        float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else float("nan")
    )
    return float(array.mean()), error, int(array.size)


def paired(
    data: pd.DataFrame,
    metric: str,
    grid: str,
    dgp: str,
    claimant: str,
    comparator: str,
    target: str | None = None,
) -> tuple[float, float, float] | None:
    """Seed-paired difference, sign-corrected so negative favours the claimant."""

    left = cell_series(data, metric, grid, dgp, claimant, target)
    right = cell_series(data, metric, grid, dgp, comparator, target)
    shared = left.index.intersection(right.index)
    if len(shared) < 3:
        return None
    difference = (left.loc[shared] - right.loc[shared]).to_numpy()
    if metric in HIGHER_IS_BETTER:
        difference = -difference
    mean = float(difference.mean())
    error = float(difference.std(ddof=1) / np.sqrt(difference.size))
    won = float(np.mean(difference < 0.0))
    return mean, error, won


# --------------------------------------------------------------------- tables


def fmt(value: float, error: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "n/a"
    if not np.isfinite(error):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f}\\,({error:.{digits}f})"


def regime_table(
    data: pd.DataFrame,
    metric: str,
    grid: str,
    methods: list[str],
    caption: str,
    label: str,
    target: str | None = None,
    digits: int = 4,
    dgps: list[str] | None = None,
    note: str = "",
) -> str:
    dgps = dgps or DGPS
    header = " & ".join(["Regime"] + [TEX_LABELS[m] for m in methods]) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}l" + "r" * len(methods) + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for dgp in dgps:
        cells = []
        raw = []
        for method in methods:
            mean, error, _ = summary(data, metric, grid, dgp, method, target)
            raw.append(mean)
            cells.append(fmt(mean, error, digits))
        finite = [v for v in raw if np.isfinite(v)]
        if finite:
            best = max(finite) if metric in HIGHER_IS_BETTER else min(finite)
            for index, value in enumerate(raw):
                if np.isfinite(value) and value == best:
                    cells[index] = r"\textbf{" + cells[index] + "}"
        lines.append(" & ".join([dgp] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(r"\par\vspace{2pt}\parbox{\textwidth}{\scriptsize " + note + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def functional_longtable(
    data: pd.DataFrame,
    metric: str,
    prefix: str,
    methods: list[str],
    caption: str,
    label: str,
) -> str:
    targets = [
        f"{prefix}-grid_mean",
        f"{prefix}-grid_sd",
        f"{prefix}-grid_skewness",
        f"{prefix}-grid_upper_tail_mean",
    ]
    pretty = ["grid\\_mean", "grid\\_sd", "grid\\_skewness", "grid\\_upper\\_tail\\_mean"]
    header = (
        " & ".join(["Functional", "Regime"] + [TEX_LABELS[m] for m in methods]) + r" \\"
    )
    lines = [
        r"\begingroup",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{longtable}{@{}ll" + "r" * len(methods) + r"@{}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for target, name in zip(targets, pretty):
        for index, dgp in enumerate(DGPS):
            cells = []
            raw = []
            for method in methods:
                mean, error, _ = summary(data, metric, "main", dgp, method, target)
                raw.append(mean)
                cells.append(fmt(mean, error))
            finite = [v for v in raw if np.isfinite(v)]
            if finite:
                best = min(finite)
                for position, value in enumerate(raw):
                    if np.isfinite(value) and value == best:
                        cells[position] = r"\textbf{" + cells[position] + "}"
            first = f"\\texttt{{{name}}}" if index == 0 else ""
            lines.append(" & ".join([first, dgp] + cells) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def paired_table(
    data: pd.DataFrame,
    claimant: str,
    comparators: list[str],
    entries: list[tuple[str, str, str | None, str]],
    caption: str,
    label: str,
) -> str:
    """One row per (metric, target, regime); one column block per comparator."""

    header = (
        " & ".join(
            ["Target", "Regime"]
            + [f"\\multicolumn{{2}}{{c}}{{vs {LABELS[c]}}}" for c in comparators]
        )
        + r" \\"
    )
    sub = " & ".join([" ", " "] + ["diff.", "SE"] * len(comparators)) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll" + "rr" * len(comparators) + r"@{}}",
        r"\toprule",
        header,
        sub,
        r"\midrule",
    ]
    previous = None
    for metric, dgp, target, pretty in entries:
        cells: list[str] = []
        for comparator in comparators:
            result = paired(data, metric, "main", dgp, claimant, comparator, target)
            if result is None:
                cells += ["n/a", ""]
                continue
            mean, error, _ = result
            marker = ""
            if mean < -2 * error:
                marker = r"$^{\ast}$"
            elif mean > 2 * error:
                marker = r"$^{\dagger}$"
            cells += [f"{mean:+.4f}{marker}", f"{error:.4f}"]
        first = pretty if pretty != previous else ""
        previous = pretty
        lines.append(" & ".join([first, dgp] + cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.96\textwidth}{\scriptsize Negative favours the "
        r"claimant. $\ast$ marks a claimant win and $\dagger$ a comparator win, both "
        r"at more than two paired standard errors, which is the preregistered "
        r"definition. \emph{n/a} means the comparator does not estimate that target.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def cost_table(data: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Operational cost over every cell each method ran, both tracks "
        r"pooled. Runtime is fit plus predict; peak memory is a process high-water "
        r"mark, so a zero means the worker never exceeded its own baseline.}",
        r"\label{tab:cost-all}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Method & Cells & Median (s) & Mean (s) & Max (s) & Median RAM (MB) \\",
        r"\midrule",
    ]
    runtime = data[(data.metric == "runtime") & (data.status == "ok")]
    memory = data[(data.metric == "peak_ram") & (data.status == "ok")]
    allowed = {"cwdb_r3_cvridge", "causal_drf", "drf", "sqw2_booster"}
    order = sorted(
        [m for m in runtime.method.unique() if m in allowed],
        key=lambda m: runtime[runtime.method == m].value.median(),
    )
    for method in order:
        block = runtime[runtime.method == method].value
        ram = memory[memory.method == method].value
        lines.append(
            f"{LABELS.get(method, method)} & {len(block)} & {block.median():.2f} & "
            f"{block.mean():.2f} & {block.max():.1f} & "
            f"{(ram.median() if len(ram) else float('nan')):.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def sensitivity_table(data: pd.DataFrame) -> str:
    """Particles, resolution and scaling grids, all of which are v1-only."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{The three grids that were \emph{not} re-run for the repaired "
        r"variants. Every number here belongs to C-WDB-v1 and its comparators, and "
        r"none of it supports a claim about \texttt{cwdb\_r3\_cvridge}.}",
        r"\label{tab:sensitivity}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Grid & Regime & Setting & Method & Excess energy risk & Law metric \\",
        r"\midrule",
    ]
    for dgp in ["D1", "D6"]:
        for particles in (2, 5, 10, 25):
            for method in ("cwdb_v1", "sqw2_booster"):
                energy = summary(
                    data, "arm_energy_risk", "particles", dgp, method,
                    n_particles=particles,
                )
                law = summary(
                    data, "kernel_law_error", "particles", dgp, method,
                    n_particles=particles,
                )
                if not np.isfinite(energy[0]):
                    continue
                lines.append(
                    f"\\texttt{{particles}} & {dgp} & $M={particles}$ & "
                    f"{LABELS[method]} & {fmt(*energy[:2])} & {fmt(*law[:2])} \\\\"
                )
    lines.append(r"\midrule")
    for dgp in ["D1", "D5", "D6", "D7"]:
        for method in ("cwdb_v1", "cwdb_v0", "causal_drf", "wdrft"):
            law = summary(data, "kernel_law_error", "resolution", dgp, method)
            energy = summary(data, "arm_energy_risk", "resolution", dgp, method)
            if not np.isfinite(law[0]):
                continue
            lines.append(
                f"\\texttt{{resolution}} & {dgp} & $K=49$ & {LABELS[method]} & "
                f"{fmt(*energy[:2])} & {fmt(*law[:2])} \\\\"
            )
    lines.append(r"\midrule")
    for dgp in ["D1", "D4", "D6"]:
        for method in ("cwdb_v1", "cwdb_v0", "causal_drf", "wdrft", "pta_s"):
            law = summary(data, "kernel_law_error", "scaling", dgp, method)
            mean_q = summary(data, "mean_quantile_rmse", "scaling", dgp, method)
            if not np.isfinite(mean_q[0]):
                continue
            lines.append(
                f"\\texttt{{scaling}} & {dgp} & $n=2000$ & {LABELS[method]} & "
                f"{fmt(*mean_q[:2])} & {fmt(*law[:2])} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\parbox{0.94\textwidth}{\scriptsize In the "
        r"\texttt{scaling} block the fifth column is \texttt{mean\_quantile\_rmse} "
        r"rather than excess energy risk, because PTA-S produces no law.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- figures


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555555",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "legend.frameon": False,
        }
    )


def regime_figure(
    data: pd.DataFrame,
    metric: str,
    methods: list[str],
    title: str,
    ylabel: str,
    filename: str,
    grid: str = "main",
    logy: bool = True,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 3.4))
    positions = np.arange(len(DGPS))
    for method in methods:
        means, errors = [], []
        for dgp in DGPS:
            mean, error, _ = summary(data, metric, grid, dgp, method)
            means.append(mean)
            errors.append(error)
        axis.errorbar(
            positions,
            means,
            yerr=2 * np.array(errors),
            label=LABELS[method],
            color=COLOURS[method],
            marker=MARKERS[method],
            markersize=4.5,
            linewidth=1.6,
            capsize=2,
            elinewidth=0.9,
        )
    if logy:
        axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(DGPS)
    axis.set_xlabel("regime")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10)
    axis.legend(ncol=3, fontsize=7.5, loc="best")
    figure.tight_layout()
    figure.savefig(FIGURES / filename)
    plt.close(figure)


def paired_figure(
    data: pd.DataFrame,
    metric: str,
    claimant: str,
    comparator: str,
    title: str,
    xlabel: str,
    filename: str,
) -> None:
    rows = []
    for dgp in DGPS:
        result = paired(data, metric, "main", dgp, claimant, comparator)
        if result is not None:
            rows.append((dgp, *result))
    rows.sort(key=lambda row: row[1])
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    errors = [2 * row[2] for row in rows]
    # Polarity, so a diverging pair with a neutral zero line: one hue for the
    # claimant side, the other for the comparator side, no rank-based recolouring.
    colours = ["#0072B2" if value < 0 else "#D55E00" for value in values]
    axis.barh(labels, values, xerr=errors, color=colours, height=0.62, capsize=2.5,
              error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
    axis.axvline(0.0, color="#666666", linewidth=0.9)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(FIGURES / filename)
    plt.close(figure)


def functional_figure(data: pd.DataFrame, methods: list[str], filename: str) -> None:
    targets = [
        ("TCATE-K-grid_mean", "grid_mean"),
        ("TCATE-K-grid_sd", "grid_sd"),
        ("TCATE-K-grid_skewness", "grid_skewness"),
        ("TCATE-K-grid_upper_tail_mean", "grid_upper_tail_mean"),
    ]
    regimes = ["D5", "D6", "D7"]
    figure, axes = plt.subplots(1, 4, figsize=(7.6, 2.9), sharey=False)
    width = 0.8 / len(methods)
    for axis, (target, name) in zip(axes, targets):
        positions = np.arange(len(regimes))
        for index, method in enumerate(methods):
            means, errors, missing = [], [], []
            for dgp in regimes:
                mean, error, _ = summary(
                    data, "tcate_functional_rmse", "main", dgp, method, target
                )
                missing.append(not np.isfinite(mean))
                means.append(0.0 if not np.isfinite(mean) else mean)
                errors.append(0.0 if not np.isfinite(error) else error)
            offsets = positions + index * width - 0.4 + width / 2
            axis.bar(
                offsets,
                means,
                width=width * 0.86,
                yerr=2 * np.array(errors),
                color=COLOURS[method],
                label=LABELS[method] if axis is axes[0] else None,
                capsize=1.5,
                error_kw={"elinewidth": 0.7},
            )
            # A method that cannot estimate the functional is labelled, so an
            # absent bar is never read as an error of zero.
            for offset, absent in zip(offsets, missing):
                if absent:
                    axis.text(offset, 0.0, "n/a", rotation=90, fontsize=5.5,
                              ha="center", va="bottom", color="#777777")
        axis.set_xticks(positions)
        axis.set_xticklabels(regimes)
        axis.set_title(name.replace("_", "\\_") if False else name, fontsize=8.5)
        axis.tick_params(labelsize=7.5)
    axes[0].set_ylabel("TCATE RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=len(methods), fontsize=7,
                  loc="lower center", bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(
        "Moderator-conditional functional error; a missing bar means the method "
        "cannot estimate that functional", fontsize=9)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure.savefig(FIGURES / filename)
    plt.close(figure)


def cost_figure(data: pd.DataFrame) -> None:
    runtime = data[(data.metric == "runtime") & (data.status == "ok")]
    allowed = {"cwdb_r3_cvridge", "causal_drf", "drf", "sqw2_booster"}
    order = sorted(
        [m for m in runtime.method.unique() if m in allowed],
        key=lambda m: runtime[runtime.method == m].value.median(),
    )
    medians = [runtime[runtime.method == m].value.median() for m in order]
    figure, axis = plt.subplots(figsize=(7.0, 2.8))
    axis.bar(
        [LABELS.get(m, m) for m in order],
        medians,
        color="#0072B2",
        width=0.62,
    )
    for index, value in enumerate(medians):
        axis.text(index, value * 1.06, f"{value:.1f}", ha="center", fontsize=7.5)
    axis.set_yscale("log")
    axis.set_ylabel("median seconds per cell")
    axis.set_title("Cost per fitted cell, both tracks pooled", fontsize=10)
    axis.tick_params(axis="x", rotation=30, labelsize=7.5)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    figure.tight_layout()
    figure.savefig(FIGURES / "cost_all_methods.png")
    plt.close(figure)


def null_regime_figure(data: pd.DataFrame) -> None:
    """The rule-1 picture: D2 against D0 and D3 for every contrast rule."""

    variants = [
        "cwdb_v1",
        "cwdb_v1_pooledinit",
        "cwdb_r1_ridge",
        "cwdb_r2_threshold3",
        "cwdb_r3_cvridge",
    ]
    regimes = ["D0", "D2", "D3"]
    figure, axis = plt.subplots(figsize=(7.0, 3.0))
    positions = np.arange(len(regimes))
    width = 0.8 / len(variants)
    palette = ["#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#0072B2"]
    for index, (method, colour) in enumerate(zip(variants, palette)):
        means, errors = [], []
        for dgp in regimes:
            mean, error, _ = summary(data, "mean_quantile_rmse", "main", dgp, method)
            means.append(mean)
            errors.append(0.0 if not np.isfinite(error) else error)
        axis.bar(
            positions + index * width - 0.4 + width / 2,
            means,
            width=width * 0.86,
            yerr=2 * np.array(errors),
            color=colour,
            label=LABELS[method],
            capsize=2,
            error_kw={"elinewidth": 0.8},
        )
    best_baseline = min(
        summary(data, "mean_quantile_rmse", "main", "D2", m)[0]
        for m in ("causal_drf", "drf", "cwdb_v0")
    )
    axis.axhline(1.25 * best_baseline, color="#666666", linestyle="--", linewidth=1.0)
    axis.text(
        1.42, 1.25 * best_baseline * 1.03,
        f"rule-1 cap on D2: {1.25 * best_baseline:.4f}",
        fontsize=7.5, color="#444444",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        ["D0 (deterministic)", "D2 (null effect)", "D3 (separate heads)"]
    )
    axis.set_ylabel("mean\\_quantile\\_rmse".replace("\\_", "_"))
    axis.set_title(
        "What each contrast rule costs and buys\n"
        "null regime, exactly-recoverable regime, and the regime that punishes "
        "shrinkage",
        fontsize=9,
    )
    axis.legend(fontsize=7.5, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURES / "contrast_rules_d0_d2_d3.png")
    plt.close(figure)


def main() -> None:
    data = load()
    style()

    written: list[str] = []

    written.append(
        regime_table(
            data, "mean_quantile_rmse", "main", MAIN_ORDER,
            "Grid causal mean error \\texttt{mean\\_quantile\\_rmse} against "
            "\\texttt{MEANQ-A-K}, main grid, mean over 40 cells "
            "($n\\in\\{500,1000\\}$, 20 seeds) with the standard error in "
            "parentheses. Best in each regime is bold.",
            "tab:all-meanq",
            note="\\texttt{cwdb\\_r2\\_threshold} at $c=1$ is omitted: the "
            "preregistered stage-1 screen stopped it after D2, where it scored "
            "0.0790.",
        )
    )
    written.append(
        regime_table(
            data, "kernel_law_error", "main", LAW_ORDER,
            "Declared primary law metric \\texttt{kernel\\_law\\_error} (squared "
            "MMD to the true conditional law, Gaussian kernel on the rescaled "
            "coordinates, median bandwidth), main grid, for the two forest "
            "comparators and the proposed law estimator.",
            "tab:all-kernel",
        )
    )
    written.append(
        regime_table(
            data, "arm_energy_risk", "main", LAW_ORDER,
            "Excess energy risk over the true conditional law "
            "\\texttt{arm\\_energy\\_risk}, main grid. Zero is attainable only at "
            "the truth, and the finite-$M$ floor of Table~\\ref{tab:particles} is "
            "included in these values.",
            "tab:all-energy",
        )
    )
    written.append(
        regime_table(
            data, "barycenter_rmse", "main", MAIN_ORDER,
            "Arm-level barycenter error \\texttt{barycenter\\_rmse} against "
            "\\texttt{BARY-A}/\\texttt{MEANQ-A-K}, main grid. Reported separately "
            "from Table~\\ref{tab:all-meanq} and never pooled with it, since one is "
            "an arm-level quantity and the other the arm contrast.",
            "tab:all-bary",
        )
    )
    written.append(
        regime_table(
            data, "mode_coverage", "main", LAW_ORDER,
            "\\texttt{mode\\_coverage}, the fraction of true outer modes carrying "
            "predicted mass, main grid. Higher is better and D6 is the only "
            "multimodal regime, so every other row is a degenerate single-mode "
            "check that a method should pass trivially.",
            "tab:all-mode",
        )
    )
    written.append(
        regime_table(
            data, "tail_calibration", "main", LAW_ORDER,
            "\\texttt{tail\\_calibration}, absolute error of a predeclared "
            "threshold event under the grid law, main grid.",
            "tab:all-tail",
        )
    )
    written.append(
        regime_table(
            data, "reference_tcate_rmse", "main", MAIN_ORDER,
            "Conditional Wasserstein reference effect \\texttt{REF-TCATE-K} with "
            "$\\nu_\\star$ the standard normal, main grid.",
            "tab:all-reftcate",
        )
    )
    written.append(
        regime_table(
            data, "reference_effect_rmse", "main", MAIN_ORDER,
            "Marginal Wasserstein reference effect \\texttt{REF-ATE-K}, main grid.",
            "tab:all-refate",
        )
    )
    written.append(
        functional_longtable(
            data, "tcate_functional_rmse", "TCATE-K", MAIN_ORDER,
            "Moderator-conditional functional error \\texttt{tcate\\_functional"
            "\\_rmse} for all four grid functionals, main grid. "
            "\\texttt{grid\\_skewness} and \\texttt{grid\\_upper\\_tail\\_mean} are "
            "excluded from every training manifest; the law-producing methods "
            "still estimate them at evaluation time, which is the transfer test.",
            "tab:all-tcate",
        )
    )
    written.append(
        functional_longtable(
            data, "tate_functional_rmse", "TATE-K", MAIN_ORDER,
            "Marginal functional error \\texttt{tate\\_functional\\_rmse} for all "
            "four grid functionals, main grid.",
            "tab:all-tate",
        )
    )
    written.append(
        regime_table(
            data, "mean_quantile_rmse", "smallk",
            MAIN_ORDER + ["sqw2_booster"],
            "\\texttt{mean\\_quantile\\_rmse} on the \\texttt{smallk} grid "
            "($n=500$, $K=5$, 20 seeds), with the proposed C-WDB R3, the two "
            "forest comparators, and the squared-$W_2$ mechanism ablation.",
            "tab:smallk-meanq",
        )
    )
    written.append(
        regime_table(
            data, "kernel_law_error", "smallk",
            LAW_ORDER + ["sqw2_booster"],
            "\\texttt{kernel\\_law\\_error} on the \\texttt{smallk} grid, with "
            "the squared-$W_2$ comparator retained as a mechanism check.",
            "tab:smallk-kernel",
        )
    )
    written.append(
        regime_table(
            data, "mode_coverage", "smallk",
            LAW_ORDER + ["sqw2_booster"],
            "\\texttt{mode\\_coverage} on the \\texttt{smallk} grid. The D6 row is "
            "the collapse result: every method that keeps the repulsion term covers "
            "both modes and the ablation does not.",
            "tab:smallk-mode",
        )
    )
    entries = [
        ("kernel_law_error", d, None, "\\texttt{kernel\\_law\\_error}")
        for d in ["D1", "D5", "D6", "D7"]
    ] + [
        ("mean_quantile_rmse", d, None, "\\texttt{mean\\_quantile\\_rmse}")
        for d in DGPS
    ] + [
        ("tcate_functional_rmse", d, "TCATE-K-grid_skewness",
         "\\texttt{TCATE-K-grid\\_skewness}")
        for d in ["D5", "D6", "D7"]
    ] + [
        ("tcate_functional_rmse", d, "TCATE-K-grid_upper_tail_mean",
         "\\texttt{TCATE-K-grid\\_upper\\_tail\\_mean}")
        for d in ["D5", "D6", "D7"]
    ] + [
        ("tcate_functional_rmse", d, "TCATE-K-grid_mean",
         "\\texttt{TCATE-K-grid\\_mean}")
        for d in ["D5", "D6", "D7"]
    ]
    written.append(
        paired_table(
            data, "cwdb_r3_cvridge", ["causal_drf", "drf"], entries,
            "Seed-paired comparisons for the proposed claimant "
            "\\texttt{cwdb\\_r3\\_cvridge}, main grid.",
            "tab:paired-r3",
        )
    )
    written.append(cost_table(data))

    (TABLES / "all_tables.tex").write_text("\n".join(written))

    regime_figure(
        data, "mean_quantile_rmse",
        ["cwdb_r3_cvridge", "causal_drf", "drf"],
        "Grid causal mean error across regimes, main grid, two standard errors",
        "mean_quantile_rmse (lower is better)",
        "meanq_all_regimes.png",
    )
    regime_figure(
        data, "kernel_law_error",
        ["cwdb_r3_cvridge", "causal_drf", "drf"],
        "Primary law metric across regimes, main grid, two standard errors",
        "kernel_law_error (lower is better)",
        "kernel_all_regimes.png",
    )
    regime_figure(
        data, "arm_energy_risk",
        ["cwdb_r3_cvridge", "causal_drf", "drf"],
        "Excess energy risk over the true law, main grid, two standard errors",
        "excess energy risk",
        "energy_all_regimes.png",
    )
    paired_figure(
        data, "kernel_law_error", "cwdb_r3_cvridge", "causal_drf",
        "Primary law metric, R3 minus the published incumbent, worst regime first",
        "paired difference (negative favours C-WDB R3)",
        "paired_r3_vs_drf.png",
    )
    functional_figure(
        data,
        ["cwdb_r3_cvridge", "causal_drf", "drf"],
        "functional_targets.png",
    )
    null_regime_figure(data)
    cost_figure(data)

    print(f"tables written to {TABLES}")
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
