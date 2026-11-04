"""Build paper simulation assets from frozen, audited experiment outputs.

The n=1000 tables and the three simulation figures come from the fifty-
replication confirmatory run through ``confirmatory_data``.  The n=500 LS
table and the historical ten-replication ``simulation_metrics_rmse.csv``
ledger remain from the phase-6 and merged-baseline runs.

Law error is averaged over replications.  Functional-effect errors are Monte
Carlo RMSEs: squared replication-level errors are pooled (also across the four
declared functionals where applicable) before taking the square root.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

from confirmatory_data import build_ledger as build_confirmatory_ledger


ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "results"
OUT = ROOT / "figures"

METHODS = {
    "cwdb_dr": ("WCF", "#0072B2", "o"),
    "cwdb_zipt": ("Two-part WCF", "#0072B2", "o"),
    "causal_drf": ("Causal-DRF", "#D55E00", "s"),
    "drf": ("DRF", "#009E73", "^"),
}

ORDINARY_DESIGNS = ("LS1", "LS3", "S4", "S5")

METRICS = {
    "kernel_law_error": r"Conditional-law error (MMD$^2$)",
    "tate_functional_rmse": "Functional TATE RMSE",
    "tcate_functional_rmse": "Functional TCATE RMSE",
    "reference_effect_rmse": "Reference TATE RMSE",
    "reference_tcate_rmse": "Reference TCATE RMSE",
}

PAPER_DGP = {
    "IC0": "LS0", "IC1": "LS1", "IC2": "LS2", "IC3": "LS3",
    "D0": "S1", "D2": "S2", "D5": "S3", "D6": "S4", "D7": "S5", "D8": "S6",
    "ZI0": "Z0", "ZI1": "Z1", "ZI2": "Z2", "ZI3": "Z3",
}

TABLE_METRICS = {
    "kernel_law_error": "Law", "tate_functional_rmse": "TATE",
    "tcate_functional_rmse": "TCATE", "reference_effect_rmse": "Ref. TATE",
    "reference_tcate_rmse": "Ref. TCATE",
}

RAW_SPECS = (
    ("income", RAW / "merged_phase6" / "phase6_results.parquet", "income", ("cwdb_dr", "causal_drf", "drf"), ("IC0", "IC1", "IC2", "IC3"), (500, 1000)),
    ("abstract", RAW / "merged_phase6" / "phase6_results.parquet", "main", ("cwdb_dr",), ("D0", "D2", "D5", "D6", "D7", "D8"), (1000,)),
    ("abstract", RAW / "merged_original_causal_drf" / "main_results.parquet", "main", ("causal_drf",), ("D0", "D2", "D5", "D6", "D7", "D8"), (1000,)),
    ("abstract", RAW / "merged_original_drf" / "main_results.parquet", "main", ("drf",), ("D0", "D2", "D5", "D6", "D7", "D8"), (1000,)),
)


def _jackknife_rmse_se(seed_mse: pd.Series) -> float:
    values = seed_mse.to_numpy(dtype=float)
    leave_one_out = [math.sqrt((values.sum() - value) / (len(values) - 1)) for value in values]
    center = sum(leave_one_out) / len(leave_one_out)
    return math.sqrt((len(values) - 1) / len(values) * sum((x - center) ** 2 for x in leave_one_out))


def build_rmse_ledger() -> pd.DataFrame:
    """Historical ten-replication ledger, kept for the n=500 table."""

    rows = []
    for suite, path, grid, methods, dgps, sample_sizes in RAW_SPECS:
        frame = pd.read_parquet(path)
        frame = frame[
            (frame["grid"] == grid) & frame["method"].isin(methods)
            & frame["dgp"].isin(dgps) & frame["n_train"].isin(sample_sizes)
            & (frame["n_grid"] == 25) & (frame["n_particles"] == 10)
            & frame["seed"].isin(range(10)) & (frame["status"] == "ok")
            & frame["metric"].isin(METRICS)
        ].copy()
        for (dgp, method, n_train, metric), group in frame.groupby(
            ["dgp", "method", "n_train", "metric"], sort=False
        ):
            if metric == "kernel_law_error":
                seed_values = group.groupby("seed")["value"].mean()
                estimate = float(seed_values.mean())
                se = float(seed_values.std(ddof=1) / math.sqrt(len(seed_values)))
                observed = set(seed_values.index)
            else:
                seed_mse = group.assign(squared=group["value"].astype(float) ** 2).groupby("seed")["squared"].mean()
                estimate = math.sqrt(float(seed_mse.mean()))
                se = _jackknife_rmse_se(seed_mse)
                observed = set(seed_mse.index)
            if observed != set(range(10)):
                raise RuntimeError(f"Incomplete seeds for {dgp}/{method}/{n_train}/{metric}")
            rows.append({
                "suite": suite, "dgp": dgp, "method": method,
                "n_train": int(n_train), "metric": metric,
                "mean": estimate, "se": se, "n_replications": 10,
            })
    ledger = pd.DataFrame(rows)
    expected = (2 * 4 * 3 + 6 * 3) * len(METRICS)
    if len(ledger) != expected:
        raise RuntimeError(f"Expected {expected} summary rows, found {len(ledger)}")
    (ROOT / "tables").mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ROOT / "tables" / "simulation_metrics_rmse.csv", index=False)
    return ledger


def plot_simulation_summary(data: pd.DataFrame) -> None:
    """Five principal criteria on the four selected ordinary designs."""

    ordinary_methods = ("cwdb_dr", "causal_drf", "drf")
    offsets = {"cwdb_dr": -0.17, "causal_drf": 0.0, "drf": 0.17}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.15))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, METRICS.items()):
        panel = data[data["metric"] == metric].set_index(["design", "method"])
        for method in ordinary_methods:
            label, color, marker = METHODS[method]
            means = [panel.loc[(design, method), "mean"] for design in ORDINARY_DESIGNS]
            ses = [panel.loc[(design, method), "se"] for design in ORDINARY_DESIGNS]
            ax.errorbar(
                [index + offsets[method] for index in range(len(ORDINARY_DESIGNS))],
                means, yerr=ses, color=color, marker=marker, markersize=4.0,
                linestyle="none", capsize=2, label=label,
            )
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(range(len(ORDINARY_DESIGNS)), ORDINARY_DESIGNS, fontsize=7)
        ax.set_yscale("log")
        ax.grid(axis="y", color="0.88", linewidth=0.6)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Error (log scale; lower is better)", fontsize=7.5)
    axes[3].set_ylabel("Error (log scale; lower is better)", fontsize=7.5)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.09, right=0.99, wspace=0.30, hspace=0.52)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "simulation_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "simulation_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_family_table(data: pd.DataFrame, n_train: int, filename: str, caption: str) -> None:
    table_dir = ROOT / "tables"
    method_names = {key: value[0] for key, value in METHODS.items()}
    replications = int(data["n_replications"].iloc[0])
    word = {10: "ten", 50: "fifty"}.get(replications, str(replications))
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption} at $n={n_train}$. Law entries are Monte Carlo means; functional-effect entries are Monte Carlo RMSEs. Parentheses give Monte Carlo SEs over {word} replications; lower is better.}}",
        rf"\label{{tab:{filename}}}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Design & Method & Law & TATE & TCATE & Ref. TATE & Ref. TCATE \\", r"\midrule",
    ]
    for dgp in data["dgp"].drop_duplicates():
        for method in ("cwdb_dr", "causal_drf", "drf"):
            row = data[(data["dgp"] == dgp) & (data["method"] == method)]
            cells = []
            for metric in TABLE_METRICS:
                hit = row[row["metric"] == metric]
                cells.append("--" if hit.empty else f"{hit.iloc[0]['mean']:.4f}({hit.iloc[0]['se']:.4f})")
            lines.append(f"{PAPER_DGP[dgp]} & {method_names[method]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (table_dir / f"{filename}.tex").write_text("\n".join(lines), encoding="utf-8")


def build_tables(legacy: pd.DataFrame, confirmatory: pd.DataFrame) -> None:
    table_dir = ROOT / "tables"
    method_names = {key: value[0] for key, value in METHODS.items()}
    legacy_ls_500 = legacy[(legacy["suite"] == "income") & (legacy["n_train"] == 500)]
    confirmatory_ls = confirmatory[(confirmatory["suite"] == "income") & (confirmatory["n_train"] == 1000)]
    confirmatory_s = confirmatory[(confirmatory["suite"] == "abstract") & (confirmatory["n_train"] == 1000)]
    _write_family_table(
        legacy_ls_500, 500, "full_results_ls_500",
        "Location, shape, and overlap designs",
    )
    _write_family_table(
        confirmatory_ls, 1000, "full_results_ls",
        "Location, shape, and overlap designs",
    )
    _write_family_table(
        confirmatory_s, 1000, "full_results_s",
        "Law-separation and heterogeneous-effect designs",
    )

    zi = confirmatory[(confirmatory["suite"] == "zi") & (confirmatory["n_train"] == 1000)]
    zi_methods = ["cwdb_zipt", "causal_drf", "drf"]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Structural-zero designs at $n=1000$. Mass is the error in the degenerate-component probability, and mass TCATE is its moderator-stratum contrast error. Entries are Monte Carlo mean (SE) over fifty replications; lower is better.}",
        r"\label{tab:full_results_z}", r"\begin{tabular}{llrrr}", r"\toprule",
        r"Design & Method & Law & Mass & Mass TCATE \\", r"\midrule",
    ]
    for dgp in zi["dgp"].drop_duplicates():
        for method in zi_methods:
            row = zi[(zi["dgp"] == dgp) & (zi["method"] == method)]
            cells = []
            for metric in ["kernel_law_error", "zero_mass_abs_error", "mass_contrast_rmse"]:
                hit = row[row["metric"] == metric]
                cells.append("--" if hit.empty else f"{hit.iloc[0]['mean']:.4f}({hit.iloc[0]['se']:.4f})")
            lines.append(f"{PAPER_DGP[dgp]} & {method_names[method]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (table_dir / "full_results_z.tex").write_text("\n".join(lines), encoding="utf-8")


def build_sensitivity_tables() -> None:
    """Generate the fifty-replication sensitivity tables for the appendix."""

    import sensitivity_r50_tables

    sensitivity_r50_tables.build(ROOT / "tables")


def main() -> None:
    legacy = build_rmse_ledger()
    confirmatory = build_confirmatory_ledger()
    confirmatory.to_csv(ROOT / "tables" / "simulation_metrics_rmse_50.csv", index=False)
    plot_simulation_summary(confirmatory)
    build_tables(legacy, confirmatory)
    build_sensitivity_tables()


if __name__ == "__main__":
    main()
