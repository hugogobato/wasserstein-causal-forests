"""Build sample-size sensitivity figures and complete appendix tables."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confirmatory_data import build_ledger as build_confirmatory_ledger


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT.parent
    / "results/wcf_sample_size_sensitivity/all_dgps/merged/wcf_sample_size_results.parquet"
)
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

DGP_LABEL = {
    "IC0": "LS0", "IC1": "LS1", "IC2": "LS2", "IC3": "LS3",
    "D0": "S1", "D2": "S2", "D5": "S3", "D6": "S4", "D7": "S5", "D8": "S6",
    "ZI0": "Z0", "ZI1": "Z1", "ZI2": "Z2", "ZI3": "Z3",
}
FAMILY = {label: label.rstrip("0123456789") for label in DGP_LABEL.values()}
METHODS = {
    "cwdb_dr": ("WCF", "#0072B2", "o"),
    "cwdb_zipt": ("Two-part WCF", "#0072B2", "o"),
    "causal_drf": ("Causal-DRF", "#D55E00", "s"),
    "drf": ("DRF", "#009E73", "^"),
}
N_VALUES = (125, 250, 500, 1000)
ORDINARY_METRICS = (
    ("kernel_law_error", "Law"),
    ("tate_functional_rmse", "TATE"),
    ("tcate_functional_rmse", "TCATE"),
    ("reference_effect_rmse", "Ref. TATE"),
    ("reference_tcate_rmse", "Ref. TCATE"),
)
ZERO_METRICS = (
    ("kernel_law_error", "Law"),
    ("zero_mass_abs_error", "Mass"),
    ("mass_contrast_rmse", "Mass TCATE"),
)


def jackknife_rmse_se(seed_mse: pd.Series) -> float:
    values = seed_mse.to_numpy(dtype=float)
    leave = [math.sqrt((values.sum() - value) / (len(values) - 1)) for value in values]
    center = np.mean(leave)
    return float(math.sqrt((len(values) - 1) / len(values) * np.sum((leave - center) ** 2)))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wanted = {metric for metric, _ in ORDINARY_METRICS + ZERO_METRICS}
    frame = frame[(frame["status"] == "ok") & frame["metric"].isin(wanted)].copy()
    for (dgp, n_train, method, metric), group in frame.groupby(
        ["dgp", "n_train", "method", "metric"], sort=False
    ):
        if metric == "kernel_law_error":
            seed_values = group.groupby("seed")["value"].mean()
            estimate = float(seed_values.mean())
            se = float(seed_values.std(ddof=1) / math.sqrt(len(seed_values)))
        else:
            seed_mse = group.assign(squared=group["value"].astype(float) ** 2).groupby("seed")["squared"].mean()
            estimate = math.sqrt(float(seed_mse.mean()))
            se = jackknife_rmse_se(seed_mse)
        if len(seed_values if metric == "kernel_law_error" else seed_mse) != 10:
            raise RuntimeError(f"Expected ten seeds for {dgp}/{n_train}/{method}/{metric}")
        label = DGP_LABEL[dgp]
        rows.append({
            "design": label, "family": FAMILY[label], "n_train": int(n_train),
            "method": method, "metric": metric, "mean": estimate, "se": se,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "sample_size_sensitivity_summary.csv", index=False)
    return out


def summarize_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Seed-paired family averages used by the summary figure.

    Functional summaries pool squared errors across designs within a seed
    before taking the square root, so the plotted standard errors respect the
    shared replication indices.
    """
    wanted = {metric for metric, _ in ORDINARY_METRICS + ZERO_METRICS}
    frame = frame[(frame["status"] == "ok") & frame["metric"].isin(wanted)].copy()
    family_dgps = {
        family: tuple(
            code for code, label in DGP_LABEL.items() if FAMILY[label] == family
        )
        for family in ("LS", "S", "Z")
    }
    rows = []
    for family, dgps in family_dgps.items():
        subset = frame[frame["dgp"].isin(dgps)]
        for (n_train, method, metric), group in subset.groupby(
            ["n_train", "method", "metric"], sort=False
        ):
            if metric == "kernel_law_error":
                seed_values = group.groupby("seed")["value"].mean()
                estimate = float(seed_values.mean())
                se = float(seed_values.std(ddof=1) / math.sqrt(len(seed_values)))
            else:
                seed_mse = group.assign(squared=group["value"].astype(float) ** 2).groupby("seed")["squared"].mean()
                estimate = math.sqrt(float(seed_mse.mean()))
                se = jackknife_rmse_se(seed_mse)
            if len(seed_values if metric == "kernel_law_error" else seed_mse) != 10:
                raise RuntimeError(f"Expected ten seeds for {family}/{n_train}/{method}/{metric}")
            rows.append({
                "family": family, "n_train": int(n_train), "method": method,
                "metric": metric, "mean": estimate, "se": se,
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "sample_size_sensitivity_family.csv", index=False)
    return out


def write_family_table(summary: pd.DataFrame, family: str) -> None:
    metrics = ZERO_METRICS if family == "Z" else ORDINARY_METRICS
    method_order = ("cwdb_zipt", "causal_drf", "drf") if family == "Z" else ("cwdb_dr", "causal_drf", "drf")
    subset = summary[summary["family"] == family]
    columns = "r" * len(metrics)
    lines = [
        r"\begingroup",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{longtable}{lll" + columns + "}",
        rf"\caption{{Sample-size sensitivity for the {family} family. Law is mean MMD$^2$; other entries are Monte Carlo RMSE. Parentheses are Monte Carlo SEs over ten replications.}}\label{{tab:sample-size-{family.lower()}}}\\",
        r"\toprule",
        "Design & $n$ & Method & " + " & ".join(label for _, label in metrics) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        "Design & $n$ & Method & " + " & ".join(label for _, label in metrics) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for design in subset["design"].drop_duplicates():
        for n_train in N_VALUES:
            for method in method_order:
                row = subset[(subset["design"] == design) & (subset["n_train"] == n_train) & (subset["method"] == method)]
                cells = []
                for metric, _ in metrics:
                    hit = row[row["metric"] == metric]
                    if hit.empty:
                        raise RuntimeError(f"Missing {design}/{n_train}/{method}/{metric}")
                    cells.append(f"{hit.iloc[0]['mean']:.4f}({hit.iloc[0]['se']:.4f})")
                lines.append(f"{design} & {n_train} & {METHODS[method][0]} & " + " & ".join(cells) + r" \\")
            lines.append(r"\addlinespace")
        lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    (TABLES / f"sample_size_{family.lower()}.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_family_summary(summary: pd.DataFrame) -> None:
    panels = (
        ("LS", "kernel_law_error", "LS: conditional-law MMD$^2$"),
        ("LS", "reference_tcate_rmse", "LS: reference TCATE RMSE"),
        ("S", "kernel_law_error", "S: conditional-law MMD$^2$"),
        ("S", "reference_tcate_rmse", "S: reference TCATE RMSE"),
        ("Z", "kernel_law_error", "Z: conditional-law MMD$^2$"),
        ("Z", "mass_contrast_rmse", "Z: mass TCATE RMSE"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))
    for ax, (family, metric, title) in zip(axes.ravel(), panels):
        methods = ("cwdb_zipt", "causal_drf", "drf") if family == "Z" else ("cwdb_dr", "causal_drf", "drf")
        panel = summary[(summary["family"] == family) & (summary["metric"] == metric)]
        for method in methods:
            line = panel[panel["method"] == method].set_index("n_train").reindex(N_VALUES)
            label, color, marker = METHODS[method]
            ax.errorbar(
                N_VALUES, line["mean"], yerr=line["se"], color=color,
                marker=marker, linewidth=1.2, markersize=4, capsize=1.7, label=label,
            )
        ax.set_title(title, fontsize=8.5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(color="0.88", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Mean across designs", fontsize=7.5)
    axes[1, 0].set_ylabel("Mean across designs", fontsize=7.5)
    for ax in axes[1, :]:
        ax.set_xlabel("Training sample size", fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.08, right=0.99, wspace=0.29, hspace=0.42)
    fig.savefig(FIGURES / "sample_size_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "sample_size_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_all_designs(summary: pd.DataFrame) -> None:
    """Show every design at n=1000 without averaging across families."""
    ordinary = summary[(summary["n_train"] == 1000) & summary["family"].isin(("LS", "S"))]
    designs = tuple(f"LS{i}" for i in range(4)) + tuple(f"S{i}" for i in range(1, 7))
    methods = ("cwdb_dr", "causal_drf", "drf")
    offsets = {"cwdb_dr": -0.19, "causal_drf": 0.0, "drf": 0.19}
    metric_titles = (
        ("kernel_law_error", r"Conditional-law error (MMD$^2$)"),
        ("tate_functional_rmse", "Functional TATE RMSE"),
        ("tcate_functional_rmse", "Functional TCATE RMSE"),
        ("reference_effect_rmse", "Reference TATE RMSE"),
        ("reference_tcate_rmse", "Reference TCATE RMSE"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.35))
    axes = axes.ravel()
    x = np.arange(len(designs))
    for ax, (metric, title) in zip(axes, metric_titles):
        panel = ordinary[ordinary["metric"] == metric].set_index(["design", "method"])
        for method in methods:
            label, color, marker = METHODS[method]
            means = [panel.loc[(design, method), "mean"] for design in designs]
            ses = [panel.loc[(design, method), "se"] for design in designs]
            ax.errorbar(
                x + offsets[method], means, yerr=ses, color=color, marker=marker,
                markersize=3.5, linestyle="none", capsize=1.7, label=label,
            )
        ax.axvline(3.5, color="0.72", linewidth=0.7)
        ax.set_title(title, fontsize=8.3)
        ax.set_xticks(x, designs, fontsize=6.5, rotation=45, ha="right")
        ax.set_yscale("log")
        ax.grid(axis="y", color="0.88", linewidth=0.6)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Error (log scale)", fontsize=7.5)
    axes[3].set_ylabel("Error (log scale)", fontsize=7.5)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.94, bottom=0.11, left=0.085, right=0.99, wspace=0.29, hspace=0.58)
    fig.savefig(FIGURES / "all_dgp_summary.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "all_dgp_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    zero = summary[(summary["n_train"] == 1000) & (summary["family"] == "Z")]
    zero_designs = tuple(f"Z{i}" for i in range(4))
    zero_methods = ("cwdb_zipt", "causal_drf", "drf")
    zero_offsets = {"cwdb_zipt": -0.15, "causal_drf": 0.0, "drf": 0.15}
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.15))
    for ax, (metric, title) in zip(axes, ZERO_METRICS):
        panel = zero[zero["metric"] == metric].set_index(["design", "method"])
        for method in zero_methods:
            label, color, marker = METHODS[method]
            means = [panel.loc[(design, method), "mean"] for design in zero_designs]
            ses = [panel.loc[(design, method), "se"] for design in zero_designs]
            ax.errorbar(
                np.arange(4) + zero_offsets[method], means, yerr=ses,
                color=color, marker=marker, markersize=4.0, linestyle="none",
                capsize=2, label=label,
            )
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(np.arange(4), zero_designs, fontsize=7)
        ax.set_yscale("log")
        ax.grid(axis="y", color="0.88", linewidth=0.6)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Error (log scale)", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.88, bottom=0.27, left=0.075, right=0.99, wspace=0.28)
    fig.savefig(FIGURES / "all_dgp_zero_summary.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "all_dgp_zero_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = summarize(pd.read_parquet(SOURCE))
    for family in ("LS", "S", "Z"):
        write_family_table(summary, family)
    plot_family_summary(summarize_families(pd.read_parquet(SOURCE)))
    # The all-design n=1000 panels use the fifty-replication confirmatory run.
    plot_all_designs(build_confirmatory_ledger())


if __name__ == "__main__":
    main()
