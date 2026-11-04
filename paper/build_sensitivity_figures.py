"""Build the fifty-replication sensitivity figures for the paper appendix.

Inputs (read only, through ``sensitivity_r50_data``)::

    results/wcf_sensitivity_r50/merged/wcf_sensitivity_r50_results.parquet

Outputs (overwritten, under ``paper/figures``)::

    sensitivity_resolution.pdf, sensitivity_assignment.pdf,
    sensitivity_nulls.pdf

Every curve is a seed summary computed from the frozen r50 result file:
law entries are Monte Carlo means and functional entries are Monte Carlo
RMSEs over fifty paired replications, following the confirmatory convention.
Nothing is fitted and no numeric result is hardcoded.  Run from the
repository root with::

    python3 paper/build_sensitivity_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sensitivity_r50_data as S


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

METHODS = {
    "cwdb_dr": ("WCF", "#0072B2", "o"),
    "causal_drf": ("Causal-DRF", "#D55E00", "s"),
    "drf": ("DRF", "#009E73", "^"),
}
METHOD_ORDER = ("cwdb_dr", "causal_drf", "drf")
FACTORIES = (
    ("Logistic", "cwdb_dr", "#0072B2", "o"),
    ("Random forest", "cwdb_dr_flex_rf", "#56B4E9", "s"),
    ("Gradient boosting", "cwdb_dr_flex", "#CC79A7", "D"),
    ("Oracle", "cwdb_dr_oracle", "#7F7F7F", "^"),
)
DESIGNS = tuple(S.DESIGN_DGP)
DESIGN_COLORS = {
    "LS0": "#999999", "LS1": "#0072B2", "LS2": "#009E73", "LS3": "#D55E00",
}
K_VALUES = (5, 25, 49)
M_VALUES = (5, 10, 25)
RESOLUTION_COLUMNS = (
    ("Law (native)", "kernel_law_error", "LAW-A-K", True),
    ("Mean-quantile", "mean_quantile_rmse", "MEANQ-A-K", False),
    ("Ref. TCATE (AIPW)", "aipw_reference_tcate_rmse", "REF-TCATE-K", False),
)
NULL_TARGETS = (
    ("TATE grid mean", "tate"),
    ("TCATE grid mean", "tcate"),
    ("Ref. TATE", "ref_tate"),
    ("Ref. TCATE", "ref_tcate"),
)


def _estimate(
    frame: pd.DataFrame,
    metric: str,
    target: str,
    *,
    method: str = "cwdb_dr",
    law: bool = False,
    **filters: object,
) -> tuple[float, float]:
    values = S.seed_values(frame, metric, target, method=method, **filters)
    estimate, se, _ = S.summarize(values, law=law)
    return estimate, se


def _style_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.grid(color="0.88", linewidth=0.6)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_resolution(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))
    for row, coordinate in enumerate(("n_grid", "n_particles")):
        values = K_VALUES if coordinate == "n_grid" else M_VALUES
        for column, (title, metric, target, law) in enumerate(RESOLUTION_COLUMNS):
            ax = axes[row, column]
            for design in DESIGNS:
                means, ses = [], []
                for value in values:
                    filters = {
                        "grid": S.GRID_PRIMARY, "dgp": S.DESIGN_DGP[design],
                        "n_train": 1000,
                    }
                    if coordinate == "n_grid":
                        filters.update(n_grid=value, n_particles=10)
                    else:
                        filters.update(n_grid=25, n_particles=value)
                    mean, se = _estimate(frame, metric, target, law=law, **filters)
                    means.append(mean)
                    ses.append(se)
                ax.errorbar(
                    values, means, yerr=ses, color=DESIGN_COLORS[design],
                    marker="o", linewidth=1.2, markersize=4, capsize=1.7,
                    label=design,
                )
            ax.set_title(title, fontsize=8.5)
            ax.set_xticks(values, [str(value) for value in values])
            _style_axis(ax)
    axes[0, 0].set_xlabel("$K$ (particle budget $M=10$)", fontsize=7.5)
    axes[0, 2].set_xlabel("$K$ (particle budget $M=10$)", fontsize=7.5)
    axes[1, 0].set_xlabel("$M$ (grid size $K=25$)", fontsize=7.5)
    axes[1, 2].set_xlabel("$M$ (grid size $K=25$)", fontsize=7.5)
    axes[0, 0].set_ylabel("Error (log scale)", fontsize=7.5)
    axes[1, 0].set_ylabel("Error (log scale)", fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.99, wspace=0.32, hspace=0.52)
    fig.savefig(FIGURES / "sensitivity_resolution.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "sensitivity_resolution.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_assignment(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5))
    for row, n_train in ((0, 500), (1, 1000)):
        ax = axes[row, 0]
        for method in METHOD_ORDER:
            label, color, marker = METHODS[method]
            means, ses = [], []
            for dgp in S.SYM_DGPS:
                mean, se = _estimate(
                    frame, S.end_to_end_metric(method, "ref_tcate"),
                    S.FUNCTIONAL_TARGET["ref_tcate"], method=method,
                    grid=S.GRID_SYM, dgp=dgp, n_train=n_train,
                    n_grid=25, n_particles=10,
                )
                means.append(mean)
                ses.append(se)
            ax.errorbar(
                np.arange(len(S.SYM_DGPS)), means, yerr=ses, color=color,
                marker=marker, markersize=4, linestyle="none", capsize=2,
                label=label,
            )
        ax.set_title(rf"Assignment mechanism, $n={n_train}$", fontsize=8.5)
        ax.set_xticks(np.arange(len(S.SYM_DGPS)), S.SYM_DGPS, fontsize=7)
        _style_axis(ax)
        ax = axes[row, 1]
        offsets = np.linspace(-0.12, 0.12, len(FACTORIES))
        for (label, method, color, marker), offset in zip(FACTORIES, offsets):
            means, ses = [], []
            for dgp in S.PROPENSITY_DGPS:
                grid = S.GRID_SYM if method == "cwdb_dr" else (
                    S.GRID_FLEX_RF if method == "cwdb_dr_flex_rf" else S.GRID_FLEX
                )
                mean, se = _estimate(
                    frame, S.end_to_end_metric(method, "ref_tcate"),
                    S.FUNCTIONAL_TARGET["ref_tcate"], method=method,
                    grid=grid, dgp=dgp, n_train=n_train,
                    n_grid=25, n_particles=10,
                )
                means.append(mean)
                ses.append(se)
            ax.errorbar(
                np.arange(len(S.PROPENSITY_DGPS)) + offset, means, yerr=ses,
                color=color, marker=marker, markersize=4, linestyle="none",
                capsize=2, label=label,
            )
        ax.set_title(rf"Propensity factory, $n={n_train}$", fontsize=8.5)
        ax.set_xticks(np.arange(len(S.PROPENSITY_DGPS)), S.PROPENSITY_DGPS, fontsize=7)
        _style_axis(ax)
    axes[0, 0].set_ylabel("Reference-TCATE error", fontsize=7.5)
    axes[1, 0].set_ylabel("Reference-TCATE error", fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.87, bottom=0.14, left=0.09, right=0.99, wspace=0.30, hspace=0.48)
    fig.savefig(FIGURES / "sensitivity_assignment.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "sensitivity_assignment.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_nulls(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, n_train in zip(axes, (500, 1000)):
        for method in METHOD_ORDER:
            label, color, marker = METHODS[method]
            means, ses = [], []
            for _, functional in NULL_TARGETS:
                metric = S.end_to_end_metric(method, functional)
                target = S.FUNCTIONAL_TARGET[functional]
                pooled = pd.concat([
                    S.seed_values(
                        frame, metric, target, method=method,
                        grid=grid, dgp=dgp, n_train=n_train,
                        n_grid=25, n_particles=10,
                    )
                    for grid, dgp in S.NULL_COORDINATES
                ])
                estimate, se, _ = S.pooled_summary(pooled)
                means.append(estimate)
                ses.append(se)
            ax.errorbar(
                np.arange(len(NULL_TARGETS)), means, yerr=ses, color=color,
                marker=marker, markersize=4, linestyle="none", capsize=2,
                label=label,
            )
        ax.set_title(rf"Null companions, $n={n_train}$", fontsize=8.5)
        ax.set_xticks(
            np.arange(len(NULL_TARGETS)),
            [label for label, _ in NULL_TARGETS],
            fontsize=7,
        )
        _style_axis(ax)
    axes[0].set_ylabel("False-effect error", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.85, bottom=0.26, left=0.09, right=0.99, wspace=0.28)
    fig.savefig(FIGURES / "sensitivity_nulls.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "sensitivity_nulls.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = S.load_per_seed()
    plot_resolution(frame)
    plot_assignment(frame)
    plot_nulls(frame)


if __name__ == "__main__":
    main()
