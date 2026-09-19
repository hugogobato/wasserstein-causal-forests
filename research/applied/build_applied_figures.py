#!/usr/bin/env python3
"""Density figures for the applied studies.

Each figure is built from the unit-level quantile vectors that define the
outcomes. For one arm, the displayed curve is a Gaussian kernel density
estimate of the equal-unit mixture of the unit laws, formed from the pooled
unit quantile coordinates; the bandwidth is Silverman's rule applied to the
pooled arm coordinates and is common to all curves in a panel. Reference
curves apply the same estimator to the reference quantile vector interpolated
to a fine level grid. This mixture representation keeps the support of the
pooled units, unlike a Wasserstein barycenter, whose support endpoints are the
averages of the per-unit tail quantiles and can produce boundary spikes. No
covariate adjustment enters these curves; they are descriptive.

For the state minimum-wage study the same construction is repeated on the
full sample and on the sample retained by the standard overlap band
[0.05, 0.95], and a companion forest plot reports every declared functional
for the two samples with one-standard-error bars. Star and Kenya forest plots
report the calibrated functionals with influence-function standard errors.

Outputs (report/figures_generated/):
    applied_star_densities.pdf
    applied_star_estimates.pdf
    applied_minwage_densities.pdf   (requires the overlap-restriction fits)
    applied_minwage_estimates.pdf
    applied_egger_densities.pdf
    applied_egger_estimates.pdf

Usage::

    PYTHONPATH=src python research/applied/build_applied_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "applied_study_exploration"
STAR_DIR = SRC / "project_star"
MINWAGE_DIR = SRC / "primary_state_minwage"
EGGER_DIR = SRC / "egger_kenya"
FIGURE_DIR = ROOT / "report" / "figures_generated"

sys.path.insert(0, str(ROOT / "src"))
from wasserstein_causal_forests.applied.adapter import grid_weights  # noqa: E402
from wasserstein_causal_forests.g3.dgps import moderator_bins  # noqa: E402

TREATED_COLOUR = "#0072B2"
CONTROL_COLOUR = "#D55E00"
REF_COLOURS = ("#000000", "#666666")
REF_STYLES = ("--", ":")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------------
# Density construction from quantile vectors
# ---------------------------------------------------------------------------

def silverman_bandwidth(sample: np.ndarray) -> float:
    """Silverman's rule bandwidth for an equal-weight sample."""

    values = np.asarray(sample, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 1.0
    std = float(np.std(values, ddof=1))
    q75, q25 = np.percentile(values, [75.0, 25.0])
    spread = min(std, float(q75 - q25) / 1.34)
    if spread <= 0.0:
        spread = std if std > 0.0 else 1.0
    return 0.9 * spread * values.size ** (-0.2)


def mixture_density(
    sample: np.ndarray, x_grid: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Gaussian kernel density estimate of an equal-weight sample."""

    values = np.asarray(sample, dtype=float)
    z = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1)
    return density / (values.size * bandwidth * np.sqrt(2.0 * np.pi))


def _expand_reference(q_star: np.ndarray, n_fine: int = 200) -> np.ndarray:
    """Interpolate a reference quantile vector to a fine level grid."""

    from scipy.interpolate import PchipInterpolator

    star = np.asarray(q_star, dtype=float).ravel()
    k = star.size
    u = (np.arange(k) + 0.5) / k
    fine = np.linspace(u[0], u[-1], n_fine)
    return PchipInterpolator(u, star)(fine)


def reference_distances(
    q: np.ndarray, a: np.ndarray, q_star: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Mean grid-Wasserstein distance to the benchmark by arm and the contrast."""

    w = grid_weights(q.shape[1])
    w = w / w.sum()
    distance = np.sqrt(((q - q_star[None, :]) ** 2) @ w)
    if mask is None:
        mask = np.ones(q.shape[0], dtype=bool)
    d_control = float(distance[mask & (a == 0)].mean())
    d_treated = float(distance[mask & (a == 1)].mean())
    return d_control, d_treated, d_treated - d_control


def _density_curves(
    ax: plt.Axes,
    arms: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    x_grid: np.ndarray,
    log_coordinates: bool,
    title: str,
    fade_control: bool = True,
    both_vivid: bool = False,
    annotation: str | None = None,
    legend: bool = False,
) -> None:
    pooled = np.concatenate(
        [np.asarray(v, dtype=float).ravel() for v in arms.values()]
        + [np.asarray(v, dtype=float).ravel() for v in references.values()]
    )
    positive = pooled[pooled > 0.0]
    floor = float(positive.min()) * 0.5 if positive.size else 1.0

    def to_plot(values: np.ndarray) -> np.ndarray:
        block = np.asarray(values, dtype=float)
        return np.log(np.maximum(block, floor)) if log_coordinates else block

    def from_plot(values: np.ndarray) -> np.ndarray:
        return np.exp(values) if log_coordinates else values

    arm_pools = {label: to_plot(qs).ravel() for label, qs in arms.items()}
    bandwidth = silverman_bandwidth(np.concatenate(list(arm_pools.values())))

    curves: dict[str, np.ndarray] = {}
    for label, sample in arm_pools.items():
        curves[label] = mixture_density(sample, x_grid, bandwidth)
    for label, q_star in references.items():
        fine = _expand_reference(q_star)
        curves[label] = mixture_density(to_plot(fine), x_grid, bandwidth)

    all_finite = np.concatenate(
        [curve[np.isfinite(curve)] for curve in curves.values()]
    )
    scale = float(np.max(all_finite)) if all_finite.size else 1.0

    x_plot = from_plot(x_grid)

    for label, colour, alpha, lw in (
        (
            "control",
            CONTROL_COLOUR,
            0.55 if (both_vivid or not fade_control) else 0.30,
            1.8,
        ),
        ("treated", TREATED_COLOUR, 0.55 if both_vivid else 0.30, 2.0),
    ):
        curve = curves[label] / scale
        ax.fill_between(
            x_plot,
            0.0,
            np.nan_to_num(curve),
            color=colour,
            alpha=alpha,
            linewidth=0.0,
        )
        ax.plot(x_plot, curve, color=colour, lw=lw, label=label.capitalize())
    for index, (label, q_star) in enumerate(references.items()):
        curve = curves[label] / scale
        ax.plot(
            x_plot,
            curve,
            color=REF_COLOURS[index % len(REF_COLOURS)],
            ls=REF_STYLES[index % len(REF_STYLES)],
            lw=1.6,
            label=label,
        )

    if log_coordinates:
        ax.set_xscale("log")
    ax.set_xlim(x_plot[0], x_plot[-1])
    ax.set_ylim(0.0, 1.15)
    ax.set_title(title)
    ax.set_ylabel("Density (scaled)")
    ax.grid(alpha=0.22, lw=0.5)
    if annotation:
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.6),
        )
    if both_vivid or legend:
        ax.legend(frameon=False, loc="upper right")


def _reference_grid(qs: np.ndarray, lo: float, hi: float, log_coordinates: bool) -> np.ndarray:
    stacked = np.asarray(qs).ravel()
    stacked = stacked[np.isfinite(stacked)]
    if log_coordinates:
        stacked = stacked[stacked > 0.0]
        lo_v, hi_v = np.log(lo), np.log(hi)
    else:
        lo_v, hi_v = lo, hi
    return np.linspace(lo_v, hi_v, 400)


# ---------------------------------------------------------------------------
# STAR
# ---------------------------------------------------------------------------

def star_figure() -> Path:
    data = np.load(STAR_DIR / "data" / "dataset.npz")
    q, a, q_star = data["Q"], data["A"], data["q_star"]
    bins = moderator_bins(data["X"])
    w = grid_weights(q.shape[1])
    w = w / w.sum()

    summary = json.load(open(STAR_DIR / "summary.json"))
    calibrated_bins = summary["primary"]["functionals"]["reference"]["bins_mean"]
    calibrated_marginal = summary["primary"]["functionals"]["reference"]["marginal_mean"]

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), constrained_layout=True)

    grid = _reference_grid(q, -3.7, 4.1, log_coordinates=False)

    d = np.sqrt(((q - q_star[None, :]) ** 2) @ w)

    def annotation_for(mask: np.ndarray) -> str:
        d0 = float(d[mask & (a == 0)].mean())
        d1 = float(d[mask & (a == 1)].mean())
        return f"$d_0={d0:.3f}$, $d_1={d1:.3f}$\n$\\Delta d = {d1 - d0:+.3f}$"

    quartile_titles = [
        "(a) Free-lunch Q1 (lowest)",
        "(b) Free-lunch Q2",
        "(c) Free-lunch Q3",
        "(d) Free-lunch Q4 (highest)",
    ]
    for index in range(4):
        ax = axes[0, index] if index < 3 else axes[1, 0]
        mask = bins == index
        _density_curves(
            ax,
            {"control": q[mask & (a == 0)], "treated": q[mask & (a == 1)]},
            {"Benchmark": q_star},
            grid,
            log_coordinates=False,
            title=quartile_titles[index],
            annotation=annotation_for(mask),
            legend=index == 0,
        )
        ax.set_xlabel("Within-grade $z$-score")

    ax = axes[1, 1]
    _density_curves(
        ax,
        {"control": q[a == 0], "treated": q[a == 1]},
        {"Benchmark": q_star},
        grid,
        log_coordinates=False,
        title="(e) Marginal",
        annotation=annotation_for(np.ones(q.shape[0], dtype=bool)),
        legend=True,
    )
    ax.set_xlabel("Within-grade $z$-score")

    ax = axes[1, 2]
    labels = ["Q1", "Q2", "Q3", "Q4", "Marg."]
    raw = []
    for index in range(4):
        _, _, diff = reference_distances(q, a, q_star, bins == index)
        raw.append(diff)
    _, _, marginal_diff = reference_distances(q, a, q_star)
    raw.append(marginal_diff)
    positions = np.arange(len(labels))
    colours = ["#B2182B" if value < 0 else "#2166AC" for value in raw]
    ax.bar(positions, raw, color=colours, alpha=0.65, width=0.6, label="Raw contrast")
    calibrated_se = (
        summary["primary"]["functionals"]["reference"]["bins_if_se_seed0"]
        + [summary["primary"]["functionals"]["reference"]["if_se_mean"]]
    )
    ax.errorbar(
        positions,
        calibrated_bins + [calibrated_marginal],
        yerr=calibrated_se,
        fmt="D",
        mfc="white",
        mec="black",
        ecolor="black",
        capsize=2,
        ms=5,
        zorder=3,
        label="Calibrated (DR)",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title("(f) Reference contrast by quartile")
    ax.set_ylabel("$d_1-d_0$")
    ax.legend(frameon=False, loc="upper right")

    path = FIGURE_DIR / "applied_star_densities.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Minimum wage
# ---------------------------------------------------------------------------

def _selection_propensity(x: np.ndarray, a: np.ndarray, seed: int = 0) -> np.ndarray:
    from wasserstein_causal_forests.cwdb.dr_calibration import FunctionalAIPW

    probe = FunctionalAIPW(n_bins=4).fit(
        observed={},
        oof_arm_means={},
        X=x,
        treatment=a,
        random_state=seed + 31,
    )
    return np.asarray(probe.ehat_train_, dtype=float)


_MINWAGE_FUNCTIONALS = (
    "mean",
    "sd",
    "skew",
    "tail",
    "reference",
    "p10",
    "p90",
    "ref_us2016",
)


def _read_fit_marginals(paths: list[Path]) -> dict[str, float]:
    values = {name: [] for name in _MINWAGE_FUNCTIONALS}
    for path in paths:
        payload = json.load(open(path))
        for name in values:
            values[name].append(float(payload["marginal_dr"][name]))
    return {name: float(np.mean(vals)) for name, vals in values.items()}


def _read_fit_se(path: Path) -> dict[str, float]:
    payload = json.load(open(path))
    return {
        name: float(payload["if_se"][name]) for name in _MINWAGE_FUNCTIONALS
    }


def _minwage_sample_paths() -> list[tuple[str, list[Path]]]:
    runs = MINWAGE_DIR / "results"
    sample_paths = [
        ("Full sample", [runs / f"fit_primary_seed{seed}.json" for seed in range(3)]),
        (
            "Trim [0.05,0.95]",
            [runs / f"fit_trim005_095_seed{seed}.json" for seed in range(3)],
        ),
    ]
    for name, paths in sample_paths:
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise SystemExit(f"missing fits: {missing}")
    return sample_paths


def minwage_figure() -> Path:
    data = np.load(MINWAGE_DIR / "data" / "dataset.npz")
    q, a = data["Q"], data["A"]
    x = data["X"]
    benchmarks = np.load(MINWAGE_DIR / "data" / "benchmarks.npz")
    nordic = benchmarks["nordic_scaled"]
    us2016 = benchmarks["us2016"]

    ehat = _selection_propensity(x, a, seed=0)
    variants = [
        ("Full sample", np.ones_like(a, dtype=bool), "n=867 (117/750)"),
        ("Trimprop $[0.05,0.95]$", (ehat >= 0.05) & (ehat <= 0.95), None),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), constrained_layout=True)
    grid = _reference_grid(q, 2.0, 80.0, log_coordinates=True)

    for column, (title, mask, fixed_label) in enumerate(variants):
        ax = axes[column]
        d0, d1, diff = reference_distances(q, a, nordic, mask)
        n_treated = int(np.sum(mask & (a == 1)))
        n_control = int(np.sum(mask & (a == 0)))
        label = fixed_label or f"n={int(mask.sum())} ({n_treated}/{n_control})"
        annotation = f"{label}\n$d_0={d0:.2f}$, $d_1={d1:.2f}$\n$\\Delta d={diff:+.2f}$"
        _density_curves(
            ax,
            {"control": q[mask & (a == 0)], "treated": q[mask & (a == 1)]},
            {"Nordic (scaled)": nordic, "US 2016": us2016},
            grid,
            log_coordinates=True,
            title=title,
            annotation=annotation,
            legend=column == 0,
        )
        ax.set_xlabel("Hourly wage, 2016 USD (log scale)")

    path = FIGURE_DIR / "applied_minwage_densities.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


def minwage_estimates_figure() -> Path:
    sample_paths = _minwage_sample_paths()
    order = (
        ("mean", "Mean"),
        ("sd", "SD"),
        ("skew", "Skewness"),
        ("tail", "Upper-half mean"),
        ("reference", "Ref. (Nordic)"),
        ("p10", "$p_{10}$"),
        ("p90", "$p_{90}$"),
        ("ref_us2016", "Ref. (US 2016)"),
    )
    ys = np.arange(len(order))
    full_paths = sample_paths[0][1]
    trim_paths = sample_paths[1][1]
    full_values = [_read_fit_marginals(full_paths)[f] for f, _ in order]
    full_se = [_read_fit_se(full_paths[0])[f] for f, _ in order]
    trim_values = [_read_fit_marginals(trim_paths)[f] for f, _ in order]
    trim_se = [_read_fit_se(trim_paths[0])[f] for f, _ in order]

    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    offset = 0.23
    ys_full = ys - offset
    ys_trim = ys + offset
    ax.errorbar(
        full_values,
        ys_full,
        xerr=full_se,
        fmt="o",
        color=TREATED_COLOUR,
        ecolor="#333333",
        capsize=3,
        ms=5,
        label="Full sample",
    )
    ax.errorbar(
        trim_values,
        ys_trim,
        xerr=trim_se,
        fmt="D",
        mfc="white",
        mec="#444444",
        ecolor="#888888",
        capsize=3,
        ms=5,
        label="Trim $[0.05,0.95]$",
    )
    ax.axvline(0.0, color="black", lw=0.8)
    for y in ys:
        ax.axhline(y, color="#f0f0f0", lw=0.6, zorder=0)
    for y, value in zip(ys_full, full_values):
        ax.annotate(
            f"{value:+.3f}",
            (value, y),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
        )
    for y, value in zip(ys_trim, trim_values):
        ax.annotate(
            f"{value:+.3f}",
            (value, y),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
            color="#444444",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label in order])
    ax.set_ylim(len(order) - 0.42, -0.85)
    ax.set_xlabel("Contrast (2016 USD per hour; skewness dimensionless)")
    ax.set_title("State minimum wage: calibrated contrasts by sample")
    ax.grid(alpha=0.22, lw=0.5, axis="x")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    path = FIGURE_DIR / "applied_minwage_estimates.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Kenya cash transfers
# ---------------------------------------------------------------------------

def egger_figure() -> Path:
    data = np.load(EGGER_DIR / "data" / "dataset.npz")
    q, a, q_star = data["Q"], data["A"], data["q_star"]
    bench = np.load(
        EGGER_DIR / "benchmark_external" / "kihbs_2015_national_benchmark.npz"
    )
    external = bench["q_star_ksh_per_year"]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), constrained_layout=True)
    grid = _reference_grid(q, 3.0e3, 3.0e5, log_coordinates=True)

    ax = axes[0]
    _density_curves(
        ax,
        {"control": q[a == 0], "treated": q[a == 1]},
        {},
        grid,
        log_coordinates=True,
        title="(a) Village densities, both arms",
        both_vivid=True,
    )
    ax.set_xlabel("Per-capita consumption, KSH (log scale)")

    ax = axes[1]
    d0, d1, diff = reference_distances(q, a, q_star)
    _density_curves(
        ax,
        {"control": q[a == 0], "treated": q[a == 1]},
        {"Control benchmark": q_star},
        grid,
        log_coordinates=True,
        title="(b) Control benchmark",
        annotation=f"$d_0={d0:.0f}$, $d_1={d1:.0f}$\n$\\Delta d={diff:+.0f}$",
        legend=True,
    )
    ax.set_xlabel("Per-capita consumption, KSH (log scale)")

    ax = axes[2]
    d0, d1, diff = reference_distances(q, a, external)
    _density_curves(
        ax,
        {"control": q[a == 0], "treated": q[a == 1]},
        {"KIHBS 2015/16": external},
        grid,
        log_coordinates=True,
        title="(c) External national benchmark",
        annotation=f"$d_0={d0:.0f}$, $d_1={d1:.0f}$\n$\\Delta d={diff:+.0f}$",
        legend=True,
    )
    ax.set_xlabel("Per-capita consumption, KSH (log scale)")

    path = FIGURE_DIR / "applied_egger_densities.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Uncertainty summaries
# ---------------------------------------------------------------------------

def star_estimates_figure() -> Path:
    summary = json.load(open(STAR_DIR / "summary.json"))
    primary = summary["primary"]["functionals"]
    placebo = summary["placebo"]["functionals"]
    order = (
        ("mean", "Mean"),
        ("tail", "Upper-half mean"),
        ("p10", "$p_{10}$"),
        ("p90", "$p_{90}$"),
        ("reference", "Reference"),
        ("sd", "SD"),
        ("skew", "Skewness"),
    )
    ys = np.arange(len(order))
    dr = np.asarray([primary[f]["marginal_mean"] for f, _ in order])
    se = np.asarray([primary[f]["if_se_mean"] for f, _ in order])
    pl = np.asarray([placebo[f]["marginal_mean"] for f, _ in order])

    fig, ax = plt.subplots(figsize=(7.8, 3.9), constrained_layout=True)
    ax.errorbar(
        dr,
        ys,
        xerr=se,
        fmt="o",
        color=TREATED_COLOUR,
        ecolor="#333333",
        capsize=3,
        ms=5,
        label="Calibrated (DR, $\\pm 1$ SE)",
    )
    ax.plot(pl, ys, "o", mfc="white", mec="#666666", ms=5, ls="none", label="Placebo")
    ax.axvline(0.0, color="black", lw=0.8)
    ax.axvline(0.2, color="#666666", lw=0.9, ls="--")
    ax.text(
        0.205,
        len(order) - 0.7,
        "literature\n~0.2 SD",
        fontsize=8,
        color="#444444",
        va="top",
    )
    for y, value, error in zip(ys, dr, se):
        ax.annotate(
            f"{value:+.3f} ({error:.3f})",
            (value, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label in order])
    ax.invert_yaxis()
    ax.set_xlabel("Contrast (within-grade SD; skewness dimensionless)")
    ax.set_title("Project STAR: calibrated functionals with influence-function SEs")
    ax.grid(alpha=0.22, lw=0.5, axis="x")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    path = FIGURE_DIR / "applied_star_estimates.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


def egger_estimates_figure() -> Path:
    payload = json.load(open(EGGER_DIR / "summary.json"))
    summary = payload["primary"]
    capped = payload["capped_reference"]
    raw = json.load(open(EGGER_DIR / "raw_checks.json"))
    external = json.load(open(EGGER_DIR / "runs" / "external" / "results.json"))
    bench = json.load(
        open(EGGER_DIR / "benchmark_external" / "kihbs_2015_national_benchmark.json")
    )
    order = (
        ("mean", "Mean"),
        ("sd", "SD"),
        ("skew", "Skewness"),
        ("tail", "Upper-half mean"),
        ("p10", "$p_{10}$"),
        ("p90", "$p_{90}$"),
        ("reference", "Ref. (control)"),
    )
    dr = [summary[f]["dr_mean"] for f, _ in order]
    se = [summary[f]["if_se"] for f, _ in order]
    raw_values = [raw["levels"][f]["raw_diff"] for f, _ in order]
    raw_p = [raw["permutation"]["p_value_two_sided"][f] for f, _ in order]
    labels = [label for _, label in order]

    dr.append(float(capped["dr_mean"]))
    se.append(float(capped["if_se"]))
    raw_values.append(float("nan"))
    raw_p.append(float("nan"))
    labels.append("Ref. (capped p75)")

    external_dr = external["aggregate"]["marginal_dr"]["reference"]["mean"]
    external_se = float(
        np.mean([run["if_se"]["reference"] for run in external["per_seed"]])
    )
    external_raw = bench["raw_contrast_to_external_benchmark"]
    dr.append(external_dr)
    se.append(external_se)
    raw_values.append(external_raw["contrast"])
    raw_p.append(external_raw["permutation_p"])
    labels.append("Ref. (KIHBS 2015/16)")

    dr = np.asarray(dr)
    se = np.asarray(se)
    raw_values = np.asarray(raw_values)
    raw_p = np.asarray(raw_p)
    raws_ok = np.isfinite(raw_values)
    ys = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    ax.errorbar(
        dr,
        ys,
        xerr=se,
        fmt="o",
        color=TREATED_COLOUR,
        ecolor="#333333",
        capsize=3,
        ms=5,
        label="Calibrated (DR, $\\pm 1$ SE)",
    )
    ax.plot(
        raw_values[raws_ok],
        ys[raws_ok],
        "D",
        mfc="white",
        mec="#666666",
        ms=5,
        ls="none",
        label="Raw contrast",
    )
    ax.axvline(0.0, color="black", lw=0.8)
    for y in ys:
        ax.axhline(y, color="#eeeeee", lw=0.6, zorder=0)
    p_column = 6_000.0
    for y, p_value in zip(ys, raw_p):
        if not np.isfinite(p_value):
            continue
        text = f"{p_value:.3f}" if p_value >= 0.01 else f"{p_value:.4f}"
        ax.annotate(
            f"$p={text}$",
            (p_column, y),
            ha="left",
            va="center",
            fontsize=8,
            color="#444444",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-8_800, 9_200)
    ax.set_xlabel("Contrast (KSH per capita; skewness dimensionless)")
    ax.set_title("Kenya cash transfers: calibrated and raw contrasts")
    ax.grid(alpha=0.22, lw=0.5, axis="x")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    path = FIGURE_DIR / "applied_egger_estimates.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("wrote", star_figure())
    print("wrote", star_estimates_figure())
    print("wrote", egger_figure())
    print("wrote", egger_estimates_figure())
    trim_paths = [
        MINWAGE_DIR / "results" / f"fit_trim005_095_seed{seed}.json"
        for seed in range(3)
    ]
    if all(path.exists() for path in trim_paths):
        print("wrote", minwage_figure())
        print("wrote", minwage_estimates_figure())
    else:
        print("trimmed fits not ready; skipping minimum-wage figures")


if __name__ == "__main__":
    main()
