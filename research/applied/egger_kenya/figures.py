"""Figures for the Egger Kenya WCF applied study.

Outputs (results/applied_study_exploration/egger_kenya/figures):
  fig1_quantile_curves.png   arm-mean quantile curves, benchmark, coordinate raw diffs
  fig2_functional_effects.png DR marginals with IF SE, placebo and Q-noise spreads
  fig3_bin_contrasts.png     DR 4-bin contrasts vs raw bin contrasts

Run from the repo root:
    PYTHONPATH=src /usr/bin/python3 research/applied/egger_kenya/figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.applied.adapter import AppliedDataset  # noqa: E402

OUT = ROOT / "results/applied_study_exploration/egger_kenya"
FIGS = OUT / "figures"
FUNC_ORDER = ("mean", "sd", "skew", "tail", "reference", "p10", "p90")


def load(stage: str) -> dict:
    with open(OUT / "runs" / stage / "results.json") as fh:
        return json.load(fh)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    ds = AppliedDataset.load(OUT / "data")
    with open(OUT / "summary.json") as fh:
        summary = json.load(fh)
    raw = summary["raw_checks"]

    u = np.asarray(raw["coordinate_raw"]["u"])
    t_mean = np.asarray(raw["coordinate_raw"]["treated_mean_q"])
    c_mean = np.asarray(raw["coordinate_raw"]["control_mean_q"])
    diff = np.asarray(raw["coordinate_raw"]["raw_diff"])
    pvals = np.asarray(raw["coordinate_raw"]["p_value_two_sided"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(u, c_mean / 1000, marker="o", ms=3, label="control villages (mean of Q)")
    axes[0].plot(u, t_mean / 1000, marker="s", ms=3, label="treated villages (mean of Q)")
    axes[0].plot(u, ds.q_star / 1000, ls="--", color="0.4", label="q* pooled control (primary)")
    capped = np.asarray(ds.meta["q_star_capped"])
    axes[0].plot(u, capped / 1000, ls=":", color="0.6", label="q* capped p75 (secondary)")
    axes[0].set_xlabel("quantile level u")
    axes[0].set_ylabel("KSH per capita (thousands)")
    axes[0].set_title("Village quantile curves at endline")
    axes[0].legend(fontsize=7)
    sig = pvals < 0.05
    axes[1].bar(u, diff / 1000, width=0.032, color=np.where(sig, "tab:red", "0.6"))
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("quantile level u")
    axes[1].set_ylabel("treated - control (KSH pc, thousands)")
    axes[1].set_title("Raw coordinate differences (red: perm p<0.05)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_quantile_curves.png", dpi=160)
    plt.close(fig)

    primary = summary["primary"]
    placebo = summary["placebo"]
    noise = summary["noise"]
    names = [n for n in FUNC_ORDER if n in primary]
    x = np.arange(len(names))
    est = np.array([primary[n]["dr_mean"] for n in names])
    se = np.array([primary[n]["if_se"] for n in names])
    scale = 1000.0
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.errorbar(x, est / scale, yerr=1.96 * se / scale, fmt="o", capsize=4, label="DR marginal +/- 1.96 IF SE")
    for i, n in enumerate(names):
        if n in placebo:
            ax.scatter([i] * 3, np.array(placebo[n]["values"]) / scale, marker="x", color="tab:red", label="placebo seeds" if i == 0 else None)
        if n in noise:
            ax.scatter([i] * 3, np.array(noise[n]["values"]) / scale, marker="^", facecolors="none", color="tab:green", label="Q-bootstrap seeds" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("effect (KSH pc, thousands; p90 log scale caveat)")
    ax.set_title("WCF DR marginals, placebo and Q-measurement-noise seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_functional_effects.png", dpi=160)
    plt.close(fig)

    raw_bins = raw["unadjusted_bins"]
    prim = load("primary_extras")["aggregate"]["bin_contrasts_dr"]
    show = ("mean", "tail", "p10", "reference")
    fig, axes = plt.subplots(1, len(show), figsize=(13, 3.6), sharex=True)
    for ax, name in zip(axes, show, strict=True):
        dr = np.array([np.nan if v is None else v for v in prim[name]["mean"]])
        dr_sd = np.array([np.nan if v is None else v for v in prim[name]["std"]])
        rb = np.array(raw_bins[name])
        xi = np.arange(4)
        ax.errorbar(xi, dr / scale, yerr=1.96 * dr_sd / scale, fmt="o", capsize=3, label="DR bin mean")
        ax.scatter(xi, rb / scale, marker="s", color="0.4", label="raw bin mean")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(name)
        ax.set_xticks(xi, ["Q1\n(poorest)", "Q2", "Q3", "Q4\n(richest)"], fontsize=7)
    axes[0].set_ylabel("effect (KSH pc, thousands)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Moderator-bin contrasts by baseline village assets", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_bin_contrasts.png", dpi=160)
    plt.close(fig)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
