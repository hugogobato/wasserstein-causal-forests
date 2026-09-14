"""Summarize the Project STAR WCF runs into JSON, Markdown, and figures.

    PYTHONPATH=src OMP_NUM_THREADS=1 python research/applied/project_star/05_summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_checks import trusted_scalar  # noqa: E402
from star_common import RESULTS_DIR  # noqa: E402

FUNCTIONAL_ORDER = ("mean", "sd", "skew", "tail", "reference", "p10", "p90")


def load_runs(pattern: str) -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob(pattern)):
        with open(path) as handle:
            run = json.load(handle)
        run["_file"] = str(path)
        runs.append(run)
    return runs


def summarize_runs(runs: list[dict]) -> dict:
    if not runs:
        return {"n_runs": 0}
    names = [n for n in FUNCTIONAL_ORDER if n in runs[0]["marginal_dr"]]
    summary: dict = {
        "n_runs": len(runs),
        "files": [r["_file"] for r in runs],
        "wall_seconds": [r["wall_seconds"] for r in runs],
        "functionals": {},
        "selected_contrast_shrinkage": [r["selected_contrast_shrinkage"] for r in runs],
        "propensity": {},
        "moderator_bin_counts": runs[0]["moderator_bin_counts"],
    }
    for name in names:
        marginals = np.asarray([r["marginal_dr"][name] for r in runs])
        if_ses = np.asarray([r["if_se"][name] for r in runs])
        bins = np.asarray(
            [[np.nan if v is None else v for v in r["bin_contrasts_dr"][name]] for r in runs],
            dtype=float,
        )
        det = runs[0].get("aipw_details", {}).get(name, {})
        bin_se = np.asarray(
            [np.nan if v is None else v for v in det.get("bin_if_se", [np.nan] * 4)],
            dtype=float,
        )
        summary["functionals"][name] = {
            "marginal_mean": float(marginals.mean()),
            "marginal_sd": float(marginals.std(ddof=1)) if marginals.size > 1 else 0.0,
            "marginal_values": marginals.tolist(),
            "if_se_seed0": float(if_ses[0]),
            "if_se_mean": float(if_ses.mean()),
            "bins_mean": np.nanmean(bins, axis=0).tolist(),
            "bins_sd": (
                np.nanstd(bins, axis=0, ddof=1).tolist()
                if bins.shape[0] > 1
                else [0.0] * bins.shape[1]
            ),
            "bins_if_se_seed0": [None if np.isnan(v) else float(v) for v in bin_se],
            "plugin_marginal_mean": float(
                np.mean([r["marginal_plugin"][name] for r in runs])
            ),
            "plugin_bins_mean": np.mean(
                np.asarray([r["bin_contrasts_plugin"][name] for r in runs]), axis=0
            ).tolist(),
        }
    props = np.asarray(
        [[r["propensity"][k] for k in ("min", "max", "mean")] for r in runs]
    )
    summary["propensity"] = {
        "min": float(props[:, 0].min()),
        "max": float(props[:, 1].max()),
        "mean": float(props[:, 2].mean()),
        "max_share_outside_0p1_0p9": float(
            max(r["propensity"]["share_outside_0p1_0p9"] for r in runs)
        ),
        "max_share_clipped": float(
            max(
                r["propensity"]["share_at_clip_low"] + r["propensity"]["share_at_clip_high"]
                for r in runs
            )
        ),
        "treated_mean": float(np.mean([r["propensity"]["treated_mean"] for r in runs])),
        "control_mean": float(np.mean([r["propensity"]["control_mean"] for r in runs])),
    }
    return summary


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown_table(summary: dict) -> str:
    lines = [
        "| functional | DR marginal | seed SD | IF SE | plug-in | bins b1..b4 (DR) | bin IF SE |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in summary["functionals"].items():
        bins = ", ".join(fmt(v) for v in s["bins_mean"])
        bse = ", ".join(fmt(v) for v in s["bins_if_se_seed0"])
        lines.append(
            f"| {name} | {fmt(s['marginal_mean'])} | {fmt(s['marginal_sd'])} | "
            f"{fmt(s['if_se_mean'])} | {fmt(s['plugin_marginal_mean'])} | {bins} | {bse} |"
        )
    return "\n".join(lines)


def make_figure(primary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["mean", "tail", "p10", "p90", "sd", "reference"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        s = primary["functionals"][name]
        lo = np.asarray(s["bins_if_se_seed0"], dtype=float)
        x = np.arange(4)
        ax.errorbar(x, s["bins_mean"], yerr=1.96 * lo, fmt="o", capsize=3, color="#1f4e79")
        ax.axhline(0.0, color="grey", lw=0.8, ls="--")
        ax.set_title(name)
        ax.set_xticks(x, ["Q1 low FL", "Q2", "Q3", "Q4 high FL"], fontsize=8)
    fig.suptitle("Project STAR WCF: DR TCATE by school free-lunch quartile (math)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    primary = summarize_runs(load_runs("primary/primary_seed*.json"))
    placebo = summarize_runs(load_runs("placebo/placebo_seed*.json"))
    sensitivity = {
        name: summarize_runs(load_runs(f"sensitivity/{name}_seed*.json"))
        for name in ("min15", "reg2", "reading", "qstarall", "schoolsat")
    }
    smoke_runs = load_runs("misc/smoke_seed*.json")
    smoke = summarize_runs(smoke_runs) if smoke_runs else {"n_runs": 0}

    checks = trusted_scalar("math")
    checks_reading = trusted_scalar("reading")

    comparison = {}
    if primary["n_runs"]:
        for name in FUNCTIONAL_ORDER:
            if name not in primary["functionals"]:
                continue
            p = primary["functionals"][name]
            pl = placebo["functionals"][name] if name in placebo.get("functionals", {}) else None
            comparison[name] = {
                "primary_marginal": p["marginal_mean"],
                "primary_if_se": p["if_se_mean"],
                "placebo_marginal": None if pl is None else pl["marginal_mean"],
                "placebo_if_se": None if pl is None else pl["if_se_mean"],
                "false_effect_ratio": (
                    None
                    if pl is None or p["if_se_mean"] == 0
                    else float(pl["marginal_mean"] / p["if_se_mean"])
                ),
            }

    out = {
        "primary": primary,
        "placebo": placebo,
        "sensitivity": sensitivity,
        "smoke": smoke,
        "trusted_scalar_math": checks,
        "trusted_scalar_reading": checks_reading,
        "comparison": comparison,
    }
    with open(RESULTS_DIR / "summary.json", "w") as handle:
        json.dump(out, handle, indent=2, default=str)

    lines = [
        "# Project STAR WCF applied study: results summary",
        "",
        f"Primary runs: {primary['n_runs']} seeds; placebo runs: {placebo['n_runs']} seeds.",
        "",
        "## Primary (math, control q_star, frozen defaults)",
        "",
        markdown_table(primary),
        "",
        "Selected contrast shrinkage per primary run: "
        + ", ".join(str(v) for v in primary["selected_contrast_shrinkage"]),
        "",
        f"Propensity across primary seeds: min={primary['propensity']['min']:.3f}, "
        f"max={primary['propensity']['max']:.3f}, "
        f"treated mean={primary['propensity']['treated_mean']:.3f}, "
        f"control mean={primary['propensity']['control_mean']:.3f}, "
        f"max share outside [0.1,0.9]={primary['propensity']['max_share_outside_0p1_0p9']:.3f}, "
        f"max share clipped={primary['propensity']['max_share_clipped']:.3f}.",
        "",
        "## Placebo (permuted treatment)",
        "",
        markdown_table(placebo) if placebo["n_runs"] else "not run",
        "",
        "False-effect ratio (placebo mean / primary IF SE): "
        + ", ".join(
            f"{k}={v['false_effect_ratio']:.2f}"
            for k, v in comparison.items()
            if v["false_effect_ratio"] is not None
        ),
        "",
        "## Trusted scalar (within-grade SD units)",
        "",
        f"- student-level difference in means with grade FE: "
        f"{checks['student_grade_fe']['diff']:.3f} "
        f"(cluster SE {checks['student_grade_fe']['cluster_se']:.3f})",
        f"- unit-level mean with grade FE: {checks['unit_mean_grade_fe']['diff']:.3f} "
        f"(SE {checks['unit_mean_grade_fe']['hc_se']:.3f})",
        "",
        "By grade (student-level, cluster SE): "
        + "; ".join(
            f"{g} {v['diff']:.3f} ({v['cluster_se']:.3f})"
            for g, v in checks["student_by_grade"].items()
        ),
        "",
        "## Sensitivity variants",
        "",
        "| variant | seeds | mean | tail | reference | p10 | p90 | sd | shrink |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in sensitivity.items():
        if not s["n_runs"]:
            continue
        f = s["functionals"]
        lines.append(
            f"| {name} | {s['n_runs']} | {f['mean']['marginal_mean']:.3f} "
            f"| {f['tail']['marginal_mean']:.3f} | {f['reference']['marginal_mean']:.3f} "
            f"| {f['p10']['marginal_mean']:.3f} | {f['p90']['marginal_mean']:.3f} "
            f"| {f['sd']['marginal_mean']:.4f} "
            f"| {', '.join(str(v) for v in s['selected_contrast_shrinkage'])} |"
        )
    checks_json = RESULTS_DIR / "misc" / "design_checks.json"
    if checks_json.exists():
        with open(checks_json) as handle:
            design = json.load(handle)
        lines += ["", "## Inner-n threshold feasibility", ""]
        for thr, v in design["threshold_feasibility"].items():
            lines.append(
                f"- tested_n >= {thr}: {v['small_units']} small / {v['regular_units']} "
                f"control units; adapter minimum satisfied: {v['adapter_min_arm_5_satisfied']}"
            )
        lines += ["", "Covariate balance (standardized differences), small minus control:"]
        lines.append(
            ", ".join(
                f"{k}={v['std_diff']:.3f}" for k, v in design["balance"].items()
            )
        )
    fig_path = RESULTS_DIR / "figures" / "bin_contrasts_math.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    if primary["n_runs"]:
        make_figure(primary, fig_path)
        lines += ["", f"Figure: `{fig_path}`"]
    with open(RESULTS_DIR / "RESULTS.md", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
