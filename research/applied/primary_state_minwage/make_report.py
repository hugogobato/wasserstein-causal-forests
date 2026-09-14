#!/usr/bin/env python3
"""Build the consolidated results report for the primary applied study.

Reads every fit JSON in the results directory, aggregates the three primary
seeds and the three placebo seeds through the shared adapter helpers, and
writes `results/REPORT.md` plus CSV tables.

Run from the repository root:

    PYTHONPATH=src python research/applied/primary_state_minwage/make_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY = REPO_ROOT / "results" / "applied_study_exploration" / "primary_state_minwage"
RES = STUDY / "results"

sys.path.insert(0, str(REPO_ROOT / "src"))
from wasserstein_causal_forests.applied.adapter import aggregate_replications  # noqa: E402

FUNCS = ["mean", "sd", "skew", "tail", "reference", "p10", "p90",
         "ref_us2016", "ref_mean_norm"]
BINS = ["Q1", "Q2", "Q3", "Q4"]


def load(name: str) -> dict:
    with open(RES / name) as handle:
        return json.load(handle)


def fmt(value: float, nd: int = 3) -> str:
    return f"{value:+.{nd}f}"


def marginal_block(records: list[dict], label: str) -> list[str]:
    lines = [f"### {label}", "",
             "| functional | DR marginal (mean over seeds) | across-seed sd | IF SE (seed 0) "
             "| plug-in (seed 0) | DR bins Q1..Q4 (mean) |",
             "|---|---|---|---|---|---|"]
    for name in FUNCS:
        if name not in records[0]["marginal_dr"]:
            continue
        values = np.array([r["marginal_dr"][name] for r in records])
        se = records[0]["if_se"][name]
        plugin = records[0]["marginal_plugin"][name]
        bins = np.array([[np.nan if v is None else v for v in r["bin_contrasts_dr"][name]]
                         for r in records])
        bins_mean = np.nanmean(bins, axis=0)
        lines.append(
            f"| {name} | {values.mean():+.4f} | {values.std(ddof=1):.4f} | {se:.4f} | "
            f"{plugin:+.4f} | " + " ".join(f"{b:+.3f}" for b in bins_mean) + " |"
        )
    lines.append("")
    return lines


def robustness_table(records: dict[str, dict]) -> list[str]:
    lines = ["### Robustness fits (seed 0 unless noted)", "",
             "| run | n | treated | shrinkage | mean | p10 | sd | skew | reference |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, r in records.items():
        lines.append(
            f"| {label} | {r['n']} | {r['n_treated']} | "
            f"{r['selected_contrast_shrinkage']:.0f} | "
            f"{fmt(r['marginal_dr']['mean'])} | {fmt(r['marginal_dr']['p10'])} | "
            f"{fmt(r['marginal_dr']['sd'])} | {fmt(r['marginal_dr']['skew'])} | "
            f"{fmt(r['marginal_dr']['reference'])} |"
        )
    lines.append("")
    return lines


def main() -> None:
    primary = [load(f"fit_primary_seed{s}.json") for s in range(3)]
    placebo = [load(f"placebo_primary_seed{s}.json") for s in range(3)]

    agg_primary = aggregate_replications(primary)
    agg_placebo = aggregate_replications(placebo)
    with open(RES / "replications_primary_aggregated.json", "w") as handle:
        json.dump(agg_primary, handle, indent=2)
    with open(RES / "replications_placebo_aggregated.json", "w") as handle:
        json.dump(agg_placebo, handle, indent=2)

    lines = ["# Primary applied study: WCF results", "",
             "Frozen defaults: M=10 particles, 100 trees, lr 0.12, depth 4, min leaf 10, "
             "min arm leaf 5, epsilon 1e-3, contrast candidates (0, 50, 500), 3 folds. "
             "All fits below use these settings.", ""]
    lines += marginal_block(primary, "Confirmatory window 2000-2016, treatment A1, "
                                     "three seeds (0, 1, 2)")
    lines += marginal_block(placebo, "Placebo (permuted A1), three seeds (0, 1, 2)")
    lines += robustness_table({
        "A1 primary seed 0": primary[0],
        "A2 (federal increases included)": load("fit_primary_A2_seed0.json"),
        "A3 (state above federal floor)": load("fit_primary_A3_seed0.json"),
        "A1 + calendar year in X (post-hoc)": load("fit_yearfe_seed0.json"),
        "First-difference A1 seed 0 (post-hoc)": load("fit_fd_seed0.json"),
        "First-difference A1 seed 1 (post-hoc)": load("fit_fd_seed1.json"),
        "Extension 1980-2016 seed 0": load("fit_ext_1980_2016_seed0.json"),
        "Extension 1980-2016 seed 1": load("fit_ext_1980_2016_seed1.json"),
        "Extension 2017-2022 seed 0": load("fit_ext_2017_2022_seed0.json"),
        "Leave out CA": load("lo_state_CA.json"),
        "Leave out TX": load("lo_state_TX.json"),
        "Leave out NY": load("lo_state_NY.json"),
    })

    fd0, fd1 = load("fit_fd_seed0.json"), load("fit_fd_seed1.json")
    lines += ["### First-difference variant detail (seeds 0 and 1)", "",
              "| functional | seed 0 | seed 1 |", "|---|---|---|"]
    for name in fd0["marginal_dr"]:
        lines.append(f"| {name} | {fmt(fd0['marginal_dr'][name])} "
                     f"(se {fd0['if_se'][name]:.4f}) | {fmt(fd1['marginal_dr'][name])} "
                     f"(se {fd1['if_se'][name]:.4f}) |")
    lines.append("")

    lines += ["### Propensity diagnostics (seed 0, primary)", "",
              "| quantity | value |", "|---|---|"]
    for key, value in primary[0]["propensity"].items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.append("")
    lines += ["### Selected contrast shrinkage and held-out risks (primary seeds)", "",
              "| seed | selected | risks (shrinkage: risk) |", "|---|---|---|"]
    for seed, r in enumerate(primary):
        risks = ", ".join(f"{s['contrast_shrinkage']:.0f}: {s['held_out_risk']:.5f}"
                          for s in r["selection_records"])
        lines.append(f"| {seed} | {r['selected_contrast_shrinkage']:.0f} | {risks} |")
    lines.append("")

    text = "\n".join(lines)
    (RES / "REPORT.md").write_text(text)

    # tidy CSV of core marginals across all fits
    rows = []
    for path in sorted(RES.glob("*.json")):
        if "aggregated" in path.name:
            continue
        r = json.loads(path.read_text())
        if r.get("reduced_settings"):
            continue
        for name in FUNCS:
            if name in r.get("marginal_dr", {}):
                rows.append({
                    "run": path.stem, "seed": r["random_state"],
                    "n": r["n"], "treated": r["n_treated"],
                    "shrinkage": r["selected_contrast_shrinkage"],
                    "functional": name,
                    "dr": r["marginal_dr"][name], "if_se": r["if_se"][name],
                    "plugin": r["marginal_plugin"][name],
                })
    pd.DataFrame(rows).to_csv(RES / "all_fits_marginals.csv", index=False)
    print("wrote", RES / "REPORT.md", "and all_fits_marginals.csv")


if __name__ == "__main__":
    main()
