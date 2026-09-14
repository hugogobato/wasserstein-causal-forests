#!/usr/bin/env python3
"""Summarize WCF fit JSONs into markdown and CSV tables.

    PYTHONPATH=src python research/applied/primary_state_minwage/report_results.py \
        --results-dir results/applied_study_exploration/primary_state_minwage/results \
        --out-dir results/applied_study_exploration/primary_state_minwage/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FUNCTIONAL_ORDER = [
    "mean", "sd", "skew", "tail", "reference", "p10", "p90",
    "ref_us2016", "ref_mean_norm",
]
BIN_LABELS = ["Q1(low mod)", "Q2", "Q3", "Q4(high mod)"]


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as handle:
            record = json.load(handle)
        record["_path"] = str(path)
        out.append(record)
    return out


def marginal_table(records: list[dict], kind: str) -> pd.DataFrame:
    rows = []
    for record in records:
        label = f"{record.get('treatment_variant', 'A1')}_{record.get('_path', '')}"
        for name in FUNCTIONAL_ORDER:
            if name not in record["marginal_dr"]:
                continue
            rows.append({
                "record": Path(record["_path"]).name,
                "seed": record["random_state"],
                "treatment": record.get("treatment_variant", "A1"),
                "functional": name,
                "marginal": record["marginal_dr"][name],
                "if_se": record["if_se"][name],
                "plugin": record["marginal_plugin"][name],
            })
    return pd.DataFrame(rows)


def contrast_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        for name in FUNCTIONAL_ORDER:
            if name not in record["bin_contrasts_dr"]:
                continue
            for b, value in enumerate(record["bin_contrasts_dr"][name]):
                rows.append({
                    "record": Path(record["_path"]).name,
                    "seed": record["random_state"],
                    "treatment": record.get("treatment_variant", "A1"),
                    "functional": name,
                    "bin": b,
                    "dr": value,
                    "plugin": record["bin_contrasts_plugin"][name][b],
                })
    return pd.DataFrame(rows)


def summarize(records: list[dict]) -> str:
    lines = []
    lines.append(f"# WCF applied-study results ({len(records)} fits)\n")
    for record in records:
        name = Path(record["_path"]).name
        t = record.get("treatment_variant", "A1")
        lines.append(
            f"- `{name}`: treatment {t}, seed {record['random_state']}, n={record['n']} "
            f"({record['n_treated']} treated / {record['n_control']} control), "
            f"K={record['K']}, folds={record['n_folds']}, "
            f"reduced={record.get('reduced_settings')}, "
            f"selected shrinkage={record['selected_contrast_shrinkage']:.1f}, "
            f"elapsed={record.get('elapsed_seconds', float('nan')):.1f}s"
        )
    lines.append("")

    for record in records:
        name = Path(record["_path"]).name
        t = record.get("treatment_variant", "A1")
        lines.append(f"## {name} (treatment {t}, seed {record['random_state']})\n")
        lines.append("| functional | DR marginal | IF SE | plug-in | DR bins (Q1..Q4) |")
        lines.append("|---|---|---|---|---|")
        for functional in FUNCTIONAL_ORDER:
            if functional not in record["marginal_dr"]:
                continue
            bins = record["bin_contrasts_dr"][functional]
            bins_txt = " ".join("nan" if v is None else f"{v:+.4f}" for v in bins)
            lines.append(
                f"| {functional} | {record['marginal_dr'][functional]:+.4f} | "
                f"{record['if_se'][functional]:.4f} | "
                f"{record['marginal_plugin'][functional]:+.4f} | {bins_txt} |"
            )
        prop = record["propensity"]
        lines.append("")
        lines.append(
            f"Propensity: min {prop['min']:.3f}, max {prop['max']:.3f}, "
            f"mean {prop['mean']:.3f}, treated mean {prop['treated_mean']:.3f}, "
            f"control mean {prop['control_mean']:.3f}, share outside [0.1,0.9] "
            f"{prop['share_outside_0p1_0p9']:.3f}, at clip low/high "
            f"{prop['share_at_clip_low']:.4f}/{prop['share_at_clip_high']:.4f}"
        )
        selection = ", ".join(
            f"{r['contrast_shrinkage']:.0f}:{r['held_out_risk']:.5f}"
            for r in record["selection_records"]
        )
        lines.append(f"Selection risks (shrinkage:risk): {selection}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="all")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_results(args.results_dir)
    marg = marginal_table(records, "marginal")
    bins = contrast_table(records)
    marg.to_csv(args.out_dir / f"summary_marginals_{args.label}.csv", index=False)
    bins.to_csv(args.out_dir / f"summary_bin_contrasts_{args.label}.csv", index=False)
    text = summarize(records)
    with open(args.out_dir / f"summary_{args.label}.md", "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
