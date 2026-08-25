#!/usr/bin/env python3
"""Score the rule-1 repair variants against the frozen G3 gate.

Run from the repository root, after `g3 merge --track repair`:

    python research/checks/g3_repair_report.py

Reads the frozen merged table and the repair table, concatenates them, and
evaluates `compute_gate_flags` once per repair variant with that variant named
as claimant. The frozen rows are never recomputed and never modified: every
baseline number a repair variant is compared against is the number the G3 memo
already reports.

Rule 1 is reported in full because it is the rule the repair exists to fix. The
other rules are reported so a repair that fixes rule 1 by destroying rule 2 is
visible immediately rather than at the next gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.analysis import (  # noqa: E402
    CLAIMANT,
    compute_gate_flags,
    cost_summary,
    failure_rates,
    load_rows,
    mechanism_ablations,
    method_means,
    paired_comparison,
)
from wasserstein_causal_forests.g3.manifest import (  # noqa: E402
    GATE_RULES,
    PRIMARY_LAW_METRIC,
)
from wasserstein_causal_forests.g3.repair import REPAIR_METHODS  # noqa: E402

FROZEN = ROOT / "results" / "merged" / "main_results.parquet"
REPAIRED = ROOT / "results" / "merged_repair" / "main_results.parquet"
PAYLOAD = ROOT / "results" / "merged_repair" / "repair_payload.json"


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def rule_one_table(rows: list[dict], claimants: list[str]) -> list[dict]:
    """Rule 1's numbers for every claimant, on the frozen thresholds."""

    rule = GATE_RULES["rule_1_correctness"]
    table = []
    for name in claimants:
        flags = compute_gate_flags(rows, claimant=name)["rule_1_correctness"]
        table.append(
            {
                "method": name,
                "d0_mean_quantile_rmse": flags["d0_mean_quantile_rmse"],
                "d2_mean_quantile_rmse": flags["d2_mean_quantile_rmse"],
                "d2_best_baseline": flags["d2_best_baseline"],
                "d2_false_effect_ratio": flags["d2_false_effect_ratio"],
                "d2_cap": rule["d2_max_false_effect_ratio"],
                "passed": flags["passed"],
            }
        )
    return table


def against_frozen_claimant(rows: list[dict], name: str) -> list[dict]:
    """Seed-paired difference from C-WDB-v1, regime by regime.

    Same trees, same budget, same seeds: the difference isolates the contrast
    regulariser and nothing else.
    """

    comparisons = []
    dgps = sorted({row["dgp"] for row in rows if row["grid"] == "main"})
    for metric in ("mean_quantile_rmse", PRIMARY_LAW_METRIC):
        for dgp in dgps:
            comparison = paired_comparison(
                rows, metric, claimant=name, comparator=CLAIMANT,
                grid="main", dgp=dgp,
            )
            if comparison is not None:
                comparisons.append(comparison)
    return comparisons


def selected_strengths(rows: list[dict], name: str) -> dict[str, float]:
    """What the cross-fitted variant actually chose, per regime."""

    chosen: dict[str, list[float]] = {}
    for row in rows:
        if (
            row["method"] == name
            and row["metric"] == "diagnostic_selected_contrast_shrinkage"
            and row["status"] == "ok"
            and row["value"] is not None
        ):
            chosen.setdefault(row["dgp"], []).append(float(row["value"]))
    return {
        dgp: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "n": len(values),
        }
        for dgp, values in sorted(chosen.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        default="",
        help="comma-separated repair methods; default is every method present",
    )
    arguments = parser.parse_args()

    if not REPAIRED.exists():
        print(f"no repair results at {REPAIRED}", file=sys.stderr)
        return 1
    frozen_rows = load_rows(FROZEN)
    repair_rows = load_rows(REPAIRED)
    rows = frozen_rows + repair_rows

    present = sorted({row["method"] for row in repair_rows})
    claimants = (
        [m for m in arguments.methods.split(",") if m]
        if arguments.methods
        else [m for m in REPAIR_METHODS if m in present]
    )
    if not claimants:
        print("no repair methods found in the repair table", file=sys.stderr)
        return 1

    table = rule_one_table(rows, [CLAIMANT, *claimants])
    cap = GATE_RULES["rule_1_correctness"]["d2_max_false_effect_ratio"]

    print(f"frozen rows: {len(frozen_rows)}   repair rows: {len(repair_rows)}")
    print(f"\nRule 1, the null regime. Cap on the false-effect ratio: {cap}\n")
    header = f"{'method':<22}{'D2 RMSE':>10}{'baseline':>10}{'ratio':>8}{'D0 RMSE':>10}  verdict"
    print(header)
    print("-" * len(header))
    for record in table:
        print(
            f"{record['method']:<22}"
            f"{record['d2_mean_quantile_rmse']:>10.4f}"
            f"{record['d2_best_baseline']:>10.4f}"
            f"{record['d2_false_effect_ratio']:>8.2f}"
            f"{record['d0_mean_quantile_rmse']:>10.4f}"
            f"  {'PASS' if record['passed'] else 'FAIL'}"
        )

    payload = {
        "git_revision": git_revision(),
        "frozen_checksum": checksum(FROZEN),
        "repair_checksum": checksum(REPAIRED),
        "n_frozen_rows": len(frozen_rows),
        "n_repair_rows": len(repair_rows),
        "failure_rates": failure_rates(repair_rows),
        "cost_summary": cost_summary(rows),
        "rule_1_table": table,
        "gate_flags": {},
        "against_frozen_claimant": {},
        "mechanism_ablations": {},
        "selected_contrast_shrinkage": {},
    }

    for name in [CLAIMANT, *claimants]:
        payload["gate_flags"][name] = compute_gate_flags(rows, claimant=name)
        # A repair that passes the gate by destroying a mechanism ablation has
        # not repaired anything worth keeping, so the ablations travel with the
        # flags rather than being looked up separately afterwards.
        payload["mechanism_ablations"][name] = mechanism_ablations(rows, claimant=name)
        if name == CLAIMANT:
            continue
        payload["against_frozen_claimant"][name] = against_frozen_claimant(rows, name)
        chosen = selected_strengths(rows, name)
        if chosen:
            payload["selected_contrast_shrinkage"][name] = chosen

    print("\nEvery rule, per repair variant. A rule with no rows for a variant")
    print("reports as failed; widen the repair manifest before reading it.\n")
    rule_names = [n for n in GATE_RULES]
    header = f"{'method':<22}" + "".join(f"{n.split('_')[1][:9]:>11}" for n in rule_names)
    print(header)
    print("-" * len(header))
    for name in [CLAIMANT, *claimants]:
        flags = (
            compute_gate_flags(rows, claimant=name) if name != CLAIMANT
            else compute_gate_flags(rows)
        )
        marks = "".join(
            f"{'PASS' if flags[n]['passed'] else 'fail':>11}" for n in rule_names
        )
        print(f"{name:<22}{marks}")

    print("\nPaired against C-WDB-v1 on the main grid (negative favours the repair):\n")
    for name in claimants:
        print(f"  {name}")
        for comparison in payload["against_frozen_claimant"][name]:
            verdict = (
                "repair" if comparison["claimant_wins"]
                else "v1" if comparison["comparator_wins"] else "tie"
            )
            print(
                f"    {comparison['dgp']:<4}{comparison['metric']:<22}"
                f"{comparison['claimant_mean']:>9.4f}"
                f"{comparison['comparator_mean']:>10.4f}"
                f"{comparison['paired_mean_difference']:>11.4f}"
                f" +- {comparison['paired_standard_error']:<9.4f}{verdict}"
            )

    print("\nMechanism ablations. A repair that passes the gate by breaking one")
    print("of these has not repaired anything worth keeping.\n")
    for name in [CLAIMANT, *claimants]:
        ablations = payload["mechanism_ablations"][name]
        lost = [
            f"{c['dgp']}/{c['metric']} vs {c['comparator']}"
            for kind in ("repulsion", "sharing", "shrinkage")
            for c in ablations[kind]
            if c["comparator_wins"]
        ]
        print(f"  {name:<22} ablations lost: {', '.join(lost) if lost else 'none'}")

    for name, chosen in payload["selected_contrast_shrinkage"].items():
        print(f"\n  {name}: contrast strength chosen by cross-fitting")
        for dgp, record in chosen.items():
            print(f"    {dgp:<4} median {record['median']:>7.1f}  mean {record['mean']:>7.1f}")

    PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {PAYLOAD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
