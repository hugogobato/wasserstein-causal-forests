#!/usr/bin/env python3
"""Independent recomputation of the G3 gate flags.

Run from the repository root:

    python research/checks/g3_gate_flags.py

WP3-B3 requires that an independent script reproduce the gate flags the memo
reports. This one deliberately shares nothing with
`wasserstein_causal_forests.g3.analysis` except the frozen thresholds, which
must be identical for the comparison to mean anything. The paired statistics,
the cell aggregation, and the rule logic are written again here from the
merged table, so a bug in either implementation shows up as a disagreement
rather than as two copies of the same wrong answer.

Exits nonzero if the two implementations disagree on any rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.manifest import (  # noqa: E402
    DECISION_SE_MULTIPLE,
    GATE_RULES,
    HIGHER_IS_BETTER,
)

MERGED = ROOT / "results" / "merged" / "main_results.parquet"
REPAIRED = ROOT / "results" / "merged_repair" / "main_results.parquet"
ORIGINAL_CAUSAL_DRF = ROOT / "results" / "merged_original_causal_drf" / "main_results.parquet"
OUTPUT = ROOT / "results" / "merged" / "gate_flags_independent.json"
REPAIR_OUTPUT = ROOT / "results" / "merged_repair" / "gate_flags_independent.json"
CLAIMANT = "cwdb_v1"

#: Rule 1 measures the claimant against the best of the roster the first
#: tournament ran. Restated here rather than imported, because the point of this
#: script is that a bug in the analysis module shows up as a disagreement.
FROZEN_ROSTER = (
    "cwdb_v1", "cwdb_v0", "cwdb_v1_noshrink", "sqw2_booster",
    "pta_s", "pta_f", "wdrft", "causal_drf",
)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        alternative = path.with_suffix(".jsonl")
        if alternative.exists():
            return [
                json.loads(line)
                for line in alternative.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        raise SystemExit(f"no merged results at {path}")
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def series(rows, metric, grid, dgp, method, n_particles=None, target_id=None):
    """Seed -> value, averaging arm rows, for one method, target and regime."""

    by_seed: dict[int, list[float]] = {}
    for row in rows:
        if (
            row["grid"] != grid
            or row["dgp"] != dgp
            or row["method"] != method
            or row["metric"] != metric
            or row["status"] != "ok"
            or row["value"] is None
            or (target_id is not None and row["target_id"] != target_id)
        ):
            continue
        if n_particles is not None and row["n_particles"] != n_particles:
            continue
        by_seed.setdefault(row["seed"], []).append(float(row["value"]))
    return {seed: float(np.mean(values)) for seed, values in by_seed.items()}


def paired(rows, metric, grid, dgp, comparator, target_id=None, claimant=CLAIMANT):
    left = series(rows, metric, grid, dgp, claimant, target_id=target_id)
    right = series(rows, metric, grid, dgp, comparator, target_id=target_id)
    shared = sorted(set(left) & set(right))
    if len(shared) < 3:
        return None
    sign = -1.0 if metric in HIGHER_IS_BETTER else 1.0
    differences = np.array([sign * (left[s] - right[s]) for s in shared])
    mean = float(differences.mean())
    error = float(differences.std(ddof=1) / np.sqrt(differences.size))
    return {
        "dgp": dgp,
        "metric": metric,
        "target_id": target_id,
        "mean": mean,
        "standard_error": error,
        "n_seeds": len(shared),
        # See `analysis.paired_comparison`: a zero paired standard error is a
        # perfectly consistent difference, so the sign decides.
        "win": bool(mean < -DECISION_SE_MULTIPLE * error),
    }


def mean_of(rows, metric, grid, dgp, method):
    values = list(series(rows, metric, grid, dgp, method).values())
    return float(np.mean(values)) if values else float("nan")


def replace_causal_drf(rows):
    """Use the authors' rerun for Causal-DRF while preserving other rows."""

    if not ORIGINAL_CAUSAL_DRF.exists():
        raise SystemExit(f"no original-code Causal-DRF results at {ORIGINAL_CAUSAL_DRF}")
    original = read_rows(ORIGINAL_CAUSAL_DRF)
    return [row for row in rows if row["method"] != "causal_drf"] + original


def recompute(rows, claimant=CLAIMANT) -> dict:
    flags: dict[str, dict] = {}

    rule = GATE_RULES["rule_1_correctness"]
    d0 = mean_of(rows, "mean_quantile_rmse", "main", "D0", claimant)
    d2 = mean_of(rows, "mean_quantile_rmse", "main", "D2", claimant)
    others = [
        mean_of(rows, "mean_quantile_rmse", "main", "D2", method)
        for method in sorted({r["method"] for r in rows if r["grid"] == "main"})
        if method != claimant and method in FROZEN_ROSTER
    ]
    others = [value for value in others if np.isfinite(value)]
    best = min(others) if others else float("inf")
    flags["rule_1_correctness"] = {
        "passed": bool(
            d0 <= rule["d0_max_mean_quantile_rmse"]
            and d2 <= rule["d2_max_mean_quantile_rmse"]
            and (d2 / best if best > 0 else np.inf)
            <= rule["d2_max_false_effect_ratio"]
        )
    }

    rule = GATE_RULES["rule_2_law_advantage"]
    wins = [
        result
        for dgp in rule["eligible_dgps"]
        if (result := paired(rows, rule["metric"], "main", dgp, rule["comparator"],
                             claimant=claimant))
        and result["win"]
    ]
    flags["rule_2_law_advantage"] = {
        "passed": len(wins) >= rule["min_wins"],
        "winning_dgps": [w["dgp"] for w in wins],
    }

    # Per target, not per metric: `tcate_functional_rmse` spans four grid
    # functionals and the methods do not all report the same ones.
    rule = GATE_RULES["rule_3_transfer"]
    targets = {
        (metric, dgp): sorted({
            r["target_id"] for r in rows
            if r["metric"] == metric and r["grid"] == "main" and r["dgp"] == dgp
        })
        for metric in rule["metrics"] for dgp in rule["eligible_dgps"]
    }
    transfer = [
        result
        for (metric, dgp), ids in targets.items()
        for tid in ids
        if (result := paired(rows, metric, "main", dgp, rule["comparator"], tid,
                             claimant=claimant))
        and result["win"]
    ]
    flags["rule_3_transfer"] = {
        "passed": len(transfer) >= rule["min_wins"],
        "winning_targets": [f"{t['target_id']}@{t['dgp']}" for t in transfer],
    }

    rule = GATE_RULES["rule_4_beats_direct_learner"]
    against_pta = []
    for item in transfer:
        supplied = any(
            r["method"] == rule["comparator"] and r["metric"] == item["metric"]
            and r["target_id"] == item["target_id"] and r["grid"] == "main"
            and r["dgp"] == item["dgp"] and r["status"] == "ok"
            and r["value"] is not None
            for r in rows
        )
        if not supplied:
            # The comparator cannot estimate this target at all: a capability
            # win for the claimant, counted and labelled as such.
            against_pta.append({**item, "kind": "capability"})
            continue
        result = paired(rows, item["metric"], "main", item["dgp"],
                        rule["comparator"], item["target_id"], claimant=claimant)
        if result and result["win"]:
            against_pta.append({**result, "kind": "accuracy"})
    flags["rule_4_beats_direct_learner"] = {
        "passed": len(against_pta) >= rule["min_wins"],
        "winning_targets": [f"{t['target_id']}@{t['dgp']}" for t in against_pta],
        "n_capability_wins": sum(1 for t in against_pta if t["kind"] == "capability"),
    }

    rule = GATE_RULES["rule_5_no_collapse"]
    coverage = mean_of(rows, "mode_coverage", "main", "D6", claimant)
    support = mean_of(rows, "diagnostic_effective_support", "main", "D6", claimant)
    flags["rule_5_no_collapse"] = {
        "passed": bool(
            coverage >= rule["d6_min_mode_coverage"]
            and (support / 10.0) >= rule["min_effective_support_fraction"]
        )
    }

    rule = GATE_RULES["rule_6_cost"]
    claimant_runtimes = [
        float(r["value"]) for r in rows
        if r["metric"] == "runtime" and r["method"] == claimant
        and r["status"] == "ok" and r["value"] is not None
    ]
    incumbent = [
        float(r["value"]) for r in rows
        if r["metric"] == "runtime" and r["method"] == "causal_drf"
        and r["status"] == "ok" and r["value"] is not None
    ]
    ratio = (
        float(np.median(claimant_runtimes)) / float(np.median(incumbent))
        if claimant_runtimes and incumbent and np.median(incumbent) > 0
        else float("inf")
    )
    flags["rule_6_cost"] = {
        "passed": ratio <= rule["max_runtime_ratio_to_causal_drf"],
        "runtime_ratio": ratio,
    }

    names = list(flags)
    passed = [name for name in names if flags[name]["passed"]]
    flags["summary"] = {
        "n_rules": len(names),
        "n_passed": len(passed),
        "rules_failed": [n for n in names if not flags[n]["passed"]],
        "verdict": "GO" if len(passed) == len(names) else "NOT-GO",
    }
    return flags


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--claimant",
        default=CLAIMANT,
        help="method under test; a repair variant also needs --with-repair",
    )
    parser.add_argument(
        "--with-repair",
        action="store_true",
        help="append the repair table, leaving the frozen rows untouched",
    )
    parser.add_argument(
        "--original-causal-drf",
        action="store_true",
        help="replace historical Causal-DRF rows with the authors' rerun",
    )
    arguments = parser.parse_args()

    rows = read_rows(MERGED)
    if arguments.original_causal_drf:
        rows = replace_causal_drf(rows)
    if arguments.with_repair:
        rows = rows + read_rows(REPAIRED)
    independent = recompute(rows, claimant=arguments.claimant)

    from wasserstein_causal_forests.g3.analysis import compute_gate_flags

    primary = compute_gate_flags(rows, claimant=arguments.claimant)

    disagreements = []
    for name in independent:
        if name == "summary":
            continue
        if independent[name]["passed"] != primary[name]["passed"]:
            disagreements.append(
                f"{name}: independent={independent[name]['passed']} "
                f"primary={primary[name]['passed']}"
            )
    if independent["summary"]["verdict"] != primary["summary"]["verdict"]:
        disagreements.append(
            f"verdict: independent={independent['summary']['verdict']} "
            f"primary={primary['summary']['verdict']}"
        )

    certificate = {
        "check": "g3_gate_flags",
        "claimant": arguments.claimant,
        "with_repair": arguments.with_repair,
        "independent": independent,
        "primary_verdict": primary["summary"]["verdict"],
        "disagreements": disagreements,
        "status": "PASS" if not disagreements else "FAIL",
    }
    output = REPAIR_OUTPUT if arguments.with_repair else OUTPUT
    if arguments.with_repair:
        output = output.with_name(f"gate_flags_independent_{arguments.claimant}.json")
    if arguments.original_causal_drf:
        output = output.with_name(output.stem + "_original_causal_drf.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
    print(json.dumps(certificate, indent=2))
    return 0 if not disagreements else 1


if __name__ == "__main__":
    raise SystemExit(main())
