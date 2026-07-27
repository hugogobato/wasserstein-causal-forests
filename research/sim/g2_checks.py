"""Executable WP9 Gate G2 checks for long-format simulation output.

This module deliberately refuses to issue a promise verdict from an
incomplete pilot.  It applies the thresholds in the theory plan to the
prespecified method names and reports the inner-sampling criterion separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.evaluation import REQUIRED_RESULT_FIELDS, validate_result_rows


FOCAL = "odcf_composite"
BOOTSTRAP = "odcf_composite_bootstrap"
MULTI_OUTPUT = "multi_output_dr_forest"
POINTWISE = "pointwise_causal_forest"
SCALAR = "scalar_causal_forest"
CHALLENGE_DGPS = ("D3", "D4", "D5", "D8")
PRIMARY_METRIC = {
    "D3": "ise_curve",
    "D4": "rmse_functional_0",
    "D5": "worst_standardized_error",
    "D8": "worst_standardized_error",
}


def load_rows(path: str | Path) -> list[dict]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError("simulation output must be a JSON list of rows")
    validate_result_rows(rows)
    return rows


def _means(rows: list[dict]) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "runtime_seconds":
            continue
        key = (
            row["dgp_id"], row["observation_regime"],
            row["method"], row["metric"],
        )
        grouped[key].append(float(row["value"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _value(means, dgp, regime, method, metric):
    return means.get((dgp, regime, method, metric))


def check_g2(rows: list[dict]) -> dict:
    means = _means(rows)
    seeds = defaultdict(set)
    for row in rows:
        if row["metric"] != "runtime_seconds":
            seeds[(row["dgp_id"], row["observation_regime"])].add(row["seed"])

    complete_cells = {
        key: len(value) >= 30 for key, value in seeds.items()
    }
    challenge_improvements = {}
    for dgp in CHALLENGE_DGPS:
        regime = "feasible_growing_inner" if dgp == "D8" else "oracle_latent"
        metric = PRIMARY_METRIC[dgp]
        focal = _value(means, dgp, regime, FOCAL, metric)
        baseline_values = [
            value for (dd, rr, method, mm), value in means.items()
            if dd == dgp and rr == regime and mm == metric
            and method not in {FOCAL, BOOTSTRAP}
        ]
        if focal is None or not baseline_values:
            challenge_improvements[dgp] = None
        else:
            best = min(baseline_values)
            challenge_improvements[dgp] = (best - focal) / max(best, 1e-12)

    easy_d1 = {}
    focal_curve = _value(means, "D1", "oracle_latent", FOCAL, "ise_curve")
    pointwise_curve = _value(
        means, "D1", "oracle_latent", POINTWISE, "ise_curve"
    )
    easy_d1["curve"] = (
        focal_curve is not None and pointwise_curve is not None
        and focal_curve <= 1.10 * pointwise_curve
    )
    functional_checks = []
    for j in range(3):
        focal = _value(means, "D1", "oracle_latent", FOCAL, f"rmse_functional_{j}")
        scalar = _value(
            means, "D1", "oracle_latent", SCALAR, f"rmse_functional_{j}"
        )
        functional_checks.append(
            focal is not None and scalar is not None and focal <= 1.10 * scalar
        )
    easy_d1["functional"] = all(functional_checks)

    different_from_multi_output = False
    for key, focal_value in means.items():
        dgp, regime, method, metric = key
        if method != FOCAL or metric == "runtime_seconds":
            continue
        incumbent = means.get((dgp, regime, MULTI_OUTPUT, metric))
        if incumbent is not None and abs(focal_value - incumbent) > 1e-12:
            different_from_multi_output = True
            break

    oracle = _value(means, "D8", "oracle_latent", FOCAL, "worst_standardized_error")
    feasible = _value(means, "D8", "feasible_growing_inner", FOCAL, "worst_standardized_error")
    corrected = _value(means, "D8", "feasible_growing_inner", BOOTSTRAP, "worst_standardized_error")
    if oracle is None or feasible is None or corrected is None:
        inner_sampling = None
    else:
        plain_gap = abs(feasible - oracle)
        corrected_gap = abs(corrected - oracle)
        inner_sampling = {
            "plain_gap": plain_gap,
            "bootstrap_gap": corrected_gap,
            "reduced_by_20_percent": corrected_gap <= 0.80 * plain_gap,
        }

    required_cells = [
        (dgp, "feasible_growing_inner" if dgp == "D8" else "oracle_latent")
        for dgp in CHALLENGE_DGPS
    ] + [("D1", "oracle_latent")]
    enough_replicates = all(complete_cells.get(cell, False) for cell in required_cells)
    improvements = [value for value in challenge_improvements.values() if value is not None]
    result = {
        "enough_replicates": enough_replicates,
        "challenge_improvements": challenge_improvements,
        "criterion_1_two_fifteen_percent_wins": len(
            [value for value in improvements if value >= 0.15]
        ) >= 2,
        "criterion_2_easy_D1": bool(easy_d1["curve"] and easy_d1["functional"]),
        "criterion_3_coherent_pareto": len(
            [value for value in improvements if value >= 0.15]
        ) >= 2,
        "criterion_4_inner_sampling": inner_sampling,
        "criterion_5_not_exact_multi_output": different_from_multi_output,
        "complete_cells": {str(key): value for key, value in complete_cells.items()},
    }
    result["gate_pass"] = bool(
        enough_replicates
        and result["criterion_1_two_fifteen_percent_wins"]
        and result["criterion_2_easy_D1"]
        and result["criterion_3_coherent_pareto"]
        and result["criterion_5_not_exact_multi_output"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    args = parser.parse_args()
    result = check_g2(load_rows(args.results_json))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
