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

# Every challenge DGP is judged in the noisy regime.  Under oracle_latent the
# scores are effectively noiseless and the errors sit at 1e-7..1e-16, so a
# ratio computed there measures floating-point behavior, not performance.
CHALLENGE_REGIME = "feasible_growing_inner"

# Criterion 5 asks whether the focal method's behavior is reproduced by an
# off-the-shelf competitor with the same score and scaling.  A raw inequality
# of floating-point means is not evidence of a real difference, so the check
# below requires a difference that is material relative to the incumbent, and
# it now includes the three prior-art incumbents, not only the multi-output
# forest.
INCUMBENTS = (
    "multi_output_dr_forest",
    "causal_drf_port",
    "focal_dr_meta_learner",
    "wasserstein_random_forest",
)
MATERIAL_DIFFERENCE = 0.05

# Criterion 3 requires a coherent Pareto gain rather than a win bought by
# sacrificing every other target.  The focal method may not be worse than the
# strongest baseline by more than this margin on any secondary metric of a
# challenge DGP it claims to win.
PARETO_TOLERANCE = 0.10
SECONDARY_METRICS = (
    "ise_curve", "rmse_functional_0", "rmse_functional_1",
    "rmse_functional_2", "worst_standardized_error",
)


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
    challenge_best_baseline = {}
    for dgp in CHALLENGE_DGPS:
        metric = PRIMARY_METRIC[dgp]
        focal = _value(means, dgp, CHALLENGE_REGIME, FOCAL, metric)
        baselines = {
            method: value
            for (dd, rr, method, mm), value in means.items()
            if dd == dgp and rr == CHALLENGE_REGIME and mm == metric
            and method not in {FOCAL, BOOTSTRAP}
        }
        if focal is None or not baselines:
            challenge_improvements[dgp] = None
            challenge_best_baseline[dgp] = None
        else:
            best_method = min(baselines, key=baselines.get)
            best = baselines[best_method]
            challenge_improvements[dgp] = (best - focal) / max(best, 1e-12)
            challenge_best_baseline[dgp] = best_method

    # Criterion 3: on every DGP the focal method claims to win, it must not be
    # materially worse than the strongest baseline on any secondary metric.
    pareto_violations = []
    for dgp, improvement in challenge_improvements.items():
        if improvement is None or improvement < 0.15:
            continue
        for metric in SECONDARY_METRICS:
            if metric == PRIMARY_METRIC[dgp]:
                continue
            focal = _value(means, dgp, CHALLENGE_REGIME, FOCAL, metric)
            others = [
                value for (dd, rr, method, mm), value in means.items()
                if dd == dgp and rr == CHALLENGE_REGIME and mm == metric
                and method not in {FOCAL, BOOTSTRAP}
            ]
            if focal is None or not others:
                continue
            best = min(others)
            if focal > (1.0 + PARETO_TOLERANCE) * max(best, 1e-12):
                pareto_violations.append(
                    {"dgp": dgp, "metric": metric,
                     "focal": focal, "best_baseline": best}
                )

    easy_d1 = {}
    focal_curve = _value(means, "D1", CHALLENGE_REGIME, FOCAL, "ise_curve")
    pointwise_curve = _value(
        means, "D1", CHALLENGE_REGIME, POINTWISE, "ise_curve"
    )
    easy_d1["curve"] = (
        focal_curve is not None and pointwise_curve is not None
        and focal_curve <= 1.10 * pointwise_curve
    )
    functional_checks = []
    for j in range(3):
        focal = _value(means, "D1", CHALLENGE_REGIME, FOCAL, f"rmse_functional_{j}")
        scalar = _value(
            means, "D1", CHALLENGE_REGIME, SCALAR, f"rmse_functional_{j}"
        )
        functional_checks.append(
            focal is not None and scalar is not None and focal <= 1.10 * scalar
        )
    easy_d1["functional"] = all(functional_checks)

    # Criterion 5, against every incumbent rather than only the multi-output
    # forest, and requiring a materially different number rather than any
    # nonzero floating-point gap.
    incumbent_gaps = {}
    for incumbent_name in INCUMBENTS:
        gaps = []
        for key, focal_value in means.items():
            dgp, regime, method, metric = key
            if method != FOCAL or metric == "runtime_seconds":
                continue
            incumbent = means.get((dgp, regime, incumbent_name, metric))
            if incumbent is None:
                continue
            gaps.append(abs(focal_value - incumbent) / max(abs(incumbent), 1e-12))
        incumbent_gaps[incumbent_name] = max(gaps) if gaps else None
    measured_gaps = [gap for gap in incumbent_gaps.values() if gap is not None]
    different_from_incumbents = bool(measured_gaps) and all(
        gap >= MATERIAL_DIFFERENCE for gap in measured_gaps
    )

    oracle = _value(means, "D8", "oracle_latent", FOCAL, "worst_standardized_error")
    feasible = _value(means, "D8", CHALLENGE_REGIME, FOCAL, "worst_standardized_error")
    corrected = _value(means, "D8", CHALLENGE_REGIME, BOOTSTRAP, "worst_standardized_error")
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

    required_cells = [(dgp, CHALLENGE_REGIME) for dgp in CHALLENGE_DGPS]
    required_cells += [("D1", CHALLENGE_REGIME), ("D8", "oracle_latent")]
    enough_replicates = all(complete_cells.get(cell, False) for cell in required_cells)
    improvements = [value for value in challenge_improvements.values() if value is not None]
    wins = [value for value in improvements if value >= 0.15]
    result = {
        "enough_replicates": enough_replicates,
        "challenge_regime": CHALLENGE_REGIME,
        "challenge_improvements": challenge_improvements,
        "challenge_best_baseline": challenge_best_baseline,
        "criterion_1_two_fifteen_percent_wins": len(wins) >= 2,
        "criterion_2_easy_D1": bool(easy_d1["curve"] and easy_d1["functional"]),
        "criterion_3_coherent_pareto": bool(len(wins) >= 2 and not pareto_violations),
        "criterion_3_pareto_violations": pareto_violations,
        "criterion_4_inner_sampling": inner_sampling,
        "criterion_5_not_replicated_by_incumbents": different_from_incumbents,
        "criterion_5_incumbent_max_relative_gaps": incumbent_gaps,
        "complete_cells": {str(key): value for key, value in complete_cells.items()},
    }
    result["gate_pass"] = bool(
        enough_replicates
        and result["criterion_1_two_fifteen_percent_wins"]
        and result["criterion_2_easy_D1"]
        and result["criterion_3_coherent_pareto"]
        and result["criterion_5_not_replicated_by_incumbents"]
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
