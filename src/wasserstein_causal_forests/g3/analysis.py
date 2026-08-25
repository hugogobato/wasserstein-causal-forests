"""WP3-B3: paired comparisons, mechanism ablations, and the G3 gate flags.

Comparisons are paired by replication. Every method in a replication sees the
same training sample and the same test design, so the seed-level difference
removes the replication effect and its standard error is the honest one; an
unpaired comparison across such correlated runs would badly overstate the
uncertainty and hide real differences in both directions.

Nothing here chooses a threshold. The primary metric, the eligible regimes, the
decision multiple, and every numeric cutoff come from `GATE_RULES` in the frozen
manifest, which was written before the first decisive seed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import (
    DECISION_SE_MULTIPLE,
    FROZEN_G3_METHODS,
    GATE_RULES,
    HIGHER_IS_BETTER,
    PRIMARY_LAW_METRIC,
)

CLAIMANT = "cwdb_v1"


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _cell_value(
    rows: list[dict[str, Any]], metric: str, target_id: str | None = None
) -> float | None:
    """Collapse a cell's rows for one metric, optionally one target, into a number.

    Arm-specific metrics contribute one row per arm; the tournament compares
    methods on their average over the two arms, since no claim here is about a
    single arm in isolation.

    Averaging across *targets* is different and is only safe when both methods
    report the same ones. A law method reports all four grid functionals while
    PTA reports only the two in its manifest, so a metric-level average would
    charge C-WDB for the two hardest targets and PTA for neither. Callers
    comparing across methods must pass `target_id`.
    """

    values = [
        row["value"]
        for row in rows
        if row["metric"] == metric
        and row["status"] == "ok"
        and row["value"] is not None
        and (target_id is None or row["target_id"] == target_id)
    ]
    return float(np.mean(values)) if values else None


def index_by_cell(rows: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["grid"], row["dgp"], row["n_train"], row["n_grid"],
            row["n_particles"], row["method"], row["seed"],
        )
        grouped[key].append(row)
    return grouped


def paired_comparison(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    claimant: str = CLAIMANT,
    comparator: str,
    grid: str,
    dgp: str,
    target_id: str | None = None,
    n_train: int | None = None,
    n_grid: int | None = None,
    n_particles: int | None = None,
) -> dict[str, Any] | None:
    """Seed-paired difference `claimant - comparator` for one metric and regime.

    Pass `target_id` whenever the metric covers several targets, so both sides
    are scored on the same one.
    """

    grouped = index_by_cell(rows)
    # The pairing unit is the full design point, not the seed. Keying by seed
    # alone silently overwrites one sample size with the other whenever a
    # comparison pools them, which halves the sample and makes the surviving
    # half depend on dictionary order rather than on the design.
    values: dict[str, dict[tuple, float]] = {claimant: {}, comparator: {}}
    for key, cell_rows in grouped.items():
        (
            row_grid, row_dgp, row_n, row_k, row_m, row_method, row_seed,
        ) = key
        if row_grid != grid or row_dgp != dgp or row_method not in values:
            continue
        if n_train is not None and row_n != n_train:
            continue
        if n_grid is not None and row_k != n_grid:
            continue
        if n_particles is not None and row_m != n_particles:
            continue
        value = _cell_value(cell_rows, metric, target_id)
        if value is not None:
            values[row_method][(row_n, row_k, row_m, row_seed)] = value

    seeds = sorted(set(values[claimant]) & set(values[comparator]))
    if len(seeds) < 3:
        return None
    difference = np.array(
        [values[claimant][s] - values[comparator][s] for s in seeds]
    )
    # A lower error is better, so a negative difference is a claimant win;
    # `mode_coverage` is inverted so every reported effect has the same sign
    # convention.
    sign = -1.0 if metric in HIGHER_IS_BETTER else 1.0
    difference = sign * difference
    mean = float(difference.mean())
    standard_error = float(difference.std(ddof=1) / np.sqrt(difference.size))
    return {
        "metric": metric,
        "target_id": target_id,
        "grid": grid,
        "dgp": dgp,
        "n_train": n_train,
        "n_grid": n_grid,
        "n_particles": n_particles,
        "claimant": claimant,
        "comparator": comparator,
        # Paired design points, which equals the seed count only when the
        # comparison is restricted to one sample size.
        "n_seeds": len(seeds),
        "claimant_mean": float(np.mean([values[claimant][s] for s in seeds])),
        "comparator_mean": float(np.mean([values[comparator][s] for s in seeds])),
        "paired_mean_difference": mean,
        "paired_standard_error": standard_error,
        "seed_win_fraction": float(np.mean(difference < 0.0)),
        # A zero paired standard error means every replication produced the
        # same difference. That is the strongest evidence available, not the
        # weakest, so the sign decides; requiring a positive standard error
        # would silently reject a perfectly consistent advantage.
        "claimant_wins": bool(mean < -DECISION_SE_MULTIPLE * standard_error),
        "comparator_wins": bool(mean > DECISION_SE_MULTIPLE * standard_error),
    }


def target_ids_for(
    rows: list[dict[str, Any]], metric: str, *, grid: str, dgp: str
) -> list[str]:
    """Every target this metric reports in a regime, whoever supplied it."""

    return sorted({
        row["target_id"]
        for row in rows
        if row["metric"] == metric and row["grid"] == grid and row["dgp"] == dgp
    })


def supplies_target(
    rows: list[dict[str, Any]], method: str, metric: str, target_id: str,
    *, grid: str, dgp: str,
) -> bool:
    """True when `method` produced any usable estimate of this target."""

    return any(
        row["method"] == method and row["metric"] == metric
        and row["target_id"] == target_id and row["grid"] == grid
        and row["dgp"] == dgp and row["status"] == "ok"
        and row["value"] is not None
        for row in rows
    )


def method_means(
    rows: list[dict[str, Any]], metric: str, *, grid: str, dgp: str,
    target_id: str | None = None,
) -> dict[str, dict[str, float]]:
    """Mean and standard error of one metric per method, for a regime."""

    grouped = index_by_cell(rows)
    collected: dict[str, list[float]] = defaultdict(list)
    for key, cell_rows in grouped.items():
        row_grid, row_dgp, _, _, _, row_method, _ = key
        if row_grid != grid or row_dgp != dgp:
            continue
        value = _cell_value(cell_rows, metric, target_id)
        if value is not None:
            collected[row_method].append(value)
    return {
        method: {
            "mean": float(np.mean(values)),
            "standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else float("nan"),
            "n": len(values),
        }
        for method, values in collected.items()
    }


def failure_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-method failed-cell counts, kept explicit rather than filtered away."""

    total: dict[str, set] = defaultdict(set)
    failed: dict[str, set] = defaultdict(set)
    for row in rows:
        total[row["method"]].add(row["cell_key"])
        if row["status"] == "failed":
            failed[row["method"]].add(row["cell_key"])
    return {
        method: {
            "n_cells": len(cells),
            "n_failed": len(failed.get(method, set())),
            "failure_rate": len(failed.get(method, set())) / len(cells),
        }
        for method, cells in sorted(total.items())
    }


def cost_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Median runtime and peak RAM per method, for the cost-commensurate rule."""

    runtime: dict[str, list[float]] = defaultdict(list)
    memory: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok" or row["value"] is None:
            continue
        if row["metric"] == "runtime":
            runtime[row["method"]].append(float(row["value"]))
        elif row["metric"] == "peak_ram":
            memory[row["method"]].append(float(row["value"]))
    return {
        method: {
            "median_runtime_seconds": float(np.median(values)),
            "max_runtime_seconds": float(np.max(values)),
            "median_peak_ram_mb": float(np.median(memory.get(method, [0.0]))),
        }
        for method, values in sorted(runtime.items())
    }


def compute_gate_flags(
    rows: list[dict[str, Any]], *, claimant: str = CLAIMANT
) -> dict[str, Any]:
    """Evaluate every frozen gate rule against the merged results.

    `claimant` names the method under test. It is a parameter so a repaired
    variant can be scored against the same frozen thresholds and the same
    comparator roster, without a second copy of the rules drifting from this one.
    """

    flags: dict[str, Any] = {}

    # ------------------------------------------------- rule 1: correctness
    rule = GATE_RULES["rule_1_correctness"]
    d0 = method_means(rows, "mean_quantile_rmse", grid="main", dgp="D0")
    d2 = method_means(rows, "mean_quantile_rmse", grid="main", dgp="D2")
    claimant_d0 = d0.get(claimant, {}).get("mean", float("inf"))
    claimant_d2 = d2.get(claimant, {}).get("mean", float("inf"))
    # The reference is the best of the frozen roster, never of whatever happens
    # to be in `rows`: a repair variant sitting alongside the claimant must not
    # be able to lower the bar it is judged against.
    baseline_d2 = [
        value["mean"]
        for method, value in d2.items()
        if method != claimant and method in FROZEN_G3_METHODS
    ]
    best_baseline_d2 = min(baseline_d2) if baseline_d2 else float("inf")
    ratio = claimant_d2 / best_baseline_d2 if best_baseline_d2 > 0 else float("inf")
    flags["rule_1_correctness"] = {
        "statement": rule["statement"],
        "d0_mean_quantile_rmse": claimant_d0,
        "d2_mean_quantile_rmse": claimant_d2,
        "d2_best_baseline": best_baseline_d2,
        "d2_false_effect_ratio": ratio,
        "passed": bool(
            claimant_d0 <= rule["d0_max_mean_quantile_rmse"]
            and claimant_d2 <= rule["d2_max_mean_quantile_rmse"]
            and ratio <= rule["d2_max_false_effect_ratio"]
        ),
    }

    # -------------------------------------------- rule 2: primary law metric
    rule = GATE_RULES["rule_2_law_advantage"]
    wins: list[dict[str, Any]] = []
    details = []
    for dgp in rule["eligible_dgps"]:
        comparison = paired_comparison(
            rows, rule["metric"], claimant=claimant,
            comparator=rule["comparator"], grid="main", dgp=dgp,
        )
        if comparison is None:
            continue
        details.append(comparison)
        if comparison["claimant_wins"]:
            wins.append(comparison)
    flags["rule_2_law_advantage"] = {
        "statement": rule["statement"],
        "metric": rule["metric"],
        "n_wins": len(wins),
        "min_wins": rule["min_wins"],
        "winning_dgps": [item["dgp"] for item in wins],
        "comparisons": details,
        "passed": len(wins) >= rule["min_wins"],
    }

    # ------------------------------------------------------ rule 3: transfer
    rule = GATE_RULES["rule_3_transfer"]
    # Compared per target rather than per metric. `tcate_functional_rmse`
    # spans four grid functionals, and averaging a method's error across a set
    # of targets is only a comparison when both methods report the same set.
    transfer_wins = []
    transfer_details = []
    for metric in rule["metrics"]:
        for dgp in rule["eligible_dgps"]:
            for target in target_ids_for(rows, metric, grid="main", dgp=dgp):
                comparison = paired_comparison(
                    rows, metric, claimant=claimant,
                    comparator=rule["comparator"], grid="main", dgp=dgp,
                    target_id=target,
                )
                if comparison is None:
                    continue
                transfer_details.append(comparison)
                if comparison["claimant_wins"]:
                    transfer_wins.append(comparison)
    flags["rule_3_transfer"] = {
        "statement": rule["statement"],
        "n_wins": len(transfer_wins),
        "min_wins": rule["min_wins"],
        "winning_targets": [
            f"{item['target_id']}@{item['dgp']}" for item in transfer_wins
        ],
        "comparisons": transfer_details,
        "passed": len(transfer_wins) >= rule["min_wins"],
    }

    # ----------------------------------------- rule 4: beats the direct learner
    rule = GATE_RULES["rule_4_beats_direct_learner"]
    # Evaluated only on the targets rule 3 actually won, so a method cannot pass
    # by beating PTA-S somewhere it lost to Causal-DRF.
    #
    # A win comes in two kinds and they are recorded separately. On a target
    # PTA-S estimates, the comparison is a paired accuracy test. On a target it
    # cannot estimate at all, because the functional was named after training
    # and is outside its frozen manifest, C-WDB wins on capability: it answers a
    # question the direct target learner cannot answer without refitting every
    # head. That is the substance of the full-law claim, so it counts, but it is
    # labelled `capability` rather than `accuracy` so no reader mistakes one for
    # the other.
    pta_wins = []
    pta_details = []
    for item in transfer_wins:
        target = item["target_id"]
        if not supplies_target(rows, rule["comparator"], item["metric"], target,
                               grid="main", dgp=item["dgp"]):
            claimant_error = method_means(
                rows, item["metric"], grid="main", dgp=item["dgp"],
                target_id=target,
            ).get(claimant, {})
            record = {
                "metric": item["metric"],
                "target_id": target,
                "dgp": item["dgp"],
                "kind": "capability",
                "comparator": rule["comparator"],
                "comparator_supplies_target": False,
                "claimant_mean": claimant_error.get("mean"),
                "claimant_wins": True,
                "note": (
                    "the comparator reports no estimate of this target; the "
                    "functional was declared after training and lies outside "
                    "its frozen manifest"
                ),
            }
            pta_details.append(record)
            pta_wins.append(record)
            continue
        comparison = paired_comparison(
            rows, item["metric"], claimant=claimant,
            comparator=rule["comparator"], grid="main", dgp=item["dgp"],
            target_id=target,
        )
        if comparison is None:
            continue
        comparison["kind"] = "accuracy"
        comparison["comparator_supplies_target"] = True
        pta_details.append(comparison)
        if comparison["claimant_wins"]:
            pta_wins.append(comparison)
    flags["rule_4_beats_direct_learner"] = {
        "statement": rule["statement"],
        "n_wins": len(pta_wins),
        "n_accuracy_wins": sum(1 for w in pta_wins if w.get("kind") == "accuracy"),
        "n_capability_wins": sum(1 for w in pta_wins if w.get("kind") == "capability"),
        "min_wins": rule["min_wins"],
        "winning_targets": [f"{i['target_id']}@{i['dgp']}" for i in pta_wins],
        "comparisons": pta_details,
        "passed": len(pta_wins) >= rule["min_wins"],
    }

    # -------------------------------------------------- rule 5: no collapse
    rule = GATE_RULES["rule_5_no_collapse"]
    coverage = method_means(rows, "mode_coverage", grid="main", dgp="D6")
    support = method_means(
        rows, "diagnostic_effective_support", grid="main", dgp="D6"
    )
    claimant_coverage = coverage.get(claimant, {}).get("mean", 0.0)
    claimant_support = support.get(claimant, {}).get("mean", 0.0)
    n_particles = 10
    support_fraction = claimant_support / n_particles if n_particles else 0.0
    flags["rule_5_no_collapse"] = {
        "statement": rule["statement"],
        "d6_mode_coverage": claimant_coverage,
        "effective_support": claimant_support,
        "effective_support_fraction": support_fraction,
        "comparator_sqw2_mode_coverage": method_means(
            rows, "mode_coverage", grid="smallk", dgp="D6"
        ).get("sqw2_booster", {}).get("mean"),
        "passed": bool(
            claimant_coverage >= rule["d6_min_mode_coverage"]
            and support_fraction >= rule["min_effective_support_fraction"]
        ),
    }

    # ------------------------------------------------------- rule 6: cost
    rule = GATE_RULES["rule_6_cost"]
    costs = cost_summary(rows)
    claimant_cost = costs.get(claimant, {}).get("median_runtime_seconds", float("inf"))
    incumbent_cost = costs.get("causal_drf", {}).get("median_runtime_seconds", 0.0)
    cost_ratio = claimant_cost / incumbent_cost if incumbent_cost > 0 else float("inf")
    flags["rule_6_cost"] = {
        "statement": rule["statement"],
        "claimant_median_runtime_seconds": claimant_cost,
        "causal_drf_median_runtime_seconds": incumbent_cost,
        "runtime_ratio": cost_ratio,
        "max_allowed": rule["max_runtime_ratio_to_causal_drf"],
        "passed": cost_ratio <= rule["max_runtime_ratio_to_causal_drf"],
    }

    rule_names = list(flags)
    passed = [name for name in rule_names if flags[name]["passed"]]
    flags["summary"] = {
        "n_rules": len(rule_names),
        "n_passed": len(passed),
        "rules_passed": passed,
        "rules_failed": [n for n in rule_names if not flags[n]["passed"]],
        "verdict": "GO" if len(passed) == len(rule_names) else "NOT-GO",
    }
    return flags


def mechanism_ablations(
    rows: list[dict[str, Any]], *, claimant: str = CLAIMANT
) -> dict[str, Any]:
    """The four preregistered mechanism ablations."""

    ablations: dict[str, Any] = {}

    ablations["repulsion"] = [
        comparison
        for dgp in ("D1", "D6")
        for metric in (PRIMARY_LAW_METRIC, "mode_coverage", "arm_energy_risk")
        if (comparison := paired_comparison(
            rows, metric, claimant=claimant, comparator="sqw2_booster",
            grid="smallk", dgp=dgp,
        )) is not None
    ]
    ablations["sharing"] = [
        comparison
        for dgp in ("D3", "D4")
        for metric in ("mean_quantile_rmse", PRIMARY_LAW_METRIC)
        if (comparison := paired_comparison(
            rows, metric, claimant=claimant, comparator="cwdb_v0",
            grid="main", dgp=dgp,
        )) is not None
    ]
    ablations["shrinkage"] = [
        comparison
        for dgp in ("D2", "D8")
        for metric in ("mean_quantile_rmse", PRIMARY_LAW_METRIC)
        if (comparison := paired_comparison(
            rows, metric, claimant=claimant, comparator="cwdb_v1_noshrink",
            grid="shrinkage", dgp=dgp,
        )) is not None
    ]

    particles: dict[str, Any] = {}
    for dgp in ("D1", "D6"):
        series = {}
        for n_particles in (2, 5, 10, 25):
            values = method_means(rows, "arm_energy_risk", grid="particles", dgp=dgp)
            # method_means ignores M, so re-filter by particle count directly.
            selected = [
                _cell_value(cell_rows, "arm_energy_risk")
                for key, cell_rows in index_by_cell(rows).items()
                if key[0] == "particles" and key[1] == dgp
                and key[4] == n_particles and key[5] == claimant
            ]
            selected = [v for v in selected if v is not None]
            if selected:
                series[str(n_particles)] = {
                    "mean_excess_energy_risk": float(np.mean(selected)),
                    "standard_error": float(
                        np.std(selected, ddof=1) / np.sqrt(len(selected))
                    ) if len(selected) > 1 else float("nan"),
                    "n": len(selected),
                }
            del values
        particles[dgp] = series
    ablations["particles"] = particles
    return ablations


def worst_regime(
    rows: list[dict[str, Any]], metric: str, *, comparator: str, grid: str = "main"
) -> dict[str, Any]:
    """The regime where the claimant loses by the most, for the failure region."""

    comparisons = []
    for dgp in sorted({row["dgp"] for row in rows if row["grid"] == grid}):
        comparison = paired_comparison(
            rows, metric, comparator=comparator, grid=grid, dgp=dgp
        )
        if comparison is not None:
            comparisons.append(comparison)
    if not comparisons:
        return {}
    ranked = sorted(comparisons, key=lambda c: -c["paired_mean_difference"])
    return {
        "metric": metric,
        "comparator": comparator,
        "worst": ranked[0],
        "best": ranked[-1],
        "all": ranked,
    }
