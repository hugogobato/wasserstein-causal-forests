"""Phase 5.5 Stage 2 analysis: the `cwdb_mutau` frozen comparison.

Applies the rules of `research/simulation_preregistration_phase55_stage2.md`,
which was frozen before the first Stage 2 cell ran. Nothing here chooses a
threshold; every constant below is copied from that document, which in turn
copies them from the G3 gate rules.

Two conventions this file enforces because the preregistration requires them:

* the claimant's analysis surface is the union of the Stage 2 table with the
  Stage 1 `cwdb_mutau` rows, whose cell keys are disjoint by construction, so
  every regime reaches the twenty seeds the incumbents already carry;
* runtime, and therefore the rule 6 ratio, is read from Stage 2 rows only,
  because `mutau.py` was made roughly five times faster after Stage 1 ran
  without any accuracy metric moving (section 6.1 of the preregistration).
  Pooling cost across the two implementations would report a speed no single
  implementation ever had.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wasserstein_causal_forests.g3.analysis import method_means, paired_comparison

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results" / "merged" / "main_results.parquet"
REPAIR = ROOT / "results" / "merged_repair" / "main_results.parquet"
PHASE55 = ROOT / "results" / "merged_phase55" / "main_results.parquet"
STAGE2 = ROOT / "results" / "merged_phase55_stage2" / "main_results.parquet"
ORIGINAL_CAUSAL_DRF = ROOT / "results" / "merged_original_causal_drf" / "main_results.parquet"
ORIGINAL_DRF = ROOT / "results" / "merged_original_drf" / "main_results.parquet"
PAYLOAD = ROOT / "results" / "merged_phase55_stage2" / "stage2_analysis_payload.json"

CLAIMANT = "cwdb_mutau"
DECISION_MULTIPLE = 2.0

D0_CAP = 0.15
D2_CAP = 0.15
D2_RATIO_CAP = 1.25
D6_MODE_COVERAGE_FLOOR = 0.90
SUPPORT_FLOOR = 0.60
COST_RATIO_CAP = 60.0
N_PARTICLES = 10

ALL_DGPS = ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")
STAGE2_SEEDS = tuple(range(20))

#: The frozen baselines rule 1 measures the D2 false-effect ratio against.
FROZEN_BASELINES = ("pta_s", "cwdb_v0", "cwdb_v1", "wdrft", "causal_drf")

#: Comparators, and the contract limit that decides which targets each may be
#: compared on. A mean-only comparator is never scored on a law metric.
LAW_COMPARATORS = ("cwdb_r3_cvridge", "cwdb_v1", "causal_drf", "drf", "wdrft")
CONTRAST_COMPARATORS = ("cwdb_r3_cvridge", "pta_s", "cwdb_v1", "causal_drf", "drf")

CONTRAST_METRICS = ("mean_quantile_rmse", "barycenter_rmse",
                    "reference_effect_rmse", "reference_tcate_rmse")
LAW_METRICS = ("kernel_law_error", "arm_energy_risk", "mode_coverage",
               "tail_calibration")
FUNCTIONAL_METRICS = ("tcate_functional_rmse", "tate_functional_rmse")


def load(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def rows_for(
    rows: list[dict], *, grid: str = "main", method: str | None = None,
    dgp: str | None = None, n_train: int | None = None,
    metric: str | None = None, seeds: tuple[int, ...] | None = None,
) -> list[dict]:
    return [
        row for row in rows
        if row["grid"] == grid
        and (method is None or row["method"] == method)
        and (dgp is None or row["dgp"] == dgp)
        and (n_train is None or row["n_train"] == n_train)
        and (metric is None or row["metric"] == metric)
        and (seeds is None or row["seed"] in seeds)
    ]


def verdict(comparison: dict | None) -> str:
    """Claimant win, comparator win, or tie.

    The flags come from `g3.analysis` rather than being recomputed here, so this
    stage uses the same decision rule as every earlier one, including its
    treatment of a zero paired standard error (every replication agreeing is the
    strongest evidence available, so the sign decides).
    """

    if comparison is None:
        return "absent"
    if comparison["claimant_wins"]:
        return "claimant"
    if comparison["comparator_wins"]:
        return "comparator"
    return "tie"


def annotate(comparison: dict | None) -> dict | None:
    if comparison is None:
        return None
    error = comparison["paired_standard_error"]
    comparison["se_ratio"] = (
        float(comparison["paired_mean_difference"] / error)
        if error and np.isfinite(error) else float("nan")
    )
    comparison["verdict"] = verdict(comparison)
    return comparison


def cell_median(rows: list[dict], metric: str) -> float:
    values = [row["value"] for row in rows
              if row["metric"] == metric and row["status"] == "ok"]
    return float(np.median(values)) if values else float("nan")


def main() -> None:
    stage2 = load(STAGE2)
    if not stage2:
        raise SystemExit(
            "Stage 2 merged table not found; run `merge --track phase55_stage2` first"
        )
    stage1 = load(PHASE55)
    # The union the preregistration declares: Stage 1 contributes only the
    # claimant's rows, because no other Stage 1 method continues.
    claimant_rows = [r for r in stage1 if r["method"] == CLAIMANT] + stage2
    # The retired project-local Causal-DRF and DRF drivers are not comparators.
    # They must be dropped by source file, not by contract id: the original-code
    # reruns carry the same `G3-MAIN-v1` id, so filtering on the id removes the
    # comparator instead of the incumbent and leaves rule 2 silently unevaluable.
    frozen = [r for r in load(FROZEN) if r["method"] not in ("causal_drf", "drf")]
    combined = (
        frozen + load(REPAIR)
        + load(ORIGINAL_CAUSAL_DRF) + load(ORIGINAL_DRF)
        + claimant_rows
    )

    payload: dict = {
        "stage": 2,
        "decision_multiple": DECISION_MULTIPLE,
        "seeds": list(STAGE2_SEEDS),
        "preregistration": "research/simulation_preregistration_phase55_stage2.md",
    }

    seed_counts = {}
    for dgp in ALL_DGPS:
        seeds = {r["seed"] for r in rows_for(claimant_rows, dgp=dgp)}
        seed_counts[dgp] = len(seeds)
    payload["claimant_seed_counts"] = seed_counts

    # ------------------------------------------------ rule 1, correctness
    rule1: dict = {}
    for n in (500, 1000):
        d0 = method_means(
            rows_for(claimant_rows, dgp="D0", n_train=n),
            "mean_quantile_rmse", grid="main", dgp="D0",
        ).get(CLAIMANT, {})
        d2 = method_means(
            rows_for(claimant_rows, dgp="D2", n_train=n),
            "mean_quantile_rmse", grid="main", dgp="D2",
        ).get(CLAIMANT, {})
        baselines = method_means(
            rows_for(combined, dgp="D2", n_train=n), "mean_quantile_rmse",
            grid="main", dgp="D2",
        )
        available = {
            name: stats["mean"] for name, stats in baselines.items()
            if name in FROZEN_BASELINES
        }
        best = min(available.values()) if available else float("nan")
        rule1[str(n)] = {
            "d0_mean": d0.get("mean"),
            "d0_cap": D0_CAP,
            "d2_mean": d2.get("mean"),
            "d2_cap": D2_CAP,
            "d2_best_baseline": best,
            "d2_best_baseline_method": (
                min(available, key=available.get) if available else None
            ),
            "d2_ratio": (d2.get("mean") / best) if best == best else float("nan"),
            "d2_ratio_cap": D2_RATIO_CAP,
            "pass": bool(
                d0.get("mean", float("inf")) <= D0_CAP
                and d2.get("mean", float("inf")) <= D2_CAP
                and (d2.get("mean", float("inf")) / best) <= D2_RATIO_CAP
            ),
        }
    payload["rule_1_correctness"] = rule1

    # --------------------------- rule 2, law metric against Causal-DRF, and
    # the full law surface against every law-capable comparator.
    law: dict = {}
    for dgp in ALL_DGPS:
        block: dict = {}
        for metric in LAW_METRICS:
            block[metric] = {
                comparator: annotate(paired_comparison(
                    combined, metric, claimant=CLAIMANT, comparator=comparator,
                    grid="main", dgp=dgp,
                ))
                for comparator in LAW_COMPARATORS
            }
        law[dgp] = block
    payload["law_surface"] = law
    payload["rule_2_law_advantage"] = {
        "wins_vs_causal_drf": sorted(
            dgp for dgp in ALL_DGPS
            if law[dgp]["kernel_law_error"]["causal_drf"] is not None
            and law[dgp]["kernel_law_error"]["causal_drf"]["verdict"] == "claimant"
        ),
        "required": 2,
    }

    # ------------------------------------ the contrast surface, all regimes
    contrast: dict = {}
    for dgp in ALL_DGPS:
        block: dict = {}
        for metric in CONTRAST_METRICS:
            block[metric] = {
                comparator: annotate(paired_comparison(
                    combined, metric, claimant=CLAIMANT, comparator=comparator,
                    grid="main", dgp=dgp,
                ))
                for comparator in CONTRAST_COMPARATORS
            }
        contrast[dgp] = block
    payload["contrast_surface"] = contrast

    # ------------------------------------------- rules 3 and 4, functionals
    functionals: dict = {}
    for dgp in ALL_DGPS:
        block: dict = {}
        for metric in FUNCTIONAL_METRICS:
            block[metric] = {
                comparator: annotate(paired_comparison(
                    combined, metric, claimant=CLAIMANT, comparator=comparator,
                    grid="main", dgp=dgp,
                ))
                for comparator in ("causal_drf", "pta_s", "cwdb_r3_cvridge")
            }
        functionals[dgp] = block
    payload["functional_surface"] = functionals

    # The phase's accuracy clause: at least one accuracy win against PTA-S on a
    # target PTA-S also estimates. A capability win does not count and is not
    # counted here.
    accuracy_wins = []
    for dgp in ALL_DGPS:
        for metric in CONTRAST_METRICS + FUNCTIONAL_METRICS:
            source = contrast if metric in CONTRAST_METRICS else functionals
            comparison = source[dgp][metric].get("pta_s")
            if comparison is not None and comparison["verdict"] == "claimant":
                accuracy_wins.append(
                    {"dgp": dgp, "metric": metric,
                     "mean": comparison["paired_mean_difference"],
                     "standard_error": comparison["paired_standard_error"]}
                )
    payload["accuracy_clause"] = {
        "statement": (
            "at least one accuracy win against PTA-S on a target both estimate"
        ),
        "wins": accuracy_wins,
        "pass": bool(accuracy_wins),
    }

    # ------------------------------------------- the degradation clause
    # D3 is the separate-head-favourable regime the mu/tau reparameterisation
    # puts at risk; D7 is the pure-shape-transfer regime. A loss against R3 on
    # kernel_law_error at either sample size that crosses the decision multiple
    # is the frozen definition of "materially degrading".
    degradation: dict = {}
    for dgp in ("D3", "D7"):
        per_n = {}
        for n in (500, 1000):
            per_n[str(n)] = annotate(paired_comparison(
                combined, "kernel_law_error", claimant=CLAIMANT,
                comparator="cwdb_r3_cvridge", grid="main", dgp=dgp, n_train=n,
            ))
        degradation[dgp] = per_n
    payload["degradation_clause"] = {
        "comparisons": degradation,
        "violated": sorted(
            f"{dgp}@n{n}"
            for dgp, per_n in degradation.items()
            for n, comparison in per_n.items()
            if comparison is not None and comparison["verdict"] == "comparator"
        ),
    }

    # ------------------------------------------------- rule 5, no collapse
    support = [
        row["value"] for row in claimant_rows
        if row["metric"] == "diagnostic_effective_support" and row["status"] == "ok"
    ]
    d6_coverage = method_means(
        rows_for(claimant_rows, dgp="D6"), "mode_coverage", grid="main", dgp="D6",
    ).get(CLAIMANT, {})
    payload["rule_5_no_collapse"] = {
        "effective_support_min": float(np.min(support)) if support else None,
        "effective_support_median": float(np.median(support)) if support else None,
        "support_floor_particles": SUPPORT_FLOOR * N_PARTICLES,
        "d6_mode_coverage": d6_coverage.get("mean"),
        "d6_mode_coverage_floor": D6_MODE_COVERAGE_FLOOR,
        "pass": bool(
            support and np.min(support) >= SUPPORT_FLOOR * N_PARTICLES
            and (d6_coverage.get("mean") or 0.0) >= D6_MODE_COVERAGE_FLOOR
        ),
    }

    # ------------------------------------------------------- rule 6, cost
    # Stage 2 rows only for the claimant, and the original-code Causal-DRF on
    # the same coordinates for the denominator.
    claimant_runtime = cell_median(rows_for(stage2), "runtime")
    reference_rows = rows_for(load(ORIGINAL_CAUSAL_DRF), seeds=STAGE2_SEEDS)
    reference_runtime = cell_median(reference_rows, "runtime")
    payload["rule_6_cost"] = {
        "claimant_median_runtime_s": claimant_runtime,
        "runtime_source": "stage 2 rows only",
        "reference": "causal_drf, original-code rerun",
        "reference_median_runtime_s": reference_runtime,
        "ratio": claimant_runtime / reference_runtime,
        "cap": COST_RATIO_CAP,
        "pass": bool(claimant_runtime / reference_runtime <= COST_RATIO_CAP),
        "peak_ram_median_mb": cell_median(rows_for(stage2), "peak_ram"),
    }

    # ---------------------------------------------------------- diagnostics
    diagnostics: dict = {}
    for dgp in ALL_DGPS:
        block = {}
        for metric in ("diagnostic_effective_support",
                       "diagnostic_selected_contrast_shrinkage",
                       "diagnostic_n_boosting_steps", "diagnostic_train_risk"):
            values = [row["value"] for row in rows_for(claimant_rows, dgp=dgp)
                      if row["metric"] == metric and row["status"] == "ok"]
            block[metric] = float(np.mean(values)) if values else None
        diagnostics[dgp] = block
    payload["diagnostics"] = diagnostics

    failures = [row for row in stage2 if row["status"] == "failed"]
    payload["n_failed_rows"] = len(failures)

    PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {PAYLOAD}")
    print(json.dumps({
        "seed_counts": payload["claimant_seed_counts"],
        "rule_1": {n: block["pass"] for n, block in rule1.items()},
        "rule_2_wins": payload["rule_2_law_advantage"]["wins_vs_causal_drf"],
        "accuracy_clause": payload["accuracy_clause"]["pass"],
        "degradation_violated": payload["degradation_clause"]["violated"],
        "rule_5": payload["rule_5_no_collapse"]["pass"],
        "rule_6_ratio": payload["rule_6_cost"]["ratio"],
        "n_failed_rows": payload["n_failed_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
