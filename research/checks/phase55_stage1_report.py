"""Phase 5.5 Stage 1 analysis: mechanism screen on the frozen coordinates.

Reads the Phase 5.5 result rows and pairs them seed by seed against the frozen
G3 rows (PTA-S from `results/merged`, R3 from `results/merged_repair`), then
applies the Stage 1 screen criteria from `research_phases/Phase 5.5 -
Orthogonalized C-WDB Variants.md`:

* `cwdb_rmean` must show a credible D2/D8 contrast improvement; the null screen
  passes only if the D2 false-effect RMSE is no worse than the inherited G3 cap
  and D0 shows no material bias. D8 is the primary confounding test.
* `cwdb_mutau` must retain particle validity and D7 transfer, with the
  inherited G3 correctness and cost thresholds as the starting point.
* `cwdb_xmean` must be tested on the imbalance extension before any claim about
  its intended advantage is made; a useful X result must be localized to
  imbalance or overlap stress.

Every comparison uses the seed-paired convention of `g3.analysis`: the seed
level difference removes the replication effect and the paired standard error
is the honest one. The decision multiple is the frozen 2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wasserstein_causal_forests.g3.analysis import paired_comparison, method_means

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results" / "merged" / "main_results.parquet"
REPAIR = ROOT / "results" / "merged_repair" / "main_results.parquet"
PHASE55 = ROOT / "results" / "merged_phase55" / "main_results.parquet"

DECISION_MULTIPLE = 2.0

#: Inherited G3 correctness caps (rule 1) and the D2 reference.
D0_CAP = 0.15
D2_CAP = 0.15
D2_RATIO_CAP = 1.25
D6_MODE_COVERAGE_FLOOR = 0.90
SUPPORT_FLOOR = 0.60
COST_RATIO_CAP = 60.0

MAIN_DGPS = ("D0", "D2", "D7", "D8")
STAGE1_SEEDS = tuple(range(10))


def load(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def rows_for(
    rows: list[dict], *, grid: str = "main", method: str | None = None,
    dgp: str | None = None, n_train: int | None = None, seeds: tuple[int, ...] | None = None,
) -> list[dict]:
    selected = [
        row for row in rows
        if row["grid"] == grid
        and (method is None or row["method"] == method)
        and (dgp is None or row["dgp"] == dgp)
        and (n_train is None or row["n_train"] == n_train)
        and (seeds is None or row["seed"] in seeds)
    ]
    return selected


def main() -> None:
    frozen = load(FROZEN)
    repair = load(REPAIR)
    phase55 = load(PHASE55)
    # Paired comparisons need both sides' rows in one table: the frozen rows
    # for PTA-S and the repair rows for R3 sit in their own tracks.
    combined = frozen + repair + phase55
    payload: dict = {
        "stage": 1,
        "decision_multiple": DECISION_MULTIPLE,
        "seeds": list(STAGE1_SEEDS),
    }

    # --------------------------------------------- rmean mechanism screen
    payload["rmean"] = {}
    for n in (500, 1000):
        block: dict = {}
        for dgp in ("D0", "D2", "D8"):
            block[dgp] = {
                "vs_pta_s": paired_comparison(
                    combined, "mean_quantile_rmse", claimant="cwdb_rmean",
                    comparator="pta_s", grid="main", dgp=dgp, n_train=n,
                ),
                "vs_r3": paired_comparison(
                    combined, "mean_quantile_rmse", claimant="cwdb_rmean",
                    comparator="cwdb_r3_cvridge", grid="main", dgp=dgp, n_train=n,
                ),
            }
        payload["rmean"][str(n)] = block

    # rule 1 style checks on the stage-1 seeds. One convention, applied
    # identically to every variant: the D2 false-effect ratio always divides a
    # single n's D2 mean by the best frozen baseline at that same n. Pooling the
    # baseline across sample sizes, or mixing a per-n numerator with a pooled
    # denominator, is not the convention and must not appear in the memo.
    def rule1_block(method: str) -> dict:
        out: dict = {}
        for n in (500, 1000):
            d0 = method_means(
                rows_for(phase55, method=method, dgp="D0", n_train=n, seeds=STAGE1_SEEDS),
                "mean_quantile_rmse", grid="main", dgp="D0",
            )
            d2 = method_means(
                rows_for(phase55, method=method, dgp="D2", n_train=n, seeds=STAGE1_SEEDS),
                "mean_quantile_rmse", grid="main", dgp="D2",
            )
            d2_frozen = method_means(
                rows_for(frozen, dgp="D2", n_train=n, seeds=STAGE1_SEEDS),
                "mean_quantile_rmse", grid="main", dgp="D2",
            )
            baseline = min(
                value["mean"] for candidate, value in d2_frozen.items()
                if candidate in {"pta_s", "cwdb_v0", "cwdb_v1", "wdrft", "causal_drf"}
            )
            d0_mean = d0.get(method, {}).get("mean", np.inf)
            d2_mean = d2.get(method, {}).get("mean", np.inf)
            ratio = d2_mean / baseline if baseline > 0 else float("inf")
            out[str(n)] = {
                "d0_mean": d0.get(method, {}).get("mean"),
                "d2_mean": d2.get(method, {}).get("mean"),
                "d2_best_frozen_baseline": baseline,
                "d2_false_effect_ratio": ratio,
                "d0_pass": d0_mean <= D0_CAP,
                "d2_pass": d2_mean <= D2_CAP and ratio <= D2_RATIO_CAP,
            }
        return out

    payload["rmean"]["rule1"] = rule1_block("cwdb_rmean")
    payload["rmean"]["d8_primary_confounding_test"] = {
        "n500": {
            "rmean": payload["rmean"]["500"]["D8"]["vs_pta_s"],
            "pta_s_mean": payload["rmean"]["500"]["D8"]["vs_pta_s"]["comparator_mean"],
            "r3_mean": payload["rmean"]["500"]["D8"]["vs_r3"]["comparator_mean"],
        },
        "n1000": {
            "rmean": payload["rmean"]["1000"]["D8"]["vs_pta_s"],
            "pta_s_mean": payload["rmean"]["1000"]["D8"]["vs_pta_s"]["comparator_mean"],
            "r3_mean": payload["rmean"]["1000"]["D8"]["vs_r3"]["comparator_mean"],
        },
    }

    # --------------------------------------------- mutau mechanism screen
    payload["mutau"] = {}
    for dgp in ("D0", "D2", "D7", "D8"):
        for n in (500, 1000):
            payload["mutau"].setdefault(str(n), {})[dgp] = {
                "mean_q": {
                    "vs_r3": paired_comparison(
                        combined, "mean_quantile_rmse", claimant="cwdb_mutau",
                        comparator="cwdb_r3_cvridge", grid="main", dgp=dgp, n_train=n,
                    ),
                    "vs_pta_s": paired_comparison(
                        combined, "mean_quantile_rmse", claimant="cwdb_mutau",
                        comparator="pta_s", grid="main", dgp=dgp, n_train=n,
                    ),
                },
                "kernel_law_error": {
                    "vs_r3": paired_comparison(
                        combined, "kernel_law_error", claimant="cwdb_mutau",
                        comparator="cwdb_r3_cvridge", grid="main", dgp=dgp, n_train=n,
                    ),
                },
            }
    # D7 unseen-functional transfer, against R3, per functional target
    payload["mutau"]["d7_transfer"] = {}
    for n in (500, 1000):
        block = {}
        for target in ("TATE-K-grid_skewness", "TATE-K-grid_upper_tail_mean",
                       "TCATE-K-grid_skewness", "TCATE-K-grid_upper_tail_mean"):
            metric = "tate_functional_rmse" if target.startswith("TATE") else "tcate_functional_rmse"
            block[target] = paired_comparison(
                combined, metric, claimant="cwdb_mutau",
                comparator="cwdb_r3_cvridge", grid="main", dgp="D7", n_train=n,
                target_id=target,
            )
        payload["mutau"]["d7_transfer"][str(n)] = block
    # particle validity: effective support on D7/D8 (D6 arrives at Stage 2)
    support = {}
    for dgp in ("D7", "D8"):
        means = method_means(
            rows_for(phase55, method="cwdb_mutau", dgp=dgp, seeds=STAGE1_SEEDS),
            "diagnostic_effective_support", grid="main", dgp=dgp,
        )
        support[dgp] = means.get("cwdb_mutau", {}).get("mean")
    payload["mutau"]["effective_support"] = support
    payload["mutau"]["support_fraction_floor"] = SUPPORT_FLOOR
    payload["mutau"]["rule1"] = rule1_block("cwdb_mutau")

    # --------------------------------------------- xmean mechanism screen
    payload["xmean"] = {}
    for dgp in MAIN_DGPS:
        for n in (500, 1000):
            payload["xmean"].setdefault(str(n), {})[dgp] = {
                "vs_pta_s": paired_comparison(
                    combined, "mean_quantile_rmse", claimant="cwdb_xmean",
                    comparator="pta_s", grid="main", dgp=dgp, n_train=n,
                ),
                "vs_r3": paired_comparison(
                    combined, "mean_quantile_rmse", claimant="cwdb_xmean",
                    comparator="cwdb_r3_cvridge", grid="main", dgp=dgp, n_train=n,
                ),
            }
    # imbalance suite: xmean against rmean on the same cells
    payload["xmean"]["imbalance"] = {}
    for dgp in ("D2-imb", "D7-imb", "D8-imb"):
        payload["xmean"]["imbalance"][dgp] = paired_comparison(
            combined, "mean_quantile_rmse", claimant="cwdb_xmean",
            comparator="cwdb_rmean", grid="imbalance", dgp=dgp,
        )
    # the imbalance stress each cell actually saw
    ehat_stats = {}
    for dgp in ("D2-imb", "D7-imb", "D8-imb"):
        rows = rows_for(phase55, grid="imbalance", method="cwdb_xmean", dgp=dgp, seeds=STAGE1_SEEDS)
        ehats = [row["value"] for row in rows if row["metric"] == "diagnostic_ehat_mean" and row["status"] == "ok"]
        ehat_stats[dgp] = {
            "mean_ehat_mean": float(np.mean(ehats)) if ehats else None,
            "n": len(ehats),
        }
    payload["xmean"]["imbalance_stress_seen"] = ehat_stats
    payload["xmean"]["rule1"] = rule1_block("cwdb_xmean")

    # --------------------------------------------- cost
    from wasserstein_causal_forests.g3.analysis import cost_summary
    payload["cost"] = {
        "phase55": cost_summary(phase55),
        "frozen_causal_drf": cost_summary(rows_for(frozen, method="causal_drf")),
    }

    out = ROOT / "results" / "merged_phase55" / "stage1_analysis_payload.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
