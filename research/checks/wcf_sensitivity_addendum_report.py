#!/usr/bin/env python3
"""Addendum reports for the fresh-seed confirmation and the RF propensity run.

Two questions, both answered with paired seed differences:

1. ``confirm``: does the common-grid evidence that selected K=49 over K=25
   replicate on fresh seeds 20--39 (IC0--IC3, M=10, n=1000)?
2. ``flex_rf``: how does the random-forest flexible propensity factory compare
   with the logistic default, the overconfident gradient-boosting factory, and
   the oracle diagnostic on SYM-NL and SYM-MU?

Writes tidy CSVs and a JSON verdict under ``results/wcf_sensitivity/analysis/``.
The verdict is descriptive: the primary setting was selected on seeds 0--9 and
these seeds only confirm or fail to confirm it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.wcf_sensitivity_analysis import cell_metrics  # noqa: E402

ANALYSIS = ROOT / "results" / "wcf_sensitivity" / "analysis"
PRIMARY = ROOT / "results" / "wcf_sensitivity" / "merged" / "wcf_sensitivity_results.parquet"
CONFIRM = ROOT / "results" / "wcf_sensitivity" / "confirm" / "merged" / "wcf_sensitivity_results.parquet"
FLEX_RF = ROOT / "results" / "wcf_sensitivity" / "flex_rf" / "merged" / "wcf_sensitivity_results.parquet"

NATIVE_TARGETS = (
    ("reference_tcate_rmse", "REF-TCATE-K"),
    ("reference_effect_rmse", "REF-ATE-K"),
    ("kernel_law_error", "LAW-A-K"),
    ("mean_quantile_rmse", "MEANQ-A-K"),
    ("tcate_functional_rmse", "TCATE-K-grid_mean"),
)
COMMON_TARGETS = (
    ("reference_tcate_rmse", "REF-TCATE-COMMON199"),
    ("reference_effect_rmse", "REF-ATE-COMMON199"),
    ("kernel_law_error", "LAW-A-COMMON199"),
    ("mean_quantile_rmse", "MEANQ-A-COMMON199"),
    ("tcate_functional_rmse", "TCATE-COMMON199-grid_mean"),
)


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing merged parquet {path}")
    return pd.read_parquet(path)


def _series(
    cells: pd.DataFrame, method: str, target: str, coords: tuple, dgp: str, n_train: int
):
    q = cells[
        (cells["method"] == method)
        & (cells["target_id"] == target)
        & (cells["dgp"] == dgp)
        & (cells["n_train"] == n_train)
        & (cells["n_grid"] == coords[0])
        & (cells["n_particles"] == coords[1])
    ]
    return dict(zip(q["seed"], q["value"]))


def paired_table(
    cells: pd.DataFrame,
    *,
    method_a: str,
    coords_a: tuple,
    method_b: str,
    coords_b: tuple,
    targets: tuple,
    dgps: tuple,
    n_train_values: tuple = (1000,),
) -> pd.DataFrame:
    rows = []
    for n_train in n_train_values:
        for metric, target in targets:
            for dgp in dgps:
                a = _series(cells, method_a, target, coords_a, dgp, n_train)
                b = _series(cells, method_b, target, coords_b, dgp, n_train)
                common = sorted(set(a) & set(b))
                if not common:
                    continue
                delta = np.array([b[s] - a[s] for s in common], dtype=float)
                mean_a = float(np.mean([a[s] for s in common]))
                rows.append(
                    {
                        "dgp": dgp,
                        "n_train": n_train,
                        "metric": metric,
                        "target_id": target,
                        "method_a": method_a,
                        "coords_a": f"K{coords_a[0]}M{coords_a[1]}",
                        "method_b": method_b,
                        "coords_b": f"K{coords_b[0]}M{coords_b[1]}",
                        "n_pairs": len(common),
                        "mean_a": round(mean_a, 6),
                        "mean_b": round(float(np.mean([b[s] for s in common])), 6),
                        "delta_b_minus_a": round(float(delta.mean()), 6),
                        "mc_se": round(
                            float(delta.std(ddof=1) / np.sqrt(len(delta))), 6
                        ),
                        "relative_pct": round(100.0 * float(delta.mean()) / mean_a, 2),
                        "t_like": round(
                            float(delta.mean())
                            / max(
                                float(delta.std(ddof=1) / np.sqrt(len(delta))), 1e-12
                            ),
                            2,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _confirm_verdict(table: pd.DataFrame) -> dict:
    """Replicate if the K=49 common-grid law gain holds and REF-TCATE does not degrade."""
    verdict: dict[str, object] = {"regimes": {}, "replicated": None}
    law = table[table["target_id"] == "LAW-A-COMMON199"].set_index("dgp")
    ref = table[table["target_id"] == "REF-TCATE-COMMON199"].set_index("dgp")
    ok = True
    evaluated = 0
    for dgp in ("IC1", "IC3"):
        entry: dict[str, object] = {}
        if dgp in law.index:
            row = law.loc[dgp]
            entry["law_gain_pct"] = float(row["relative_pct"])
            entry["law_significant_gain"] = bool(
                row["delta_b_minus_a"] < 0 and abs(row["t_like"]) > 2.0
            )
            evaluated += 1
        if dgp in ref.index:
            row = ref.loc[dgp]
            entry["ref_delta_pct"] = float(row["relative_pct"])
            entry["ref_no_material_loss"] = bool(row["relative_pct"] <= 10.0)
        if entry:
            passed = bool(
                entry.get("law_significant_gain")
                and entry.get("ref_no_material_loss", True)
            )
            entry["passes"] = passed
            ok = ok and passed
        verdict["regimes"][dgp] = entry
    verdict["replicated"] = bool(ok and evaluated == 2)
    return verdict


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    confirm_cells = cell_metrics(_load(CONFIRM))
    confirm = paired_table(
        confirm_cells,
        method_a="cwdb_dr",
        coords_a=(25, 10),
        method_b="cwdb_dr",
        coords_b=(49, 10),
        targets=NATIVE_TARGETS + COMMON_TARGETS,
        dgps=("IC0", "IC1", "IC2", "IC3"),
    )
    confirm.to_csv(ANALYSIS / "confirm_paired.csv", index=False)
    verdict = _confirm_verdict(confirm)
    (ANALYSIS / "confirm_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print("=== fresh-seed confirmation: K=49 (b) vs K=25 (a), seeds 20-39 ===")
    print(confirm.to_string(index=False))
    print("verdict:", json.dumps(verdict))

    if FLEX_RF.exists():
        primary_cells = cell_metrics(_load(PRIMARY))
        flex_cells = cell_metrics(_load(FLEX_RF))
        combined = pd.concat([primary_cells, flex_cells], ignore_index=True)
        comparisons = (
            ("cwdb_dr", (25, 10), "cwdb_dr_flex_rf", (25, 10)),
            ("cwdb_dr_flex", (25, 10), "cwdb_dr_flex_rf", (25, 10)),
            ("cwdb_dr", (25, 10), "cwdb_dr_oracle", (25, 10)),
        )
        frames = []
        for method_a, coords_a, method_b, coords_b in comparisons:
            table = paired_table(
                combined,
                method_a=method_a,
                coords_a=coords_a,
                method_b=method_b,
                coords_b=coords_b,
                targets=NATIVE_TARGETS + COMMON_TARGETS,
                dgps=("SYM-NL", "SYM-MU"),
                n_train_values=(500, 1000),
            )
            table["comparison"] = f"{method_a} -> {method_b}"
            frames.append(table)
        flex = pd.concat(frames, ignore_index=True)
        flex.to_csv(ANALYSIS / "flex_rf_paired.csv", index=False)
        print("\n=== random-forest flexible propensity ===")
        print(
            flex[flex["target_id"].isin(["REF-TCATE-K", "REF-TCATE-COMMON199"])]
            .to_string(index=False)
        )
    else:
        print(f"note: {FLEX_RF} is absent; skipped the RF comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
