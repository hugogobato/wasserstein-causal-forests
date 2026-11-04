#!/usr/bin/env python3
"""Tutorial 02: Applied Microdata Workflow & Project STAR Case Study.

Demonstrates how to apply Wasserstein Causal Forests (WCF) to real or synthetic
hierarchical microdata where individuals (students, workers, households) are
nested within treatment units (classrooms, state-years, villages):

Part A — Building an AppliedDataset from raw individual microdata:
    1. Aggregating individual observations into unit-level K=25 midpoint
       quantile vectors Q_i using ``quantile_matrix`` and ``weighted_quantiles``.
    2. Constructing the rank-transformed moderator in column 0 of X via
       ``ecdf_moderator`` so the four bins (-1, -0.5, 0, 0.5, 1) represent
       moderator quartiles.
    3. Running ``run_wcf`` (with custom extra functionals p10 and p90) and
       ``run_placebo`` (permutation falsification test).

Part B — Reproducing the Project STAR classroom analysis from the paper:
    Loads the audited Project STAR dataset (n = 1,333 classrooms across grades
    K-3, 522 small classes vs. 811 regular/aide classes) and inspects the
    calibrated distributional contrasts and free-lunch quartile heterogeneity.

Run from the repository root:
    python tutorials/02_applied_workflow_and_project_star.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wasserstein_causal_forests import (
    AppliedDataset,
    ecdf_moderator,
    midpoint_levels,
    quantile_matrix,
    run_placebo,
    run_wcf,
    weighted_quantiles,
)

ROOT = Path(__file__).resolve().parents[1]


def part_a_microdata_to_wcf() -> None:
    print("=" * 72)
    print("Part A: Constructing an AppliedDataset from Individual Microdata")
    print("=" * 72)

    rng = np.random.default_rng(123)
    n_units = 160  # e.g., 160 classrooms or villages
    individuals_per_unit = rng.integers(15, 35, size=n_units)

    # Unit-level covariates and treatment assignment
    poverty_share = rng.uniform(0.10, 0.90, size=n_units)
    urban = rng.binomial(1, 0.45, size=n_units).astype(float)
    teacher_exp = rng.gamma(shape=3.0, scale=3.0, size=n_units)
    A = rng.binomial(1, 0.45, size=n_units).astype(np.int64)

    # Simulate individual-level outcomes nested inside each unit
    unit_ids_list = []
    outcomes_list = []
    weights_list = []
    for unit_id in range(n_units):
        m_i = int(individuals_per_unit[unit_id])
        unit_shift = (
            -0.40 * poverty_share[unit_id]
            + 0.02 * teacher_exp[unit_id]
            + A[unit_id] * (0.18 + 0.15 * poverty_share[unit_id])
            + rng.normal(0.0, 0.15)
        )
        y_ij = unit_shift + rng.normal(0.0, 1.0, size=m_i)
        w_ij = rng.uniform(0.8, 1.2, size=m_i)
        unit_ids_list.append(np.full(m_i, unit_id, dtype=int))
        outcomes_list.append(y_ij)
        weights_list.append(w_ij)

    all_unit_ids = np.concatenate(unit_ids_list)
    all_outcomes = np.concatenate(outcomes_list)
    all_weights = np.concatenate(weights_list)

    # Step 1: Convert individual microdata into unit-level K=25 quantile vectors
    u = midpoint_levels(25)
    kept_units, Q = quantile_matrix(
        values=all_outcomes,
        sample_weights=all_weights,
        unit_ids=all_unit_ids,
        levels=u,
    )
    assert np.array_equal(kept_units, np.arange(n_units))

    # Step 2: Define benchmark q_star (e.g., pooled control-group individual quantiles)
    control_mask_indiv = np.isin(all_unit_ids, np.where(A == 0)[0])
    q_star = weighted_quantiles(
        all_outcomes[control_mask_indiv],
        all_weights[control_mask_indiv],
        levels=u,
    )

    # Step 3: Put rank-transformed moderator in column 0 of X
    X = np.column_stack(
        [
            ecdf_moderator(poverty_share),
            urban,
            teacher_exp,
        ]
    )

    dataset = AppliedDataset(
        study="synthetic_classroom_demo",
        X=X,
        A=A,
        Q=Q,
        moderator_raw=poverty_share,
        q_star=q_star,
        feature_names=["poverty_share_ecdf", "urban", "teacher_exp"],
        meta={"description": "Synthetic classroom microdata demonstration"},
    ).validate()

    # Optional extra functionals (e.g., 10th and 90th percentiles)
    idx_p10 = int(np.argmin(np.abs(u - 0.10)))
    idx_p90 = int(np.argmin(np.abs(u - 0.90)))
    extra = {
        "p10": lambda q, k=idx_p10: np.asarray(q, dtype=float)[:, k],
        "p90": lambda q, k=idx_p90: np.asarray(q, dtype=float)[:, k],
    }

    # Step 4: Fit WCF and a permutation placebo test
    _, results = run_wcf(
        dataset,
        random_state=0,
        n_estimators=50,
        extra_functionals=extra,
    )
    placebo_runs = run_placebo(
        dataset,
        seeds=(0,),
        n_estimators=50,
        extra_functionals=extra,
    )
    placebo = placebo_runs[0]

    print(
        f"Built AppliedDataset: n = {results['n']} units "
        f"({results['n_treated']} treated, {results['n_control']} control), K = {results['K']}"
    )
    print(f"Selected contrast shrinkage: {results['selected_contrast_shrinkage']}")
    print(f"{'Functional':<12} {'WCF (DR)':>10} {'IF SE':>10} {'Plug-in':>10} {'Placebo DR':>12}")
    print("-" * 60)
    for func in ("mean", "sd", "skew", "tail", "reference", "p10", "p90"):
        print(
            f"{func:<12} "
            f"{results['marginal_dr'][func]:+10.3f} "
            f"{results['if_se'][func]:10.3f} "
            f"{results['marginal_plugin'][func]:+10.3f} "
            f"{placebo['marginal_dr'][func]:+12.3f}"
        )


def part_b_project_star_case_study() -> None:
    print("\n" + "=" * 72)
    print("Part B: Project STAR Classroom Study (AISTATS 2027 Paper Section 4)")
    print("=" * 72)

    star_dir = ROOT / "results" / "applied_study_exploration" / "project_star"
    ds = AppliedDataset.load(star_dir / "data")
    summary = json.loads((star_dir / "summary.json").read_text(encoding="utf-8"))

    print(
        f"Loaded Project STAR: n = {ds.X.shape[0]} classroom-grades "
        f"(treated small classes = {int(ds.A.sum())}, "
        f"control regular/aide = {int((ds.A == 0).sum())})"
    )
    print(f"Covariates ({ len(ds.feature_names) }): {', '.join(ds.feature_names)}")

    primary = summary["primary"]["functionals"]
    placebo = summary["placebo"]["functionals"]

    print("\nAudited 3-Seed Project STAR Contrasts (within-grade SD units):")
    print(f"{'Functional':<14} {'WCF (DR)':>10} {'IF SE':>10} {'Plug-in':>10} {'Placebo':>10}")
    print("-" * 60)
    for func in ("mean", "sd", "skew", "tail", "reference", "p10", "p90"):
        print(
            f"{func:<14} "
            f"{primary[func]['marginal_mean']:+10.3f} "
            f"{primary[func]['if_se_mean']:10.3f} "
            f"{primary[func]['plugin_marginal_mean']:+10.3f} "
            f"{placebo[func]['marginal_mean']:+10.3f}"
        )

    print("\nFree-Lunch Moderator Quartile Contrasts (Q1 = lowest poverty, Q4 = highest):")
    for func in ("mean", "tail", "reference", "p10", "p90"):
        bins = primary[func]["bins_mean"]
        ses = primary[func]["bins_if_se_seed0"]
        cells = " | ".join(f"Q{q + 1}: {bins[q]:+.3f} ({ses[q]:.3f})" for q in range(4))
        print(f"  {func:<10} -> {cells}")


def main() -> None:
    part_a_microdata_to_wcf()
    part_b_project_star_case_study()


if __name__ == "__main__":
    main()
