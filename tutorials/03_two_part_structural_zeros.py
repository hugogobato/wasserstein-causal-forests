#!/usr/bin/env python3
"""Tutorial 03: Two-Part Wasserstein Causal Forests for Structural Zeros.

When outcome distributions have a non-trivial point mass at zero (e.g., labor
earnings with non-employment or medical spending with zero-expenditure spells),
some units realize the degenerate zero quantile curve Q_i = 0_K.

Standard smooth particle boosting cannot represent an exact atom at 0_K without
distorting the positive support. Two-part WCF (``ZIPTAdapter`` / ``cwdb_zipt``)
decomposes the conditional distribution into:
1. A cross-fitted per-arm classifier for the degenerate zero-outcome event
   1 - pi_hat_a(x) = P_hat(Q_i = 0_K | X_i = x, A_i = a), and
2. A cross-fitted CWDB particle booster fit on positive-stratum units (I_i = 1),
assembling the two-part mixture law:
   P_hat_{a,ZI}^K(x) = (1 - pi_hat_a(x)) * delta_{0_K} + pi_hat_a(x) * P_hat_{a,+}(x).

For any functional h, the identity
   h(Q_i) = h(0_K) + I_i * (h(Q_i) - h(0_K))
enables doubly robust AIPW calibration across the full sample.

Run from the repository root:
    python tutorials/03_two_part_structural_zeros.py
"""

from __future__ import annotations

import numpy as np

from wasserstein_causal_forests.g3 import runner
from wasserstein_causal_forests.g3.confirmatory_evaluation import evaluate_confirmatory
from wasserstein_causal_forests.g3.manifest import TRAINING_FUNCTIONALS, Cell
from wasserstein_causal_forests.g3.phase65_dgps import register_phase65_dgps
from wasserstein_causal_forests.g3.phase65_methods import ZIPTAdapter


def main() -> None:
    print("=" * 72)
    print("Tutorial 03: Two-Part WCF (ZIPTAdapter) for Structural Zero Outcomes")
    print("=" * 72)

    register_phase65_dgps()
    cell = Cell("tutorial", "ZI1", 400, 25, 10, "cwdb_zipt", 0)
    dgp = runner.build_dgp(cell.dgp, cell.n_grid)
    train = dgp.sample(cell.n_train, seed=cell.seed)
    test = dgp.sample(300, seed=cell.test_seed)

    is_zero = np.max(np.abs(train.quantiles), axis=1) <= 1e-12
    print(
        f"Generated Z1 (ZI1) training sample: n = {train.n_rows}, K = {dgp.grid.n_grid}"
    )
    print(
        f"Structural-zero share: overall = {is_zero.mean():.1%} "
        f"(control = {is_zero[train.treatment == 0].mean():.1%}, "
        f"treated = {is_zero[train.treatment == 1].mean():.1%})"
    )

    # Fit Two-Part WCF (ZIPTAdapter)
    adapter = ZIPTAdapter(
        n_particles=10,
        classifier_c=1.0,
        n_folds=3,
        contrast_candidates=(0.0, 50.0, 500.0),
        n_estimators=60,
        learning_rate=0.12,
        max_depth=4,
        min_samples_leaf=10,
        min_arm_leaf=5,
        collision_epsilon=1e-3,
    )
    output = adapter.fit_predict(
        train=train,
        X_test=test.X,
        dgp=dgp,
        functionals=TRAINING_FUNCTIONALS,
        seed=cell.seed,
    )

    print("\nFitted Two-Part WCF Diagnostics:")
    print(
        f"  Predicted mean positive-outcome rate: "
        f"arm 0 = {output.diagnostics['positive_rate_arm0']:.3f}, "
        f"arm 1 = {output.diagnostics['positive_rate_arm1']:.3f}"
    )
    print(
        f"  Selected positive-stratum contrast shrinkage: "
        f"lambda* = {output.diagnostics.get('selected_contrast_shrinkage', 0.0):.1f}"
    )
    print(
        f"  Mixture law atoms shape (n_test, 1 + M, K): "
        f"{output.law[0].atoms.shape} (atom 0 is 0_K; atoms 1..M are positive particles)"
    )

    # Evaluate against oracle truth using the paper's confirmatory evaluation protocol
    rows = evaluate_confirmatory(
        cell,
        output,
        train,
        dgp,
        test.X,
        runner.evaluation_manifest(cell.n_grid),
    )

    by_metric: dict[str, list[float]] = {}
    for row in rows:
        if row["status"] == "ok" and row["value"] is not None:
            by_metric.setdefault(row["metric"], []).append(float(row["value"]))

    print("\nOracle Evaluation on Held-Out Test Covariates:")
    for key, label in (
        ("kernel_law_error", "Conditional-law error (MMD^2)"),
        ("mean_quantile_rmse", "Mean-quantile RMSE"),
        ("aipw_tate_functional_rmse", "Mean functional TATE RMSE (AIPW)"),
        ("aipw_tcate_functional_rmse", "Mean functional TCATE RMSE (AIPW)"),
        ("aipw_reference_effect_rmse", "Reference-distance TATE RMSE (AIPW)"),
        ("aipw_reference_tcate_rmse", "Reference-distance TCATE RMSE (AIPW)"),
    ):
        if key in by_metric:
            print(f"  {label:<40}: {np.mean(by_metric[key]):.5f}")


if __name__ == "__main__":
    main()
