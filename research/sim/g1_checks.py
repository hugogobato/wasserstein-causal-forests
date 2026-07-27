import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sim.dgps import (
    D0Null, D1LocationShift, D4PureNonlinear,
    u_matrix, FW, FUNCTIONAL_GRID, QUANTILE_GRID,
    empirical_u_vector, sample_inner_from_quantile,
)

from wp3_odcf import (
    ODCFEstimator,
    oracle_dr_scores,
    cross_fitted_dr_scores,
    fit_arm_curve_forests,
)
from sim.evaluation import (
    integrated_squared_error,
    integrated_squared_error_curve_only,
    rmse_functional,
    monotonicity_violations,
)
from sim.baselines import _cross_fit_nuisances, _observed_U, _dr_scores


def check_d0_null_bias():
    ns = [200, 500, 1000]
    bias_values = []
    for n in ns:
        replicate_predictions = []
        for replicate in range(3):
            dgp = D0Null().sample(n, seed=42 + replicate, regime="oracle_latent")
            z = dgp.Z
            U_obs = np.where(z[:, None] == 1, dgp.Q1_log, dgp.Q0_log)
            raw_obs = np.where(z[:, None] == 1, dgp.Q1_raw_func, dgp.Q0_raw_func)
            U_full = u_matrix(U_obs, raw_obs, FW, FUNCTIONAL_GRID)
            e_cf, m0_cf, m1_cf = _cross_fit_nuisances(
                dgp.X, z, U_full, n_folds=5, seed=99 + replicate
            )
            scores = _dr_scores(U_full, z, e_cf, m0_cf, m1_cf)
            model = ODCFEstimator(
                K=dgp.K, J=3, variant="composite", n_trees=50,
                min_leaf=5, max_depth=6, random_state=99 + replicate,
            ).fit(dgp.X, scores)
            replicate_predictions.append(model.predict(dgp.X_eval))
        mean_prediction = np.mean(replicate_predictions, axis=0)
        bias_values.append(
            integrated_squared_error(
                mean_prediction, np.zeros_like(mean_prediction), 49
            )
        )

    assert bias_values[-1] < bias_values[0], (
        f"D0 null bias did not shrink from n={ns[0]} to n={ns[-1]}: {bias_values}"
    )
    print(f"  G1-1 PASS: D0 null bias shrinks: {list(zip(ns, bias_values))}")


def check_d1_oracle_imse():
    ns = [200, 500, 1000]
    ise_values = []
    for n in ns:
        replicate_ise = []
        for replicate in range(3):
            dgp = D1LocationShift().sample(
                n, seed=42 + replicate, regime="oracle_latent"
            )
            z = dgp.Z
            U_obs = np.where(z[:, None] == 1, dgp.Q1_log, dgp.Q0_log)
            raw_obs = np.where(z[:, None] == 1, dgp.Q1_raw_func, dgp.Q0_raw_func)
            U_full = u_matrix(U_obs, raw_obs, FW, FUNCTIONAL_GRID)
            scores = oracle_dr_scores(
                U_full, z, dgp.true_propensity, dgp.true_m0, dgp.true_m1
            )
            model = ODCFEstimator(
                K=dgp.K, J=3, variant="composite", n_trees=50,
                min_leaf=5, max_depth=6, random_state=99 + replicate,
            ).fit(dgp.X, scores)
            pred = model.predict(dgp.X_eval)
            replicate_ise.append(
                integrated_squared_error(pred, dgp.true_theta_eval, dgp.K)
            )
        ise_values.append(float(np.mean(replicate_ise)))

    assert ise_values[-1] < ise_values[0], (
        f"D1 oracle curve ISE did not decrease: n={ns} ISEs={ise_values}"
    )
    print(f"  G1-2 PASS: D1 oracle curve IMSE decreases with n: {list(zip(ns, ise_values))}")


def check_d4_nonlinear_detection():
    gini_curve, gini_composite, spurious_curve, spurious_composite = [], [], [], []
    for replicate in range(3):
        dgp = D4PureNonlinear().sample(
            500, seed=42 + replicate, regime="oracle_latent"
        )
        z = dgp.Z
        U_obs = np.where(z[:, None] == 1, dgp.Q1_log, dgp.Q0_log)
        raw_obs = np.where(z[:, None] == 1, dgp.Q1_raw_func, dgp.Q0_raw_func)
        U_full = u_matrix(U_obs, raw_obs, FW, FUNCTIONAL_GRID)
        scores = oracle_dr_scores(
            U_full, z, dgp.true_propensity, dgp.true_m0, dgp.true_m1
        )
        curve_pred = ODCFEstimator(
            K=dgp.K, J=3, variant="curve_only", n_trees=100,
            min_leaf=5, max_depth=6, random_state=99 + replicate,
        ).fit(dgp.X, scores).predict(dgp.X_eval)
        composite_pred = ODCFEstimator(
            K=dgp.K, J=3, variant="composite", n_trees=100,
            min_leaf=5, max_depth=6, random_state=99 + replicate,
        ).fit(dgp.X, scores).predict(dgp.X_eval)
        gini_curve.append(rmse_functional(curve_pred, dgp.true_theta_eval, dgp.K, 0))
        gini_composite.append(
            rmse_functional(composite_pred, dgp.true_theta_eval, dgp.K, 0)
        )
        spurious_curve.append(
            np.max(
                np.abs(
                    np.mean(
                        curve_pred[:, :dgp.K] - dgp.true_theta_eval[:, :dgp.K],
                        axis=0,
                    )
                )
            )
        )
        spurious_composite.append(
            np.max(
                np.abs(
                    np.mean(
                        composite_pred[:, :dgp.K]
                        - dgp.true_theta_eval[:, :dgp.K],
                        axis=0,
                    )
                )
            )
        )

    print(f"  D4: curve-only Gini RMSE={np.mean(gini_curve):.4f}")
    print(f"  D4: composite Gini RMSE={np.mean(gini_composite):.4f}")
    print(f"  D4: max pointwise quantile errors={spurious_curve}/{spurious_composite}")

    assert np.mean(gini_composite) < np.mean(gini_curve), (
        f"Composite should beat curve-only for Gini: {gini_composite} vs {gini_curve}"
    )
    assert max(spurious_curve + spurious_composite) < 0.05, (
        "D4 mean quantile effects must remain close to their exact zero target"
    )
    print("  G1-4 PASS: D4 composite ODCF detects Gini heterogeneity; curve-only does not fabricate")


def check_arm_projection():
    dgp = D1LocationShift().sample(500, seed=42, regime="oracle_latent")
    z = dgp.Z
    U_obs = np.where(z[:, None] == 1, dgp.Q1_log, dgp.Q0_log)
    raw_obs = np.where(z[:, None] == 1, dgp.Q1_raw_func, dgp.Q0_raw_func)
    U_full = u_matrix(U_obs, raw_obs, FW, FUNCTIONAL_GRID)
    arm_forest = fit_arm_curve_forests(
        dgp.X,
        U_full,
        z,
        dgp.true_propensity,
        dgp.true_m0,
        dgp.true_m1,
        K=dgp.K,
        n_trees=30,
        min_leaf=5,
        max_depth=6,
        random_state=99,
    )
    unprojected = arm_forest.predict_arms(dgp.X_eval, project=False)
    projected = arm_forest.predict_arms(dgp.X_eval, project=True)
    assert monotonicity_violations(projected[0]) == 0.0
    assert monotonicity_violations(projected[1]) == 0.0
    print(
        "  G1-3 PASS: projected arm curves have zero monotonicity violations "
        f"(raw fractions {monotonicity_violations(unprojected[0]):.3g}, "
        f"{monotonicity_violations(unprojected[1]):.3g})"
    )


def check_oracle_large_inner_coincidence():
    n = 500
    dgp = D1LocationShift().sample(n, seed=42, regime="oracle_latent")

    z = dgp.Z
    U_obs = np.where(z[:, None] == 1, dgp.Q1_log, dgp.Q0_log)
    raw_obs = np.where(z[:, None] == 1, dgp.Q1_raw_func, dgp.Q0_raw_func)
    U_oracle = u_matrix(U_obs, raw_obs, FW, FUNCTIONAL_GRID)
    scores_oracle = oracle_dr_scores(
        U_oracle, z, dgp.true_propensity, dgp.true_m0, dgp.true_m1
    )

    rng = np.random.default_rng(991)
    U_large = np.array([
        empirical_u_vector(
            sample_inner_from_quantile(raw_obs[i], 10000, rng),
            QUANTILE_GRID,
            FUNCTIONAL_GRID,
            FW,
        )
        for i in range(n)
    ])
    scores_large = oracle_dr_scores(
        U_large, z, dgp.true_propensity, dgp.true_m0, dgp.true_m1
    )

    mse = np.mean((scores_oracle - scores_large) ** 2)
    print(f"  G1-5: m=10000 versus latent DR score MSE = {mse:.6f}")
    assert mse < 0.02, (
        f"large-inner DR scores deviate too much from oracle: MSE={mse:.4f}"
    )
    print("  G1-5 PASS: large-inner path approximates the latent oracle")


def _observed_u_from_inner(sample, grid, f_grid, w):
    from sim.dgps import empirical_u_vector
    return empirical_u_vector(sample, grid, f_grid, w)


def main():
    print("=== G1 Construction Checks ===")
    check_d0_null_bias()
    check_d1_oracle_imse()
    check_arm_projection()
    check_d4_nonlinear_detection()
    check_oracle_large_inner_coincidence()
    print("\n=== G1 checks: ALL PASS ===")


if __name__ == "__main__":
    main()
