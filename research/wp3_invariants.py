"""Executable WP3 construction checks.

Run with ``python3 research/wp3_invariants.py`` from the project root.  These
checks are finite-grid implementation checks, not evidence for an asymptotic
theorem.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from wp3_odcf import (  # noqa: E402
    ODCFEstimator,
    CoordinateScaler,
    DEFAULT_RANDOM_STATE,
    arm_dr_scores,
    assert_cross_fit_disjointness,
    bootstrap_bias_corrected_unit,
    cross_fitted_dr_scores,
    dr_scores,
    empirical_unit_vector,
    effect_curve_from_arms,
    exhaustive_gain_identity,
    fit_specialized_forests,
    fit_arm_curve_forests,
    fit_odcf_from_inner_samples,
    oracle_dr_scores,
    pava,
    project_arm_mean_curves,
    split_gain_from_means,
    trapezoidal_grid_weights,
)


def synthetic_scores(n: int = 96, K: int = 7, J: int = 3, seed: int = 17):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    propensity = 1.0 / (1.0 + np.exp(-0.4 * X[:, 0]))
    Z = rng.binomial(1, propensity)
    curve = (
        0.35 * X[:, [0]]
        + 0.12 * X[:, [1]] * np.linspace(-1, 1, K)[None, :]
        + rng.normal(scale=0.15, size=(n, K))
    )
    functionals = np.c_[
        0.8 * X[:, 0] + rng.normal(scale=0.2, size=n),
        -0.5 * X[:, 1] + rng.normal(scale=0.2, size=n),
        0.3 * X[:, 2] + rng.normal(scale=0.2, size=n),
    ][:, :J]
    U = np.c_[curve, functionals]
    m0 = 0.2 * X[:, [0]] + np.zeros((n, K + J))
    m1 = m0 + np.c_[0.1 + 0.1 * X[:, [0]], np.zeros((n, K - 1 + J))]
    observed = np.where(Z[:, None] == 1, m1, m0) + rng.normal(scale=0.3, size=(n, K + J))
    return X, Z, U, observed, propensity, m0, m1


def check_split_gain_algebra():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(7, 4))
    weights = np.array([0.15, 0.1, 0.2, 0.55])
    error = exhaustive_gain_identity(values, weights)
    assert error < 1e-12, error
    left_mean = np.array([1.0, 2.0, 0.0, -1.0])
    right_mean = np.array([0.0, 2.0, 1.0, -1.0])
    gain = split_gain_from_means(left_mean, right_mean, 3, 2, weights)
    assert gain > 0
    # The positive contribution comes from one coordinate and all coordinate
    # weights are strictly positive, which is C3.1's finite-node content.
    assert np.isclose(gain, (3 * 2 / 5) * (0.15 + 0.2), atol=1e-12)


def check_pure_functional_signal():
    # Corrected WP2-D4 construction.  Control distributions are {1,3};
    # treated distributions in the X=1 group are equally often delta_1 and
    # {1,7}.  Their expected log-quantile vector is unchanged, while mean
    # unit-level Gini changes by -1/16.
    rows = []
    for x in (0.0, 1.0):
        for z in (0.0, 1.0):
            for state in (0, 1):
                for _ in range(20):
                    rows.append((x, z, state))
    design = np.asarray(rows)
    X = design[:, [0]]
    Z = design[:, 1]
    state = design[:, 2].astype(int)
    control_curve = np.array([np.log(2.0), np.log(4.0)])
    control_gini = 1.0 / 4.0
    m0 = np.tile(np.r_[control_curve, control_gini], (len(X), 1))
    m1 = m0.copy()
    m1[X[:, 0] == 1, -1] = 3.0 / 16.0
    observed = m0.copy()
    treated_x1 = (Z == 1) & (X[:, 0] == 1)
    observed[treated_x1 & (state == 0)] = np.r_[
        np.log(2.0), np.log(2.0), 0.0
    ]
    observed[treated_x1 & (state == 1)] = np.r_[
        np.log(2.0), np.log(8.0), 3.0 / 8.0
    ]
    realized_scores = oracle_dr_scores(observed, Z, 0.5, m0, m1)
    scores = m1 - m0
    for x in (0.0, 1.0):
        in_group = X[:, 0] == x
        assert np.allclose(
            np.mean(realized_scores[in_group], axis=0),
            np.mean(scores[in_group], axis=0),
            atol=1e-12,
        )
    curve_only = ODCFEstimator(
        K=2, J=1, variant="curve_only", n_trees=12, min_leaf=5, max_depth=2, random_state=1
    ).fit(X, scores)
    composite = ODCFEstimator(
        K=2, J=1, variant="composite", n_trees=12, min_leaf=5, max_depth=2, random_state=1
    ).fit(X, scores)
    assert all(tree.root.is_leaf for tree in curve_only.trees)
    assert any(tree.root.gain > 0 for tree in composite.trees)
    x1_effect = np.mean(realized_scores[X[:, 0] == 1], axis=0)
    assert np.allclose(x1_effect[:2], 0.0, atol=1e-12)
    assert np.isclose(x1_effect[-1], -1.0 / 16.0, atol=1e-12)


def check_cross_fit_and_honesty():
    X, Z, U, observed, _, _, _ = synthetic_scores()
    _, _, _, _, true_propensity, true_m0, true_m1 = synthetic_scores(seed=17)
    oracle_scores = oracle_dr_scores(
        observed, Z, true_propensity, true_m0, true_m1
    )
    oracle_arm0, oracle_arm1 = arm_dr_scores(
        observed, Z, true_propensity, true_m0, true_m1
    )
    assert np.allclose(oracle_arm1 - oracle_arm0, oracle_scores)
    assert np.max(np.abs(np.mean(oracle_arm0 - true_m0, axis=0))) < 0.15
    assert np.max(np.abs(np.mean(oracle_arm1 - true_m1, axis=0))) < 0.15
    oracle_truth = true_m1 - true_m0
    assert np.max(
        np.abs(np.mean(oracle_scores - oracle_truth, axis=0))
    ) < 0.15
    oracle_model = ODCFEstimator(
        K=7, J=3, variant="composite", n_trees=8, min_leaf=4, max_depth=3, random_state=4
    ).fit(X, oracle_scores)
    assert oracle_model.predict(X[:2]).shape == (2, 10)
    result = cross_fitted_dr_scores(X, Z, observed, n_folds=4, random_state=9)
    assert_cross_fit_disjointness(result, len(X))
    for training, _ in result.folds:
        assert set(np.unique(Z[training])) == {0.0, 1.0}
    with patch(
        "sklearn.ensemble.RandomForestClassifier.fit",
        side_effect=AssertionError("known propensity must bypass classifier fitting"),
    ):
        known_result = cross_fitted_dr_scores(
            X, Z, observed, n_folds=4, random_state=9, known_propensity=0.37
        )
    assert np.allclose(known_result.propensity, 0.37)
    assert all(
        backend.endswith(":known_propensity")
        for backend in known_result.nuisance_backends
    )
    with patch(
        "sklearn.ensemble.RandomForestRegressor.fit",
        side_effect=RuntimeError("nuisance failure sentinel"),
    ):
        try:
            cross_fitted_dr_scores(
                X, Z, observed, n_folds=4, random_state=9, known_propensity=0.37
            )
        except RuntimeError as error:
            assert "nuisance failure sentinel" in str(error)
        else:
            raise AssertionError("unexpected nuisance failures were silently swallowed")
    model = ODCFEstimator(
        K=7, J=3, variant="composite", n_trees=16, min_leaf=4, max_depth=3, random_state=4
    ).fit(X, result.scores, treatment=Z, propensity=result.propensity)
    report = model.honesty_report()
    assert report["all_split_estimation_disjoint"]
    assert report["all_leaf_populations_from_estimation"]
    assert report["all_children_satisfy_balance"]
    assert report["all_local_leaves_nonempty"]
    prediction = model.predict(X[:5])
    assert prediction.shape == (5, 10)
    assert np.allclose(np.sum(model.weights_at(X[0])), 1.0)
    arm_forest = fit_arm_curve_forests(
        X,
        observed,
        Z,
        result.propensity,
        result.m0,
        result.m1,
        K=7,
        n_trees=4,
        min_leaf=4,
        max_depth=2,
        random_state=14,
    )
    arm0, arm1 = arm_forest.predict_arms(X[:8], project=True)
    assert np.all(np.diff(arm0, axis=1) >= -1e-12)
    assert np.all(np.diff(arm1, axis=1) >= -1e-12)
    assert np.allclose(
        arm_forest.predict_effect(X[:8], project_arms=True),
        arm1 - arm0,
    )
    try:
        fit_arm_curve_forests(
            X,
            observed,
            Z,
            result.propensity,
            result.m0,
            result.m1,
            K=7,
            variant="mmd_score",
            n_trees=2,
            min_leaf=4,
        )
    except ValueError as error:
        assert "fixes variant" in str(error)
    else:
        raise AssertionError("arm-forest helper silently accepted a conflicting variant")
    try:
        dr_scores(
            observed,
            Z,
            result.propensity,
            result.m0[:, :1],
            result.m1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("DR nuisance shape broadcasting was not rejected")


def check_tree_scaling_uses_split_side_only():
    X, _, U, _, _, _, _ = synthetic_scores(n=80, K=4, J=3, seed=91)
    original = ODCFEstimator(
        K=4, J=3, n_trees=4, min_leaf=4, max_depth=3, random_state=12
    ).fit(X, U)
    estimation_only = int(original.trees[0].estimation_indices[0])
    perturbed_scores = U.copy()
    perturbed_scores[estimation_only, -1] += 1e6
    perturbed = ODCFEstimator(
        K=4, J=3, n_trees=4, min_leaf=4, max_depth=3, random_state=12
    ).fit(X, perturbed_scores)

    def structure(node):
        if node.is_leaf:
            return ("leaf",)
        return (
            node.feature,
            node.threshold,
            structure(node.left),
            structure(node.right),
        )

    assert np.array_equal(
        original.trees[0].split_indices,
        perturbed.trees[0].split_indices,
    )
    assert np.array_equal(
        original.trees[0].estimation_indices,
        perturbed.trees[0].estimation_indices,
    )
    assert np.allclose(
        original.trees[0].coordinate_scales,
        perturbed.trees[0].coordinate_scales,
    )
    assert structure(original.trees[0].root) == structure(perturbed.trees[0].root)


def check_grid_duplication_invariance():
    X, _, U, _, _, _, _ = synthetic_scores(n=72, K=4, J=3)
    original_weights = np.array([0.1, 0.2, 0.3, 0.3])
    duplicate_curve = np.repeat(U[:, :4], 2, axis=1)
    duplicate_scores = np.c_[duplicate_curve, U[:, 4:]]
    duplicate_weights = np.repeat(original_weights / 2.0, 2)
    def structure(node):
        if node.is_leaf:
            return ("leaf",)
        return (
            "node",
            node.feature,
            node.threshold,
            structure(node.left),
            structure(node.right),
        )

    for variant in ("composite", "mmd_score"):
        original = ODCFEstimator(
            K=4, J=3, variant=variant, n_trees=10, min_leaf=4, max_depth=3,
            random_state=21, quadrature_weights=original_weights,
        ).fit(X, U)
        duplicate = ODCFEstimator(
            K=8, J=3, variant=variant, n_trees=10, min_leaf=4, max_depth=3,
            random_state=21, quadrature_weights=duplicate_weights,
        ).fit(X, duplicate_scores)
        original_prediction = original.predict(X[:12])
        duplicate_prediction = duplicate.predict(X[:12])
        assert np.max(
            np.abs(original_prediction[:, :4] - duplicate_prediction[:, :8:2])
        ) < 1e-8
        assert np.max(
            np.abs(original_prediction[:, 4:] - duplicate_prediction[:, 8:])
        ) < 1e-8
        original_structures = [structure(tree.root) for tree in original.trees]
        duplicate_structures = [structure(tree.root) for tree in duplicate.trees]
        assert original_structures == duplicate_structures


def check_scaling_and_pava():
    X, Z, U, _, propensity, _, _ = synthetic_scores()
    trapezoid = trapezoidal_grid_weights(7)
    assert np.isclose(np.sum(trapezoid), 0.9)
    assert np.isclose(trapezoid[0], 0.9 / (2 * 6))
    assert np.isclose(trapezoid[-1], trapezoid[0])
    for rule in ("robust_sd", "mad", "null_score_se"):
        scaler = CoordinateScaler.fit(
            U, K=7, quadrature_weights=trapezoid, rule=rule,
            treatment=Z, propensity=propensity,
        )
        assert np.all(np.isfinite(scaler.scales))
        assert np.all(scaler.scales > 0)
    projected = pava(np.array([3.0, 1.0, 2.0]))
    assert np.all(np.diff(projected) >= -1e-12)
    arm_curves = project_arm_mean_curves(
        np.array([[3.0, 0.0, 2.0], [1.0, 2.0, 3.0]]),
        quadrature_weights=np.array([1.0, 2.0, 1.0]),
    )
    assert np.allclose(arm_curves[0], np.array([1.0, 1.0, 2.0]))
    assert np.all(np.diff(arm_curves, axis=1) >= -1e-12)
    effect = effect_curve_from_arms(arm_curves[:1], arm_curves[1:])
    assert np.any(np.diff(effect[0]) < 0), "effect curves must remain unconstrained"


def check_common_api_and_inner_variant():
    X, Z, U, _, _, _, _ = synthetic_scores(n=60, K=4, J=3)
    models = [
        ODCFEstimator(K=4, J=3, variant="curve_only", n_trees=6, min_leaf=3, random_state=2),
        ODCFEstimator(K=4, J=3, variant="composite", n_trees=6, min_leaf=3, random_state=2),
        ODCFEstimator(K=4, J=3, variant="mmd_score", n_trees=6, min_leaf=3, random_state=2),
    ]
    for model in models:
        assert model.fit(X, U).predict(X[:3]).shape == (3, 7)
    specialized = fit_specialized_forests(
        X,
        U,
        K=4,
        J=3,
        groups={"curve": np.arange(4), "gini": [4], "theil": [5], "atkinson": [6]},
        n_trees=5,
        min_leaf=3,
        random_state=2,
    )
    assert specialized.predict(X[:3]).shape == (3, 7)
    assert [
        model.random_state for model in specialized.models.values()
    ] == [2 + offset for offset in range(4)]
    default_specialized = fit_specialized_forests(
        X,
        U,
        K=4,
        J=3,
        groups={"curve": np.arange(4), "functionals": [4, 5, 6]},
        n_trees=1,
        min_leaf=3,
    )
    assert [
        model.random_state for model in default_specialized.models.values()
    ] == [DEFAULT_RANDOM_STATE, DEFAULT_RANDOM_STATE + 1]
    try:
        models[1].weights_at(np.array([np.nan, 0.0, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite prediction covariates were not rejected")

    for invalid_model in (
        ODCFEstimator(
            K=4,
            J=3,
            active_coordinates=[0.9, 1.1],
            n_trees=1,
            min_leaf=3,
        ),
        ODCFEstimator(K=4, J=3, n_trees=1, min_leaf=3.5),
    ):
        try:
            invalid_model.fit(X, U)
        except ValueError:
            pass
        else:
            raise AssertionError("an invalid integer-valued API input was accepted")
    try:
        CoordinateScaler.fit(U, K=4, rule="null_score_se")
    except ValueError:
        pass
    else:
        raise AssertionError("null_score_se accepted missing design inputs")
    try:
        ODCFEstimator(
            K=4,
            J=3,
            variant="mmd_score",
            n_trees=1,
            min_leaf=3,
            inner_noise_correction=True,
        ).fit(X, U, noise_variances=np.ones_like(U))
    except ValueError as error:
        assert "SSE-only" in str(error)
    else:
        raise AssertionError("MMD silently accepted an unsupported noise correction")

    rng = np.random.default_rng(22)
    inner_samples = [rng.lognormal(size=36) for _ in range(len(X))]
    convention_check = empirical_unit_vector(
        np.array([1.0, 7.0]), np.array([0.5])
    )
    assert np.isclose(convention_check[0], np.log(2.0))
    result = bootstrap_bias_corrected_unit(inner_samples[0], np.linspace(0.1, 0.9, 4), n_replicates=8)
    assert result.corrected_vector.shape == (7,)
    assert np.all(result.estimated_noise_variance >= 0)
    provisional, cross_fit, bootstrap_outputs = fit_odcf_from_inner_samples(
        X,
        Z,
        inner_samples,
        np.linspace(0.1, 0.9, 4),
        ODCFEstimator(
            K=4,
            J=3,
            n_trees=4,
            min_leaf=3,
            random_state=2,
            inner_noise_correction=True,
        ),
        nuisance_folds=3,
        random_state=2,
        n_bootstrap_replicates=4,
        known_propensity=0.41,
    )
    assert provisional.predict(X[:2]).shape == (2, 7)
    assert len(cross_fit.folds) == 3
    assert np.allclose(cross_fit.propensity, 0.41)
    direct_coefficient = Z / cross_fit.propensity + (1 - Z) / (1 - cross_fit.propensity)
    expected_noise = np.vstack(
        [item.estimated_noise_variance for item in bootstrap_outputs]
    ) * direct_coefficient[:, None] ** 2
    assert np.allclose(provisional.noise_variances, expected_noise)


def main():
    check_split_gain_algebra()
    check_pure_functional_signal()
    check_cross_fit_and_honesty()
    check_tree_scaling_uses_split_side_only()
    check_grid_duplication_invariance()
    check_scaling_and_pava()
    check_common_api_and_inner_variant()
    print("WP3 invariants: PASS")
    print("N3a: curve-only root gain is zero on pure-functional signal; composite root gain is positive")
    print("N3b/N3h: duplicated quadrature grid preserves SSE and MMD-score structures and predictions")
    print("N3c/N3e: split/populate indices are disjoint and all local population leaves are nonempty")
    print("N3d: known propensity bypasses classifier fitting; nuisance failures propagate")
    print("N3f: weighted arm PAVA is correct and effect curves remain unconstrained")
    print("N3g: optional SSE noise correction uses the DR-score scale; MMD rejects it")


if __name__ == "__main__":
    main()
