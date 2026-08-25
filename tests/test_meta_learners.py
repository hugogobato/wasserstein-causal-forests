"""WP5.5-A and WP5.5-B/D unit checks: folds, nuisances, R-loss, X pseudo-outcomes.

The Phase 5.5 contract lives in `research_phases/Phase 5.5 - Orthogonalized
C-WDB Variants.md`. These tests pin the leak-free infrastructure (WP5.5-A), the
vector R-learner's loss and exact-nuisance behaviour (WP5.5-B, claims C55-1 and
C55-2), and the vector X-learner's imputation honesty (WP5.5-D, claim C55-5).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wasserstein_causal_forests.cwdb.geometry import from_rescaled, to_rescaled
from wasserstein_causal_forests.g3.dgps import build_dgp
from wasserstein_causal_forests.g3.manifest import enumerate_cells
from wasserstein_causal_forests.g3.phase55 import (
    PHASE55_METHOD_REGISTRY,
    PHASE55_METHODS,
    build_phase55_manifest,
    enumerate_phase55_cells,
)
from wasserstein_causal_forests.g3.repair import enumerate_repair_cells
from wasserstein_causal_forests.meta_learners.nuisance import (
    CrossFittedNuisance,
    FoldPlan,
)
from wasserstein_causal_forests.meta_learners.r_learner import RLossTree, VectorRLearner
from wasserstein_causal_forests.meta_learners.x_learner import VectorXLearner


def synthetic_effect(
    n: int = 600, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Noiseless data with known m, e, and tau: Z = m(X) + (A - e(X)) tau(X)."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 5))
    e = np.clip(0.5 + 0.3 * np.tanh(X[:, 0]), 0.05, 0.95)
    A = rng.binomial(1, e)
    tau = np.stack([0.5 * np.sin(np.pi * X[:, 0]) + 0.2 * X[:, 1]] * 4, axis=1)
    m = 0.6 * np.sin(np.pi * X[:, 0])
    Z = m[:, None] + (A - e)[:, None] * tau
    return X, A, Z, e, tau, m


# ------------------------------------------------------------------- WP5.5-A


def test_fold_plan_is_deterministic_and_arm_stratified() -> None:
    rng = np.random.default_rng(0)
    treatment = rng.binomial(1, 0.4, size=500)
    X = rng.uniform(-1.0, 1.0, size=(500, 5))
    first = FoldPlan.stratified(treatment, 5, keys=X, random_state=99)
    second = FoldPlan.stratified(treatment, 5, keys=X, random_state=99)
    assert np.array_equal(first.labels, second.labels)
    assert first.random_state == 99
    assert first.n_folds == 5
    assert len(first.treatment_counts) == 5
    # every fold holds both arms, and the counts sum to the arm totals
    for fold, (n0, n1) in enumerate(first.treatment_counts):
        rows = first.labels == fold
        assert np.sum(treatment[rows] == 0) == n0
        assert np.sum(treatment[rows] == 1) == n1
        assert n0 > 0 and n1 > 0
    assert sum(n0 for n0, _ in first.treatment_counts) == np.sum(treatment == 0)
    assert sum(n1 for _, n1 in first.treatment_counts) == np.sum(treatment == 1)


def test_fold_plan_matches_an_independent_implementation() -> None:
    """Fold assignments must reproduce under a differently written split."""

    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(300, 5))
    treatment = rng.binomial(1, 0.6, size=300)
    plan = FoldPlan.stratified(treatment, 4, keys=X, random_state=123)
    combined = np.column_stack((X, treatment))
    order = np.lexsort(combined[:, ::-1].T)
    independent = np.empty(300, dtype=np.int64)
    for arm in (0, 1):
        positions = order[treatment[order] == arm]
        independent[positions] = np.arange(positions.size) % 4
    assert np.array_equal(plan.labels, independent)


def test_nuisance_predictions_are_permutation_invariant() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, size=(300, 5))
    A = rng.binomial(1, 0.5 + 0.2 * X[:, 0])
    Z = rng.normal(size=(300, 6)) + X[:, :1]
    order = rng.permutation(300)
    inverse = np.argsort(order)
    base = CrossFittedNuisance(random_state=5)
    base.fit(X, A, Z)
    permuted = CrossFittedNuisance(random_state=5)
    permuted.fit(X[order], A[order], Z[order])
    assert np.allclose(base.ehat_oof_, permuted.ehat_oof_[inverse])
    assert np.allclose(base.mhat_oof_, permuted.mhat_oof_[inverse])


def test_no_row_influences_its_own_nuisance_prediction() -> None:
    """Flipping one row's outcome must not move its own out-of-fold prediction."""

    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, size=(400, 5))
    A = rng.binomial(1, 0.5 + 0.2 * X[:, 0])
    Z = rng.normal(size=(400, 6)) + X[:, :1]
    base = CrossFittedNuisance(random_state=9).fit(X, A, Z)
    flipped = Z.copy()
    flipped[17] = flipped[17] + 1000.0
    altered = CrossFittedNuisance(random_state=9).fit(X, A, flipped)
    assert np.allclose(base.ehat_oof_[17], altered.ehat_oof_[17])
    assert np.allclose(base.mhat_oof_[17], altered.mhat_oof_[17])


def test_randomized_constant_outcome_collapses_to_zero_contrast() -> None:
    """The WP5.5-A collapse check: constant outcome, e = 1/2, no effect."""

    rng = np.random.default_rng(3)
    n = 400
    X = rng.uniform(-1.0, 1.0, size=(n, 5))
    A = rng.binomial(1, 0.5, size=n)
    Z = np.tile(np.arange(1.0, 7.0), (n, 1))  # constant outcome law per row
    learner = VectorRLearner(random_state=0)
    learner.fit(X, A, Z, np.ones(6) / 6.0)
    contrast = learner.predict_contrast(X)
    assert np.linalg.norm(contrast) / np.sqrt(contrast.size) < 1e-10


# ------------------------------------------------------------------- WP5.5-B


def test_rloss_tree_matches_its_reference_implementation() -> None:
    X, A, Z, e, _, _ = synthetic_effect(n=400, seed=4)
    w = A - e
    fast = RLossTree(max_depth=3, min_samples_leaf=10).fit(X, Z, w)
    reference = RLossTree(max_depth=3, min_samples_leaf=10).fit_reference(X, Z, w)
    assert np.array_equal(fast.node_feature_, reference.node_feature_)
    assert np.allclose(
        fast.node_threshold_, reference.node_threshold_, equal_nan=True
    )
    assert np.allclose(fast.predict(X), reference.predict(X))


def test_rloss_leaf_value_is_the_exact_weighted_minimizer() -> None:
    X, A, Z, e, _, _ = synthetic_effect(n=120, seed=5)
    w = A - e
    tree = RLossTree(max_depth=0, min_samples_leaf=2).fit(X, Z, w)
    value = tree.predict(X)[0]
    leaf = w
    expected = np.einsum("i,ij->j", leaf, Z) / float(np.dot(leaf, leaf))
    assert np.allclose(value, expected)


def test_rlearner_recovers_tau_with_exact_nuisances() -> None:
    """C55-1: with the true m and e, the R-loss minimizer is tau_Z."""

    X, A, Z, e, tau, m = synthetic_effect(seed=6)
    grid_weights = np.ones(Z.shape[1]) / Z.shape[1]
    # The Stage 1 budget stops at 20 steps because noisy targets overfit
    # afterwards; this noiseless exact-nuisance check is precisely the case
    # where the R-loss can be minimized further, so it uses the full budget.
    learner = VectorRLearner(
        random_state=0,
        contrast_budget={"n_estimators": 100, "learning_rate": 0.12,
                         "max_depth": 4, "min_samples_leaf": 10},
    )
    # Nuisances live in the rescaled coordinates the learner works in.
    learner.fit(X, A, Z, grid_weights, ehat=e, mhat=m * np.sqrt(grid_weights[0]))
    recovered = learner.predict_contrast(X)
    # Noiseless data: the boosting path converges to the truth.
    assert np.sqrt(np.mean((recovered - to_rescaled(tau, grid_weights)) ** 2)) < 5e-2


def test_rlearner_never_uses_an_in_sample_nuisance() -> None:
    """C55-2: flipping a row's outcome leaves its own residualization intact."""

    X, A, Z, _, _, _ = synthetic_effect(n=300, seed=7)
    grid_weights = np.ones(Z.shape[1]) / Z.shape[1]
    base = VectorRLearner(random_state=1)
    base.fit(X, A, Z, grid_weights)
    flipped = Z.copy()
    flipped[23] = flipped[23] + 1000.0
    altered = VectorRLearner(random_state=1)
    altered.fit(X, A, flipped, grid_weights)
    # The row's own treatment weight and residualised outcome cannot change.
    assert np.allclose(base.ehat_train_[23], altered.ehat_train_[23])
    assert np.allclose(base.mhat_train_[23], altered.mhat_train_[23])


def test_rmean_output_is_mean_only_by_contract() -> None:
    from wasserstein_causal_forests.g3.evaluation import evaluate
    from wasserstein_causal_forests.g3.methods import MethodOutput

    entry = PHASE55_METHOD_REGISTRY["cwdb_rmean"]
    assert entry["produces_law"] is False
    assert entry["target_ids"] == ["MEANQ-A-K"]
    assert entry["inference"] is None
    assert entry["cross_fitted"] is True


def test_mean_only_output_marks_law_metrics_not_applicable() -> None:
    from wasserstein_causal_forests.g3.evaluation import EvaluationManifest, evaluate
    from wasserstein_causal_forests.g3.methods import MethodOutput

    dgp = build_dgp("D0", 25)
    train = dgp.sample(200, seed=0)
    test = dgp.sample(100, seed=900_000)
    grid_weights = dgp.grid.weights
    learner = VectorRLearner(random_state=0)
    learner.fit(train.X, train.treatment, train.quantiles, grid_weights)
    output = MethodOutput(
        mean_quantiles={
            arm: learner.predict_mean_quantiles(test.X, arm) for arm in (0, 1)
        },
        functionals={},
        reference=None,
        law=None,
        supported_functionals=(),
        n_atoms=0,
        fit_seconds=0.0,
        predict_seconds=0.0,
        peak_ram_mb=0.0,
    )
    manifest = EvaluationManifest(
        manifest_id="test",
        functionals=("grid_mean", "grid_sd"),
        tail_level_index=24,
        tail_threshold=1.5,
        mode_radius=1.0,
        mode_mass_floor=0.15,
    )
    rows = evaluate(output, dgp, test.X, manifest)
    law_rows = [row for row in rows if row["metric"] == "kernel_law_error"]
    assert law_rows and all(row["status"] == "not_applicable" for row in law_rows)
    assert any(
        row["metric"] == "mean_quantile_rmse" and row["status"] == "ok" for row in rows
    )


# ------------------------------------------------------------------- WP5.5-D


def test_x_learner_pseudo_outcomes_have_the_conditional_contrast_mean() -> None:
    """C55-5: with exact arm nuisances, both pseudo-outcome means equal tau_Z.

    On the noiseless synthetic data, D^(1) = Z - mu0(X) equals tau_Z exactly
    per row once the arm nuisances are exact, so the effect regressions recover
    the contrast up to their own approximation error. The full effect budget is
    used here because the noiseless setting is exactly where more steps do not
    overfit; the Stage 1 budget of three steps is frozen for noisy targets.
    """

    X, A, Z, e, tau, m = synthetic_effect(n=600, seed=8)
    grid_weights = np.ones(Z.shape[1]) / Z.shape[1]
    # The exact-nuisance identity, checked directly on the pseudo-outcomes the
    # learner defines: with mu0 exact, every treated row's D^(1) is tau_Z.
    scale = np.sqrt(grid_weights[0])
    mu0_exact = (m[:, None] - e[:, None] * tau) * scale
    pseudo_d1 = (to_rescaled(Z, grid_weights) - mu0_exact)[A == 1]
    assert np.allclose(pseudo_d1, to_rescaled(tau, grid_weights)[A == 1])
    learner = VectorXLearner(
        random_state=0,
        effect_budget={"n_estimators": 100, "learning_rate": 0.12,
                       "max_depth": 4, "min_samples_leaf": 10},
    )
    learner.fit(X, A, Z, grid_weights)
    recovered = learner.predict_contrast(X)
    expected = to_rescaled(tau, grid_weights)
    assert np.sqrt(np.mean((recovered - expected) ** 2)) < 5e-2


def test_x_learner_imputation_uses_no_in_sample_prediction() -> None:
    """No unit may create its own imputed outcome through an in-sample fit."""

    X, A, Z, _, _, _ = synthetic_effect(n=300, seed=9)
    grid_weights = np.ones(Z.shape[1]) / Z.shape[1]
    base = VectorXLearner(random_state=2)
    base.fit(X, A, Z, grid_weights)
    flipped = Z.copy()
    flipped[41] = flipped[41] + 1000.0
    altered = VectorXLearner(random_state=2)
    altered.fit(X, A, flipped, grid_weights)
    assert np.allclose(base.mu0_oof_[41], altered.mu0_oof_[41])
    assert np.allclose(base.mu1_oof_[41], altered.mu1_oof_[41])


# ------------------------------------------------------------------ registry


def test_phase55_cells_never_collide_with_frozen_or_repair_cells() -> None:
    frozen_keys = {cell.key for cell in enumerate_cells()}
    repair_keys = {cell.key for cell in enumerate_repair_cells()}
    phase55_keys = {cell.key for cell in enumerate_phase55_cells()}
    assert not phase55_keys & frozen_keys
    assert not phase55_keys & repair_keys


def test_phase55_manifest_freeze_is_stable_and_scoped() -> None:
    document = build_phase55_manifest()
    assert document["manifest_contract_id"] == "G3-PHASE55-v1"
    assert document["stage"] == 1
    assert document["n_cells"] == 300
    main_dgps = {
        cell["dgp"]
        for cell in document["cells"]
        if cell["grid"] == "main"
    }
    assert main_dgps == {"D0", "D2", "D7", "D8"}
    # re-freezing produces the same checksum
    assert build_phase55_manifest()["manifest_checksum"] == document["manifest_checksum"]


def test_every_phase55_method_registers_its_schema_fields() -> None:
    for method in PHASE55_METHODS:
        entry = PHASE55_METHOD_REGISTRY[method]
        assert set(entry) >= {"role", "adapter", "produces_law", "target_ids",
                              "inference", "cross_fitted"}
        assert entry["cross_fitted"] is True
    assert PHASE55_METHOD_REGISTRY["cwdb_mutau"]["produces_law"] is True
    assert PHASE55_METHOD_REGISTRY["cwdb_rmean"]["produces_law"] is False
    assert PHASE55_METHOD_REGISTRY["cwdb_xmean"]["produces_law"] is False
