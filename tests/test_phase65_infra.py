"""Phase 6.5 infrastructure tests: adapters, ZI truth, collapse checks.

The tests use deliberately tiny grids and budgets so the whole file runs in
well under a minute. They cover the four claims the phase's honesty rests on:
the log control changes geometry only; the ZI oracle machinery matches Monte
Carlo; the two-part assembly degenerates correctly at the mixture's boundaries;
and the row-weight generalisation of the metric layer leaves every frozen-suite
number untouched.
"""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
import pytest

from wasserstein_causal_forests.g3.dgps import build_dgp
from wasserstein_causal_forests.g3.evaluation import (
    EvaluationManifest,
    implied_zero_mass,
    evaluate,
)
from wasserstein_causal_forests.g3 import phase65_dgps
from wasserstein_causal_forests.g3 import phase6_dgps
from wasserstein_causal_forests.g3.phase65_methods import (
    BANDWIDTH_CANDIDATES,
    SELECTION_SEEDS,
)
from wasserstein_causal_forests.g3.laws import LawPrediction

phase6_dgps.register_phase6_dgps()
phase65_dgps.register_phase65_dgps()
from wasserstein_causal_forests.g3.laws import LawPrediction

phase65_dgps.register_phase65_dgps()

TINY_BUDGET = {
    "n_estimators": 12,
    "learning_rate": 0.3,
    "max_depth": 2,
    "min_samples_leaf": 5,
    "min_arm_leaf": 3,
}


def _manifest() -> EvaluationManifest:
    return EvaluationManifest(
        manifest_id="TEST",
        functionals=("grid_mean", "grid_sd"),
        tail_level_index=8,
        tail_threshold=6.0,
        mode_radius=1.0,
        mode_mass_floor=0.15,
        zero_mass_tolerance=0.05,
    )


# ------------------------------------------------------------------- Track D


def test_ablated_regimes_are_registered_and_sample() -> None:
    for name in ("DAskew", "DArand", "DAunit", "DAref", "DAdim"):
        dgp = build_dgp(name, 9)
        sample = dgp.sample(64, seed=0)
        assert sample.quantiles.shape == (64, 9)
        assert np.all(np.diff(sample.quantiles, axis=1) >= -1e-12)


def test_daunit_is_unit_scale_rendering_of_ic1() -> None:
    """Dividing IC1 by the divisor then rescaling must reproduce DAunit."""

    ic1 = build_dgp("IC1", 9)
    daunit = build_dgp("DAunit", 9)
    X = np.random.default_rng(1).uniform(-1, 1, size=(128, 6))
    left = ic1.mean_quantiles(X, 1) / phase65_dgps.UNIT_SCALE_DIVISOR
    right = daunit.mean_quantiles(X, 1)
    assert np.allclose(left, right, atol=1e-10)


def test_dadim_uses_two_covariates() -> None:
    dgp = build_dgp("DAdim", 7)
    assert dgp.spec.n_features == 2
    sample = dgp.sample(32, seed=2)
    assert sample.X.shape[1] == 2


def test_daref_uses_the_standard_normal_reference() -> None:
    from wasserstein_causal_forests.g3.dgps import GridSpec
    from wasserstein_causal_forests.g3.phase6_dgps import IncomeGridSpec

    assert isinstance(build_dgp("DAref", 11).grid, GridSpec)
    assert not isinstance(build_dgp("DAref", 11).grid, IncomeGridSpec)
    assert isinstance(build_dgp("DAskew", 11).grid, IncomeGridSpec)


# ------------------------------------------------------------------- Track E


@pytest.mark.parametrize("name", ["ZI0", "ZI1", "ZI2", "ZI3"])
def test_zi_samples_mix_exact_zeros_with_positive_laws(name) -> None:
    dgp = build_dgp(name, 9)
    sample = dgp.sample(400, seed=3)
    row_norms = np.max(np.abs(sample.quantiles), axis=1)
    n_zero = int(np.sum(row_norms <= 1e-12))
    assert 5 < n_zero < 395, "both mixture components must be populated"
    # Monotonicity survives the pinning.
    assert np.all(np.diff(sample.quantiles, axis=1) >= -1e-12)
    # The positive branch never crosses zero by construction check.
    positive_rows = sample.quantiles[row_norms > 1e-12]
    assert float(positive_rows.min()) > 0.0


def test_zi_truth_matches_monte_carlo() -> None:
    dgp = build_dgp("ZI1", 9)
    X = np.random.default_rng(4).uniform(-1, 1, size=(24, 6))
    quadrature = dgp.zero_type_probability(X, 1)

    draws = 120_000
    rng = np.random.default_rng(7)
    arm = 1
    p = dgp.participation(X, arm)
    component = rng.random((draws, 24)) < p[None, :]
    eta = rng.normal(size=(draws, 24)) * 0.10
    location = (
        3.80 + 0.35 * X[None, :, 1] - 0.25 * X[None, :, 3]
    )
    scale = np.exp(0.10 + 0.06 * X[None, :, 5] - 0.03 * X[None, :, 1] + eta)
    z = dgp.grid.base_z[None, None, :]
    gamma = np.clip(
        0.50 + 0.06 * X[None, :, 3], 0.05, 0.85
    )
    hermite = (z * z - 1.0) / 2.0 + (z * z * z - 3.0 * z) / 6.0
    top_coordinate = location + scale * (z[..., 0] + gamma * hermite[..., 0])
    realised = np.where(component, top_coordinate, 0.0)
    monte_carlo = np.mean(realised <= 1e-12, axis=0)

    assert np.max(np.abs(quadrature - monte_carlo)) < 0.04


def test_zi_conditional_expectation_matches_row_weight_contraction() -> None:
    dgp = build_dgp("ZI2", 9)
    X = np.random.default_rng(5).uniform(-1, 1, size=(16, 6))
    direct = dgp.conditional_expectation(X, 0, lambda block: block)
    weights_matrix = dgp.law_node_weights(X, 0)
    blocks = [block for _, block in dgp.iter_law_nodes(X, 0)]
    manual = sum(
        weights_matrix[:, [j]] * block
        for j, block in enumerate(blocks)
    )
    assert np.allclose(direct, manual, atol=1e-12)
    row_sums = weights_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12)


def test_zi_mean_quantile_contrast_is_null_on_zi0() -> None:
    dgp = build_dgp("ZI0", 9)
    X = np.random.default_rng(6).uniform(-1, 1, size=(48, 6))
    contrast = dgp.mean_quantile_contrast(X)
    assert float(np.max(np.abs(contrast))) < 1e-12
    mass_contrast = (
        dgp.zero_type_probability(X, 1) - dgp.zero_type_probability(X, 0)
    )
    assert float(np.max(np.abs(mass_contrast))) < 1e-12


def test_frozen_suite_declares_no_degenerate_component() -> None:
    dgp = build_dgp("D5", 9)
    X = np.random.default_rng(8).uniform(-1, 1, size=(8, 5))
    assert dgp.zero_type_probability(X, 0) is None
    assert dgp.law_node_weights(X, 0) is None


# ------------------------------------------------------- two-part assembly


def _zi_output(name: str):
    from wasserstein_causal_forests.g3.phase65_methods import ZIPTAdapter

    dgp = build_dgp(name, 9)
    train = dgp.sample(160, seed=11)
    test_X = np.random.default_rng(12).uniform(-1, 1, size=(40, 6))
    adapter = ZIPTAdapter(n_particles=5, **TINY_BUDGET)
    output = adapter.fit_predict(train, test_X, dgp, ("grid_mean",), seed=11)
    return output, dgp, test_X


def test_zipt_assembles_a_valid_mixture_law() -> None:
    output, _, _ = _zi_output("ZI1")
    assert output.produces_law
    for arm in (0, 1):
        law = output.law[arm]
        assert law.atoms.shape[1] == 6  # one spike atom + five particles
        assert np.allclose(law.weights.sum(axis=1), 1.0, atol=1e-12)
        # The spike atom sits at exactly the zero vector.
        assert float(np.max(np.abs(law.atoms[:, 0, :]))) == 0.0
        # Row weights put the remaining mass uniformly on the particles.
        assert np.allclose(
            law.weights[:, 0], 1.0 - law.weights[:, 1:].sum(axis=1)
        )


def test_zipt_reduces_to_point_mass_when_everything_is_degenerate() -> None:
    from wasserstein_causal_forests.g3.phase65_methods import ZIPTAdapter

    dgp = build_dgp("ZI0", 9)
    train = dgp.sample(80, seed=13)
    # Force an all-degenerate training sample.
    train = type(train)(
        X=train.X, treatment=train.treatment,
        quantiles=np.zeros_like(train.quantiles),
        propensity=train.propensity, dgp_id=train.dgp_id, seed=train.seed,
    )
    test_X = np.random.default_rng(14).uniform(-1, 1, size=(20, 6))
    output = ZIPTAdapter(n_particles=4, **TINY_BUDGET).fit_predict(
        train, test_X, dgp, ("grid_mean",), seed=13
    )
    for arm in (0, 1):
        law = output.law[arm]
        assert np.allclose(law.weights[:, 0], 1.0, atol=1e-9)
        assert float(np.max(np.abs(law.mean_quantiles()))) == 0.0


def test_zipt_evaluation_produces_zero_mass_rows() -> None:
    output, dgp, test_X = _zi_output("ZI1")
    rows = evaluate(output, dgp, test_X, _manifest(), cache_key=None)
    metrics = {row["metric"] for row in rows}
    assert "zero_mass_abs_error" in metrics
    assert "mass_contrast_rmse" in metrics
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["zero_mass_abs_error"]["status"] == "ok"
    assert by_metric["zero_mass_abs_error"]["value"] >= 0.0


# ------------------------------------------------------------ metric layer


def test_implied_zero_mass_handles_shared_and_particle_banks() -> None:
    atoms_shared = np.array([[0.0, 0.0], [1.0, 2.0]])
    weights_shared = np.array([[0.25, 0.75], [0.5, 0.5]])
    shared = LawPrediction(atoms=atoms_shared, weights=weights_shared,
                           shared_atoms=True)
    loose, exact = implied_zero_mass(shared, tolerance=0.05)
    # Both rows share the atom bank, so each row's mass on the zero atom is
    # its weight on the first training row.
    assert np.allclose(loose, [0.25, 0.5])
    assert np.allclose(exact, [0.25, 0.5])

    particles = np.array([[[0.0, 0.01], [3.0, 3.0]]])
    particle_weights = np.array([[0.4, 0.6]])
    row_specific = LawPrediction(atoms=particles,
                                 weights=particle_weights,
                                 shared_atoms=False)
    loose, exact = implied_zero_mass(row_specific, tolerance=0.05)
    assert np.allclose(loose, [0.4])
    assert np.allclose(exact, [0.0])  # 0.01 exceeds the strict 1e-9 bar


def test_row_weight_generalisation_matches_shared_formula() -> None:
    """A (n, J) weight matrix with identical rows must equal the (J,) path."""

    rng = np.random.default_rng(21)
    nodes = rng.normal(size=(6, 4, 3))
    grid_weights = np.full(3, 1.0 / 3.0)
    shared = rng.random(4)
    shared = shared / shared.sum()
    matrix = np.tile(shared, (6, 1))
    prediction = LawPrediction.from_particles(rng.normal(size=(6, 5, 3)))

    from wasserstein_causal_forests.g3.laws import (
        energy_risk_against_truth,
        kernel_law_error,
    )

    bandwidth = 1.3
    epsilon = 1e-3
    left = energy_risk_against_truth(prediction, nodes, shared, grid_weights,
                                     epsilon=epsilon)
    right = energy_risk_against_truth(prediction, nodes, matrix, grid_weights,
                                      epsilon=epsilon)
    assert np.allclose(left, right, atol=1e-12)
    left = kernel_law_error(prediction, nodes, shared, grid_weights,
                            bandwidth=bandwidth)
    right = kernel_law_error(prediction, nodes, matrix, grid_weights,
                             bandwidth=bandwidth)
    assert np.allclose(left, right, atol=1e-12)


def test_log_adapter_shifts_and_returns_the_original_bank(monkeypatch,
                                                          tmp_path) -> None:
    """The shifted-log control must land its weights on the untouched bank."""

    from wasserstein_causal_forests.g3 import phase65_methods as p65m
    from wasserstein_causal_forests.g3.r_bridge import ForestBaselineResult

    dgp = build_dgp("IC1", 7)
    train = dgp.sample(60, seed=17)
    assert float(train.quantiles.min()) < 0.0, "the shift must be exercised"
    test_X = np.random.default_rng(18).uniform(-1, 1, size=(24, 6))

    seen: dict = {}

    def fake_fit_predict(method, **kwargs):
        seen.update(kwargs, method=method)
        n_test = kwargs["X_test"].shape[0]
        n_train = kwargs["Q_train"].shape[0]
        weights = {
            arm: np.full((n_test, n_train), 1.0 / n_train) for arm in (0, 1)
        }
        return ForestBaselineResult(
            weights=weights, fit_seconds=0.1, total_seconds=0.2,
            peak_ram_mb=10.0,
        )

    monkeypatch.setattr(p65m.r_bridge, "fit_predict", fake_fit_predict)
    adapter = p65m.LogForestAdapter("causal_drf", cache_directory=tmp_path)
    output = adapter.fit_predict(train, test_X, dgp, ("grid_mean",), seed=17)

    assert output.produces_law
    for arm in (0, 1):
        # The atom bank is byte-for-byte the original-scale training sample.
        assert np.array_equal(output.law[arm].atoms, train.quantiles)
    # The bridge saw log(Q - floor) with the frozen floor rule.
    floor = float(np.min(train.quantiles)) - 0.05 * float(
        np.std(train.quantiles)
    )
    assert np.allclose(
        seen["Q_train"], np.log(train.quantiles - floor), atol=1e-12
    )
    assert output.diagnostics["log_floor"] == pytest.approx(floor)
    # The composite map is a bijection: exp inverts the shifted log exactly.
    assert np.allclose(
        np.exp(seen["Q_train"]) + floor, train.quantiles, atol=1e-12
    )


# ------------------------------------------------------------------ manifest


def test_phase65_manifest_enumerates_the_full_grid() -> None:
    from wasserstein_causal_forests.g3.phase65 import (
        enumerate_phase65_cells,
        build_phase65_manifest,
    )

    cells = enumerate_phase65_cells()
    # c_controls: 4 x 3 x 2 x 10 = 240; c_scaling: 2 x 3 x 2 x 5 = 60;
    # d_ablation: 5 x 4 x 1 x 10 = 200; e_zi: 4 x 5 x 2 x 10 = 400.
    assert len(cells) == 900
    document = build_phase65_manifest()
    assert document["n_cells"] == 900
    assert len(document["manifest_checksum"]) == 64


def test_selection_constants_are_frozen() -> None:
    assert BANDWIDTH_CANDIDATES == (0.25, 0.5, 1.0, 2.0, 4.0)
    assert SELECTION_SEEDS == (100, 101)


def test_runner_resolves_every_phase65_adapter(tmp_path) -> None:
    from wasserstein_causal_forests.g3.manifest import Cell
    from wasserstein_causal_forests.g3.runner import build_adapter

    for method in ("causal_drf_log", "drf_log"):
        cell = Cell("c_controls", "IC0", 500, 25, 10, method, 0)
        adapter = build_adapter(cell, tmp_path)
        assert adapter.produces_law
    cell = Cell("c_controls", "IC0", 500, 25, 10, "causal_drf_retn", 0)
    assert build_adapter(cell, tmp_path).produces_law
    cell = Cell("e_zi", "ZI0", 500, 25, 10, "cwdb_zipt", 0)
    assert build_adapter(cell, tmp_path).produces_law
