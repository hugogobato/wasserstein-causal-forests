"""Common-grid adapter: interpolation, dense reference, and row coverage.

The adapter's reason to exist is that native-grid scores change estimand with
K. These tests pin the pieces that make the common target well defined: the
interpolation rule (exact on source levels, constant outside), the dense
reference for the income regimes, the distinct target identifiers, and the
DR payload branch.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from wasserstein_causal_forests.g3 import phase6_dgps
from wasserstein_causal_forests.g3.common_grid import (
    COMMON_LEVELS,
    INTERIOR_LEVELS,
    LevelGridSpec,
    build_dgp_at_levels,
    evaluate_common_grid,
    interpolate_quantile_curves,
    subsample_law,
)
from wasserstein_causal_forests.g3.dgps import build_dgp, resolve_dgp_spec
from wasserstein_causal_forests.g3.laws import LawPrediction
from wasserstein_causal_forests.g3.manifest import Cell
from wasserstein_causal_forests.g3.methods import _output_from_laws
from wasserstein_causal_forests.pta_bcf.targets import GRID_FUNCTIONALS

phase6_dgps.register_phase6_dgps()

NATIVE_TARGET_IDS = (
    {"MEANQ-A-K", "LAW-A-K", "REF-ATE-K", "REF-TCATE-K"}
    | {f"TATE-K-{name}" for name in GRID_FUNCTIONALS}
    | {f"TCATE-K-{name}" for name in GRID_FUNCTIONALS}
)

ROW_KEYS = {
    "metric",
    "target_id",
    "arm",
    "detail",
    "value",
    "status",
    "failure_reason",
}


def test_interpolation_reproduces_source_levels_exactly() -> None:
    rng = np.random.default_rng(0)
    source = (np.arange(9) + 0.5) / 9
    values = np.sort(rng.normal(size=(4, 9)), axis=-1)
    result = interpolate_quantile_curves(values, source, source)
    np.testing.assert_allclose(result, values)


def test_interpolation_continues_constantly_outside_the_source_range() -> None:
    source = np.array([0.2, 0.4, 0.6, 0.8])
    values = np.array([[1.0, 2.0, 3.0, 4.0]])
    target = np.array([-1.0, 0.05, 0.2, 0.8, 0.99, 1.5])
    result = interpolate_quantile_curves(values, source, target)
    np.testing.assert_allclose(result, np.array([[1.0, 1.0, 1.0, 4.0, 4.0, 4.0]]))


def test_interpolation_supports_vectorised_leading_shapes() -> None:
    rng = np.random.default_rng(1)
    source = np.linspace(0.05, 0.95, 11)
    target = np.linspace(0.0, 1.0, 23)
    for shape in ((11,), (7, 11), (3, 5, 11)):
        values = np.sort(rng.normal(size=shape), axis=-1)
        result = interpolate_quantile_curves(values, source, target)
        assert result.shape == shape[:-1] + (target.size,)
        assert result.dtype == np.float64


def test_interpolation_preserves_monotone_rows() -> None:
    rng = np.random.default_rng(2)
    source = np.linspace(0.1, 0.9, 9)
    target = np.linspace(0.0, 1.0, 41)
    values = np.sort(rng.normal(size=(6, 9)), axis=-1)
    result = interpolate_quantile_curves(values, source, target)
    assert np.all(np.diff(result, axis=-1) >= 0.0)


def test_subsample_law_keeps_weights_and_shared_flag() -> None:
    rng = np.random.default_rng(3)
    atoms = np.sort(rng.normal(size=(6, 3, 11)), axis=-1)
    weights = np.full((6, 3), 1.0 / 3.0)
    law = LawPrediction(atoms=atoms, weights=weights, shared_atoms=False)

    result = subsample_law(law, COMMON_LEVELS)
    assert result.atoms.shape == (6, 3, COMMON_LEVELS.size)
    np.testing.assert_allclose(result.weights, weights)
    assert result.shared_atoms is False

    native = (np.arange(11) + 0.5) / 11
    np.testing.assert_allclose(subsample_law(law, native).atoms, atoms)


def test_level_grid_spec_uses_the_income_reference_for_ic_ids() -> None:
    spec = LevelGridSpec(COMMON_LEVELS, "IC1")
    z = norm.ppf(COMMON_LEVELS)
    hermite = (z * z - 1.0) / 2.0 + (z**3 - 3.0 * z) / 6.0
    expected = phase6_dgps.REFERENCE_LOCATION + phase6_dgps.REFERENCE_LOG_SCALE * (
        z + phase6_dgps.REFERENCE_SHAPE * hermite
    )
    np.testing.assert_allclose(spec.reference_quantiles(), expected)
    assert spec.n_grid == COMMON_LEVELS.size
    np.testing.assert_allclose(spec.weights, np.full(spec.n_grid, 1.0 / spec.n_grid))


def test_level_grid_spec_defaults_to_the_standard_normal() -> None:
    spec = LevelGridSpec(INTERIOR_LEVELS, "D1")
    np.testing.assert_allclose(spec.reference_quantiles(), norm.ppf(INTERIOR_LEVELS))


def test_build_dgp_at_levels_uses_the_supplied_levels() -> None:
    dense = build_dgp_at_levels("IC1", INTERIOR_LEVELS)
    assert isinstance(dense.grid, LevelGridSpec)
    assert dense.grid.n_grid == INTERIOR_LEVELS.size
    np.testing.assert_allclose(dense.grid.levels, INTERIOR_LEVELS)
    assert np.all(INTERIOR_LEVELS > 0.1)
    assert np.all(INTERIOR_LEVELS < 0.9)


def _small_case():
    dgp = build_dgp("IC1", 25)
    train = dgp.sample(n_rows=120, seed=3)
    test = dgp.sample(n_rows=80, seed=900003)
    rng = np.random.default_rng(7)
    laws = {
        arm: LawPrediction.from_particles(
            np.sort(rng.normal(size=(test.n_rows, 5, 25)), axis=-1)
        )
        for arm in (0, 1)
    }
    output = _output_from_laws(
        laws,
        dgp.grid.weights,
        dgp.grid.reference_quantiles(),
        ("grid_mean", "grid_sd"),
        fit_seconds=0.0,
        predict_seconds=0.0,
        peak_ram_mb=0.0,
    )
    cell = Cell("test_grid", "IC1", 120, 25, 5, "cwdb_dr", 0)
    assert train.n_rows == 120
    return dgp, test, output, cell


def test_evaluate_common_grid_emits_distinct_dense_target_ids() -> None:
    dgp, test, output, cell = _small_case()
    rows = evaluate_common_grid(cell, output, dgp, test.X)

    targets = {row["target_id"] for row in rows}
    for suffix in ("COMMON199", "INTERIOR"):
        assert f"MEANQ-A-{suffix}" in targets
        assert f"LAW-A-{suffix}" in targets
    for name in GRID_FUNCTIONALS:
        assert f"TATE-COMMON199-{name}" in targets
        assert f"TCATE-COMMON199-{name}" in targets
    assert "REF-ATE-COMMON199" in targets
    assert "REF-TCATE-COMMON199" in targets
    assert "REF-TCATE-INTERIOR" in targets
    assert not (targets & NATIVE_TARGET_IDS)

    for row in rows:
        assert set(row) == ROW_KEYS
        if row["status"] == "ok":
            assert row["value"] is not None
            assert np.isfinite(row["value"])
        else:
            assert row["value"] is None
            assert row["failure_reason"]


def test_law_less_output_keeps_every_row_with_a_reason() -> None:
    dgp, test, output, cell = _small_case()
    object.__setattr__(output, "law", None)

    rows = evaluate_common_grid(cell, output, dgp, test.X)
    law_rows = [row for row in rows if row["metric"] == "kernel_law_error"]
    assert law_rows
    assert all(
        row["status"] == "not_applicable" and row["failure_reason"]
        for row in law_rows
    )
    functional_rows = [
        row
        for row in rows
        if row["metric"] in {"tate_functional_rmse", "tcate_functional_rmse"}
    ]
    assert len(functional_rows) == 2 * len(GRID_FUNCTIONALS)
    assert all(
        row["status"] == "not_applicable" and row["failure_reason"]
        for row in functional_rows
    )
    mean_rows = [row for row in rows if row["metric"] == "mean_quantile_rmse"]
    assert mean_rows and all(row["status"] == "ok" for row in mean_rows)


def test_evaluate_common_grid_uses_the_dr_payload_when_present() -> None:
    dgp, test, output, cell = _small_case()
    marginal = {name: 0.25 for name in GRID_FUNCTIONALS}
    marginal["reference"] = 0.4
    bin_contrasts = {
        name: np.array([0.1, 0.2, 0.3, 0.4]) for name in GRID_FUNCTIONALS
    }
    bin_contrasts["reference"] = np.array([0.05, 0.1, 0.15, 0.2])
    payload = {
        suffix: {
            "levels": levels,
            "marginal": marginal,
            "bin_contrasts": bin_contrasts,
        }
        for suffix, levels in (
            ("COMMON199", COMMON_LEVELS),
            ("INTERIOR", INTERIOR_LEVELS),
        )
    }
    object.__setattr__(output, "common_grid", payload)
    assert output.law is not None

    rows = evaluate_common_grid(cell, output, dgp, test.X)
    dr_rows = [row for row in rows if row["detail"] == "dense-grid DR calibration"]
    assert dr_rows

    dense = build_dgp_at_levels("IC1", COMMON_LEVELS)
    truth = dense.functional_contrast(test.X, "grid_mean")
    expected = abs(0.25 - float(np.mean(truth)))
    row = next(
        row
        for row in rows
        if row["metric"] == "tate_functional_rmse"
        and row["target_id"] == "TATE-COMMON199-grid_mean"
    )
    assert row["detail"] == "dense-grid DR calibration"
    assert row["value"] == pytest.approx(expected)

    law_rows = [
        row
        for row in rows
        if row["metric"] == "kernel_law_error" and row["status"] == "ok"
    ]
    assert law_rows
    assert all(row["detail"] != "dense-grid DR calibration" for row in law_rows)


def test_resolve_dgp_spec_reads_frozen_and_registered_regimes() -> None:
    assert resolve_dgp_spec("D1").dgp_id == "D1"
    assert resolve_dgp_spec("IC1").dgp_id == "IC1"
    with pytest.raises(ValueError):
        resolve_dgp_spec("not-a-registered-regime")
