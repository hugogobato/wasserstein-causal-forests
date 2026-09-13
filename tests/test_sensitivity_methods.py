"""Dense-grid DR calibration payload and the sensitivity method identifiers.

The flexible and oracle adapters must produce the common-grid payload the
`common_grid` evaluator consumes, without changing the frozen default path.
"""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.g3 import runner

from wasserstein_causal_forests.g3.common_grid import (
    COMMON_LEVELS,
    INTERIOR_LEVELS,
)
from wasserstein_causal_forests.g3.dgps import build_dgp
from wasserstein_causal_forests.g3.manifest import Cell
from wasserstein_causal_forests.g3.phase6 import PHASE6_METHOD_REGISTRY
from wasserstein_causal_forests.g3.phase6_methods import DRAdapter
from wasserstein_causal_forests.g3.sensitivity_methods import (
    SENSITIVITY_METHOD_REGISTRY,
    register_sensitivity_methods,
)
from wasserstein_causal_forests.g3.wcf_sensitivity import (
    build_wcf_method_registry,
)

NATIVE_GRID = 5
FUNCTIONALS = ("grid_mean", "grid_sd")
FAST_PARAMETERS = dict(
    n_particles=3,
    n_folds=2,
    contrast_candidates=(0.0,),
    n_estimators=10,
    learning_rate=0.2,
    max_depth=2,
    min_samples_leaf=5,
    min_arm_leaf=2,
    arm_shrinkage=5.0,
)


@pytest.fixture(scope="module")
def dgp():
    return build_dgp("IC1", NATIVE_GRID)


@pytest.fixture(scope="module")
def train(dgp):
    return dgp.sample(n_rows=200, seed=1)


@pytest.fixture(scope="module")
def test_design(dgp):
    return dgp.sample(n_rows=60, seed=900001)


def _fit(adapter, dgp, train, test_design, seed=11):
    return adapter.fit_predict(train, test_design.X, dgp, FUNCTIONALS, seed=seed)


@pytest.fixture(scope="module")
def flexible_output(dgp, train, test_design):
    adapter = DRAdapter(
        propensity_factory="hist_gradient_boosting",
        common_grid_levels="COMMON199+INTERIOR",
        **FAST_PARAMETERS,
    )
    return _fit(adapter, dgp, train, test_design)


@pytest.fixture(scope="module")
def oracle_output(dgp, train, test_design):
    adapter = DRAdapter(
        oracle_propensity=True,
        common_grid_levels="COMMON199+INTERIOR",
        **FAST_PARAMETERS,
    )
    return _fit(adapter, dgp, train, test_design)


@pytest.fixture(scope="module")
def default_output(dgp, train, test_design):
    return _fit(DRAdapter(**FAST_PARAMETERS), dgp, train, test_design)


def test_flexible_adapter_emits_both_dense_level_sets(flexible_output):
    payload = flexible_output.common_grid
    assert set(payload) == {"COMMON199", "INTERIOR"}
    np.testing.assert_array_equal(payload["COMMON199"]["levels"], COMMON_LEVELS)
    np.testing.assert_array_equal(payload["INTERIOR"]["levels"], INTERIOR_LEVELS)
    for entry in payload.values():
        for name in ("grid_mean", "reference"):
            marginal = entry["marginal"][name]
            assert isinstance(marginal, float)
            assert np.isfinite(marginal)
            contrasts = entry["bin_contrasts"][name]
            assert contrasts.shape == (4,)
            assert np.any(np.isfinite(contrasts))


def test_flexible_adapter_records_the_dense_level_count(flexible_output):
    value = flexible_output.diagnostics["dense_dr_n_levels"]
    assert isinstance(value, float)
    assert value == float(COMMON_LEVELS.size)


def test_oracle_adapter_runs_and_emits_the_payload(oracle_output):
    payload = oracle_output.common_grid
    assert set(payload) == {"COMMON199", "INTERIOR"}
    for entry in payload.values():
        assert np.isfinite(entry["marginal"]["reference"])
        assert entry["bin_contrasts"]["reference"].shape == (4,)
    assert oracle_output.diagnostics["dense_dr_n_levels"] == float(
        INTERIOR_LEVELS.size
    )


def test_default_adapter_stays_free_of_the_dense_payload(default_output):
    assert not hasattr(default_output, "common_grid")
    assert set(default_output.functionals) == {
        "grid_mean",
        "grid_sd",
        "grid_skewness",
        "grid_upper_tail_mean",
    }
    assert "ehat_mean" in default_output.diagnostics
    assert "dense_dr_n_levels" not in default_output.diagnostics


def test_static_registry_mirrors_the_frozen_wcf_parameters():
    frozen = build_wcf_method_registry()
    for name in ("cwdb_dr_flex", "cwdb_dr_oracle"):
        entry = SENSITIVITY_METHOD_REGISTRY[name]
        assert entry["adapter"] == "cwdb_dr"
        assert entry["produces_law"] is True
        assert entry["cross_fitted"] is True
        assert entry["parameters"] == frozen[name]["parameters"]
        assert entry["target_ids"] == PHASE6_METHOD_REGISTRY["cwdb_dr"]["target_ids"]
    assert SENSITIVITY_METHOD_REGISTRY["cwdb_dr_flex"]["role"] == "variant"
    assert SENSITIVITY_METHOD_REGISTRY["cwdb_dr_oracle"]["role"] == "diagnostic"


def test_registration_is_idempotent_and_resolves_through_build_adapter():
    register_sensitivity_methods()
    register_sensitivity_methods()
    flex_cell = Cell(
        "wcf_sensitivity_v1_flex", "IC1", 200, NATIVE_GRID, 3, "cwdb_dr_flex", 0
    )
    flex_adapter = runner.build_adapter(flex_cell, None)
    assert isinstance(flex_adapter, DRAdapter)
    assert flex_adapter.propensity_factory == "hist_gradient_boosting"
    assert flex_adapter.oracle_propensity is False
    assert flex_adapter.contrast_candidates == (0.0, 50.0, 500.0)
    assert flex_adapter.n_folds == 3
    assert flex_adapter.common_grid_level_sets == ("COMMON199", "INTERIOR")

    oracle_cell = Cell(
        "wcf_sensitivity_v1_flex", "IC1", 200, NATIVE_GRID, 3, "cwdb_dr_oracle", 0
    )
    oracle_adapter = runner.build_adapter(oracle_cell, None)
    assert isinstance(oracle_adapter, DRAdapter)
    assert oracle_adapter.oracle_propensity is True
    assert oracle_adapter.propensity_factory is None
