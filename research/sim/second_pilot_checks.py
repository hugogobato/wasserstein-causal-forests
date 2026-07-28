"""Executable checks for the second-pilot changes to the WP9 harness.

Run before regenerating shard notebooks or launching Colab sessions:

    python3 research/sim/second_pilot_checks.py

Each check fails loudly rather than warning, because every one of them guards a
defect that silently corrupted the first pilot.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.baselines import _dr_inputs, _cross_fit_nuisances, _observed_U, _known_design_propensity
from sim.config import (
    DEFAULT_J, DEFAULT_K, METHOD_NAMES, PRIOR_ART_METHODS,
    frozen_coordinate_scales,
)
from sim.dgps import sample_dgp
from sim.evaluation import compute_metrics, worst_coordinate_standardized_error
from sim.incumbents import focal_spline_basis, run_incumbent
from sim.runner import build_simulation_tasks, run_simulation_cell


def check_frozen_scales() -> None:
    scales = frozen_coordinate_scales(DEFAULT_K, DEFAULT_J)
    assert scales.shape == (DEFAULT_K + DEFAULT_J,), scales.shape
    assert np.all(scales > 0) and np.all(np.isfinite(scales))
    assert len(np.unique(scales[:DEFAULT_K])) == 1, "curve scale must be constant"

    pred = np.zeros((10, DEFAULT_K + DEFAULT_J))
    true = np.zeros((10, DEFAULT_K + DEFAULT_J))
    try:
        worst_coordinate_standardized_error(pred, true, scales=None)
    except ValueError:
        pass
    else:  # pragma: no cover - the guard is the point of the check
        raise AssertionError("a missing standardizer must be rejected")
    print("  CHECK-1 PASS: frozen standardizer is positive and mandatory")


def check_d4_metric_is_finite_scale() -> None:
    """D4 inflated worst_standardized_error to 1e6-1e7 in the first pilot."""
    dgp = sample_dgp("D4", 200, 0, "feasible_growing_inner")
    prediction = np.zeros_like(dgp.true_theta_eval)
    rows = compute_metrics(prediction, dgp, "zero_predictor")
    worst = [row for row in rows if row["metric"] == "worst_standardized_error"]
    assert len(worst) == 1, worst
    value = worst[0]["value"]
    assert 0.0 < value < 100.0, f"D4 worst-coordinate error is off scale: {value}"
    print(f"  CHECK-2 PASS: D4 worst_standardized_error is on scale ({value:.3f})")


def check_regime_grid() -> None:
    tasks = build_simulation_tasks()
    grid: dict[str, set[str]] = {}
    for dgp_name, _n, regime, *_rest in tasks:
        grid.setdefault(dgp_name, set()).add(regime)
    for dgp_name in ("D0", "D1", "D2", "D3", "D4", "D5"):
        assert grid[dgp_name] == {"feasible_growing_inner"}, (dgp_name, grid[dgp_name])
    assert grid["D8"] == {
        "oracle_latent", "feasible_growing_inner", "empirical_proxy",
    }, grid["D8"]
    print(f"  CHECK-3 PASS: regime grid is noisy by default ({len(tasks)} cells)")


def check_incumbents_present() -> None:
    for method in PRIOR_ART_METHODS:
        assert method in METHOD_NAMES, method
    assert "specialized_forest" in METHOD_NAMES, "the primary adversary was dropped"

    basis = focal_spline_basis()
    assert np.allclose(basis.sum(axis=1), 1.0), "spline basis is not a partition of unity"

    dgp = sample_dgp("D1", 200, 0, "feasible_growing_inner")
    for method in PRIOR_ART_METHODS:
        prediction = run_incumbent(dgp, method, n_trees=20, n_folds=3, seed=0).prediction
        assert prediction.shape == dgp.true_theta_eval.shape, (method, prediction.shape)
        assert np.all(np.isfinite(prediction)), method
    print(f"  CHECK-4 PASS: {len(PRIOR_ART_METHODS)} incumbents run and are finite")


def check_nuisance_cache_is_exact() -> None:
    """Caching must be arithmetic-preserving, not an approximation."""
    dgp = sample_dgp("D1", 200, 0, "feasible_growing_inner")
    U_cached, e_cached, m0_cached, m1_cached = _dr_inputs(dgp, 3, 0)
    again = _dr_inputs(dgp, 3, 0)
    for cached, repeated in zip((U_cached, e_cached, m0_cached, m1_cached), again):
        assert np.array_equal(cached, repeated), "cache returned a different array"

    fresh = sample_dgp("D1", 200, 0, "feasible_growing_inner")
    U_fresh = _observed_U(fresh)
    e_fresh, m0_fresh, m1_fresh = _cross_fit_nuisances(
        fresh.X, fresh.Z, U_fresh, 3, 0,
        known_propensity=_known_design_propensity(fresh),
    )
    for cached, uncached in (
        (U_cached, U_fresh), (e_cached, e_fresh),
        (m0_cached, m0_fresh), (m1_cached, m1_fresh),
    ):
        assert np.array_equal(cached, uncached), "cache changed the nuisance values"

    # Mutating a returned array must not poison the cache.
    e_cached[:] = 0.0
    assert not np.array_equal(_dr_inputs(dgp, 3, 0)[1], e_cached)
    print("  CHECK-5 PASS: nuisance cache is exact and mutation-safe")


def check_mmd_node_cap() -> None:
    """The cap must bound cost without changing small-node behavior."""
    from wp3_odcf import DEFAULT_MMD_MAX_NODE_SAMPLE, ODCFEstimator

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    scores = rng.normal(size=(60, DEFAULT_K + DEFAULT_J))

    def fit(cap):
        return ODCFEstimator(
            K=DEFAULT_K, J=DEFAULT_J, variant="mmd_score", n_trees=4,
            random_state=0, mmd_max_node_sample=cap,
        ).fit(X, scores).predict(X[:10])

    # Every node here is smaller than the cap, so the cap must not bind.
    assert len(X) < DEFAULT_MMD_MAX_NODE_SAMPLE
    assert np.array_equal(fit(DEFAULT_MMD_MAX_NODE_SAMPLE), fit(None)), (
        "the MMD cap changed results on nodes smaller than the cap"
    )

    capped = fit(16)
    assert capped.shape == (10, DEFAULT_K + DEFAULT_J)
    assert np.all(np.isfinite(capped))
    print(
        f"  CHECK-6 PASS: MMD node cap ({DEFAULT_MMD_MAX_NODE_SAMPLE}) is inert "
        "below the cap and finite above it"
    )


def check_contract_tag_and_merge_guard() -> None:
    rows = run_simulation_cell(
        "D8", 200, "feasible_growing_inner", 0,
        n_trees=10, n_folds=3, methods=("odcf_composite", "causal_drf_port"),
    )
    versions = {row["evaluation_manifest_id"].split("-")[1] for row in rows}
    assert versions == {"v3"}, versions

    from sim.merge_results import load_rows

    stale = [dict(row) for row in rows]
    for row in stale:
        row["evaluation_manifest_id"] = row["evaluation_manifest_id"].replace(
            "eval-v3", "eval-v2", 1
        )
    with tempfile.TemporaryDirectory() as tmp:
        new_path = Path(tmp) / "new.json"
        old_path = Path(tmp) / "old.json"
        new_path.write_text(json.dumps(rows, default=str))
        old_path.write_text(json.dumps(stale, default=str))
        try:
            load_rows([new_path, old_path])
        except ValueError as exc:
            assert "contract" in str(exc), exc
        else:  # pragma: no cover - the guard is the point of the check
            raise AssertionError("mixed evaluation contracts were merged")
    print("  CHECK-7 PASS: contract is v3 and mixed-contract merges are rejected")


def main() -> None:
    print("=== WP9 second-pilot harness checks ===")
    check_frozen_scales()
    check_d4_metric_is_finite_scale()
    check_regime_grid()
    check_incumbents_present()
    check_nuisance_cache_is_exact()
    check_mmd_node_cap()
    check_contract_tag_and_merge_guard()
    print("=== second-pilot checks: ALL PASS ===")


if __name__ == "__main__":
    main()
