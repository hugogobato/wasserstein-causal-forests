"""WP2-B3 verification for the PTA-DIAGNOSTIC partial-residual prototype."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wasserstein_causal_forests.pta_bcf import dgps
from wasserstein_causal_forests.pta_bcf.diagnostic_partial import (
    CrossoverConfiguration,
    DiagnosticConfiguration,
    PTADiagnostic,
    run_crossover,
    summarize_crossover,
)
from wasserstein_causal_forests.pta_bcf.mvbcf import (
    MVBCFBudget,
    bridge_available,
    bridge_path,
    repository_root,
    rscript_executable,
)
from wasserstein_causal_forests.pta_bcf.separate_heads import HeadBudget
from wasserstein_causal_forests.pta_bcf.targets import ScaleManifest

requires_bridge = pytest.mark.skipif(
    not bridge_available(), reason="the pinned mvbcf R bridge is unavailable"
)

FAST_DIAGNOSTIC = DiagnosticConfiguration(
    n_folds=3,
    mvbcf_budget=MVBCFBudget(n_iter=200, n_burn=100, n_tree=20, n_tree_tau=10),
    head_budget=HeadBudget(
        num_trees_prognostic=20,
        num_trees_treatment=10,
        num_gfr=5,
        num_burnin=10,
        num_mcmc=40,
    ),
)


class _RecordingDiagnostic(PTADiagnostic):
    """Records the rows each forced-shared call was fitted and scored on."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def _shared_fit_predict(
        self, X_train, targets_train, treatment_train, propensity_train,
        X_eval, propensity_eval, seed,
    ):
        self.calls.append((np.array(X_train, copy=True), np.array(X_eval, copy=True)))
        return super()._shared_fit_predict(
            X_train, targets_train, treatment_train, propensity_train,
            X_eval, propensity_eval, seed,
        )


def _row_set(array: np.ndarray) -> set[tuple[float, ...]]:
    return {tuple(row) for row in array}


@requires_bridge
def test_binary_exchange_preserves_array_orientation():
    """A transposed read would silently swap targets and observations."""

    values = np.arange(12, dtype=float).reshape(4, 3) * 1.5 + 0.25
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.bin"
        echoed = Path(directory) / "echo.bin"
        values.ravel(order="F").tofile(source)
        script = (
            f'source("{bridge_path()}")\n'
            f'a <- read_array_bin("{source}", c(4L, 3L))\n'
            f'stopifnot(identical(dim(a), c(4L, 3L)))\n'
            f'write_array_bin(a, "{echoed}")\n'
        )
        completed = subprocess.run(
            [rscript_executable(), "-e", script],
            capture_output=True,
            text=True,
            cwd=repository_root(),
        )
        assert completed.returncode == 0, completed.stderr
        restored = np.fromfile(echoed).reshape((4, 3), order="F")
    np.testing.assert_allclose(restored, values)


@requires_bridge
def test_cross_fitting_never_scores_a_row_with_its_own_model():
    manifest = dgps.pta_manifest(2, functionals=())
    data = dgps.sample_dataset(120, "separate", 1, n_grid=2)
    model = _RecordingDiagnostic(
        manifest, configuration=FAST_DIAGNOSTIC, random_state=0
    )
    model.fit_predict(
        data["X"], data["treatment"], data["quantiles"], data["X"][:30]
    )

    assert len(model.calls) == FAST_DIAGNOSTIC.n_folds + 1
    for fitted_rows, scored_rows in model.calls[: FAST_DIAGNOSTIC.n_folds]:
        assert not _row_set(fitted_rows) & _row_set(scored_rows)
        assert fitted_rows.shape[0] + scored_rows.shape[0] == data["X"].shape[0]
    # The final call is the full-data component evaluated on new rows only.
    fitted_rows, scored_rows = model.calls[-1]
    assert fitted_rows.shape[0] == data["X"].shape[0]


@requires_bridge
def test_scale_manifest_matches_the_separate_head_contract():
    from wasserstein_causal_forests.pta_bcf.separate_heads import PTASeparateHeads

    manifest = dgps.pta_manifest(2, functionals=())
    data = dgps.sample_dataset(120, "shared", 4, n_grid=2)
    diagnostic = PTADiagnostic(
        manifest, configuration=FAST_DIAGNOSTIC, random_state=0
    )
    diagnostic.fit_predict(
        data["X"], data["treatment"], data["quantiles"], data["X"][:20]
    )
    separate = PTASeparateHeads(
        manifest, budget=FAST_DIAGNOSTIC.head_budget, random_state=0
    ).fit(data["X"], data["treatment"], data["quantiles"])

    assert (
        diagnostic.scale_manifest_.source_fingerprint
        == separate.scale_manifest_.source_fingerprint
    )
    np.testing.assert_allclose(
        diagnostic.scale_manifest_.scale, separate.scale_manifest_.scale
    )


def test_uninformative_residual_predictions_switch_the_component_off():
    manifest = dgps.pta_manifest(2, functionals=())
    model = PTADiagnostic(manifest, configuration=FAST_DIAGNOSTIC)
    rng = np.random.default_rng(0)
    residual = rng.normal(size=(500, manifest.dimension))
    noise = rng.normal(size=(500, manifest.dimension))

    np.testing.assert_array_equal(
        model._tune_weights(residual, noise),
        np.zeros(manifest.dimension),
    )
    np.testing.assert_array_equal(
        model._tune_weights(residual, np.zeros_like(residual)),
        np.zeros(manifest.dimension),
    )
    # A residual that the heads predict exactly must keep the component.
    np.testing.assert_array_equal(
        model._tune_weights(residual, residual),
        np.ones(manifest.dimension),
    )


@requires_bridge
def test_target_specific_signal_improves_on_the_forced_shared_endpoint():
    manifest = dgps.pta_manifest(3, functionals=("grid_sd",))
    train = dgps.sample_dataset(200, "separate", 0, n_grid=3)
    test = dgps.sample_dataset(200, "separate", 77, n_grid=3)
    prediction = PTADiagnostic(
        manifest, configuration=FAST_DIAGNOSTIC, random_state=0
    ).fit_predict(train["X"], train["treatment"], train["quantiles"], test["X"])

    truth = dgps.true_target_contrast(
        test["X"], "separate", manifest, n_monte_carlo=200
    )
    scale = ScaleManifest.fit(manifest.build(train["quantiles"]), manifest).scale

    def error(estimate: np.ndarray) -> float:
        return float(np.sqrt(np.mean(((estimate - truth) / scale) ** 2)))

    assert error(prediction.total_contrast) < error(prediction.shared_contrast)
    assert prediction.weights.max() > 0.0
    assert np.all(prediction.weights >= 0.0) and np.all(prediction.weights <= 1.0)


@requires_bridge
def test_null_regime_keeps_the_residual_component_small():
    manifest = dgps.pta_manifest(3, functionals=("grid_sd",))
    train = dgps.sample_dataset(200, "null", 2, n_grid=3)
    test = dgps.sample_dataset(200, "null", 88, n_grid=3)
    prediction = PTADiagnostic(
        manifest, configuration=FAST_DIAGNOSTIC, random_state=2
    ).fit_predict(train["X"], train["treatment"], train["quantiles"], test["X"])

    target_scale = ScaleManifest.fit(
        manifest.build(train["quantiles"]), manifest
    ).scale
    relative = np.abs(prediction.residual_contrast).mean(axis=0) / target_scale
    assert np.all(relative < 0.15), relative


@requires_bridge
def test_crossover_rows_carry_no_posterior_uncertainty_fields():
    configuration = CrossoverConfiguration(
        n_train=120,
        n_test=120,
        n_grid=2,
        functionals=(),
        n_folds=3,
        mvbcf_budget=FAST_DIAGNOSTIC.mvbcf_budget,
        head_budget=FAST_DIAGNOSTIC.head_budget,
        truth_monte_carlo=100,
    )
    frame = run_crossover(
        seeds=(0,), regimes=("null",), configuration=configuration
    )
    assert set(frame["method"]) == {"PTA-S", "PTA-F", "PTA-DIAGNOSTIC"}
    assert (frame["status"] == "ok").all(), frame["failure_reason"].tolist()
    forbidden = ("coverage", "interval", "credible", "posterior_sd", "ci_")
    assert not [
        column
        for column in frame.columns
        if any(token in column.lower() for token in forbidden)
    ]
    assert (frame["inference"] == "point-prediction-only").all()


def test_crossover_decision_rule_is_two_sided():
    def frame(shared_diag: float, separate_diag: float, null_diag: float):
        rows = []
        base = {
            "D4": {"PTA-S": 1.0, "PTA-F": 0.8, "PTA-DIAGNOSTIC": shared_diag},
            "D3": {"PTA-S": 0.8, "PTA-F": 1.0, "PTA-DIAGNOSTIC": separate_diag},
            "D2": {"PTA-S": 1.0, "PTA-F": 1.0, "PTA-DIAGNOSTIC": null_diag},
        }
        for regime, methods in base.items():
            for method, value in methods.items():
                rows.append(
                    {
                        "dgp": regime,
                        "method": method,
                        "value": value,
                        "status": "ok",
                        "residual_weight_mean": 0.5,
                    }
                )
        return pd.DataFrame(rows)

    passing = summarize_crossover(frame(0.90, 0.90, 1.00))
    assert passing["decision"] == "ENABLE-WP2-B4"

    # No gain in the shared-favorable regime.
    assert (
        summarize_crossover(frame(1.00, 0.90, 1.00))["decision"]
        == "RETAIN-STRONGEST-ENDPOINT"
    )
    # Material loss under the null.
    failing_null = summarize_crossover(frame(0.90, 0.90, 1.20))
    assert failing_null["decision"] == "RETAIN-STRONGEST-ENDPOINT"
    assert failing_null["retained_endpoint"] in {"PTA-S", "PTA-F", "PTA-DIAGNOSTIC"}

    incomplete = summarize_crossover(
        frame(0.9, 0.9, 1.0).query("method != 'PTA-F'")
    )
    assert incomplete["decision"] == "INDETERMINATE"
