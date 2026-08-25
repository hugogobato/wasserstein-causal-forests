"""PTA-DIAGNOSTIC: forced-shared fit plus cross-fitted target-specific residuals.

This is a sequential composition, not a joint posterior. The forced-shared
endpoint (PTA-F) is fitted with cross-fitting, its held-out residuals feed
target-specific scalar heads, and a training-only weight decides how much of
the residual component to keep. Nothing here produces posterior uncertainty
for the composed predictor, so no coverage or interval field is emitted
anywhere in this module or in the artifacts it writes.

The forced-shared sampler is reached through the R bridge, which fits and
predicts in one call. The diagnostic therefore exposes `fit_predict` rather
than a separate `predict`, so no component is ever refitted on evaluation
rows.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from . import dgps
from .mvbcf import MVBCFBudget, MVBCFForcedShared
from .separate_heads import (
    CrossFittedPropensity,
    HeadBudget,
    PTASeparateHeads,
)
from .targets import FoldPlan, ScaleManifest, TargetManifest, assert_disjoint, make_folds

PTA_DIAGNOSTIC_METHOD_ID = "PTA-DIAGNOSTIC"

#: Candidate weights on the target-specific residual component. Zero must be
#: reachable so a null residual is switched off rather than fitted to noise.
DEFAULT_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class DiagnosticConfiguration:
    n_folds: int = 5
    weight_grid: tuple[float, ...] = DEFAULT_WEIGHT_GRID
    mvbcf_budget: MVBCFBudget = MVBCFBudget()
    head_budget: HeadBudget = HeadBudget()

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("cross-fitting needs at least two folds")
        grid = np.asarray(self.weight_grid, dtype=float)
        if grid.size == 0 or np.any(grid < 0.0) or np.any(grid > 1.0):
            raise ValueError("weight_grid must be a nonempty subset of [0, 1]")


@dataclass(frozen=True)
class DiagnosticPrediction:
    """Conditional target contrasts on the original target scale.

    `shared_contrast` is the forced-shared endpoint PTA-F evaluated on the same
    rows, so the crossover comparison never refits it.
    """

    shared_contrast: NDArray[np.float64]
    residual_contrast: NDArray[np.float64]
    total_contrast: NDArray[np.float64]
    weights: NDArray[np.float64]
    crossfitted_residual_norm: NDArray[np.float64]
    runtime_seconds: float
    diagnostics: dict[str, object] = field(default_factory=dict)


def _prediction_at_own_arm(
    control: NDArray[np.float64],
    contrast: NDArray[np.float64],
    treatment: NDArray[np.int64],
) -> NDArray[np.float64]:
    return control + treatment[:, None] * contrast


class PTADiagnostic:
    """Cross-fitted shared-plus-specific composition of PTA-F and PTA-S."""

    def __init__(
        self,
        manifest: TargetManifest,
        *,
        configuration: DiagnosticConfiguration = DiagnosticConfiguration(),
        random_state: int = 0,
    ) -> None:
        self.manifest = manifest
        self.configuration = configuration
        self.random_state = int(random_state)

    def _shared_fit_predict(
        self,
        X_train: NDArray[np.float64],
        targets_train: NDArray[np.float64],
        treatment_train: NDArray[np.int64],
        propensity_train: NDArray[np.float64],
        X_eval: NDArray[np.float64],
        propensity_eval: NDArray[np.float64],
        seed: int,
    ):
        # The propensity enters the prognostic design, as in the published model.
        control_design = np.column_stack([X_train, propensity_train])
        control_design_eval = np.column_stack([X_eval, propensity_eval])
        return MVBCFForcedShared(
            budget=self.configuration.mvbcf_budget, random_state=seed
        ).fit_predict(
            control_design,
            targets_train,
            treatment_train,
            X_train,
            X_control_test=control_design_eval,
            X_moderator_test=X_eval,
        )

    def _residual_head_model(self, seed: int) -> PTASeparateHeads:
        return PTASeparateHeads(
            self.manifest,
            budget=self.configuration.head_budget,
            n_folds=self.configuration.n_folds,
            random_state=seed,
        )

    def _tune_weights(
        self,
        residual: NDArray[np.float64],
        residual_prediction: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Pick the residual weight per coordinate from training rows only."""

        grid = np.asarray(self.configuration.weight_grid, dtype=float)
        errors = np.stack(
            [
                np.mean((residual - weight * residual_prediction) ** 2, axis=0)
                for weight in grid
            ]
        )
        return grid[np.argmin(errors, axis=0)]

    def fit_predict(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        X_evaluation: ArrayLike,
        *,
        folds: FoldPlan | None = None,
    ) -> DiagnosticPrediction:
        started = time.perf_counter()
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        x_eval = np.asarray(X_evaluation, dtype=float)
        if x_eval.ndim != 2 or x_eval.shape[1] != x.shape[1]:
            raise ValueError("evaluation rows must have the training covariates")
        targets = self.manifest.build(quantiles)

        self.folds_ = folds or make_folds(
            x.shape[0],
            a,
            n_folds=self.configuration.n_folds,
            random_state=self.random_state,
        )
        # Scaling and both components use the fitting rows only.
        self.scale_manifest_ = ScaleManifest.fit(targets, self.manifest)
        scaled = self.scale_manifest_.transform(targets)

        self.propensity_model_ = CrossFittedPropensity(
            random_state=self.random_state
        ).fit(x, a, self.folds_)
        propensity = self.propensity_model_.train_scores_
        propensity_eval = self.propensity_model_.predict(x_eval)

        # Stage 1. Cross-fitted forced-shared predictions. Each row is scored
        # by a sampler that never saw it, so the residuals below are honest.
        shared_control = np.empty_like(scaled)
        shared_contrast = np.empty_like(scaled)
        for fold in self.folds_.fold_ids:
            train = self.folds_.train_index(fold)
            evaluate = self.folds_.test_index(fold)
            if evaluate.size == 0:
                continue
            assert_disjoint(train, evaluate)
            result = self._shared_fit_predict(
                x[train],
                scaled[train],
                a[train],
                propensity[train],
                x[evaluate],
                propensity[evaluate],
                self.random_state + 100 + fold,
            )
            shared_control[evaluate] = result.control_mean("test")
            shared_contrast[evaluate] = result.contrast_mean("test")
        self.crossfitted_shared_control_ = shared_control
        self.crossfitted_shared_contrast_ = shared_contrast

        # Stage 2. What forced sharing left behind at each row's own arm.
        residual = scaled - _prediction_at_own_arm(shared_control, shared_contrast, a)
        self.crossfitted_residual_ = residual

        # Stage 3. Out-of-fold residual-head predictions decide the weight.
        # Fitting the weight on in-sample residual predictions would drive it
        # to one whether or not a target-specific signal exists.
        residual_out_of_fold = np.empty_like(residual)
        for fold in self.folds_.fold_ids:
            train = self.folds_.train_index(fold)
            evaluate = self.folds_.test_index(fold)
            if evaluate.size == 0:
                continue
            inner_folds = make_folds(
                train.size,
                a[train],
                n_folds=self.configuration.n_folds,
                random_state=self.random_state + fold,
            )
            heads = self._residual_head_model(
                self.random_state + 200 + fold
            ).fit_target_matrix(
                x[train],
                a[train],
                residual[train],
                folds=inner_folds,
                propensity=propensity[train],
            )
            draws = heads.predict_draws(
                x[evaluate], propensity=propensity[evaluate]
            )
            residual_out_of_fold[evaluate] = _prediction_at_own_arm(
                draws["control"].mean(axis=2),
                draws["contrast"].mean(axis=2),
                a[evaluate],
            )
        self.weights_ = self._tune_weights(residual, residual_out_of_fold)

        # Stage 4. Full-data components evaluated on the requested rows.
        shared_evaluation = self._shared_fit_predict(
            x,
            scaled,
            a,
            propensity,
            x_eval,
            propensity_eval,
            self.random_state + 1,
        )
        self.residual_heads_ = self._residual_head_model(
            self.random_state + 300
        ).fit_target_matrix(
            x, a, residual, folds=self.folds_, propensity=propensity
        )
        shared_contrast_eval = shared_evaluation.contrast_mean("test")
        residual_contrast_eval = self.residual_heads_.predict_contrast(
            x_eval, propensity=propensity_eval
        )
        total = shared_contrast_eval + self.weights_ * residual_contrast_eval

        inverse = self.scale_manifest_.inverse_transform_contrast
        return DiagnosticPrediction(
            shared_contrast=inverse(shared_contrast_eval),
            residual_contrast=inverse(self.weights_ * residual_contrast_eval),
            total_contrast=inverse(total),
            weights=self.weights_.copy(),
            crossfitted_residual_norm=np.sqrt(np.mean(residual**2, axis=0)),
            runtime_seconds=time.perf_counter() - started,
            diagnostics={
                "method_id": PTA_DIAGNOSTIC_METHOD_ID,
                "inference": "point-prediction-only",
                "n_folds": self.configuration.n_folds,
                "shared_sampler_seconds": float(
                    shared_evaluation.meta["elapsed_seconds"]
                ),
                "n_shared_draws": int(shared_evaluation.n_draws),
            },
        )


# ---------------------------------------------------------------------------
# WP2-B3 crossover cells
# ---------------------------------------------------------------------------

REGIME_LABELS = {"null": "D2", "separate": "D3", "shared": "D4"}


@dataclass(frozen=True)
class CrossoverConfiguration:
    n_train: int = 400
    n_test: int = 500
    n_grid: int = 5
    functionals: tuple[str, ...] = ("grid_mean", "grid_sd")
    n_folds: int = 4
    mvbcf_budget: MVBCFBudget = MVBCFBudget(
        n_iter=600, n_burn=300, n_tree=50, n_tree_tau=20
    )
    head_budget: HeadBudget = HeadBudget(
        num_trees_prognostic=50,
        num_trees_treatment=20,
        num_gfr=10,
        num_burnin=100,
        num_mcmc=200,
    )
    truth_monte_carlo: int = 400


def _scaled_contrast_rmse(
    estimate: NDArray[np.float64],
    truth: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> float:
    """Coordinate-standardized RMSE so all D targets contribute comparably."""

    error = (estimate - truth) / scale
    return float(np.sqrt(np.mean(error**2)))


def run_crossover(
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    regimes: tuple[str, ...] = ("shared", "separate", "null"),
    configuration: CrossoverConfiguration = CrossoverConfiguration(),
) -> pd.DataFrame:
    """Run the D3/D4/null cells comparing PTA-S, PTA-F, and the diagnostic."""

    manifest = dgps.pta_manifest(
        configuration.n_grid, functionals=configuration.functionals
    )
    diagnostic_configuration = DiagnosticConfiguration(
        n_folds=configuration.n_folds,
        mvbcf_budget=configuration.mvbcf_budget,
        head_budget=configuration.head_budget,
    )
    hyperparameters = json.dumps(
        {
            "crossover": asdict(configuration),
            "diagnostic": {
                "n_folds": diagnostic_configuration.n_folds,
                "weight_grid": list(diagnostic_configuration.weight_grid),
            },
        },
        sort_keys=True,
        default=str,
    )

    rows: list[dict[str, object]] = []
    for regime in regimes:
        for seed in seeds:
            train = dgps.sample_dataset(
                configuration.n_train, regime, seed, n_grid=configuration.n_grid
            )
            test = dgps.sample_dataset(
                configuration.n_test,
                regime,
                5000 + seed,
                n_grid=configuration.n_grid,
            )
            truth = dgps.true_target_contrast(
                test["X"],
                regime,
                manifest,
                n_monte_carlo=configuration.truth_monte_carlo,
            )
            scale = ScaleManifest.fit(
                manifest.build(train["quantiles"]), manifest
            ).scale

            estimates: dict[str, tuple[NDArray[np.float64], float, dict]] = {}
            failures: dict[str, str] = {}

            started = time.perf_counter()
            try:
                separate = PTASeparateHeads(
                    manifest,
                    budget=configuration.head_budget,
                    n_folds=configuration.n_folds,
                    random_state=seed,
                ).fit(train["X"], train["treatment"], train["quantiles"])
                estimates["PTA-S"] = (
                    separate.predict_contrast(test["X"]),
                    time.perf_counter() - started,
                    {},
                )
            except Exception as error:
                failures["PTA-S"] = f"{type(error).__name__}: {error}"

            try:
                prediction = PTADiagnostic(
                    manifest,
                    configuration=diagnostic_configuration,
                    random_state=seed,
                ).fit_predict(
                    train["X"],
                    train["treatment"],
                    train["quantiles"],
                    test["X"],
                )
                estimates["PTA-F"] = (
                    prediction.shared_contrast,
                    float(prediction.diagnostics["shared_sampler_seconds"]),
                    {},
                )
                estimates["PTA-DIAGNOSTIC"] = (
                    prediction.total_contrast,
                    prediction.runtime_seconds,
                    {
                        "residual_weight_mean": float(prediction.weights.mean()),
                        "residual_weight_max": float(prediction.weights.max()),
                        "residual_share": float(
                            np.mean(np.abs(prediction.residual_contrast))
                            / max(np.mean(np.abs(prediction.total_contrast)), 1e-12)
                        ),
                    },
                )
            except Exception as error:
                failures["PTA-F"] = f"{type(error).__name__}: {error}"
                failures["PTA-DIAGNOSTIC"] = failures["PTA-F"]

            for method in ("PTA-S", "PTA-F", "PTA-DIAGNOSTIC"):
                if method in estimates:
                    estimate, runtime, extra = estimates[method]
                    value = _scaled_contrast_rmse(estimate, truth, scale)
                    status, reason = "ok", ""
                else:
                    value, runtime, extra = float("nan"), float("nan"), {}
                    status, reason = "failed", failures.get(method, "not attempted")
                rows.append(
                    {
                        "claim_id": "WP2-B3",
                        "dgp": REGIME_LABELS[regime],
                        "regime": regime,
                        "observation_regime": "ORACLE-V1",
                        "evaluation_manifest_id": "PTA-CROSSOVER-v1",
                        "target_id": "MEANQ-A-K+TCATE-K-j+REF-TCATE-K",
                        "inference": "point-prediction-only",
                        "n": configuration.n_train,
                        "n_test": configuration.n_test,
                        "K": configuration.n_grid,
                        "J": len(configuration.functionals),
                        "D": manifest.dimension,
                        "M": 0,
                        "seed": seed,
                        "method": method,
                        "hyperparameter_manifest_id": "PTA-CROSSOVER-HYPER-v1",
                        "hyperparameters": hyperparameters,
                        "metric": "scaled_contrast_rmse",
                        "value": value,
                        "residual_weight_mean": extra.get(
                            "residual_weight_mean", float("nan")
                        ),
                        "residual_weight_max": extra.get(
                            "residual_weight_max", float("nan")
                        ),
                        "residual_share": extra.get("residual_share", float("nan")),
                        "runtime_seconds": runtime,
                        "status": status,
                        "failure_reason": reason,
                    }
                )
    return pd.DataFrame(rows)


#: Preregistered crossover rule. The diagnostic must beat the endpoint that
#: each regime disadvantages, and must not lose materially under the null.
CROSSOVER_WIN_MARGIN = 0.02
CROSSOVER_NULL_TOLERANCE = 0.10


def summarize_crossover(results: pd.DataFrame) -> dict[str, object]:
    """Apply the preregistered WP2-B3 decision rule."""

    successful = results[results["status"] == "ok"]
    expected = {
        (regime, method)
        for regime in ("D2", "D3", "D4")
        for method in ("PTA-S", "PTA-F", "PTA-DIAGNOSTIC")
    }
    observed = set(zip(successful["dgp"], successful["method"], strict=False))
    if not expected.issubset(observed):
        return {
            "decision": "INDETERMINATE",
            "reason": "missing method-by-regime cells",
            "missing": sorted(str(cell) for cell in expected - observed),
        }

    means = successful.groupby(["dgp", "method"])["value"].mean()

    def relative_gain(regime: str, endpoint: str) -> float:
        reference = float(means.loc[(regime, endpoint)])
        diagnostic = float(means.loc[(regime, "PTA-DIAGNOSTIC")])
        return (reference - diagnostic) / reference

    shared_gain = relative_gain("D4", "PTA-S")
    separate_gain = relative_gain("D3", "PTA-F")
    null_best = min(
        float(means.loc[("D2", "PTA-S")]), float(means.loc[("D2", "PTA-F")])
    )
    null_loss = (float(means.loc[("D2", "PTA-DIAGNOSTIC")]) - null_best) / null_best

    passed = (
        shared_gain >= CROSSOVER_WIN_MARGIN
        and separate_gain >= CROSSOVER_WIN_MARGIN
        and null_loss <= CROSSOVER_NULL_TOLERANCE
    )
    dominant = None
    if not passed:
        endpoint_means = {
            method: float(
                np.mean([means.loc[(regime, method)] for regime in ("D2", "D3", "D4")])
            )
            for method in ("PTA-S", "PTA-F", "PTA-DIAGNOSTIC")
        }
        dominant = min(endpoint_means, key=endpoint_means.get)

    return {
        "decision": "ENABLE-WP2-B4" if passed else "RETAIN-STRONGEST-ENDPOINT",
        "inference": "point-prediction-only",
        "required_win_margin": CROSSOVER_WIN_MARGIN,
        "allowed_null_loss": CROSSOVER_NULL_TOLERANCE,
        "shared_gain_over_pta_s": shared_gain,
        "separate_gain_over_pta_f": separate_gain,
        "null_relative_loss": null_loss,
        "retained_endpoint": dominant,
        "mean_scaled_contrast_rmse": {
            f"{regime}:{method}": float(means.loc[(regime, method)])
            for regime in ("D2", "D3", "D4")
            for method in ("PTA-S", "PTA-F", "PTA-DIAGNOSTIC")
        },
        "mean_residual_weight": {
            regime: float(
                successful[
                    (successful["dgp"] == regime)
                    & (successful["method"] == "PTA-DIAGNOSTIC")
                ]["residual_weight_mean"].mean()
            )
            for regime in ("D2", "D3", "D4")
        },
    }
