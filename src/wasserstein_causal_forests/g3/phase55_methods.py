"""Method adapters for the Phase 5.5 variants.

Each adapter fits on one training sample and returns a `MethodOutput`. The
mean-only variants fill only the barycenter-level slots and report every law,
functional, and reference target as absent, which the evaluation layer turns
into explicit ``not_applicable`` rows: a mean-only variant must be rejected if
a law metric is requested, and the rejection is structural, not conventional.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from ..cwdb.mutau import MutauCWDBRegressor
from ..meta_learners.r_learner import VectorRLearner
from ..meta_learners.x_learner import VectorXLearner
from .dgps import DGPSample, DistributionalDGP
from .laws import LawPrediction
from .methods import MethodOutput, _output_from_laws, peak_ram_mb
from .phase55 import RMEAN_SELECTION_FOLDS, RMEAN_SHRINKAGE_CANDIDATES


def _mean_only_output(
    arm_means: dict[int, NDArray[np.float64]],
    *,
    diagnostics: dict[str, float] | None = None,
    fit_seconds: float,
    predict_seconds: float,
    peak_ram_mb_value: float,
) -> MethodOutput:
    """A barycenter-only output: no law, no functionals, no reference."""

    return MethodOutput(
        mean_quantiles={arm: arm_means[arm] for arm in (0, 1)},
        functionals={},
        reference=None,
        law=None,
        supported_functionals=(),
        n_atoms=0,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        peak_ram_mb=peak_ram_mb_value,
        diagnostics=diagnostics or {},
    )


class RMetaLearnerAdapter:
    """`cwdb_rmean`: cross-fitted vector R-learner on rescaled quantiles."""

    produces_law = False

    def fit_predict(
        self,
        train: DGPSample,
        X_test: NDArray[np.float64],
        dgp: DistributionalDGP,
        functionals: tuple[str, ...],
        *,
        seed: int,
    ) -> MethodOutput:
        before = peak_ram_mb()
        started = time.perf_counter()
        model = VectorRLearner(
            contrast_candidates=RMEAN_SHRINKAGE_CANDIDATES,
            n_selection_folds=RMEAN_SELECTION_FOLDS,
            random_state=seed,
        )
        model.fit(train.X, train.treatment, train.quantiles, dgp.grid.weights)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        arm_means = {
            arm: model.predict_mean_quantiles(X_test, arm) for arm in (0, 1)
        }
        predict_seconds = time.perf_counter() - started
        return _mean_only_output(
            arm_means,
            diagnostics={
                "selected_contrast_shrinkage": float(
                    model.selected_contrast_shrinkage_
                ),
                "train_risk": model.train_risk_,
            },
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb_value=max(peak_ram_mb() - before, 0.0),
        )


class XMetaLearnerAdapter:
    """`cwdb_xmean`: cross-fitted vector X-learner on rescaled quantiles."""

    produces_law = False

    def fit_predict(
        self,
        train: DGPSample,
        X_test: NDArray[np.float64],
        dgp: DistributionalDGP,
        functionals: tuple[str, ...],
        *,
        seed: int,
    ) -> MethodOutput:
        before = peak_ram_mb()
        started = time.perf_counter()
        model = VectorXLearner(random_state=seed)
        model.fit(train.X, train.treatment, train.quantiles, dgp.grid.weights)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        arm_means = {
            arm: model.predict_mean_quantiles(X_test, arm) for arm in (0, 1)
        }
        predict_seconds = time.perf_counter() - started
        return _mean_only_output(
            arm_means,
            diagnostics={
                # The mean fitted propensity quantifies the imbalance stress
                # each cell actually saw, so the imbalance suite can be
                # audited against its declared range rather than assumed.
                "ehat_mean": float(np.mean(model.ehat_train_)),
                "ehat_sd": float(np.std(model.ehat_train_)),
            },
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb_value=max(peak_ram_mb() - before, 0.0),
        )


class MutauAdapter:
    """`cwdb_mutau`: particle booster with prognostic/contrast leaf fields.

    The only Phase 5.5 variant that produces a law; it is held to the same
    particle validity checks as C-WDB R3 and reports every law-level target.
    """

    produces_law = True

    def __init__(
        self,
        *,
        contrast_candidates: tuple[float, ...] | None = None,
        n_folds: int = 3,
        n_particles: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.12,
        max_depth: int = 4,
        min_samples_leaf: int = 10,
        min_arm_leaf: int = 5,
        arm_shrinkage: float = 5.0,
        collision_epsilon: float = 1e-3,
    ) -> None:
        self.parameters = {
            "architecture": "v1",
            "n_particles": n_particles,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_arm_leaf": min_arm_leaf,
            "arm_shrinkage": arm_shrinkage,
            "sharing": "partial",
            "init_sharing": "pooled",
            "collision_epsilon": collision_epsilon,
        }
        self.contrast_candidates = contrast_candidates
        self.n_folds = n_folds

    def fit_predict(
        self,
        train: DGPSample,
        X_test: NDArray[np.float64],
        dgp: DistributionalDGP,
        functionals: tuple[str, ...],
        *,
        seed: int,
    ) -> MethodOutput:
        weights = dgp.grid.weights
        model = MutauCWDBRegressor(
            contrast_candidates=self.contrast_candidates,
            n_folds=self.n_folds,
            random_state=seed,
            **self.parameters,
        )
        before = peak_ram_mb()
        started = time.perf_counter()
        model.fit(train.X, train.treatment, train.quantiles, weights)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        laws = {
            arm: LawPrediction.from_particles(model.predict_particles(X_test, arm))
            for arm in (0, 1)
        }
        predict_seconds = time.perf_counter() - started

        output = _output_from_laws(
            laws,
            weights,
            dgp.grid.reference_quantiles(),
            functionals,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
        )
        diagnostics = {
            **output.diagnostics,
            "n_boosting_steps": float(len(model.training_history_)),
            "train_risk": float(model.train_risk_),
            "selected_contrast_shrinkage": float(
                model.selected_contrast_shrinkage_
            ),
        }
        object.__setattr__(output, "diagnostics", diagnostics)
        return output
