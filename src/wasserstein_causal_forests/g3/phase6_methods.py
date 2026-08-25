"""Method adapters for the Phase 6 variants.

Each adapter fits on one training sample and returns a `MethodOutput`. The
contract rules carry over from the earlier phases unchanged: a variant that
produces a law is scored on every law metric; one that does not reports
`not_applicable` there by construction. The two layers of `cwdb_dr` are kept
visible in its diagnostics so no reader can mistake the DR-aggregated
functional columns for law integrals.
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np
from numpy.typing import NDArray

from ..cwdb.dr_calibration import DRCalibratedCWDB
from ..cwdb.krr_booster import KRRArmParticleBooster
from ..cwdb.smoothing import SmoothedCWDB
from ..meta_learners.functional_r_learner import FunctionalRLearner
from ..pta_bcf.targets import GRID_FUNCTIONALS
from .dgps import DGPSample, DistributionalDGP, moderator_bins
from .laws import LawPrediction
from .methods import MethodOutput, _output_from_laws, peak_ram_mb
from .phase6 import PHASE6_CONTRAST_CANDIDATES, PHASE6_SELECTION_FOLDS


def _declared_functionals(
    weights: NDArray[np.float64], reference: NDArray[np.float64]
) -> dict[str, object]:
    """The four grid functionals plus the reference distance, weight-closed."""

    functions: dict[str, object] = {
        name: partial(h, w=weights) for name, h in GRID_FUNCTIONALS.items()
    }
    difference_scale = np.sqrt(weights)

    def reference_distance(block: NDArray[np.float64]) -> NDArray[np.float64]:
        scaled = (block - reference) * difference_scale
        return np.sqrt(np.sum(scaled * scaled, axis=-1))

    functions["reference"] = reference_distance
    return functions


class DRAdapter:
    """`cwdb_dr`: R3 law plus a doubly-robust functional calibration layer."""

    produces_law = True

    def __init__(
        self,
        *,
        contrast_candidates: tuple[float, ...] | None = None,
        n_folds: int = PHASE6_SELECTION_FOLDS,
        n_particles: int = 10,
        **budget: object,
    ) -> None:
        self.contrast_candidates = (
            PHASE6_CONTRAST_CANDIDATES
            if contrast_candidates is None
            else tuple(contrast_candidates)
        )
        self.n_folds = n_folds
        self.n_particles = n_particles
        self.budget = dict(budget)

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
        model = DRCalibratedCWDB(
            functionals=_declared_functionals(weights, dgp.grid.reference_quantiles()),
            contrast_candidates=self.contrast_candidates,
            n_folds=self.n_folds,
            n_particles=self.n_particles,
            architecture="v1",
            sharing="partial",
            init_sharing="pooled",
            arm_shrinkage=5.0,
            random_state=seed,
            **self.budget,
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
            laws, weights, dgp.grid.reference_quantiles(), functionals,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
        )

        # The DR layer replaces the g-computation columns for the declared
        # functionals and the reference target only; mean quantiles and every
        # law-level quantity stay with the booster.
        bins = moderator_bins(X_test)
        zero = np.zeros(X_test.shape[0])
        dr_functionals: dict[str, dict[int, NDArray[np.float64]]] = {}
        for name in ("grid_mean", "grid_sd", "grid_skewness", "grid_upper_tail_mean"):
            broadcast = np.where(
                np.isfinite(model.aipw_.bin_contrasts_[name][bins]),
                model.aipw_.bin_contrasts_[name][bins],
                0.0,
            )
            dr_functionals[name] = {1: broadcast, 0: zero}
        reference_broadcast = np.where(
            np.isfinite(model.aipw_.bin_contrasts_["reference"][bins]),
            model.aipw_.bin_contrasts_["reference"][bins],
            0.0,
        )
        dr_reference = {1: reference_broadcast, 0: zero}
        object.__setattr__(output, "functionals", dr_functionals)
        object.__setattr__(output, "reference", dr_reference)
        diagnostics = {
            **output.diagnostics,
            "n_boosting_steps": float(len(model.training_history_)),
            "train_risk": float(model.train_risk_),
            "selected_contrast_shrinkage": float(
                model.selected_contrast_shrinkage_
            ),
            "ehat_mean": float(np.mean(model.aipw_.ehat_train_)),
            "dr_if_se_reference": float(model.dr_if_se("reference")),
        }
        object.__setattr__(output, "diagnostics", diagnostics)
        return output


class SmoothAdapter:
    """`cwdb_smooth`: dispersion repair chosen on held-out energy score."""

    produces_law = True

    def __init__(
        self,
        *,
        contrast_candidates: tuple[float, ...] | None = None,
        n_folds: int = PHASE6_SELECTION_FOLDS,
        n_particles: int = 10,
        **budget: object,
    ) -> None:
        self.contrast_candidates = (
            PHASE6_CONTRAST_CANDIDATES
            if contrast_candidates is None
            else tuple(contrast_candidates)
        )
        self.n_folds = n_folds
        self.n_particles = n_particles
        self.budget = dict(budget)

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
        model = SmoothedCWDB(
            contrast_candidates=self.contrast_candidates,
            n_folds=self.n_folds,
            n_particles=self.n_particles,
            architecture="v1",
            sharing="partial",
            init_sharing="pooled",
            arm_shrinkage=5.0,
            random_state=seed,
            **self.budget,
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
            laws, weights, dgp.grid.reference_quantiles(), functionals,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
        )
        diagnostics = {
            **output.diagnostics,
            "selected_transform": float(
                0.0 if model.selected_transform_ == "scale" else 1.0
            ),
            "transform_value": float(model.transform_value_),
            "n_boosting_steps": float(len(model.training_history_)),
            "train_risk": float(model.train_risk_),
        }
        object.__setattr__(output, "diagnostics", diagnostics)
        return output


class KRRAdapter:
    """`cwdb_krr`: independent-arm particle boosting on ridge directions."""

    produces_law = True

    def __init__(
        self,
        *,
        n_particles: int = 10,
        **budget: object,
    ) -> None:
        self.n_particles = n_particles
        # The kernel learner has no tree geometry, so the tree-only budget
        # entries are dropped rather than ignored silently.
        self.parameters = {
            key: value
            for key, value in budget.items()
            if key in {"n_estimators", "learning_rate", "collision_epsilon"}
        }

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
        before = peak_ram_mb()
        started = time.perf_counter()
        arm_models = {}
        for arm in (0, 1):
            mask = train.treatment == arm
            arm_models[arm] = KRRArmParticleBooster(
                n_particles=self.n_particles, random_state=seed + 10_000 * arm,
                **self.parameters,
            ).fit(train.X[mask], train.quantiles[mask], weights)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        laws = {
            arm: LawPrediction.from_particles(arm_models[arm].predict_particles(X_test))
            for arm in (0, 1)
        }
        predict_seconds = time.perf_counter() - started

        output = _output_from_laws(
            laws, weights, dgp.grid.reference_quantiles(), functionals,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
        )
        diagnostics = {
            **output.diagnostics,
            "train_risk": float(np.mean([m.train_risk_ for m in arm_models.values()])),
            "n_accepted_steps": float(
                np.mean([m.n_accepted_steps_ for m in arm_models.values()])
            ),
        }
        object.__setattr__(output, "diagnostics", diagnostics)
        return output


class FRLAdapter:
    """`cwdb_frl`: joint scalar R-losses over functionals and coordinates."""

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
        weights = dgp.grid.weights
        declared = _declared_functionals(weights, dgp.grid.reference_quantiles())
        # The coordinate columns give the learner the full quantile vector, so
        # the common mean-quantile target stays comparable with cwdb_rmean.
        functions = dict(declared)
        n_grid = dgp.grid.n_grid
        for k in range(n_grid):
            functions[f"coord_{k}"] = partial(_column, index=k)
        model = FunctionalRLearner(functionals=functions, random_state=seed)

        before = peak_ram_mb()
        started = time.perf_counter()
        model.fit(train.X, train.treatment, train.quantiles)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        arm_means = {arm: model.predict_arm_means(X_test, arm) for arm in (0, 1)}
        predict_seconds = time.perf_counter() - started

        mean_quantiles = {
            arm: np.column_stack([
                arm_means[arm][f"coord_{k}"] for k in range(n_grid)
            ])
            for arm in (0, 1)
        }
        declared_names = ("grid_mean", "grid_sd", "grid_skewness",
                          "grid_upper_tail_mean")
        output = MethodOutput(
            mean_quantiles=mean_quantiles,
            functionals={
                name: {arm: arm_means[arm][name] for arm in (0, 1)}
                for name in declared_names
            },
            reference={arm: arm_means[arm]["reference"] for arm in (0, 1)},
            law=None,
            supported_functionals=declared_names,
            n_atoms=0,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
            diagnostics={
                "selected_shrinkage": float(model.selected_shrinkage_),
                # Column layout is the five declared functionals first, so the
                # reference distance is column 4.
                "shrinkage_reference_column": float(
                    model.shrinkage_vector_[4]
                ),
                "ehat_mean": float(np.mean(model.nuisance_.ehat_oof_)),
                "train_risk": float(model.train_risk_),
            },
        )
        return output


def _column(block: NDArray[np.float64], *, index: int) -> NDArray[np.float64]:
    return block[:, index]
