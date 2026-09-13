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

from ..cwdb.dr_calibration import DRCalibratedCWDB, FunctionalAIPW
from ..cwdb.krr_booster import KRRArmParticleBooster
from ..cwdb.smoothing import SmoothedCWDB
from ..meta_learners.functional_r_learner import FunctionalRLearner
from ..pta_bcf.targets import GRID_FUNCTIONALS
from .common_grid import (
    COMMON_LEVELS,
    INTERIOR_LEVELS,
    build_dgp_at_levels,
    interpolate_quantile_curves,
)
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


#: Dense level sets a DR payload can declare, in registry order.
_COMMON_GRID_LEVEL_SETS: dict[str, NDArray[np.float64]] = {
    "COMMON199": COMMON_LEVELS,
    "INTERIOR": INTERIOR_LEVELS,
}


def _parse_common_grid_levels(value: str | None) -> tuple[str, ...]:
    """Parse ``"COMMON199+INTERIOR"`` into the requested level-set names."""

    if value is None:
        return ()
    text = value.strip()
    if not text:
        return ()
    names: list[str] = []
    for token in text.split("+"):
        name = token.strip()
        if not name:
            continue
        if name not in _COMMON_GRID_LEVEL_SETS:
            raise ValueError(
                f"unknown common-grid level set {name!r}; expected any of "
                f"{sorted(_COMMON_GRID_LEVEL_SETS)} joined by '+'"
            )
        if name not in names:
            names.append(name)
    return tuple(names)


class DRAdapter:
    """`cwdb_dr`: R3 law plus a doubly-robust functional calibration layer.

    `propensity_factory` selects the AIPW propensity model by registry name or
    as a callable factory, `oracle_propensity` substitutes the sampled true
    propensity for the fitted one, and `common_grid_levels` requests a
    dense-grid calibration payload under `output.common_grid`. All three
    default to the frozen behaviour, so a phase 6 cell is unaffected.
    """

    produces_law = True

    def __init__(
        self,
        *,
        contrast_candidates: tuple[float, ...] | None = None,
        n_folds: int = PHASE6_SELECTION_FOLDS,
        n_particles: int = 10,
        propensity_factory: str | None = None,
        oracle_propensity: bool = False,
        common_grid_levels: str | None = None,
        **budget: object,
    ) -> None:
        self.contrast_candidates = (
            PHASE6_CONTRAST_CANDIDATES
            if contrast_candidates is None
            else tuple(contrast_candidates)
        )
        self.n_folds = n_folds
        self.n_particles = n_particles
        self.propensity_factory = propensity_factory
        self.oracle_propensity = bool(oracle_propensity)
        self.common_grid_levels = common_grid_levels
        self.common_grid_level_sets = _parse_common_grid_levels(common_grid_levels)
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
        budget = dict(self.budget)
        arm_shrinkage = float(budget.pop("arm_shrinkage", 5.0))
        model = DRCalibratedCWDB(
            functionals=_declared_functionals(weights, dgp.grid.reference_quantiles()),
            contrast_candidates=self.contrast_candidates,
            n_folds=self.n_folds,
            n_particles=self.n_particles,
            architecture="v1",
            sharing="partial",
            init_sharing="pooled",
            arm_shrinkage=arm_shrinkage,
            random_state=seed,
            propensity_factory=self.propensity_factory,
            oracle_propensity=self.oracle_propensity,
            **budget,
        )
        before = peak_ram_mb()
        started = time.perf_counter()
        model.fit(
            train.X,
            train.treatment,
            train.quantiles,
            weights,
            true_propensity=(train.propensity if self.oracle_propensity else None),
        )
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

        if self.common_grid_level_sets:
            dense_payload, n_dense_levels = self._dense_calibration_payload(
                model, train, dgp
            )
            object.__setattr__(output, "common_grid", dense_payload)
            object.__setattr__(
                output,
                "diagnostics",
                {
                    **output.diagnostics,
                    "dense_dr_n_levels": float(n_dense_levels),
                },
            )
        return output

    def _dense_calibration_payload(
        self,
        model: DRCalibratedCWDB,
        train: DGPSample,
        dgp: DistributionalDGP,
    ) -> tuple[dict[str, dict[str, object]], int]:
        """Calibrate the declared functionals on each requested dense level set.

        The functional targets, the reference, and the observation curves are
        all rebuilt on the dense levels, and the existing cross-fitted
        propensity is reused as the oracle, so the payload's contrast targets
        match the evaluator's dense truth exactly.
        """

        native_levels = np.asarray(dgp.grid.levels, dtype=float)
        payload: dict[str, dict[str, object]] = {}
        n_levels = 0
        for level_set_name in self.common_grid_level_sets:
            levels = _COMMON_GRID_LEVEL_SETS[level_set_name]
            dense = build_dgp_at_levels(dgp.spec.dgp_id, levels)
            dense_functionals = _declared_functionals(
                dense.grid.weights, dense.grid.reference_quantiles()
            )
            observed = {
                name: np.asarray(
                    h(
                        interpolate_quantile_curves(
                            train.quantiles, native_levels, levels
                        )
                    ),
                    dtype=float,
                )
                for name, h in dense_functionals.items()
            }
            oof: dict[str, dict[int, NDArray[np.float64]]] = {
                name: {} for name in dense_functionals
            }
            for arm in (0, 1):
                particles = interpolate_quantile_curves(
                    model.oof_particles_[arm], native_levels, levels
                )
                flat = particles.reshape(-1, levels.size)
                for name, h in dense_functionals.items():
                    values = np.asarray(h(flat), dtype=float).reshape(
                        particles.shape[:2]
                    )
                    means = np.nanmean(values, axis=1)
                    finite = np.isfinite(means)
                    if not np.all(finite):
                        fallback = (
                            float(np.mean(means[finite]))
                            if np.any(finite)
                            else np.nan
                        )
                        means = np.where(finite, means, fallback)
                    oof[name][arm] = means
            aipw = FunctionalAIPW(n_bins=4)
            aipw.fit(
                observed=observed,
                oof_arm_means=oof,
                X=train.X,
                treatment=train.treatment,
                random_state=model.random_state + 31,
                oracle_propensity=model.ehat_train_,
            )
            payload[level_set_name] = {
                "levels": levels,
                "marginal": {
                    name: float(value) for name, value in aipw.marginal_.items()
                },
                "bin_contrasts": {
                    name: np.asarray(value, dtype=float)
                    for name, value in aipw.bin_contrasts_.items()
                },
            }
            n_levels = int(levels.size)
        return payload, n_levels


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
