"""Method adapters: one interface over every model entering the G3 tournament.

Each adapter fits on one training sample and returns a `MethodOutput` for a
fixed test design. What an adapter may put in that output is constrained by
what its method actually estimates, and the constraint is enforced here rather
than left to the analysis:

* C-WDB and the two forest baselines estimate a conditional *law*, so they fill
  `law` and every downstream quantity is an integral against it.
* The two PTA endpoints estimate conditional *means* of a fixed target vector.
  Under `research/estimand_contract.md` section 4 a posterior draw of a mean
  surface may not be relabelled as an outcome draw, so they leave `law` empty
  and declare only the functionals inside their frozen manifest. Asking a PTA
  adapter for a functional it was not trained on returns nothing, which is the
  point of the D7 transfer claim rather than a gap to be patched.

The squared-W2 booster is the preregistered comparator for the repulsion claim.
It shares C-WDB's tree machinery and differs only in the loss, so any D6
difference between them isolates the repulsion term.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..cwdb.cross_fitted import CrossFittedCWDBRegressor
from ..cwdb.model import CWDBRegressor
from ..pta_bcf.mvbcf import MVBCFBudget, MVBCFForcedShared
from ..pta_bcf.separate_heads import HeadBudget, PTASeparateHeads
from ..pta_bcf.targets import GRID_FUNCTIONALS, TargetManifest, uniform_grid_manifest
from . import r_bridge
from .dgps import DGPSample, DistributionalDGP
from .laws import LawPrediction


def peak_ram_mb() -> float:
    """Process high-water mark. Monotone, so only differences are meaningful."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass(frozen=True)
class MethodOutput:
    """Everything the metric layer may read from one fitted method."""

    mean_quantiles: dict[int, NDArray[np.float64]]
    functionals: dict[str, dict[int, NDArray[np.float64]]]
    reference: dict[int, NDArray[np.float64]] | None
    law: dict[int, LawPrediction] | None
    supported_functionals: tuple[str, ...]
    n_atoms: int
    fit_seconds: float
    predict_seconds: float
    peak_ram_mb: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def produces_law(self) -> bool:
        return self.law is not None


def _weighted_reference_distance(
    grid_weights: NDArray[np.float64], reference: NDArray[np.float64]
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def distance(block: NDArray[np.float64]) -> NDArray[np.float64]:
        difference = block - reference
        return np.sqrt(np.sum(grid_weights * difference * difference, axis=-1))

    return distance


def _output_from_laws(
    laws: dict[int, LawPrediction],
    grid_weights: NDArray[np.float64],
    reference: NDArray[np.float64],
    functionals: tuple[str, ...],
    **telemetry: float,
) -> MethodOutput:
    """Integrate every declared functional against a law estimate.

    A law-producing method needs no target manifest: it evaluates h_j on its own
    atoms and averages, which is `E{h_j(q(Y^a)) | X = x}` by construction rather
    than h_j applied to a barycenter.
    """

    distance = _weighted_reference_distance(grid_weights, reference)
    # A law estimate is integrated against EVERY declared grid functional, not
    # only the ones in the training manifest. That is the whole content of the
    # D7 transfer claim: a method holding a conditional law can evaluate a
    # functional first named at evaluation time, while a method holding fixed
    # target coordinates cannot. Restricting this to `functionals` would make
    # the claim untestable by construction.
    evaluated = tuple(GRID_FUNCTIONALS)
    return MethodOutput(
        mean_quantiles={arm: law.mean_quantiles() for arm, law in laws.items()},
        functionals={
            name: {
                arm: law.scalar_expectation(
                    lambda block, h=GRID_FUNCTIONALS[name]: h(block, grid_weights)
                )
                for arm, law in laws.items()
            }
            for name in evaluated
        },
        reference={
            arm: law.scalar_expectation(distance) for arm, law in laws.items()
        },
        law=laws,
        supported_functionals=evaluated,
        n_atoms=laws[0].n_atoms,
        diagnostics={
            "effective_support": float(
                np.mean([law.effective_support().mean() for law in laws.values()])
            )
        },
        **telemetry,
    )


class CWDBAdapter:
    """C-WDB-v1, C-WDB-v0, and the ablations and repairs built on them.

    `init_sharing` and the `contrast_*` arguments default to the frozen G3
    configuration, so an adapter constructed without them is byte-identical to
    the one the first tournament ran. `contrast_candidates` switches on the
    cross-fitted selector of `cwdb.cross_fitted`, which chooses the contrast
    strength on held-out energy risk instead of taking it frozen.
    """

    produces_law = True

    def __init__(
        self,
        *,
        architecture: str = "v1",
        n_particles: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.12,
        max_depth: int = 4,
        min_samples_leaf: int = 10,
        min_arm_leaf: int = 5,
        arm_shrinkage: float = 5.0,
        sharing: str = "partial",
        init_sharing: str = "per_arm",
        contrast_rule: str = "arm_shrinkage",
        contrast_shrinkage: float = 0.0,
        contrast_threshold_scale: float = 1.0,
        contrast_damping: float = 1.0,
        collision_epsilon: float = 1e-3,
        contrast_candidates: tuple[float, ...] | None = None,
        n_folds: int = 3,
    ) -> None:
        self.parameters = {
            "architecture": architecture,
            "n_particles": n_particles,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_arm_leaf": min_arm_leaf,
            "arm_shrinkage": arm_shrinkage,
            "sharing": sharing,
            "init_sharing": init_sharing,
            "contrast_rule": contrast_rule,
            "contrast_shrinkage": contrast_shrinkage,
            "contrast_threshold_scale": contrast_threshold_scale,
            "contrast_damping": contrast_damping,
            "collision_epsilon": collision_epsilon,
        }
        self.contrast_candidates = contrast_candidates
        self.n_folds = n_folds

    def _build(self, seed: int) -> CWDBRegressor:
        if self.contrast_candidates is None:
            return CWDBRegressor(random_state=seed, **self.parameters)
        return CrossFittedCWDBRegressor(
            contrast_candidates=self.contrast_candidates,
            n_folds=self.n_folds,
            random_state=seed,
            **self.parameters,
        )

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
        model = self._build(seed)
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
        # v0 fits one booster per arm and reports no pooled training risk, so
        # the diagnostic is taken from whichever the architecture exposes.
        if hasattr(model, "train_risk_"):
            train_risk = float(model.train_risk_)
        else:
            train_risk = float(
                np.mean([arm.train_risk_ for arm in model.arm_models_.values()])
            )
        diagnostics = {
            **output.diagnostics,
            "n_boosting_steps": float(len(model.training_history_)),
            "train_risk": train_risk,
        }
        # The cross-fitted variant's chosen strength is the thing a reader will
        # ask about first, so it travels with the result rather than the log.
        if hasattr(model, "selected_contrast_shrinkage_"):
            diagnostics["selected_contrast_shrinkage"] = float(
                model.selected_contrast_shrinkage_
            )
        object.__setattr__(output, "diagnostics", diagnostics)
        return output


class SquaredW2BoosterAdapter:
    """The preregistered repulsion comparator: the same booster, squared loss.

    Minimising the mean squared W_{2,K} distance to the observed grid vector is
    separable across particles and is solved by the conditional barycenter, so
    every particle converges to the same point and the fitted law is a Dirac.
    That is exactly the collapse the energy score's repulsion term is claimed to
    prevent, which is why this is the right comparator for D6 rather than a
    weaker or differently-tuned booster.
    """

    produces_law = True

    def __init__(
        self,
        *,
        n_particles: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.12,
        max_depth: int = 4,
        min_samples_leaf: int = 10,
        min_arm_leaf: int = 5,
        arm_shrinkage: float = 5.0,
    ) -> None:
        self.n_particles = n_particles
        self.parameters = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_arm_leaf": min_arm_leaf,
            "arm_shrinkage": arm_shrinkage,
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
        from ..cwdb.arm_shared_tree import ArmSharedTreeRegressor
        from ..cwdb.geometry import project_quantiles

        weights = dgp.grid.weights
        before = peak_ram_mb()
        started = time.perf_counter()

        current = np.tile(train.quantiles.mean(axis=0), (train.X.shape[0], 1))
        estimators: list[ArmSharedTreeRegressor] = []
        learning_rate = self.parameters["learning_rate"]
        for iteration in range(self.parameters["n_estimators"]):
            # Gradient of 0.5 ||p - y||_W^2 in the rescaled coordinates is the
            # residual itself, so the booster is ordinary gradient boosting of
            # the conditional mean quantile vector.
            residual = train.quantiles - current
            estimator = ArmSharedTreeRegressor(
                max_depth=self.parameters["max_depth"],
                min_samples_leaf=self.parameters["min_samples_leaf"],
                min_arm_leaf=self.parameters["min_arm_leaf"],
                arm_shrinkage=self.parameters["arm_shrinkage"],
                sharing="partial",
                random_state=seed + iteration,
            )
            estimator.fit(train.X, train.treatment, residual)
            current = project_quantiles(
                current + learning_rate * estimator.predict(train.X, train.treatment),
                weights,
            )
            estimators.append(estimator)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        laws = {}
        for arm in (0, 1):
            centre = np.tile(train.quantiles.mean(axis=0), (X_test.shape[0], 1))
            for estimator in estimators:
                centre = project_quantiles(
                    centre + learning_rate * estimator.predict(X_test, arm), weights
                )
            # M identical particles: the squared loss carries no repulsion, so
            # the fitted law really is a point mass and is reported as one.
            laws[arm] = LawPrediction.from_particles(
                np.repeat(centre[:, None, :], self.n_particles, axis=1)
            )
        predict_seconds = time.perf_counter() - started

        return _output_from_laws(
            laws,
            weights,
            dgp.grid.reference_quantiles(),
            functionals,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
        )


def _pta_manifest(
    dgp: DistributionalDGP, functionals: tuple[str, ...]
) -> TargetManifest:
    return uniform_grid_manifest(
        dgp.grid.n_grid,
        functionals=functionals,
        reference_quantiles=dgp.grid.reference_quantiles(),
    )


def _output_from_target_means(
    manifest: TargetManifest,
    arm_means: dict[int, NDArray[np.float64]],
    functionals: tuple[str, ...],
    **telemetry: float,
) -> MethodOutput:
    """Split a fitted (n, D) target vector into its declared blocks.

    Every coordinate is a conditional mean of a declared function of q(Y), so
    the functional coordinates are outcome-level `TATE-K-j` ingredients rather
    than functionals of a barycenter. Only coordinates that were in the manifest
    at fit time are reported.
    """

    names = manifest.column_names
    quantile_block = manifest.quantile_slice
    reference_index = manifest.reference_index
    return MethodOutput(
        mean_quantiles={arm: value[:, quantile_block] for arm, value in arm_means.items()},
        functionals={
            name: {arm: value[:, names.index(name)] for arm, value in arm_means.items()}
            for name in functionals
        },
        reference=(
            None
            if reference_index is None
            else {arm: value[:, reference_index] for arm, value in arm_means.items()}
        ),
        law=None,
        supported_functionals=functionals,
        n_atoms=0,
        **telemetry,
    )


class PTASeparateAdapter:
    """PTA-S: one independently tuned scalar BCF head per target coordinate."""

    produces_law = False

    def __init__(self, *, budget: HeadBudget | None = None, n_folds: int = 5) -> None:
        self.budget = budget or HeadBudget()
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
        manifest = _pta_manifest(dgp, functionals)
        model = PTASeparateHeads(
            manifest, budget=self.budget, n_folds=self.n_folds, random_state=seed
        )
        before = peak_ram_mb()
        started = time.perf_counter()
        model.fit(train.X, train.treatment, train.quantiles)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        arm_means = {arm: model.predict_arm_mean(X_test, arm) for arm in (0, 1)}
        predict_seconds = time.perf_counter() - started

        return _output_from_target_means(
            manifest,
            arm_means,
            functionals,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
        )


class PTAForcedAdapter:
    """PTA-F: forced-shared multivariate BCF over the whole target vector."""

    produces_law = False

    def __init__(
        self, *, budget: MVBCFBudget | None = None, timeout_seconds: float = 7200.0
    ) -> None:
        self.budget = budget or MVBCFBudget()
        self.timeout_seconds = timeout_seconds

    def fit_predict(
        self,
        train: DGPSample,
        X_test: NDArray[np.float64],
        dgp: DistributionalDGP,
        functionals: tuple[str, ...],
        *,
        seed: int,
    ) -> MethodOutput:
        manifest = _pta_manifest(dgp, functionals)
        targets = manifest.build(train.quantiles)
        model = MVBCFForcedShared(
            budget=self.budget,
            random_state=seed,
            timeout_seconds=self.timeout_seconds,
        )
        before = peak_ram_mb()
        started = time.perf_counter()
        # The package checks that the control and moderator test designs have
        # matching row counts, so both must be supplied even though they are
        # the same design here.
        result = model.fit_predict(
            train.X,
            targets,
            train.treatment,
            X_control_test=X_test,
            X_moderator_test=X_test,
        )
        elapsed = time.perf_counter() - started

        control = result.control_mean("test")
        arm_means = {0: control, 1: control + result.contrast_mean("test")}
        return _output_from_target_means(
            manifest,
            arm_means,
            functionals,
            fit_seconds=elapsed,
            predict_seconds=0.0,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
        )


class ForestBaselineAdapter:
    """R-backed W-DRF-T, paper DRF, and Causal-DRF adapters."""

    produces_law = True

    def __init__(
        self,
        method: str,
        *,
        hyperparameters: dict[str, object] | None = None,
        cache_directory: Path | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        if method not in {"wdrft", "causal_drf", "drf"}:
            raise ValueError("method must be 'wdrft', 'causal_drf', or 'drf'")
        self.method = method
        self.hyperparameters = hyperparameters or {}
        self.cache_directory = cache_directory
        self.timeout_seconds = timeout_seconds

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
        result = r_bridge.fit_predict(
            self.method,
            X_train=train.X,
            treatment=train.treatment,
            Q_train=train.quantiles,
            X_test=X_test,
            quad_weights=weights,
            reference_quantiles=dgp.grid.reference_quantiles(),
            hyperparameters=self.hyperparameters,
            seed=seed,
            cache_directory=self.cache_directory,
            timeout_seconds=self.timeout_seconds,
        )
        laws = {
            arm: LawPrediction.from_forest_weights(train.quantiles, matrix)
            for arm, matrix in result.weights.items()
        }
        return _output_from_laws(
            laws,
            weights,
            dgp.grid.reference_quantiles(),
            functionals,
            fit_seconds=result.fit_seconds,
            predict_seconds=max(result.total_seconds - result.fit_seconds, 0.0),
            peak_ram_mb=max(
                result.peak_ram_mb, max(peak_ram_mb() - before, 0.0)
            ),
        )
