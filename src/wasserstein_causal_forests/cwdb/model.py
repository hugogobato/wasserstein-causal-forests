"""Independent-arm and shared-partition C-WDB estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common.quantiles import (
    canonical_training_order,
    canonicalize_particles,
    validate_quantiles,
    validate_weights,
)
from .arm_shared_tree import ArmSharedTreeRegressor
from .energy import empirical_energy_risk, energy_gradient, energy_score
from .geometry import ProjectionDiagnostics, project_quantiles
from .weak_learners import MultiOutputTreeRegressor


@dataclass(frozen=True)
class BoostingStep:
    """Auditable training diagnostics for one accepted weak learner."""

    iteration: int
    arm: int | None
    loss_before: float
    loss_after: float
    step_size: float
    backtracks: int
    projection_max: float
    projection_changed: int


def _validate_X(X: ArrayLike, n_features: int | None = None) -> NDArray[np.float64]:
    result = np.asarray(X, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError("X must have nonzero shape (n, p)")
    if n_features is not None and result.shape[1] != n_features:
        raise ValueError(f"X must have {n_features} features")
    if not np.all(np.isfinite(result)):
        raise ValueError("X must be finite")
    return result


def compute_init_base(
    quantiles: ArrayLike, n_particles: int
) -> NDArray[np.float64]:
    """Deterministic spread-preserving empirical initialization."""

    q = validate_quantiles(quantiles)
    if q.ndim != 2:
        raise ValueError("quantiles must have shape (n, K)")
    if n_particles < 1:
        raise ValueError("n_particles must be positive")
    order = np.lexsort(q[:, ::-1].T)
    sorted_q = q[order]
    indices = np.floor(
        (np.arange(n_particles, dtype=float) + 0.5)
        * sorted_q.shape[0]
        / n_particles
    ).astype(int)
    indices = np.clip(indices, 0, sorted_q.shape[0] - 1)
    return canonicalize_particles(sorted_q[indices])


def _preconditioned_descent_target(
    gradient: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Negative gradient in the rescaled Wasserstein coordinates."""

    n_particles = gradient.shape[-2]
    return -n_particles * gradient / weights


def _project_candidate(
    particles: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ProjectionDiagnostics]:
    projected, diagnostics = project_quantiles(
        particles, weights, return_diagnostics=True
    )
    return projected, diagnostics


class ArmParticleBooster:
    """Proper-score particle booster for one observed treatment arm."""

    def __init__(
        self,
        *,
        n_particles: int = 5,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        collision_epsilon: float = 1e-3,
        max_backtracks: int = 12,
        descent_tolerance: float = 1e-12,
        random_state: int = 0,
    ) -> None:
        if n_particles < 1 or n_estimators < 0:
            raise ValueError("n_particles must be positive and n_estimators nonnegative")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if collision_epsilon < 0.0:
            raise ValueError("collision_epsilon must be nonnegative")
        self.n_particles = n_particles
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.collision_epsilon = collision_epsilon
        self.max_backtracks = max_backtracks
        self.descent_tolerance = descent_tolerance
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        quantiles: ArrayLike,
        weights: ArrayLike,
        initial_particles: ArrayLike | None = None,
    ) -> "ArmParticleBooster":
        x = _validate_X(X)
        q = validate_quantiles(quantiles)
        if q.ndim != 2 or q.shape[0] != x.shape[0]:
            raise ValueError("quantiles must have shape (n, K)")
        w = validate_weights(weights, q.shape[1], require_normalized=True)
        dummy_arm = np.zeros(x.shape[0], dtype=int)
        order = canonical_training_order(x, dummy_arm, q)
        x, q = x[order], q[order]

        self.n_features_in_ = x.shape[1]
        self.n_coordinates_ = q.shape[1]
        self.weights_ = w.copy()
        # A caller fitting both arms may supply one shared base so the two arms
        # start from the same law; on its own the booster uses its own sample.
        if initial_particles is None:
            self.initial_particles_ = compute_init_base(q, self.n_particles)
        else:
            self.initial_particles_ = validate_quantiles(
                initial_particles, self.n_coordinates_
            )
        current = np.broadcast_to(
            self.initial_particles_, (x.shape[0],) + self.initial_particles_.shape
        ).copy()
        self.estimators_: list[MultiOutputTreeRegressor] = []
        self.step_sizes_: list[float] = []
        self.training_history_: list[BoostingStep] = []

        # The accepted candidate becomes `current`, so the next iteration's
        # loss_before is the value just computed. Carrying it forward removes
        # one full risk evaluation per boosting step.
        loss_before = empirical_energy_risk(
            current, q, w, epsilon=self.collision_epsilon
        )
        for iteration in range(self.n_estimators):
            gradient = energy_gradient(
                current, q, w, epsilon=self.collision_epsilon
            )
            target = _preconditioned_descent_target(gradient, w)
            estimator = MultiOutputTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + iteration,
            )
            estimator.fit(x, target)
            direction = estimator.predict(x)

            accepted = False
            step = self.learning_rate
            selected_diagnostics: ProjectionDiagnostics | None = None
            for backtracks in range(self.max_backtracks + 1):
                candidate, diagnostics = _project_candidate(
                    current + step * direction, w
                )
                loss_after = empirical_energy_risk(
                    candidate, q, w, epsilon=self.collision_epsilon
                )
                if loss_after <= loss_before + self.descent_tolerance:
                    accepted = True
                    selected_diagnostics = diagnostics
                    break
                step *= 0.5
            if not accepted or selected_diagnostics is None:
                break

            self.estimators_.append(estimator)
            self.step_sizes_.append(step)
            self.training_history_.append(
                BoostingStep(
                    iteration=iteration,
                    arm=None,
                    loss_before=loss_before,
                    loss_after=loss_after,
                    step_size=step,
                    backtracks=backtracks,
                    projection_max=selected_diagnostics.max_weighted_l2_adjustment,
                    projection_changed=selected_diagnostics.n_changed,
                )
            )
            current = candidate
            loss_before = loss_after
        self.train_risk_ = loss_before
        return self

    def _predict_labeled(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("the model has not been fitted")
        x = _validate_X(X, self.n_features_in_)
        particles = np.broadcast_to(
            self.initial_particles_, (x.shape[0],) + self.initial_particles_.shape
        ).copy()
        for estimator, step in zip(
            self.estimators_, self.step_sizes_, strict=True
        ):
            particles = project_quantiles(
                particles + step * estimator.predict(x), self.weights_
            )
        return particles

    def predict_particles(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return a canonical representation of the unordered empirical law."""

        return canonicalize_particles(self._predict_labeled(X))

    def score_samples(
        self, X: ArrayLike, quantiles: ArrayLike
    ) -> NDArray[np.float64]:
        q = validate_quantiles(quantiles, self.n_coordinates_)
        particles = self._predict_labeled(X)
        if q.shape != (particles.shape[0], self.n_coordinates_):
            raise ValueError("quantiles must have one row per X")
        return energy_score(
            particles, q, self.weights_, epsilon=self.collision_epsilon
        )


class CWDBRegressor:
    """Treatment-aware C-WDB with independent or shared tree partitions."""

    def __init__(
        self,
        *,
        architecture: str = "v0",
        n_particles: int = 5,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_leaf: int = 5,
        min_arm_leaf: int = 2,
        arm_shrinkage: float = 5.0,
        sharing: str = "partial",
        init_sharing: str = "per_arm",
        contrast_rule: str = "arm_shrinkage",
        contrast_shrinkage: float = 0.0,
        contrast_threshold_scale: float = 1.0,
        contrast_damping: float = 1.0,
        collision_epsilon: float = 1e-3,
        max_backtracks: int = 12,
        descent_tolerance: float = 1e-12,
        random_state: int = 0,
    ) -> None:
        if architecture not in {"v0", "v1"}:
            raise ValueError("architecture must be 'v0' or 'v1'")
        if init_sharing not in {"per_arm", "pooled"}:
            raise ValueError("init_sharing must be 'per_arm' or 'pooled'")
        self.architecture = architecture
        self.n_particles = n_particles
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_arm_leaf = min_arm_leaf
        self.arm_shrinkage = arm_shrinkage
        self.sharing = sharing
        self.init_sharing = init_sharing
        self.contrast_rule = contrast_rule
        self.contrast_shrinkage = contrast_shrinkage
        self.contrast_threshold_scale = contrast_threshold_scale
        self.contrast_damping = contrast_damping
        self.collision_epsilon = collision_epsilon
        self.max_backtracks = max_backtracks
        self.descent_tolerance = descent_tolerance
        self.random_state = random_state

    def _arm_parameters(self, arm: int) -> dict[str, object]:
        return {
            "n_particles": self.n_particles,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "collision_epsilon": self.collision_epsilon,
            "max_backtracks": self.max_backtracks,
            "descent_tolerance": self.descent_tolerance,
            "random_state": self.random_state + 10_000 * arm,
        }

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        weights: ArrayLike,
    ) -> "CWDBRegressor":
        x = _validate_X(X)
        a = np.asarray(treatment, dtype=int)
        q = validate_quantiles(quantiles)
        if a.shape != (x.shape[0],) or q.shape[0] != x.shape[0] or q.ndim != 2:
            raise ValueError("expected treatment (n,) and quantiles (n,K)")
        if not np.all(np.isin(a, (0, 1))):
            raise ValueError("treatment must contain only 0 and 1")
        if not all(np.any(a == arm) for arm in (0, 1)):
            raise ValueError("both treatment arms must be observed")
        w = validate_weights(weights, q.shape[1], require_normalized=True)
        order = canonical_training_order(x, a, q)
        x, a, q = x[order], a[order], q[order]

        self.n_features_in_ = x.shape[1]
        self.n_coordinates_ = q.shape[1]
        self.weights_ = w.copy()
        if self.architecture == "v0" or self.sharing == "none":
            self._fit_v0(x, a, q)
            self.fitted_architecture_ = "v0"
        else:
            self._fit_v1(x, a, q)
            self.fitted_architecture_ = "v1"
        return self

    def _initial_particles(
        self, treatment: NDArray[np.int64], Q: NDArray[np.float64]
    ) -> dict[int, NDArray[np.float64]]:
        """Base law each arm starts from.

        A per-arm base is a marginal quantity, so under confounding it carries
        the arm gap in the covariate distribution straight into the initial
        contrast, and the booster then has to spend its budget removing an
        offset it created. The pooled base starts both arms at the same law, so
        an estimated contrast can only come from a fitted tree.
        """

        if self.init_sharing == "pooled":
            base = compute_init_base(Q, self.n_particles)
            return {arm: base for arm in (0, 1)}
        return {
            arm: compute_init_base(Q[treatment == arm], self.n_particles)
            for arm in (0, 1)
        }

    def _fit_v0(
        self, X: NDArray[np.float64], treatment: NDArray[np.int64], Q: NDArray[np.float64]
    ) -> None:
        shared = (
            compute_init_base(Q, self.n_particles)
            if self.init_sharing == "pooled"
            else None
        )
        self.arm_models_: dict[int, ArmParticleBooster] = {}
        for arm in (0, 1):
            model = ArmParticleBooster(**self._arm_parameters(arm))
            mask = treatment == arm
            model.fit(X[mask], Q[mask], self.weights_, initial_particles=shared)
            self.arm_models_[arm] = model
        self.training_history_ = [
            BoostingStep(**{**asdict(step), "arm": arm})
            for arm in (0, 1)
            for step in self.arm_models_[arm].training_history_
        ]

    def _fit_v1(
        self, X: NDArray[np.float64], treatment: NDArray[np.int64], Q: NDArray[np.float64]
    ) -> None:
        self.initial_particles_ = self._initial_particles(treatment, Q)
        current = np.empty((X.shape[0], self.n_particles, self.n_coordinates_))
        for arm in (0, 1):
            current[treatment == arm] = self.initial_particles_[arm]

        self.estimators_: list[ArmSharedTreeRegressor] = []
        self.step_sizes_: list[float] = []
        self.training_history_: list[BoostingStep] = []
        # See `ArmParticleBooster.fit`: the accepted loss carries forward.
        loss_before = empirical_energy_risk(
            current, Q, self.weights_, epsilon=self.collision_epsilon
        )
        for iteration in range(self.n_estimators):
            gradient = energy_gradient(
                current, Q, self.weights_, epsilon=self.collision_epsilon
            )
            target = _preconditioned_descent_target(gradient, self.weights_)
            estimator = ArmSharedTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_arm_leaf=self.min_arm_leaf,
                arm_shrinkage=self.arm_shrinkage,
                sharing=self.sharing,
                contrast_rule=self.contrast_rule,
                contrast_shrinkage=self.contrast_shrinkage,
                contrast_threshold_scale=self.contrast_threshold_scale,
                contrast_damping=self.contrast_damping,
                random_state=self.random_state + iteration,
            )
            estimator.fit(X, treatment, target)
            direction = estimator.predict(X, treatment)

            accepted = False
            step = self.learning_rate
            selected_diagnostics: ProjectionDiagnostics | None = None
            for backtracks in range(self.max_backtracks + 1):
                candidate, diagnostics = _project_candidate(
                    current + step * direction, self.weights_
                )
                loss_after = empirical_energy_risk(
                    candidate, Q, self.weights_, epsilon=self.collision_epsilon
                )
                if loss_after <= loss_before + self.descent_tolerance:
                    accepted = True
                    selected_diagnostics = diagnostics
                    break
                step *= 0.5
            if not accepted or selected_diagnostics is None:
                break
            self.estimators_.append(estimator)
            self.step_sizes_.append(step)
            self.training_history_.append(
                BoostingStep(
                    iteration=iteration,
                    arm=None,
                    loss_before=loss_before,
                    loss_after=loss_after,
                    step_size=step,
                    backtracks=backtracks,
                    projection_max=selected_diagnostics.max_weighted_l2_adjustment,
                    projection_changed=selected_diagnostics.n_changed,
                )
            )
            current = candidate
            loss_before = loss_after
        self.train_risk_ = loss_before

    def _predict_labeled(
        self, X: ArrayLike, arm: int
    ) -> NDArray[np.float64]:
        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1")
        if not hasattr(self, "fitted_architecture_"):
            raise RuntimeError("the model has not been fitted")
        x = _validate_X(X, self.n_features_in_)
        if self.fitted_architecture_ == "v0":
            return self.arm_models_[arm]._predict_labeled(x)
        particles = np.broadcast_to(
            self.initial_particles_[arm],
            (x.shape[0],) + self.initial_particles_[arm].shape,
        ).copy()
        for estimator, step in zip(
            self.estimators_, self.step_sizes_, strict=True
        ):
            particles = project_quantiles(
                particles + step * estimator.predict(x, arm), self.weights_
            )
        return particles

    def predict_particles(
        self, X: ArrayLike, arm: int
    ) -> NDArray[np.float64]:
        """Return canonical particles representing an unordered arm law."""

        return canonicalize_particles(self._predict_labeled(X, arm))

    def predict_mean_quantile(
        self, X: ArrayLike, arm: int
    ) -> NDArray[np.float64]:
        """Integrate the identity functional over the predicted arm law."""

        return np.mean(self._predict_labeled(X, arm), axis=1)

    def predict_mean_quantile_effect(self, X: ArrayLike) -> NDArray[np.float64]:
        """Difference of arm-specific mean quantile vectors.

        This is a difference of law-invariant arm means. It is not a matched
        particle or individual treatment-effect distribution.
        """

        return self.predict_mean_quantile(X, 1) - self.predict_mean_quantile(X, 0)

    def predict_integral(
        self,
        X: ArrayLike,
        arm: int,
        functional: Callable[[NDArray[np.float64]], ArrayLike],
    ) -> NDArray[np.float64]:
        """Average a user-supplied grid functional over the empirical arm law."""

        particles = self._predict_labeled(X, arm)
        values = np.asarray(functional(particles), dtype=float)
        if values.shape[:2] != particles.shape[:2]:
            raise ValueError(
                "functional must preserve the (n, n_particles) leading axes"
            )
        return np.mean(values, axis=1)

    def score_samples(
        self, X: ArrayLike, arm: int, quantiles: ArrayLike
    ) -> NDArray[np.float64]:
        q = validate_quantiles(quantiles, self.n_coordinates_)
        particles = self._predict_labeled(X, arm)
        if q.shape != (particles.shape[0], self.n_coordinates_):
            raise ValueError("quantiles must have one row per X")
        return energy_score(
            particles, q, self.weights_, epsilon=self.collision_epsilon
        )

    def score(self, X: ArrayLike, arm: int, quantiles: ArrayLike) -> float:
        """Negative held-out energy risk, following estimator score convention."""

        return -float(np.mean(self.score_samples(X, arm, quantiles)))

