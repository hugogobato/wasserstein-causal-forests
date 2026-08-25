"""Kernel-ridge weak learners inside the particle booster.

Every C-WDB variant so far regresses the energy gradient block onto a depth-4
regression tree. Trees are piecewise constant, the truth surfaces in all ten
frozen regimes are smooth in x, and a forest baseline buys back smoothness by
averaging hundreds of randomised partitions. This module asks the direct
question the roster could not: if the weak learner itself is smooth, how much
of the remaining gap closes?

The design keeps everything except the base learner frozen: the same certified
energy gradient, the same preconditioned descent target, the same projected
line search, the same monotone cone. The weak learner is kernel ridge
regression with a Gaussian kernel on the raw covariates, median-distance
bandwidth, and a fixed landmark subsample of 150 rows whose ridge system is
Cholesky-factorised once per arm and reused across boosting steps, so the cost
stays within a small multiple of the tree booster's.

Sharing is deliberately off here: one independent booster per arm. This is a
mechanism probe, not a claimant - any gain it shows is attributable to smoother
base learners, and it cannot be credited to the shared partition because there
is none.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common.quantiles import (
    canonical_training_order,
    canonicalize_particles,
    validate_quantiles,
    validate_weights,
)
from .energy import empirical_energy_risk, energy_gradient
from .geometry import project_quantiles
from .model import compute_init_base

#: Landmark count for the reduced ridge system. At n <= 1000 this keeps the
#: Cholesky trivial while retaining enough support for smooth surfaces.
N_LANDMARKS = 150

#: Ridge strength relative to the kernel diagonal (which equals one). The
#: pilot on D5 at manifest coordinates but seeds outside the manifest showed
#: why this must be large: the energy gradient block carries a large
#: idiosyncratic per-row component, and an interpolating smoother memorises it
#: instead of averaging it away, so the replayed direction field is noise at
#: test points (mean-quantile RMSE above 0.8 at lambda = 1e-2 against 0.25 for
#: the tree booster). At lambda = 1 the learner averages like a very smooth
#: leaf and the probe becomes interpretable; anything weaker measures
#: memorisation, not smoothing.
RIDGE_RELATIVE = 1.0


def _preconditioned_descent_target(
    gradient: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Negative gradient in the rescaled coordinates (model.py's rule)."""

    n_particles = gradient.shape[-2]
    return -n_particles * gradient / weights


def _validate_x(X: ArrayLike, n_features: int | None = None) -> NDArray[np.float64]:
    result = np.asarray(X, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError("X must have nonzero shape (n, p)")
    if n_features is not None and result.shape[1] != n_features:
        raise ValueError(f"X must have {n_features} features")
    if not np.all(np.isfinite(result)):
        raise ValueError("X must be finite")
    return result


def _median_bandwidth(X: NDArray[np.float64], rng: np.random.Generator) -> float:
    n_rows = min(X.shape[0], 256)
    index = rng.choice(X.shape[0], size=n_rows, replace=False)
    sample = X[index]
    diffs = sample[:, None, :] - sample[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
    upper = distances[np.triu_indices(n_rows, k=1)]
    median = float(np.median(upper)) if upper.size else 1.0
    return median if median > 0.0 else 1.0


class KernelRidgeLearner:
    """A fixed-kernel reduced ridge regression reused across boosting steps.

    The ridge system lives on a landmark subsample, so its Cholesky factor is
    built once; each boosting step only solves against that factor with the
    current target block as right-hand side.
    """

    def __init__(
        self,
        X: NDArray[np.float64],
        *,
        random_state: int,
        n_landmarks: int = N_LANDMARKS,
        ridge_relative: float = RIDGE_RELATIVE,
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.bandwidth_ = _median_bandwidth(X, rng)
        self.n_landmarks_ = int(min(n_landmarks, X.shape[0]))
        self.landmarks_ = rng.choice(X.shape[0], size=self.n_landmarks_, replace=False)
        landmark_coordinates = X[self.landmarks_]
        scale = 1.0 / self.bandwidth_
        diffs = (
            landmark_coordinates[:, None, :] - landmark_coordinates[None, :, :]
        )
        gram = np.exp(-np.sum(diffs * diffs, axis=-1) * scale * scale)
        gram[np.diag_indices_from(gram)] += ridge_relative
        self.chol_ = np.linalg.cholesky(gram)
        cross = X[:, None, :] - landmark_coordinates[None, :, :]
        self.knm_train_ = np.exp(-np.sum(cross * cross, axis=-1) * scale * scale)
        self.landmark_coordinates_ = landmark_coordinates

    def coefficients(self, target: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ridge coefficients for an (n, D) target block."""

        solved = np.linalg.solve(self.chol_, target[self.landmarks_])
        return np.linalg.solve(self.chol_.T, solved)

    def fitted_train_values(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.knm_train_ @ coefficients

    def predict_with(
        self, coefficients: NDArray[np.float64], X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        scale = 1.0 / self.bandwidth_
        diffs = X[:, None, :] - self.landmark_coordinates_[None, :, :]
        knm = np.exp(-np.sum(diffs * diffs, axis=-1) * scale * scale)
        return knm @ coefficients


class KRRArmParticleBooster:
    """Independent-arm particle booster with kernel-ridge weak learners.

    The boosting loop, the projected line search, and the acceptance rule are
    byte-for-byte the independent-arm booster's rules applied to a different
    direction field; only the weak learner changes.
    """

    def __init__(
        self,
        *,
        n_particles: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.12,
        collision_epsilon: float = 1e-3,
        max_backtracks: int = 12,
        descent_tolerance: float = 1e-12,
        random_state: int = 0,
    ) -> None:
        self.n_particles = n_particles
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.collision_epsilon = collision_epsilon
        self.max_backtracks = max_backtracks
        self.descent_tolerance = descent_tolerance
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        quantiles: ArrayLike,
        weights: ArrayLike,
    ) -> "KRRArmParticleBooster":
        x = _validate_x(X)
        q = validate_quantiles(quantiles)
        w = validate_weights(weights, q.shape[1], require_normalized=True)
        dummy_arm = np.zeros(x.shape[0], dtype=int)
        order = canonical_training_order(x, dummy_arm, q)
        x, q = x[order], q[order]

        self.n_features_in_ = x.shape[1]
        self.weights_ = w.copy()
        self.initial_particles_ = compute_init_base(q, self.n_particles)
        current = np.broadcast_to(
            self.initial_particles_, (x.shape[0],) + self.initial_particles_.shape
        ).copy()

        learner = KernelRidgeLearner(x, random_state=self.random_state)
        flat_columns = self.n_particles * q.shape[1]
        loss_before = empirical_energy_risk(current, q, w, epsilon=self.collision_epsilon)
        self.coefficients_: list[NDArray[np.float64]] = []
        self.step_sizes_: list[float] = []
        accepted_steps = 0
        for _iteration in range(self.n_estimators):
            gradient = energy_gradient(current, q, w, epsilon=self.collision_epsilon)
            target = _preconditioned_descent_target(gradient, w)
            alpha = learner.coefficients(target.reshape(x.shape[0], flat_columns))
            direction = learner.fitted_train_values(alpha).reshape(target.shape)

            accepted = False
            step = self.learning_rate
            for _backtracks in range(self.max_backtracks + 1):
                candidate = project_quantiles(current + step * direction, w)
                loss_after = empirical_energy_risk(
                    candidate, q, w, epsilon=self.collision_epsilon
                )
                if loss_after <= loss_before + self.descent_tolerance:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            # float32 storage keeps a hundred-step coefficient history near
            # thirty megabytes instead of sixty.
            self.coefficients_.append(alpha.astype(np.float32))
            self.step_sizes_.append(step)
            current = candidate
            loss_before = loss_after
            accepted_steps += 1
        self.train_risk_ = loss_before
        self.n_accepted_steps_ = accepted_steps
        self.learner_ = learner
        return self

    def predict_particles(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "learner_"):
            raise RuntimeError("the model has not been fitted")
        x = _validate_x(X, self.n_features_in_)
        particles = np.broadcast_to(
            self.initial_particles_,
            (x.shape[0],) + self.initial_particles_.shape,
        ).copy()
        n_grid = particles.shape[-1]
        for alpha, step in zip(self.coefficients_, self.step_sizes_, strict=True):
            direction = self.learner_.predict_with(alpha, x).reshape(
                x.shape[0], self.n_particles, n_grid
            )
            particles = project_quantiles(particles + step * direction, self.weights_)
        return canonicalize_particles(particles)
