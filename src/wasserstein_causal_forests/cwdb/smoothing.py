"""Dispersion repair of the boosted particle cloud, chosen out of sample.

The diagnostic that opened Phase 6 measured the mechanism behind the D5 and
D6 reference-effect losses: the fitted particle cloud under-disperses the
conditional law, arm-specifically. On D5 arm 0, whose outer location law has
standard deviation 0.40, the predicted cloud's spread falls short of what the
true conditional law needs, and every convex functional of the spread - first
among them the Wasserstein distance to a reference economy - is biased low.
The energy score's repulsion term prevents collapse but does not guarantee
calibrated dispersion at M = 10 atoms with a tree weak learner.

This module repairs dispersion after the booster rather than inside it. Two
families of transforms are considered, both operating on a row's particle
cloud in the declared geometry:

* radial scaling by c around the cloud barycenter, followed by monotone
  projection - a pure variance correction;
* Gaussian jitter of standard deviation sigma in the rescaled coordinates,
  followed by monotone projection, replicated S times per particle - a
  kernel smoothing of the law that can also fill gaps between modes.

The transform and its strength are selected on held-out energy score over an
arm-stratified calibration split whose rows never enter any fit that predicts
them, then the booster is refitted on the whole sample. The candidate grid is
frozen here and identical in every cell; no parameter may depend on the regime.

Nothing about the training objective changes: the selected transform is a
post-processing of predictions, so the propriety of the score the booster
minimises is untouched, and the output remains a genuine predictive law.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cross_fitted import CrossFittedCWDBRegressor, stratified_folds
from .energy import empirical_energy_risk
from .geometry import project_quantiles
from .model import CWDBRegressor

#: Frozen transform candidates. Scale 1.0 with no jitter is the identity, so
#: "no repair helps" is inside the searched set and must win outright where it
#: is true.
SCALE_CANDIDATES: tuple[float, ...] = (1.0, 1.15, 1.3, 1.45)
JITTER_CANDIDATES: tuple[float, ...] = (0.0, 0.08, 0.16, 0.28)
JITTER_REPLICATES = 4


def scale_cloud(
    particles: NDArray[np.float64], weights: NDArray[np.float64], scale: float
) -> NDArray[np.float64]:
    """Radial scaling around each row's barycenter, projected to the cone."""

    if scale == 1.0:
        return particles
    barycenter = particles.mean(axis=1, keepdims=True)
    scaled = barycenter + scale * (particles - barycenter)
    n_rows = particles.shape[0]
    flat = scaled.reshape(n_rows * particles.shape[1], -1)
    projected = project_quantiles(flat, weights)
    return projected.reshape(particles.shape)


def jitter_cloud(
    particles: NDArray[np.float64],
    weights: NDArray[np.float64],
    sigma: float,
    replicates: int,
    random_state: int,
) -> NDArray[np.float64]:
    """Gaussian jitter in rescaled coordinates, projected, with replicates.

    Output atom count is replicates times the input count, uniformly weighted;
    `LawPrediction.from_particles` supplies the uniform weights downstream.
    """

    if sigma <= 0.0:
        return particles
    rng = np.random.default_rng(random_state)
    n_rows, n_atoms, n_grid = particles.shape
    root_w = np.sqrt(weights)
    z = root_w * particles
    noise = rng.normal(size=(n_rows, n_atoms, replicates, n_grid)) * sigma
    expanded = np.broadcast_to(z[:, :, None, :], (n_rows, n_atoms, replicates, n_grid)) + noise
    back = expanded / root_w
    projected = project_quantiles(back.reshape(-1, n_grid), weights)
    return projected.reshape(n_rows, n_atoms * replicates, n_grid)


def candidate_transforms() -> tuple[tuple[str, float], ...]:
    """The frozen candidate list, identity first."""

    return tuple([("scale", s) for s in SCALE_CANDIDATES]
                 + [("jitter", j) for j in JITTER_CANDIDATES[1:]])


class SmoothedCWDB(CrossFittedCWDBRegressor):
    """Cross-fitted booster whose predictions receive one calibrated transform.

    The selection split is carved out before anything is fitted: the contrast
    scan runs inside the fit split only, its fold models produce honest
    predictions on the calibration rows, the transform minimises held-out
    energy score there, and the final model refits on the full sample.
    """

    def __init__(
        self,
        *,
        jitter_replicates: int = JITTER_REPLICATES,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        self.jitter_replicates = jitter_replicates

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        weights: ArrayLike,
    ) -> "SmoothedCWDB":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        q = np.asarray(quantiles, dtype=float)
        w = np.asarray(weights, dtype=float)

        outer_folds = stratified_folds(a, 5, self.random_state + 3)
        calib = outer_folds == 0
        fit_rows = ~calib

        inner_folds = stratified_folds(a[fit_rows], self.n_folds, self.random_state)
        records = []
        for candidate in self.contrast_candidates:
            risk, _ = self._held_out_risk(
                x[fit_rows], a[fit_rows], q[fit_rows], w,
                inner_folds, candidate,
            )
            records.append((candidate, risk))
        best_strength = min(records, key=lambda item: (item[1], -item[0]))[0]
        self.selected_contrast_shrinkage_ = best_strength
        self.contrast_shrinkage = best_strength

        # Each fold model is honest for every calibration row - the split was
        # carved out before any fit - so the ensemble prediction averages the
        # per-fold clouds on the calibration rows, each row scored against the
        # law of its own observed arm.
        calib_indices = np.flatnonzero(calib)
        cloud_accumulator = np.zeros(
            (calib_indices.size, self.n_particles, q.shape[1])
        )
        for fold in range(self.n_folds):
            holdout = inner_folds == fold
            model = CWDBRegressor(**self._candidate_parameters(best_strength))
            rows_fit = np.flatnonzero(fit_rows)[~holdout]
            model.fit(x[rows_fit], a[rows_fit], q[rows_fit], w)
            for arm in (0, 1):
                mask = a[calib_indices] == arm
                if not np.any(mask):
                    continue
                cloud_accumulator[mask] = model.predict_particles(
                    x[calib_indices[mask]], arm
                )
        oof_particles = cloud_accumulator

        best = None
        epsilon = self.collision_epsilon
        for kind, value in candidate_transforms():
            if kind == "scale":
                transformed = scale_cloud(oof_particles, w, value)
            else:
                transformed = jitter_cloud(
                    oof_particles, w, value, self.jitter_replicates,
                    self.random_state + 101,
                )
            risk = empirical_energy_risk(
                transformed, q[calib], w, epsilon=epsilon
            )
            if best is None or risk < best[0]:
                best = (risk, kind, value)
        _, self.selected_transform_, self.transform_value_ = best

        super().fit(x, a, q, w)
        return self

    def predict_particles(self, X: ArrayLike, arm: int) -> NDArray[np.float64]:
        from ..common.quantiles import canonicalize_particles

        raw = self._predict_labeled(X, arm)
        if self.selected_transform_ == "scale":
            transformed = scale_cloud(raw, self.weights_, float(self.transform_value_))
        elif float(self.transform_value_) > 0.0:
            transformed = jitter_cloud(
                raw, self.weights_, float(self.transform_value_),
                self.jitter_replicates, self.random_state + 202,
            )
        else:
            transformed = raw
        return canonicalize_particles(transformed)
