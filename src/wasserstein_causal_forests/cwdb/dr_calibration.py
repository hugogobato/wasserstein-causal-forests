"""Doubly-robust calibration of law-implied functional contrasts.

The cross-fitted particle booster fits a conditional law whose energy-score
quality is high almost everywhere, and yet its implied reference-distance
contrast loses to the forest baselines on D5 by a factor of four. The recorded
diagnostic explains why: the fitted cloud under-disperses the conditional law,
arm-specifically, and any convex spread-sensitive functional inherits that
attenuation as bias. No amount of additional shrinkage tuning repairs it,
because the bias lives in the plug-in step, not in the contrast surface.

This module adds the missing layer without touching the law. For a functional
h and unit i with covariates X_i, realised arm A_i, outcome q(Y_i), and
out-of-fold arm-law estimates muhat_a(X_i) = E_hat{h(q) | X_i, A = a},

    psi_i = muhat_1(X_i) - muhat_0(X_i)
            + A_i / e_i * ( h(q(Y_i)) - muhat_1(X_i) )
            - (1 - A_i) / (1 - e_i) * ( h(q(Y_i)) - muhat_0(X_i) )

is the augmented inverse-propensity-weighted score for the marginal contrast
E{h(q(Y^1))} - E{h(q(Y^0))}; averaging psi within a moderator bin gives the
Hajek estimator of the conditional contrast at that bin. The estimator is
doubly robust: consistent if either the law-implied means or the propensity is
consistent, and efficient when both are.

Two properties matter for this project's contract. First, the correction is
computed from the realised arm only, so a functional first named at evaluation
time - skewness, an upper-tail mean, a distance to a reference economy nobody
declared at fit time - remains fully eligible: h(q(Y)) is observed whatever h
is, which keeps the D7 transfer story intact while making the transferred
number accurate rather than merely available. Second, nothing here modifies
the predicted laws; every law-level metric of the underlying cross-fitted
booster passes through unchanged, and the diagnostics record that the two
layers are distinct.

The out-of-fold arm-law means come from refitting the selected strength on the
selection folds, which is exactly what the held-out-risk scan already did; the
layer therefore costs n_folds additional booster fits, not a new method family.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression

from .cross_fitted import (
    CrossFittedCWDBRegressor,
    SelectionRecord,
    stratified_folds,
)
from .model import CWDBRegressor

#: Propensity clipping, identical to the Phase 5.5 stack so that every
#: orthogonalized variant in the project shares one overlap rule.
PROPENSITY_CLIP = (0.02, 0.98)

#: Nuisance folds for the propensity. Five matches the Phase 5.5 convention.
AIPW_N_FOLDS = 5


def _moderator_bins(X: NDArray[np.float64]) -> NDArray[np.int64]:
    """The frozen four-bin moderator, imported lazily to keep `cwdb`
    importable without the simulation stack's heavier dependencies."""

    from ..g3.dgps import moderator_bins

    return moderator_bins(X)


def hajek_bin_means(
    values: NDArray[np.float64], bins: NDArray[np.int64], n_bins: int
) -> NDArray[np.float64]:
    """Bin-wise mean of `values`, NaN where a bin is empty."""

    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        rows = bins == b
        if np.any(rows):
            out[b] = float(np.mean(values[rows]))
    return out


def aipw_scores(
    h_observed: NDArray[np.float64],
    mu0: NDArray[np.float64],
    mu1: NDArray[np.float64],
    ehat: NDArray[np.float64],
    treatment: NDArray[np.int64],
) -> NDArray[np.float64]:
    """The AIPW score psi_i for one functional's marginal contrast."""

    treatment = np.asarray(treatment, dtype=int)
    return (mu1 - mu0) + (
        treatment / ehat * (h_observed - mu1)
        - (1 - treatment) / (1.0 - ehat) * (h_observed - mu0)
    )


def _clip_propensity(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(values, PROPENSITY_CLIP[0], PROPENSITY_CLIP[1])


class FunctionalAIPW:
    """One AIPW aggregation per declared functional, over training rows."""

    def __init__(self, *, n_bins: int) -> None:
        self.n_bins = n_bins

    def fit(
        self,
        *,
        observed: dict[str, NDArray[np.float64]],
        oof_arm_means: dict[str, dict[int, NDArray[np.float64]]],
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        random_state: int,
    ) -> "FunctionalAIPW":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        folds = stratified_folds(a, AIPW_N_FOLDS, random_state)
        e_oof = np.empty(x.shape[0])
        for fold in range(AIPW_N_FOLDS):
            held_out = folds == fold
            if not np.any(held_out):
                continue
            model = LogisticRegression(max_iter=2000, solver="lbfgs")
            model.fit(x[~held_out], a[~held_out])
            e_oof[held_out] = _clip_propensity(
                model.predict_proba(x[held_out])[:, 1]
            )
        self.ehat_train_ = e_oof

        self.marginal_: dict[str, float] = {}
        self.bin_contrasts_: dict[str, NDArray[np.float64]] = {}
        self.if_se_: dict[str, float] = {}
        for name, h_values in observed.items():
            scores = aipw_scores(
                h_values,
                oof_arm_means[name][0],
                oof_arm_means[name][1],
                self.ehat_train_,
                a,
            )
            self.marginal_[name] = float(np.mean(scores))
            self.bin_contrasts_[name] = hajek_bin_means(scores, _moderator_bins(x), self.n_bins)
            self.if_se_[name] = float(np.std(scores) / np.sqrt(scores.size))
        return self


class DRCalibratedCWDB(CrossFittedCWDBRegressor):
    """Cross-fitted law plus an AIPW calibration layer for declared functionals.

    ``functionals`` maps a name to a callable mapping an (m, K) quantile block
    to an (m,) value. The reference distance is passed in like any other
    functional: the layer does not know it is special, which is the point.
    """

    def __init__(
        self,
        *,
        functionals: dict[str, object],
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        if not functionals:
            raise ValueError("at least one functional is required")
        self.dr_functionals = dict(functionals)

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        weights: ArrayLike,
    ) -> "DRCalibratedCWDB":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        q = np.asarray(quantiles, dtype=float)
        w = np.asarray(weights, dtype=float)
        folds = stratified_folds(a, self.n_folds, self.random_state)

        records: list[SelectionRecord] = []
        for candidate in self.contrast_candidates:
            risk, n_scored = self._held_out_risk(x, a, q, w, folds, candidate)
            records.append(SelectionRecord(candidate, risk, n_scored))
        self.selection_records_ = tuple(records)
        # Same tie-break as the parent class: toward the stronger regulariser.
        best = min(records, key=lambda record: (record.held_out_risk, -record.contrast_shrinkage))
        self.selected_contrast_shrinkage_ = best.contrast_shrinkage
        self.contrast_shrinkage = best.contrast_shrinkage

        oof_values = {
            name: {arm: np.empty(x.shape[0]) for arm in (0, 1)}
            for name in self.dr_functionals
        }
        for fold in range(self.n_folds):
            held_out = folds == fold
            if not np.any(held_out):
                continue
            model = CWDBRegressor(**self._candidate_parameters(self.contrast_shrinkage))
            model.fit(x[~held_out], a[~held_out], q[~held_out], w)
            for arm in (0, 1):
                particles = model.predict_particles(x[held_out], arm)
                flat = particles.reshape(-1, particles.shape[-1])
                for name, h in self.dr_functionals.items():
                    values = np.asarray(h(flat), dtype=float).reshape(particles.shape[:2])
                    oof_values[name][arm][held_out] = values.mean(axis=1)

        observed = {
            name: np.asarray(h(q), dtype=float)
            for name, h in self.dr_functionals.items()
        }
        self.aipw_ = FunctionalAIPW(n_bins=4)
        self.aipw_.fit(
            observed=observed,
            oof_arm_means=oof_values,
            X=x,
            treatment=a,
            random_state=self.random_state + 31,
        )
        super().fit(x, a, q, w)
        return self

    def dr_marginal(self, name: str) -> float:
        """The AIPW estimate of the functional's marginal contrast."""

        return self.aipw_.marginal_[name]

    def dr_bin_contrasts(self, name: str, X_test: ArrayLike) -> NDArray[np.float64]:
        """DR TCATE-bin contrasts broadcast onto a test design.

        The estimator targets the training population's bin means; the test
        design comes from the same covariate distribution, so broadcasting each
        bin's estimate across its own test rows is the interface the metric
        layer expects, not a pointwise claim at every row.
        """

        bin_estimates = self.aipw_.bin_contrasts_[name]
        return bin_estimates[_moderator_bins(np.asarray(X_test, dtype=float))]

    def dr_if_se(self, name: str) -> float:
        """Influence-function standard error of the marginal estimate."""

        return self.aipw_.if_se_[name]
