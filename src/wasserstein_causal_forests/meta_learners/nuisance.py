"""WP5.5-A: the cross-fitting splitter and the shared nuisance stack.

Every Phase 5.5 variant shares one leak-free infrastructure layer: the same
arm-stratified folds, the same propensity estimate, the same pooled prognostic
mean, the same outcome transformation, and the same declared clipping rule.
Sharing one implementation is what makes a later comparison between variants a
comparison of the orthogonalization mechanisms and not of their plumbing.

Leakage rule (A15, enforced here rather than by convention): every training-row
prediction of e or m comes from a model fitted on the complement of the row's
own fold, and no evaluation row ever influences any prediction made on it. New
points are predicted by averaging the fold models; they were in no training
fold, so every fold model is honest for them, and averaging uses all of the
data rather than discarding a fold.

The propensity model is a plain logistic regression on the raw covariates,
which every tournament regime's propensity is (a clipped logistic of a linear
index) exactly. The clipping bounds are declared in the manifest before any
fit; a variant must never let an overlap rule depend on the observed effect or
on a held-out metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression

from .boosted import BoostedMultiOutputRegressor

#: Propensity clipping, frozen in `configs/simulation_phase55.yaml`. The R-loss
#: is stable only when A - e is bounded away from zero, so the clip is part of
#: the overlap rule, not an outcome-dependent trim.
PROPENSITY_CLIP_LOW = 0.02
PROPENSITY_CLIP_HIGH = 0.98

#: Number of folds for nuisance cross-fitting. Five is the standard choice in
#: the R-learner literature and keeps the complement large enough at n = 500.
DEFAULT_NUISANCE_FOLDS = 5

#: Frozen budget for the nuisance regressions. The nuisances are interchangeable
#: parts, so their budget is deliberately smaller than the contrast budget.
NUISANCE_BUDGET = {
    "n_estimators": 50,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 10,
}


@dataclass(frozen=True)
class FoldPlan:
    """Deterministic arm-stratified folds, with the audit trail attached.

    Each arm's rows are dealt round-robin over a per-arm permutation, so every
    fold holds close to the same number of treated and control rows and no
    fold's complement can lose an arm entirely. The plan records the seed and
    the per-fold treatment counts so the audit can verify stratification and
    reproducibility without re-running the split.
    """

    labels: NDArray[np.int64]
    n_folds: int
    random_state: int
    treatment_counts: tuple[tuple[int, int], ...]

    @classmethod
    def stratified(
        cls,
        treatment: ArrayLike,
        n_folds: int,
        *,
        keys: ArrayLike,
        random_state: int,
    ) -> "FoldPlan":
        """Arm-stratified folds that are a pure function of the data.

        Rows are ordered canonically by their covariates (and treatment) before
        the arm-wise round-robin, so the fold assignment does not depend on the
        input row order: permuting the rows leaves every prediction invariant,
        and re-running with the same data reproduces the same folds exactly.
        The random seed is recorded for the audit but no longer drives the
        split, because a random split would make the plan a function of row
        order. Covariates and treatment are pre-treatment, so the plan never
        depends on an outcome.
        """

        treatment = np.asarray(treatment, dtype=int)
        keys = np.asarray(keys, dtype=float)
        if treatment.ndim != 1 or not np.all(np.isin(treatment, (0, 1))):
            raise ValueError("treatment must be a binary vector")
        if keys.ndim != 2 or keys.shape[0] != treatment.shape[0]:
            raise ValueError("keys must have shape (n, p)")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if np.any(np.bincount(treatment, minlength=2) < n_folds):
            raise ValueError("each arm must have at least n_folds observations")

        combined = np.column_stack((keys, treatment))
        order = np.lexsort(combined[:, ::-1].T)
        labels = np.empty(treatment.shape[0], dtype=np.int64)
        for arm in (0, 1):
            positions = order[treatment[order] == arm]
            labels[positions] = np.arange(positions.size) % n_folds
        counts: list[tuple[int, int]] = []
        for fold in range(n_folds):
            rows = labels == fold
            counts.append(
                (
                    int(np.sum(treatment[rows] == 0)),
                    int(np.sum(treatment[rows] == 1)),
                )
            )
        return cls(
            labels=labels,
            n_folds=n_folds,
            random_state=random_state,
            treatment_counts=tuple(counts),
        )

    def fold_rows(self, fold: int) -> NDArray[np.int64]:
        """Indices of one fold's rows."""
        return np.flatnonzero(self.labels == fold)


def clip_propensity(values: ArrayLike) -> NDArray[np.float64]:
    """Declared propensity clipping, the project-wide overlap rule."""
    return np.clip(np.asarray(values, dtype=float), PROPENSITY_CLIP_LOW, PROPENSITY_CLIP_HIGH)


def fit_propensity_predictor(
    X: ArrayLike, treatment: ArrayLike, random_state: int
) -> LogisticRegression:
    """One deterministic logistic propensity model."""
    x = np.asarray(X, dtype=float)
    a = np.asarray(treatment, dtype=int)
    if x.ndim != 2 or a.shape != (x.shape[0],):
        raise ValueError("expected X (n, p) and treatment (n,)")
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(x, a)
    return model


class CrossFittedNuisance:
    """Cross-fitted propensity e(x) and pooled prognostic mean m(x) = E[Z|X].

    ``Z`` is the outcome in rescaled coordinates; the caller performs the
    ``W^{1/2}q(Y)`` transformation once so that every variant shares it.
    """

    def __init__(
        self,
        *,
        n_folds: int = DEFAULT_NUISANCE_FOLDS,
        propensity_clip: tuple[float, float] = (
            PROPENSITY_CLIP_LOW,
            PROPENSITY_CLIP_HIGH,
        ),
        prognostic_budget: dict[str, int | float] | None = None,
        random_state: int = 0,
    ) -> None:
        self.n_folds = n_folds
        self.propensity_clip = tuple(float(c) for c in propensity_clip)
        self.prognostic_budget = dict(NUISANCE_BUDGET if prognostic_budget is None else prognostic_budget)
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        Z: ArrayLike,
    ) -> "CrossFittedNuisance":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        z = np.asarray(Z, dtype=float)
        if x.ndim != 2 or a.shape != (x.shape[0],) or z.ndim != 2 or z.shape[0] != x.shape[0]:
            raise ValueError("expected X (n, p), treatment (n,), Z (n, K)")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)):
            raise ValueError("X and Z must be finite")
        self.n_features_in_ = x.shape[1]

        self.folds_ = FoldPlan.stratified(
            a, self.n_folds, keys=x, random_state=self.random_state
        )
        e_models: list[LogisticRegression] = []
        m_models: list[BoostedMultiOutputRegressor] = []
        e_oof = np.empty(a.shape[0])
        m_oof = np.empty(z.shape)
        for fold in range(self.n_folds):
            train_rows = self.folds_.labels != fold
            holdout = self.folds_.labels == fold
            e_model = fit_propensity_predictor(
                x[train_rows], a[train_rows], self.random_state + fold
            )
            m_model = BoostedMultiOutputRegressor(
                random_state=self.random_state + 10_000 + fold,
                **self.prognostic_budget,
            )
            m_model.fit(x[train_rows], z[train_rows])
            e_models.append(e_model)
            m_models.append(m_model)
            e_oof[holdout] = np.clip(
                e_model.predict_proba(x[holdout])[:, 1],
                *self.propensity_clip,
            )
            m_oof[holdout] = m_model.predict(x[holdout])
        self.e_models_ = tuple(e_models)
        self.m_models_ = tuple(m_models)
        self.ehat_oof_ = e_oof
        self.mhat_oof_ = m_oof
        return self

    def ehat(self, X: ArrayLike) -> NDArray[np.float64]:
        """Propensity at new points: the average of the fold models."""
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        predictions = np.stack(
            [model.predict_proba(x)[:, 1] for model in self.e_models_], axis=0
        )
        return np.clip(
            predictions.mean(axis=0), *self.propensity_clip
        )

    def mhat(self, X: ArrayLike) -> NDArray[np.float64]:
        """Pooled prognostic mean at new points: the average of the fold models."""
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        return np.mean(np.stack(
            [model.predict(x) for model in self.m_models_], axis=0
        ), axis=0)
