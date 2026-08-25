"""WP5.5-D: the vector X-learner ``cwdb_xmean``.

The X-learner's rationale is treatment imbalance: when one arm is more
informative or the effect surface is simpler than either response surface, the
imputation of the counterfactual arm mean buys precision. The pseudo-outcomes
are imputation devices for regression only:

    D_i^(1) = Z_i - mu0(X_i)   on treated rows,
    D_i^(0) = mu1(X_i) - Z_i   on control rows,

with mu0, mu1 cross-fitted arm mean nuisances, so no unit helps create its own
imputed outcome. Two vector regressions estimate tau1 and tau0, combined as

    tau_X(x) = e(x) tau0(x) + (1 - e(x)) tau1(x),

with the weighting convention recorded in the manifest.

Guardrail: even with perfect nuisance fits, the pseudo-outcome distribution is
not the distribution of Y^1 - Y^0, because the cross-arm potential-outcome
coupling is not identified (A12/A14 and the estimand contract). This method
estimates a vector conditional mean only, reports ``LAW-A-K`` targets as
``not_applicable``, and gains its intended advantage only under declared
imbalance or overlap stress, never on a balanced design alone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..cwdb.geometry import from_rescaled, to_rescaled
from .boosted import BoostedMultiOutputRegressor
from .nuisance import (
    CrossFittedNuisance,
    FoldPlan,
    NUISANCE_BUDGET,
    PROPENSITY_CLIP_HIGH,
    PROPENSITY_CLIP_LOW,
)

#: Frozen budget for the effect regressions. The pseudo-outcomes carry the
#: imputation noise of the arm nuisances, so the effect regressions accumulate
#: noise along the boosting path even faster than the R-loss: the pilot scan
#: (seeds 100-104) found the imbalance-suite contrast error minimized at three
#: steps and rising monotonically afterwards. Three steps is the frozen rule.
EFFECT_BUDGET = {
    "n_estimators": 3,
    "learning_rate": 0.12,
    "max_depth": 4,
    "min_samples_leaf": 10,
}


class VectorXLearner:
    """Cross-fitted vector X-learner for the rescaled quantile vector."""

    def __init__(
        self,
        *,
        n_nuisance_folds: int = 5,
        effect_budget: dict[str, int | float] | None = None,
        nuisance_budget: dict[str, int | float] | None = None,
        propensity_clip: tuple[float, float] = (PROPENSITY_CLIP_LOW, PROPENSITY_CLIP_HIGH),
        random_state: int = 0,
    ) -> None:
        self.n_nuisance_folds = n_nuisance_folds
        self.effect_budget = dict(EFFECT_BUDGET if effect_budget is None else effect_budget)
        self.nuisance_budget = dict(NUISANCE_BUDGET if nuisance_budget is None else nuisance_budget)
        self.propensity_clip = tuple(float(c) for c in propensity_clip)
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        grid_weights: ArrayLike,
    ) -> "VectorXLearner":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        q = np.asarray(quantiles, dtype=float)
        w = np.asarray(grid_weights, dtype=float)
        if x.ndim != 2 or a.shape != (x.shape[0],) or q.ndim != 2 or q.shape[0] != x.shape[0]:
            raise ValueError("expected X (n, p), treatment (n,), quantiles (n, K)")
        z = to_rescaled(q, w)
        self.n_features_in_ = x.shape[1]
        self.n_coordinates_ = q.shape[1]
        self.grid_weights_ = w.copy()

        self.folds_ = FoldPlan.stratified(
            a, self.n_nuisance_folds, keys=x, random_state=self.random_state
        )
        mu0_models: list[BoostedMultiOutputRegressor] = []
        mu1_models: list[BoostedMultiOutputRegressor] = []
        mu0_oof = np.empty(z.shape)
        mu1_oof = np.empty(z.shape)
        for fold in range(self.n_nuisance_folds):
            train_rows = self.folds_.labels != fold
            holdout = self.folds_.labels == fold
            control = train_rows & (a == 0)
            treated = train_rows & (a == 1)
            mu0_model = BoostedMultiOutputRegressor(
                random_state=self.random_state + 20_000 + fold,
                **self.nuisance_budget,
            )
            mu0_model.fit(x[control], z[control])
            mu1_model = BoostedMultiOutputRegressor(
                random_state=self.random_state + 30_000 + fold,
                **self.nuisance_budget,
            )
            mu1_model.fit(x[treated], z[treated])
            mu0_models.append(mu0_model)
            mu1_models.append(mu1_model)
            mu0_oof[holdout] = mu0_model.predict(x[holdout])
            mu1_oof[holdout] = mu1_model.predict(x[holdout])
        self.mu0_models_ = tuple(mu0_models)
        self.mu1_models_ = tuple(mu1_models)
        self.mu0_oof_ = mu0_oof
        self.mu1_oof_ = mu1_oof

        # Imputed pseudo-outcomes, computed only for regression, and only from
        # out-of-fold nuisances.
        treated_rows = a == 1
        control_rows = a == 0
        d1 = z[treated_rows] - mu0_oof[treated_rows]
        d0 = mu1_oof[control_rows] - z[control_rows]

        self.tau1_model_ = BoostedMultiOutputRegressor(
            random_state=self.random_state + 40_000,
            **self.effect_budget,
        )
        self.tau1_model_.fit(x[treated_rows], d1)
        self.tau0_model_ = BoostedMultiOutputRegressor(
            random_state=self.random_state + 50_000,
            **self.effect_budget,
        )
        self.tau0_model_.fit(x[control_rows], d0)

        nuisance = CrossFittedNuisance(
            n_folds=self.n_nuisance_folds,
            propensity_clip=self.propensity_clip,
            prognostic_budget=self.nuisance_budget,
            random_state=self.random_state,
        )
        nuisance.fit(x, a, z)
        self.nuisance_ = nuisance
        self.ehat_train_ = nuisance.ehat_oof_
        self.tau1_train_ = self.tau1_model_.predict(x)
        self.tau0_train_ = self.tau0_model_.predict(x)
        self.contrast_train_ = (
            self.ehat_train_[:, None] * self.tau0_train_
            + (1.0 - self.ehat_train_)[:, None] * self.tau1_train_
        )
        return self

    def predict_contrast(self, X: ArrayLike) -> NDArray[np.float64]:
        """The X-learner contrast tau_X in the rescaled coordinates."""
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        e = self.nuisance_.ehat(x)
        return e[:, None] * self.tau0_model_.predict(x) + (1.0 - e)[:, None] * self.tau1_model_.predict(x)

    def predict_arm_mean(self, X: ArrayLike, arm: int) -> NDArray[np.float64]:
        """Arm mean quantile vectors in the rescaled coordinates.

        The control mean is the (more precisely estimated) control nuisance;
        the treated mean is the control nuisance plus the X-learner contrast,
        so the reported mean-quantile contrast is exactly tau_X.
        """
        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1")
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        mu0 = np.mean(
            np.stack([model.predict(x) for model in self.mu0_models_], axis=0), axis=0
        )
        if arm == 0:
            return mu0
        return mu0 + self.predict_contrast(x)

    def predict_mean_quantiles(self, X: ArrayLike, arm: int) -> NDArray[np.float64]:
        """Arm mean quantile vector in the original (unscaled) coordinates."""
        return from_rescaled(self.predict_arm_mean(X, arm), self.grid_weights_)
