"""The functional R-learner: orthogonalized contrasts for arbitrary h.

Phase 5.5 established that orthogonalisation is what removes manufactured
heterogeneity on the null regime, but its vector R-learner targets one object,
the rescaled quantile vector, and therefore answers only the grid causal mean.
Every interesting target of this project is a nonlinear functional of the law:
a standard deviation, a skewness coefficient, an upper-tail mean, a distance
to a reference economy. This module generalises the R-loss to any finite
collection of such functionals without giving up what made `cwdb_rmean`
null-safe.

For each declared functional h the semantic objective is

    L_h(t) = (1/n) sum_i ( h_i - mhat_{h,-i}(X_i)
                           - (A_i - ehat_{-i}(X_i)) t_h(X_i) )^2,

with h_i = h(q(Y_i)) observed on the realised arm. The functionals share one
contrast forest - the trees see all h-columns jointly, so a split useful for
one is available to all - and one cross-fitted nuisance plan, which keeps the
orthogonalized variants comparable as mechanisms rather than as plumbing.
Because h enters only through its realised value, a functional first named at
evaluation time is exactly as estimable as one declared at fit time: the
transfer claim survives orthogonalisation intact.

What this method is not: it produces no conditional law. Its law-level entries
are `not_applicable` by contract, and its TCATE surfaces are leaf-constant
curves, not integrals over a predictive distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .nuisance import CrossFittedNuisance, FoldPlan
from .r_learner import CONTRAST_BUDGET, RLossTree

#: Frozen candidate strengths and selection folds, identical to `cwdb_rmean`'s,
#: so the two R-learners differ in their target space and nothing else.
FRL_CANDIDATES: tuple[float, ...] = (0.0, 50.0, 500.0)
FRL_SELECTION_FOLDS = 2


class FunctionalRLearner:
    """Joint scalar R-losses over a declared functional dictionary."""

    def __init__(
        self,
        *,
        functionals: dict[str, object],
        n_nuisance_folds: int = 5,
        contrast_budget: dict[str, int | float] | None = None,
        contrast_candidates: tuple[float, ...] = FRL_CANDIDATES,
        n_selection_folds: int = FRL_SELECTION_FOLDS,
        random_state: int = 0,
    ) -> None:
        if not functionals:
            raise ValueError("at least one functional is required")
        self.functionals = dict(functionals)
        self.n_nuisance_folds = n_nuisance_folds
        self.contrast_budget = dict(
            CONTRAST_BUDGET if contrast_budget is None else contrast_budget
        )
        self.contrast_candidates = tuple(float(c) for c in contrast_candidates)
        self.n_selection_folds = n_selection_folds
        self.random_state = random_state

    # ------------------------------------------------------------------ helpers

    def _functional_matrix(self, quantiles: NDArray[np.float64]) -> NDArray[np.float64]:
        """The (n, H) block of realised functional values."""

        return np.column_stack([
            np.asarray(h(quantiles), dtype=float)
            for h in self.functionals.values()
        ])

    def _boost(
        self,
        x: NDArray[np.float64],
        weight: NDArray[np.float64],
        residual: NDArray[np.float64],
        shrinkage: float,
        seed_offset: int,
    ):
        """One boosting run of the shared-partition contrast trees."""

        learning_rate = float(self.contrast_budget["learning_rate"])
        contrast = np.zeros_like(residual)
        current = residual.copy()
        trees: list[RLossTree] = []
        for iteration in range(int(self.contrast_budget["n_estimators"])):
            tree = RLossTree(
                max_depth=int(self.contrast_budget["max_depth"]),
                min_samples_leaf=int(self.contrast_budget["min_samples_leaf"]),
                contrast_shrinkage=shrinkage,
                random_state=self.random_state + seed_offset + iteration,
            )
            tree.fit(x, current, weight)
            contrast = contrast + learning_rate * tree.predict(x)
            current = residual - weight[:, None] * contrast
            trees.append(tree)
        return trees, contrast

    def _replay(self, trees: list[RLossTree], x: NDArray[np.float64]) -> NDArray[np.float64]:
        learning_rate = float(self.contrast_budget["learning_rate"])
        contrast = np.zeros((x.shape[0], len(self.functionals)))
        for tree in trees:
            contrast = contrast + learning_rate * tree.predict(x)
        return contrast

    # ---------------------------------------------------------------------- API

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
    ) -> "FunctionalRLearner":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        q = np.asarray(quantiles, dtype=float)
        if x.ndim != 2 or a.shape != (x.shape[0],) or q.shape[0] != x.shape[0]:
            raise ValueError("expected X (n, p), treatment (n,), quantiles (n, K)")
        self.functional_names_ = tuple(self.functionals)

        h_values = self._functional_matrix(q)

        # One cross-fitted plan carries both nuisances, so every m-hat column
        # and every propensity value respect the same leakage rule.
        nuisance = CrossFittedNuisance(n_folds=self.n_nuisance_folds)
        nuisance.fit(x, a, h_values)
        self.nuisance_ = nuisance
        ehat = nuisance.ehat_oof_
        mhat = nuisance.mhat_oof_
        weight_train = a - ehat

        folds = FoldPlan.stratified(
            a, self.n_selection_folds, keys=x, random_state=self.random_state + 7
        )
        n_columns = h_values.shape[1]
        # Per-column held-out risk per candidate. A joint scalar selection
        # would couple coordinates that carry no contrast with functionals
        # whose entire signal is the contrast - the D5 failure this phase
        # diagnosed - so the ridge strength is chosen column by column from
        # the same fold runs.
        risk_by_candidate: dict[float, NDArray[np.float64]] = {}
        for candidate in self.contrast_candidates:
            risks: list[NDArray[np.float64]] = []
            for fold in range(self.n_selection_folds):
                train_rows = folds.labels != fold
                holdout = folds.labels == fold
                trees, _ = self._boost(
                    x[train_rows],
                    weight_train[train_rows],
                    h_values[train_rows] - mhat[train_rows],
                    candidate,
                    seed_offset=1000 * (fold + 1),
                )
                held_residual = (
                    h_values[holdout]
                    - mhat[holdout]
                    - (a[holdout] - ehat[holdout])[:, None]
                    * self._replay(trees, x[holdout])
                )
                risks.append(np.sum(held_residual * held_residual, axis=0))
            risk_by_candidate[candidate] = np.mean(np.stack(risks), axis=0)
        self.selection_records_ = tuple(
            (candidate, float(np.mean(risk)))
            for candidate, risk in risk_by_candidate.items()
        )
        shrinkage_vector = np.empty(n_columns)
        for column in range(n_columns):
            best = min(
                self.contrast_candidates,
                key=lambda c: (float(risk_by_candidate[c][column]), -c),
            )
            shrinkage_vector[column] = best
        self.shrinkage_vector_ = shrinkage_vector
        self.selected_shrinkage_ = float(np.mean(shrinkage_vector))

        residual = h_values - mhat
        self.estimators_, contrast_train = self._boost(
            x, weight_train, residual, shrinkage_vector, seed_offset=0
        )
        self.contrast_train_ = contrast_train
        self.train_risk_ = float(
            np.mean(
                np.sum(
                    (residual - weight_train[:, None] * contrast_train) ** 2, axis=1
                )
            )
        )
        return self

    def predict_contrasts(self, X: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """Pointwise contrast estimates t_h(x), one array per functional."""

        x = np.asarray(X, dtype=float)
        if not getattr(self, "estimators_", None):
            raise RuntimeError("the model has not been fitted")
        stacked = self._replay(list(self.estimators_), x)
        return {
            name: stacked[:, index]
            for index, name in enumerate(self.functional_names_)
        }

    def _mean_columns(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.mean(
            np.stack([model.predict(x) for model in self.nuisance_.m_models_], axis=0),
            axis=0,
        )

    def predict_arm_means(
        self, X: ArrayLike, arm: int
    ) -> dict[str, NDArray[np.float64]]:
        """Arm-level functional means reconstructed from the pooled identity:

        m_0 = m - e t and m_1 = m + (1 - e) t, evaluated per functional. The
        same reconstruction `cwdb_rmean` uses, applied per functional column.
        """

        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1")
        x = np.asarray(X, dtype=float)
        contrasts = self.predict_contrasts(x)
        e = self.nuisance_.ehat(x)
        means = self._mean_columns(x)
        return {
            name: (
                means[:, index] - e * contrasts[name]
                if arm == 0
                else means[:, index] + (1.0 - e) * contrasts[name]
            )
            for index, name in enumerate(self.functional_names_)
        }
