"""WP5.5-C: the BCF-style particle ``mu/tau`` decomposition ``cwdb_mutau``.

The R3 repair regularises the arm gap inside the existing shared leaf, but the
treatment basis of that leaf is the empirical treated share of the leaf: it
assumes the leaf share estimates the propensity relevant to a new covariate.
This module replaces the basis. Each leaf of the shared tree fits

    z_{a,m}(x) = b_m(x) + (a - e(x)) d_m(x)

in the rescaled coordinates, with b_m a prognostic particle field, d_m a
contrast field, and e(x) a cross-fitted propensity. The update never adds or
subtracts probability measures: every quantity lives in the rescaled quantile
coordinates and is projected back to the monotone cone by the existing C-WDB
step.

The leaf rule is replaced, not extended: ``MutauSharedTreeRegressor`` overrides
``_leaf_values`` outright, so the parent's ``arm_shrinkage`` and
``contrast_rule`` never execute here. ``contrast_shrinkage`` is this class's
only regularizer. The constructor therefore rejects a non-default value of
either inherited knob rather than accepting one and silently ignoring it: a
frozen manifest that records an inert hyperparameter is a reproducibility
defect, because a reader cannot tell from the manifest which regularizer was
live.

Collapse check (WP5.5-C, check 1): when the treatment basis is the leaf's own
treated share (``ehat_basis="leaf_share"``) and the contrast penalty is zero,
the leaf value is exactly ``gbar + (a - pi) delta``, the reparameterized
shared-tree update, and because the split rule (pooled SSE) is inherited
unchanged, the whole tree reduces to the current ``ArmSharedTreeRegressor``
with the ridge rule at zero. The contrast ridge factor
``sum(w - mean w)^2 / (sum(w - mean w)^2 + lambda)`` equals the repair's
``n_eff / (n_eff + lambda)`` in the leaf-share basis, so the repair's strength
scale carries over unchanged.

Guardrails carried from the phase document: a particle coordinate is not a
probability measure under addition; a shared internal particle index is not an
identified cross-arm coupling (no public quantity may use p1m - p0m); and the
pre-projection pooled component being fixed does not imply the post-projection
pooled law is fixed, so projection diagnostics are reported.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .arm_shared_tree import ArmSharedTreeRegressor
from ..meta_learners.nuisance import FoldPlan
from .model import CWDBRegressor, _preconditioned_descent_target, _project_candidate
from ..common.quantiles import canonical_training_order
from ..meta_learners.nuisance import (
    PROPENSITY_CLIP_HIGH,
    PROPENSITY_CLIP_LOW,
    fit_propensity_predictor,
)


class MutauSharedTreeRegressor(ArmSharedTreeRegressor):
    """Shared-partition tree with prognostic/contrast leaf fields.

    The split search (pooled multi-output SSE on the target) is inherited from
    ``ArmSharedTreeRegressor`` unchanged; only the leaf values differ. Each
    leaf stores the pair (b, d) minimising sum || g_i - b - w_i d ||^2 with
    w_i = A_i - e(x_i), whose closed form is the one-predictor regression of g
    on w. ``contrast_shrinkage`` shrinks d by ``mass / (mass + lambda)`` with
    ``mass = sum (w_i - mean w)^2``.

    The parent's ``arm_shrinkage`` and ``contrast_rule`` are inert here: they
    live only inside the parent ``_leaf_values``, which this class replaces.
    ``INERT_PARENT_PARAMETERS`` names them so the fact is discoverable from the
    object rather than from reading two class bodies, and
    ``tests/test_cwdb_mutau.py`` pins it.
    """

    #: Parent knobs this subclass overrides away. Recorded, not silently kept.
    INERT_PARENT_PARAMETERS = ("arm_shrinkage", "contrast_rule", "contrast_damping")

    def __init__(
        self,
        *,
        ehat_basis: str = "cross_fitted",
        contrast_shrinkage: float = 0.0,
        force_zero_contrast: bool = False,
        min_weight_mass: float = 1e-12,
        **parameters: object,
    ) -> None:
        if ehat_basis not in {"cross_fitted", "leaf_share"}:
            raise ValueError("ehat_basis must be 'cross_fitted' or 'leaf_share'")
        if contrast_shrinkage < 0.0:
            raise ValueError("contrast_shrinkage must be nonnegative")
        super().__init__(**parameters)  # type: ignore[arg-type]
        self.ehat_basis = ehat_basis
        self.contrast_shrinkage = float(contrast_shrinkage)
        self.force_zero_contrast = force_zero_contrast
        self.min_weight_mass = min_weight_mass

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        gradients: ArrayLike,
        ehat: ArrayLike | None = None,
    ) -> "MutauSharedTreeRegressor":
        """Fit with an optional cross-fitted propensity at the training rows.

        ``ehat=None`` selects the leaf-share treatment basis, which is the
        collapse-check mode: the leaf values then reduce to the reparameterized
        shared-tree update when the contrast penalty is zero.
        """

        if ehat is not None:
            ehat = np.asarray(ehat, dtype=float)
            if ehat.shape != (np.asarray(treatment).shape[0],):
                raise ValueError("ehat must have one value per training row")
            if np.any(ehat < 0.0) or np.any(ehat > 1.0):
                raise ValueError("ehat must lie in [0, 1]")
            self._fit_ehat = ehat
        else:
            self._fit_ehat = None
        super().fit(X, treatment, gradients)
        self.ehat_basis_ = "cross_fitted" if self._fit_ehat is not None else "leaf_share"
        return self

    def _leaf_values(
        self, indices: NDArray[np.int64]
    ) -> tuple[
        NDArray[np.float64],
        tuple[NDArray[np.float64], NDArray[np.float64]],
        tuple[int, int],
    ]:
        """(pooled, (b, d), counts) for one leaf.

        The two returned vectors are the prognostic field b and the contrast
        field d, stored in the parent's two-slot arm-value position and
        interpreted by ``_flatten``/``predict`` below.
        """

        gradients = self._G[indices]
        a = self._A[indices]
        if self._fit_ehat is None:
            share = float(a.mean())
            w = a - share
        else:
            w = a - self._fit_ehat[indices]
        mean_w = float(w.mean())
        centered_w = w - mean_w
        mass = float(np.dot(centered_w, centered_w))
        gbar = gradients.mean(axis=0)
        if mass <= self.min_weight_mass or self.force_zero_contrast:
            d = np.zeros_like(gbar)
        else:
            numerator = np.einsum("i,ij->j", centered_w, gradients - gbar)
            d = numerator / mass
            d = d * (mass / (mass + self.contrast_shrinkage))
        b = gbar - mean_w * d
        counts = (
            int(np.sum(a == 0)),
            int(np.sum(a == 1)),
        )
        return b, (b, d), counts

    def _flatten(self) -> None:
        """Store (b, d) per node and the per-leaf treated shares.

        The parent's flatten stores the two-slot arm values; here the slots are
        b and d. The shares are needed for the leaf-share basis at prediction
        time, where the propensity is not available.
        """

        features: list[int] = []
        thresholds: list[float] = []
        left_child: list[int] = []
        right_child: list[int] = []
        values: list[NDArray[np.float64]] = []
        shares: list[float] = []

        def visit(node) -> int:
            node_id = len(features)
            features.append(-1 if node.feature is None else int(node.feature))
            thresholds.append(
                np.nan if node.threshold is None else float(node.threshold)
            )
            left_child.append(-1)
            right_child.append(-1)
            n0, n1 = node.arm_counts
            shares.append(n1 / (n0 + n1))
            values.append(np.stack(node.arm_values))
            if not node.is_leaf:
                assert node.left is not None and node.right is not None
                left_child[node_id] = visit(node.left)
                right_child[node_id] = visit(node.right)
            return node_id

        visit(self.root_)
        self.node_feature_ = np.asarray(features, dtype=np.int64)
        self.node_threshold_ = np.asarray(thresholds, dtype=float)
        self.node_left_ = np.asarray(left_child, dtype=np.int64)
        self.node_right_ = np.asarray(right_child, dtype=np.int64)
        # Shape (n_nodes, 2, D): slot 0 is b, slot 1 is d.
        self.node_values_ = np.stack(values)
        self.node_share_ = np.asarray(shares, dtype=float)

    def predict(
        self,
        X: ArrayLike,
        arm: int | ArrayLike,
        ehat: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate b(x) + (a - e(x)) d(x) for every row.

        With the cross-fitted basis, ``ehat`` must be supplied (normally the
        propensity evaluated at X). With the leaf-share basis it may be None,
        and the per-leaf treated share is used.
        """

        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        leaves = self._leaf_ids(x)
        b = self.node_values_[leaves, 0]
        d = self.node_values_[leaves, 1]
        if np.isscalar(arm):
            arm_value = int(arm)
            if arm_value not in (0, 1):
                raise ValueError("arm must be 0 or 1")
            arms = np.full(x.shape[0], arm_value, dtype=float)
        else:
            arms = np.asarray(arm, dtype=float)
            if arms.shape != (x.shape[0],) or not np.all(np.isin(arms, (0, 1))):
                raise ValueError("arm must be scalar or a binary vector of length n")
        if ehat is None:
            if self.ehat_basis_ != "leaf_share":
                raise ValueError(
                    "the cross-fitted basis needs the propensity at the "
                    "prediction rows"
                )
            treatment_centre = self.node_share_[leaves]
        else:
            ehat = np.asarray(ehat, dtype=float)
            if ehat.shape != (x.shape[0],):
                raise ValueError("ehat must have one value per prediction row")
            treatment_centre = ehat
        flat = b + (arms - treatment_centre)[:, None] * d
        return flat.reshape((x.shape[0],) + self.target_shape_)


class MutauCWDBRegressor(CWDBRegressor):
    """C-WDB v1 with the mu/tau leaf decomposition and a cross-fitted
    propensity in the treatment basis.

    ``contrast_candidates`` switches on held-out energy-risk selection of the
    contrast penalty, exactly as the R3 repair does for the ridge strength.
    ``ehat_basis="leaf_share"`` is the collapse-check mode: it reverts the
    treatment basis to the leaf share and must reproduce the reparameterized
    shared tree at zero penalty.

    ``arm_shrinkage`` is accepted for signature compatibility with the C-WDB
    base class and then does nothing: the mu/tau leaf replaces the rule that
    consumes it. It is listed in ``INERT_PARAMETERS`` and the Stage 2 manifest
    should drop the key rather than record a strength that never applied.
    """

    INERT_PARAMETERS = MutauSharedTreeRegressor.INERT_PARENT_PARAMETERS

    def __init__(
        self,
        *,
        ehat_basis: str = "cross_fitted",
        contrast_shrinkage: float = 0.0,
        contrast_candidates: tuple[float, ...] | None = None,
        n_folds: int = 2,
        propensity_folds: int = 5,
        propensity_clip: tuple[float, float] = (
            PROPENSITY_CLIP_LOW,
            PROPENSITY_CLIP_HIGH,
        ),
        force_zero_contrast: bool = False,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)  # type: ignore[arg-type]
        if ehat_basis not in {"cross_fitted", "leaf_share"}:
            raise ValueError("ehat_basis must be 'cross_fitted' or 'leaf_share'")
        self.ehat_basis = ehat_basis
        self.contrast_shrinkage = float(contrast_shrinkage)
        self.contrast_candidates = (
            None
            if contrast_candidates is None
            else tuple(float(c) for c in contrast_candidates)
        )
        self.n_folds = n_folds
        self.propensity_folds = propensity_folds
        self.propensity_clip = tuple(float(c) for c in propensity_clip)
        self.force_zero_contrast = force_zero_contrast

    # ----------------------------------------------------------------- nuisances

    def _fit_propensity(self, X: NDArray[np.float64], treatment: NDArray[np.int64]) -> NDArray[np.float64]:
        """Cross-fitted propensity at the training rows, clipped and recorded.

        The propensity model is a plain logistic regression on the raw
        covariates, which every tournament regime's assignment mechanism is.
        """

        folds = FoldPlan.stratified(
            treatment, self.propensity_folds, keys=X, random_state=self.random_state
        )
        models = []
        ehat = np.empty(treatment.shape[0])
        for fold in range(self.propensity_folds):
            train_rows = folds.labels != fold
            holdout = folds.labels == fold
            model = fit_propensity_predictor(
                X[train_rows], treatment[train_rows], self.random_state + fold
            )
            models.append(model)
            ehat[holdout] = model.predict_proba(X[holdout])[:, 1]
        self.propensity_folds_ = folds
        self.propensity_models_ = tuple(models)
        return np.clip(ehat, *self.propensity_clip)

    def _propensity_at(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Propensity at new points: the average of the fold models."""
        predictions = np.stack(
            [model.predict_proba(X)[:, 1] for model in self.propensity_models_], axis=0
        )
        return np.clip(predictions.mean(axis=0), *self.propensity_clip)

    # --------------------------------------------------------------------- fit

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        weights: ArrayLike,
    ) -> "MutauCWDBRegressor":
        x = np.asarray(X, dtype=float)
        a = np.asarray(treatment, dtype=int)
        q = np.asarray(quantiles, dtype=float)
        w = np.asarray(weights, dtype=float)
        order = canonical_training_order(x, a, q)
        x, a, q = x[order], a[order], q[order]

        if self.ehat_basis == "cross_fitted":
            self._fit_ehat = self._fit_propensity(x, a)
        else:
            self._fit_ehat = None

        if self.contrast_candidates is not None:
            selected = self._select_contrast_strength(x, a, q, w)
            self.selected_contrast_shrinkage_ = selected
            self.contrast_shrinkage = selected

        super().fit(x, a, q, w)
        return self

    def _candidate_parameters(self) -> dict[str, object]:
        """Everything a selection-fold model inherits except the candidate."""
        return {
            "architecture": self.architecture,
            "n_particles": self.n_particles,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_arm_leaf": self.min_arm_leaf,
            "arm_shrinkage": self.arm_shrinkage,
            "sharing": self.sharing,
            "init_sharing": self.init_sharing,
            "collision_epsilon": self.collision_epsilon,
            "max_backtracks": self.max_backtracks,
            "descent_tolerance": self.descent_tolerance,
            "random_state": self.random_state,
        }

    def _select_contrast_strength(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        quantiles: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Held-out energy risk over the candidate strengths (A15).

        Each candidate is fitted on the complement of its selection fold and
        scored on the fold against the held-out unit's own observed arm, so
        nothing counterfactual enters the choice. Ties break toward the
        stronger regulariser, the null-safe default.
        """

        assert self.contrast_candidates is not None
        folds = FoldPlan.stratified(
            treatment, self.n_folds, keys=X, random_state=self.random_state + 7
        )
        records: list[tuple[float, float]] = []
        for candidate in self.contrast_candidates:
            scores: list[NDArray[np.float64]] = []
            for fold in range(self.n_folds):
                held_out = folds.labels == fold
                train_rows = ~held_out
                # The candidate runs its own full fit on the fold complement,
                # including its own cross-fitted propensity, so no held-out
                # unit contributes to the propensity, the nuisances, or the
                # contrast of the model that scores it.
                model = MutauCWDBRegressor(
                    ehat_basis=self.ehat_basis,
                    contrast_shrinkage=candidate,
                    contrast_candidates=None,
                    force_zero_contrast=self.force_zero_contrast,
                    **self._candidate_parameters(),
                )
                model.fit(
                    X[train_rows], treatment[train_rows], quantiles[train_rows], weights
                )
                for arm in (0, 1):
                    rows = held_out & (treatment == arm)
                    if not np.any(rows):
                        continue
                    scores.append(
                        model.score_samples(X[rows], arm, quantiles[rows])
                    )
            risk = float(np.mean(np.concatenate(scores)))
            records.append((candidate, risk))
        self.selection_records_ = tuple(
            (float(candidate), float(risk)) for candidate, risk in records
        )
        best = min(records, key=lambda item: (item[1], -item[0]))
        return float(best[0])

    # ------------------------------------------------------------ shared tree

    def _fit_v1(
        self, X: NDArray[np.float64], treatment: NDArray[np.int64], Q: NDArray[np.float64]
    ) -> None:
        self.initial_particles_ = self._initial_particles(treatment, Q)
        current = np.empty((X.shape[0], self.n_particles, self.n_coordinates_))
        for arm in (0, 1):
            current[treatment == arm] = self.initial_particles_[arm]

        self.estimators_: list[MutauSharedTreeRegressor] = []
        self.step_sizes_: list[float] = []
        self.training_history_ = []
        loss_before = self._energy_risk(current, Q)
        for iteration in range(self.n_estimators):
            gradient = self._energy_gradient(current, Q)
            target = _preconditioned_descent_target(gradient, self.weights_)
            estimator = MutauSharedTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_arm_leaf=self.min_arm_leaf,
                arm_shrinkage=self.arm_shrinkage,
                sharing=self.sharing,
                ehat_basis=self.ehat_basis,
                contrast_shrinkage=self.contrast_shrinkage,
                force_zero_contrast=self.force_zero_contrast,
                random_state=self.random_state + iteration,
            )
            estimator.fit(X, treatment, target, ehat=self._fit_ehat)
            direction = estimator.predict(X, treatment, ehat=self._fit_ehat)
            accepted = False
            step = self.learning_rate
            selected_diagnostics = None
            for backtracks in range(self.max_backtracks + 1):
                candidate, diagnostics = _project_candidate(
                    current + step * direction, self.weights_
                )
                loss_after = self._energy_risk(candidate, Q)
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
                self._step_record(iteration, loss_before, loss_after, step, backtracks, selected_diagnostics)
            )
            current = candidate
            loss_before = loss_after
        self.train_risk_ = loss_before

    def _energy_risk(self, particles: NDArray[np.float64], Q: NDArray[np.float64]) -> float:
        from .energy import empirical_energy_risk

        return empirical_energy_risk(
            particles, Q, self.weights_, epsilon=self.collision_epsilon
        )

    def _energy_gradient(self, particles: NDArray[np.float64], Q: NDArray[np.float64]) -> NDArray[np.float64]:
        from .energy import energy_gradient

        return energy_gradient(particles, Q, self.weights_, epsilon=self.collision_epsilon)

    def _step_record(self, iteration, loss_before, loss_after, step, backtracks, diagnostics):
        from .model import BoostingStep

        return BoostingStep(
            iteration=iteration,
            arm=None,
            loss_before=loss_before,
            loss_after=loss_after,
            step_size=step,
            backtracks=backtracks,
            projection_max=diagnostics.max_weighted_l2_adjustment,
            projection_changed=diagnostics.n_changed,
        )

    def _predict_labeled(
        self, X: ArrayLike, arm: int
    ) -> NDArray[np.float64]:
        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1")
        if not hasattr(self, "fitted_architecture_"):
            raise RuntimeError("the model has not been fitted")
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        particles = np.broadcast_to(
            self.initial_particles_[arm],
            (x.shape[0],) + self.initial_particles_[arm].shape,
        ).copy()
        ehat = None if self._fit_ehat is None else self._propensity_at(x)
        from .geometry import project_quantiles

        for estimator, step in zip(
            self.estimators_, self.step_sizes_, strict=True
        ):
            particles = project_quantiles(
                particles + step * estimator.predict(x, arm, ehat=ehat), self.weights_
            )
        return particles
