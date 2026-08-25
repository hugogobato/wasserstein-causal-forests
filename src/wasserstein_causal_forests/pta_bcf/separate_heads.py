"""PTA-S: one scalar stochtree BCF head per PTA target coordinate.

`stochtree.BCFModel` supports a multivariate treatment *basis*, not a
multivariate outcome. Each coordinate of U(Y) therefore gets its own scalar
model. All heads share the same folds, the same propensity input, and the same
tuning budget, so the only difference between heads is the response column.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

from .targets import FoldPlan, ScaleManifest, TargetManifest, make_folds

PTA_S_METHOD_ID = "PTA-S"


@dataclass(frozen=True)
class HeadBudget:
    """Tuning budget applied identically to every coordinate head."""

    num_trees_prognostic: int = 50
    num_trees_treatment: int = 20
    num_gfr: int = 10
    num_burnin: int = 100
    num_mcmc: int = 200

    def __post_init__(self) -> None:
        if min(self.num_trees_prognostic, self.num_trees_treatment) < 1:
            raise ValueError("each forest needs at least one tree")
        if self.num_mcmc < 1:
            raise ValueError("num_mcmc must be positive")
        if min(self.num_gfr, self.num_burnin) < 0:
            raise ValueError("warm-start and burn-in counts must be nonnegative")


def _validate_design(
    X: ArrayLike, treatment: ArrayLike | None = None
) -> NDArray[np.float64]:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must have shape (n, p) with positive dimensions")
    if not np.all(np.isfinite(x)):
        raise ValueError("X must be finite")
    if treatment is not None:
        a = np.asarray(treatment)
        if a.ndim != 1 or a.size != x.shape[0]:
            raise ValueError("treatment must be one-dimensional with n entries")
        if not np.isin(np.unique(a), (0, 1)).all():
            raise ValueError("treatment must be binary in {0, 1}")
    return x


class CrossFittedPropensity:
    """Out-of-fold propensity for training rows, full-fit scores for new rows."""

    def __init__(
        self,
        *,
        clip: float = 0.02,
        random_state: int = 0,
        max_iter: int = 200,
    ) -> None:
        if not 0.0 < clip < 0.5:
            raise ValueError("clip must lie in (0, 0.5)")
        self.clip = float(clip)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)

    def _new_estimator(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state,
        )

    def fit(
        self, X: ArrayLike, treatment: ArrayLike, folds: FoldPlan
    ) -> "CrossFittedPropensity":
        x = _validate_design(X, treatment)
        a = np.asarray(treatment, dtype=int)
        scores = np.empty(a.size, dtype=float)
        for fold in folds.fold_ids:
            train = folds.train_index(fold)
            evaluate = folds.test_index(fold)
            if evaluate.size == 0:
                continue
            if np.unique(a[train]).size < 2:
                scores[evaluate] = float(a[train].mean())
                continue
            estimator = self._new_estimator().fit(x[train], a[train])
            scores[evaluate] = estimator.predict_proba(x[evaluate])[:, 1]
        self.full_estimator_ = self._new_estimator().fit(x, a)
        self.train_scores_ = np.clip(scores, self.clip, 1.0 - self.clip)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        x = _validate_design(X)
        scores = self.full_estimator_.predict_proba(x)[:, 1]
        return np.clip(scores, self.clip, 1.0 - self.clip)


class PTASeparateHeads:
    """Independent scalar BCF heads on a common PTA target contract."""

    def __init__(
        self,
        manifest: TargetManifest,
        *,
        budget: HeadBudget = HeadBudget(),
        n_folds: int = 5,
        random_state: int = 0,
        propensity_clip: float = 0.02,
    ) -> None:
        self.manifest = manifest
        self.budget = budget
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.propensity_clip = float(propensity_clip)

    # -- fitting ---------------------------------------------------------

    def _head_seed(self, coordinate: int) -> int:
        # Distinct streams per head, deterministic in the model seed.
        return int(self.random_state) * 1000 + int(coordinate) + 1

    def _sample_head(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.float64],
        response: NDArray[np.float64],
        propensity: NDArray[np.float64],
        coordinate: int,
    ) -> Any:
        from stochtree import BCFModel

        model = BCFModel()
        model.sample(
            X_train=X,
            Z_train=treatment,
            y_train=response,
            propensity_train=propensity,
            num_gfr=self.budget.num_gfr,
            num_burnin=self.budget.num_burnin,
            num_mcmc=self.budget.num_mcmc,
            general_params={
                "random_seed": self._head_seed(coordinate),
                "keep_every": 1,
            },
            prognostic_forest_params={
                "num_trees": self.budget.num_trees_prognostic
            },
            treatment_effect_forest_params={
                "num_trees": self.budget.num_trees_treatment
            },
        )
        return model

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        *,
        folds: FoldPlan | None = None,
        propensity: ArrayLike | None = None,
    ) -> "PTASeparateHeads":
        """Fit one head per coordinate of U(Y) built from the grid vectors."""

        return self.fit_target_matrix(
            X,
            treatment,
            self.manifest.build(quantiles),
            folds=folds,
            propensity=propensity,
        )

    def fit_target_matrix(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        targets: ArrayLike,
        *,
        folds: FoldPlan | None = None,
        propensity: ArrayLike | None = None,
    ) -> "PTASeparateHeads":
        """Fit one head per column of a prebuilt target matrix.

        The diagnostic prototype reuses this entry point to regress
        cross-fitted residuals, which are not themselves grid vectors.
        """

        x = _validate_design(X, treatment)
        a = np.asarray(treatment, dtype=int)
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != self.manifest.dimension:
            raise ValueError(
                f"targets must have shape (n, {self.manifest.dimension})"
            )
        if targets.shape[0] != x.shape[0]:
            raise ValueError("X and targets must have the same number of rows")

        self.folds_ = folds or make_folds(
            x.shape[0], a, n_folds=self.n_folds, random_state=self.random_state
        )
        if propensity is None:
            self.propensity_model_ = CrossFittedPropensity(
                clip=self.propensity_clip, random_state=self.random_state
            ).fit(x, a, self.folds_)
            train_propensity = self.propensity_model_.train_scores_
        else:
            self.propensity_model_ = None
            train_propensity = np.clip(
                np.asarray(propensity, dtype=float),
                self.propensity_clip,
                1.0 - self.propensity_clip,
            )
            if train_propensity.shape != (x.shape[0],):
                raise ValueError("propensity must have one entry per row")

        # Scaling uses the fitting rows only; evaluation rows never contribute.
        self.scale_manifest_ = ScaleManifest.fit(targets, self.manifest)
        scaled = self.scale_manifest_.transform(targets)

        treatment_float = a.astype(float)
        self.train_propensity_ = train_propensity
        self.heads_ = [
            self._sample_head(
                x, treatment_float, scaled[:, coordinate], train_propensity, coordinate
            )
            for coordinate in range(self.manifest.dimension)
        ]
        self.n_features_ = int(x.shape[1])
        self.n_train_ = int(x.shape[0])
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "heads_"):
            raise RuntimeError("call fit before predicting")

    def _resolve_propensity(
        self, X: NDArray[np.float64], propensity: ArrayLike | None
    ) -> NDArray[np.float64]:
        if propensity is not None:
            scores = np.asarray(propensity, dtype=float)
            if scores.shape != (X.shape[0],):
                raise ValueError("propensity must have one entry per row")
            return np.clip(scores, self.propensity_clip, 1.0 - self.propensity_clip)
        if self.propensity_model_ is None:
            raise ValueError(
                "the model was fitted with supplied propensities, so prediction "
                "requires an explicit propensity argument"
            )
        return self.propensity_model_.predict(X)

    # -- prediction ------------------------------------------------------

    def predict_draws(
        self, X: ArrayLike, *, propensity: ArrayLike | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Return raw posterior draws on the original target scale.

        Shapes are (n, D, S). Nothing here is monotone-postprocessed; the
        quantile block holds raw posterior draws.
        """

        self._check_fitted()
        x = _validate_design(X)
        if x.shape[1] != self.n_features_:
            raise ValueError("X has a different number of columns than at fit time")
        scores = self._resolve_propensity(x, propensity)
        zeros = np.zeros(x.shape[0], dtype=float)

        contrasts = []
        controls = []
        for head in self.heads_:
            prediction = head.predict(x, zeros, scores)
            contrasts.append(prediction["cate"])
            controls.append(prediction["mu_hat"])
        contrast = np.stack(contrasts, axis=1)
        control = np.stack(controls, axis=1)

        scale = self.scale_manifest_.scale[None, :, None]
        center = self.scale_manifest_.center[None, :, None]
        control_original = control * scale + center
        contrast_original = contrast * scale
        return {
            "control": control_original,
            "contrast": contrast_original,
            "treated": control_original + contrast_original,
        }

    def predict_arm_draws(
        self,
        X: ArrayLike,
        arm: int,
        *,
        propensity: ArrayLike | None = None,
        project: bool = False,
    ) -> NDArray[np.float64]:
        """Raw (or monotone-projected) posterior draws of U under one arm."""

        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1")
        draws = self.predict_draws(X, propensity=propensity)
        arm_draws = draws["treated"] if arm == 1 else draws["control"]
        if not project:
            return arm_draws
        n_rows, dimension, n_draws = arm_draws.shape
        flat = np.moveaxis(arm_draws, 2, 1).reshape(n_rows * n_draws, dimension)
        projected = self.manifest.project_quantile_block(flat)
        return np.moveaxis(
            projected.reshape(n_rows, n_draws, dimension), 1, 2
        )

    def predict_arm_mean(
        self,
        X: ArrayLike,
        arm: int,
        *,
        propensity: ArrayLike | None = None,
        project: bool = False,
    ) -> NDArray[np.float64]:
        """Posterior mean of U under one arm, shape (n, D)."""

        return self.predict_arm_draws(
            X, arm, propensity=propensity, project=project
        ).mean(axis=2)

    def predict_contrast_draws(
        self, X: ArrayLike, *, propensity: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Posterior draws of the conditional target contrast, shape (n, D, S)."""

        return self.predict_draws(X, propensity=propensity)["contrast"]

    def predict_contrast(
        self, X: ArrayLike, *, propensity: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Posterior mean conditional target contrast, shape (n, D)."""

        return self.predict_contrast_draws(X, propensity=propensity).mean(axis=2)

    def predict_observed(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        *,
        propensity: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Posterior mean of U under each row's own treatment, shape (n, D)."""

        x = _validate_design(X, treatment)
        a = np.asarray(treatment, dtype=int)
        draws = self.predict_draws(x, propensity=propensity)
        control = draws["control"].mean(axis=2)
        contrast = draws["contrast"].mean(axis=2)
        return control + a[:, None] * contrast

    # -- serialization ---------------------------------------------------

    def to_json_string(self) -> str:
        """Serialize deterministically. stochtree models are not picklable."""

        self._check_fitted()
        payload = {
            "method_id": PTA_S_METHOD_ID,
            "manifest": self.manifest.to_dict(),
            "budget": asdict(self.budget),
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "propensity_clip": self.propensity_clip,
            "scale_manifest": self.scale_manifest_.to_dict(),
            "fold_assignment": self.folds_.assignment.tolist(),
            "train_propensity": self.train_propensity_.tolist(),
            "n_features": self.n_features_,
            "n_train": self.n_train_,
            "heads": [head.to_json() for head in self.heads_],
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json_string(
        cls, payload: str, *, propensity_model: CrossFittedPropensity | None = None
    ) -> "PTASeparateHeads":
        from stochtree import BCFModel

        data = json.loads(payload)
        model = cls(
            TargetManifest.from_dict(data["manifest"]),
            budget=HeadBudget(**data["budget"]),
            n_folds=int(data["n_folds"]),
            random_state=int(data["random_state"]),
            propensity_clip=float(data["propensity_clip"]),
        )
        model.scale_manifest_ = ScaleManifest.from_dict(data["scale_manifest"])
        model.folds_ = FoldPlan(
            assignment=np.asarray(data["fold_assignment"], dtype=np.int64),
            n_folds=int(data["n_folds"]),
            random_state=int(data["random_state"]),
        )
        model.train_propensity_ = np.asarray(data["train_propensity"], dtype=float)
        model.n_features_ = int(data["n_features"])
        model.n_train_ = int(data["n_train"])
        model.propensity_model_ = propensity_model
        heads = []
        for head_json in data["heads"]:
            head = BCFModel()
            head.from_json(head_json)
            heads.append(head)
        model.heads_ = heads
        return model
