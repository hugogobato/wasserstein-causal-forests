"""Contrast shrinkage chosen on held-out energy risk rather than frozen.

The G3 memo's first repair item asks for a contrast-level regulariser "with
strength selected by cross-fitting rather than frozen". This module supplies the
selector. It is deliberately thin: the regulariser itself lives in
`ArmSharedTreeRegressor`, and all that is added here is the honest choice of its
strength.

Why held-out energy risk identifies the right strength. The score is strictly
proper for the arm law that each held-out unit's *observed* arm realises, so no
counterfactual is needed. Over-shrinking the contrast when the arms genuinely
differ pushes both arm laws toward a pooled law that fits neither, and the risk
rises; under-shrinking when they do not differ fits sampling noise in the arm
gap, and the risk also rises. The minimiser is therefore informative about the
contrast rather than about the pooled fit, which is fixed by construction in the
reparameterised leaf.

Assumption A15 requires tuning to respect sample splitting. Every candidate is
scored only on folds excluded from the fit that produced it, and the refit on
the full sample uses a strength chosen without seeing any unit's own held-out
score more than once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import CWDBRegressor

#: Strengths scanned by default, spanning "no contrast regularisation" to
#: "contrast almost entirely pooled away" in roughly one order of magnitude per
#: step. A leaf holds tens of observations per arm, so `n_eff / (n_eff + lambda)`
#: moves across most of [0, 1] over this range.
DEFAULT_CONTRAST_CANDIDATES = (0.0, 5.0, 50.0, 500.0)


@dataclass(frozen=True)
class SelectionRecord:
    """Held-out risk for one candidate strength."""

    contrast_shrinkage: float
    held_out_risk: float
    n_scored: int


def stratified_folds(
    treatment: NDArray[np.int64], n_folds: int, random_state: int
) -> NDArray[np.int64]:
    """Assign fold labels, balancing both arms across folds.

    Round-robin over an arm-wise permutation, so every fold holds close to the
    same number of treated and control rows and no fold's complement can lose an
    arm entirely.
    """

    generator = np.random.default_rng(random_state)
    labels = np.empty(treatment.shape[0], dtype=np.int64)
    for arm in (0, 1):
        rows = np.flatnonzero(treatment == arm)
        labels[generator.permutation(rows)] = np.arange(rows.size) % n_folds
    return labels


class CrossFittedCWDBRegressor(CWDBRegressor):
    """C-WDB whose contrast shrinkage is selected out of sample.

    Fits `n_folds * len(contrast_candidates)` models to score the candidates and
    one final model on the whole sample at the winner, so the cost is roughly
    `1 + n_folds * len(candidates) * (n_folds - 1) / n_folds` ordinary fits.
    """

    def __init__(
        self,
        *,
        contrast_candidates: tuple[float, ...] = DEFAULT_CONTRAST_CANDIDATES,
        n_folds: int = 3,
        **parameters: object,
    ) -> None:
        parameters.setdefault("contrast_rule", "ridge")
        super().__init__(**parameters)  # type: ignore[arg-type]
        if not contrast_candidates:
            raise ValueError("at least one candidate strength is required")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self.contrast_candidates = tuple(float(c) for c in contrast_candidates)
        self.n_folds = n_folds

    def _candidate_parameters(self, contrast_shrinkage: float) -> dict[str, object]:
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
            "contrast_rule": self.contrast_rule,
            "contrast_shrinkage": contrast_shrinkage,
            "contrast_threshold_scale": self.contrast_threshold_scale,
            "contrast_damping": self.contrast_damping,
            "collision_epsilon": self.collision_epsilon,
            "max_backtracks": self.max_backtracks,
            "descent_tolerance": self.descent_tolerance,
            "random_state": self.random_state,
        }

    def _held_out_risk(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        quantiles: NDArray[np.float64],
        weights: NDArray[np.float64],
        folds: NDArray[np.int64],
        contrast_shrinkage: float,
    ) -> tuple[float, int]:
        scores: list[NDArray[np.float64]] = []
        for fold in range(self.n_folds):
            held_out = folds == fold
            if not np.any(held_out):
                continue
            model = CWDBRegressor(**self._candidate_parameters(contrast_shrinkage))
            model.fit(
                X[~held_out],
                treatment[~held_out],
                quantiles[~held_out],
                weights,
            )
            for arm in (0, 1):
                rows = held_out & (treatment == arm)
                if not np.any(rows):
                    continue
                # Each held-out unit is scored only against its own observed
                # arm, so nothing counterfactual enters the selection.
                scores.append(model.score_samples(X[rows], arm, quantiles[rows]))
        if not scores:
            raise RuntimeError("no held-out rows were scored")
        stacked = np.concatenate(scores)
        return float(np.mean(stacked)), int(stacked.size)

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        quantiles: ArrayLike,
        weights: ArrayLike,
    ) -> "CrossFittedCWDBRegressor":
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
        # Ties break toward the stronger regulariser: when the data cannot
        # distinguish two strengths, the null-safe one is the right default for
        # a method whose failure mode is manufacturing effects.
        best = min(records, key=lambda record: (record.held_out_risk, -record.contrast_shrinkage))
        self.selected_contrast_shrinkage_ = best.contrast_shrinkage
        self.contrast_shrinkage = best.contrast_shrinkage
        super().fit(X, treatment, quantiles, weights)
        return self
