"""Prior-art incumbents for the WP9 benchmark tournament.

The first pilot compared ODCF only against internal ablations and homebrew
comparators, so the three closest published competitors named in WP9.2 and in
the conditional G0 decision (WP1-D014) had never been run.  This module adds
them.

Provenance and honesty scope
----------------------------
None of these are the authors' released code.  Each is a Python
reimplementation written against the published description recorded in
``research/prior_art_matrix.csv`` and adapted to the frozen finite target
vector ``U = (Q(p_1..p_K), Gini, Theil, Atkinson)``.  Every construction below
therefore carries a `PORT` label and must be reported as such.  A port that
loses to ODCF is weaker evidence than the authors' own implementation losing to
ODCF; a port that *beats* ODCF is decisive against ODCF, because a faithful
implementation would only be stronger.

1. ``causal_drf_port`` -- Naf, Park, and Susmann (2026), Causal-DRF.
   Shared causal-forest localization with a distributional (MMD) splitting
   criterion, followed by a *local* doubly robust moment solution at each
   evaluation point.  This is structurally different from ``odcf_mmd_score``,
   which averages globally constructed AIPW scores inside MMD-split leaves.

2. ``focal_dr_meta_learner`` -- Salmaso, Testa, and Chiaromonte (2026), FOCaL.
   A doubly robust functional CATE meta-learner: cross-fitted AIPW pseudo
   outcomes on the same unscaled augmented vector, then a final learner that
   respects the functional structure by regressing B-spline coefficients of the
   quantile curve rather than the K raw coordinates.  This is the composability
   adversary demanded by WP9.2 item 10.

3. ``wasserstein_random_forest`` -- Du, Biau, Petit, and Porcher (2021), WRF.
   Arm-specific honest forests whose splits maximize the squared 2-Wasserstein
   separation between children.  Under the one-dimensional Wasserstein isometry
   the squared W2 distance between two laws is the quadrature-weighted squared
   L2 distance between their quantile functions, so the WRF split criterion is
   exactly the quadrature-weighted between-child sum of squares on the curve
   coordinates, i.e. the ``curve_only`` split rule applied to observed outcome
   vectors rather than to causal scores.  Arm predictions are then differenced.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import BSpline
from sklearn.ensemble import RandomForestRegressor

from sim.baselines import (
    BaselineResult,
    _dr_inputs,
    _dr_scores,
    _observed_U,
)
from sim.config import QUANTILE_GRID
from sim.dgps import DGPResult, trapezoidal_weights
from wp3_odcf import ODCFEstimator

FOCAL_N_BASIS = 8
FOCAL_SPLINE_DEGREE = 3


def _arm_min_leaf(count: int) -> int:
    if count < 4:
        raise ValueError("an arm-specific forest requires at least four regions")
    return max(1, min(5, count // 4))


def fit_causal_drf_port(
    dgp: DGPResult, n_trees: int = 200, n_folds: int = 5, seed: int = 0,
) -> BaselineResult:
    """PORT of Causal-DRF: MMD-split shared localization + local DR moment.

    Residualize the observed augmented vector and the treatment against
    cross-fitted nuisances, relabel with the generalized-random-forest gradient
    pseudo-outcome, grow one shared forest with the kernel/MMD split rule on
    that pseudo-outcome, and solve the local doubly robust moment condition

        tau(x) = sum_i w_i(x) Ztilde_i Utilde_i / sum_i w_i(x) Ztilde_i^2

    with the forest weights w_i(x).  The forest is shared across all K+J
    coordinates, which is the feature of Causal-DRF that makes it the relevant
    adversary for the shared-partition claim.
    """
    U_obs, e, m0, m1 = _dr_inputs(dgp, n_folds, seed)
    e = np.clip(np.asarray(e, dtype=float), 0.02, 0.98)
    z = np.asarray(dgp.Z, dtype=float)

    marginal = e[:, None] * m1 + (1.0 - e)[:, None] * m0
    U_residual = U_obs - marginal
    z_residual = z - e
    denominator = float(np.mean(z_residual ** 2))
    if denominator <= 1e-10:
        raise ValueError("treatment residuals carry no variation for Causal-DRF")

    # GRF gradient relabeling, applied coordinatewise to the augmented vector.
    pseudo = (z_residual[:, None] * U_residual) / denominator

    forest = ODCFEstimator(
        K=dgp.K, J=dgp.J, variant="mmd_score",
        n_trees=n_trees, random_state=seed,
    ).fit(dgp.X, pseudo)

    numerator_terms = z_residual[:, None] * U_residual
    denominator_terms = z_residual ** 2
    pred = np.empty((len(dgp.X_eval), dgp.K + dgp.J), dtype=float)
    for index, x in enumerate(dgp.X_eval):
        w = forest.weights_at(x)
        local_denominator = float(np.dot(w, denominator_terms))
        if local_denominator <= 1e-8:
            # Degenerate local overlap: fall back to the pooled moment rather
            # than emitting a divergent prediction.
            local_denominator = denominator
            pred[index] = np.mean(numerator_terms, axis=0) / local_denominator
            continue
        pred[index] = (w @ numerator_terms) / local_denominator
    return BaselineResult(name="causal_drf_port", prediction=pred)


def focal_spline_basis(
    grid: np.ndarray = QUANTILE_GRID,
    n_basis: int = FOCAL_N_BASIS,
    degree: int = FOCAL_SPLINE_DEGREE,
) -> np.ndarray:
    """Clamped cubic B-spline design matrix on the quantile grid."""
    grid = np.asarray(grid, dtype=float)
    if n_basis <= degree:
        raise ValueError("n_basis must exceed the spline degree")
    if len(grid) < n_basis:
        raise ValueError("the quantile grid is too coarse for the requested basis")
    lo, hi = float(grid[0]), float(grid[-1])
    pad = 1e-9 * max(1.0, hi - lo)
    interior = np.linspace(lo, hi, n_basis - degree + 1)[1:-1]
    knots = np.r_[
        np.full(degree + 1, lo - pad),
        interior,
        np.full(degree + 1, hi + pad),
    ]
    basis = BSpline.design_matrix(grid, knots, degree, extrapolate=False).toarray()
    if basis.shape != (len(grid), n_basis):
        raise ValueError(
            f"spline design matrix has shape {basis.shape}, expected "
            f"{(len(grid), n_basis)}"
        )
    return basis


def fit_focal_dr_meta_learner(
    dgp: DGPResult, n_trees: int = 200, n_folds: int = 5, seed: int = 0,
    n_basis: int = FOCAL_N_BASIS,
) -> BaselineResult:
    """PORT of a FOCaL-style doubly robust functional CATE meta-learner.

    Stage one builds cross-fitted AIPW pseudo-outcomes on the same unscaled
    augmented vector every other method consumes.  Stage two is the functional
    part: the curve block of the pseudo-outcome is projected onto a clamped
    cubic B-spline basis in p under the trapezoidal L2(0,1) inner product, the
    basis coefficients are regressed on X with a multi-output forest, and the
    curve is reconstructed.  The J nonlinear functional coordinates are learned
    by separate scalar forests, as in a coordinatewise meta-learner.

    The point of this baseline is that it smooths across p by construction, so
    it isolates whether ODCF's curve advantage is anything more than smoothing.
    """
    U_obs, e, m0, m1 = _dr_inputs(dgp, n_folds, seed)
    scores = _dr_scores(U_obs, dgp.Z, e, m0, m1)
    K, J = dgp.K, dgp.J

    basis = focal_spline_basis(QUANTILE_GRID[:K], n_basis=n_basis)
    quadrature = trapezoidal_weights(K)
    root_w = np.sqrt(quadrature)
    # Weighted least-squares projection onto the spline basis.
    weighted_basis = basis * root_w[:, None]
    projector = np.linalg.pinv(weighted_basis)
    coefficients = scores[:, :K] @ (root_w[:, None] * projector.T)

    coefficient_forest = RandomForestRegressor(
        n_estimators=n_trees, min_samples_leaf=5, max_depth=8,
        random_state=seed, n_jobs=1,
    )
    coefficient_forest.fit(dgp.X, coefficients)
    predicted_coefficients = coefficient_forest.predict(dgp.X_eval)

    pred = np.empty((len(dgp.X_eval), K + J), dtype=float)
    pred[:, :K] = predicted_coefficients @ basis.T
    for j in range(J):
        scalar_forest = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=5, max_depth=8,
            random_state=seed + 1 + j, n_jobs=1,
        )
        scalar_forest.fit(dgp.X, scores[:, K + j])
        pred[:, K + j] = scalar_forest.predict(dgp.X_eval)
    return BaselineResult(name="focal_dr_meta_learner", prediction=pred)


def fit_wasserstein_random_forest(
    dgp: DGPResult, n_trees: int = 200, seed: int = 0,
) -> BaselineResult:
    """PORT of Du et al. (2021) Wasserstein Random Forests, differenced by arm.

    Each arm gets an honest forest whose split gain is the quadrature-weighted
    between-child sum of squares on the log1p quantile coordinates, which is the
    finite-grid form of the squared W2 separation used by WRF.  Leaf means of the
    full augmented vector give the arm-conditional prediction, and the two arms
    are differenced.  This is not a doubly robust estimator; WRF is a conditional
    law estimator, and differencing arms is the causal use Du et al. describe.
    """
    U_obs = _observed_U(dgp)
    z = np.asarray(dgp.Z, dtype=float)
    predictions = []
    for arm, arm_seed in ((0.0, seed), (1.0, seed + 1)):
        mask = z == arm
        model = ODCFEstimator(
            K=dgp.K, J=dgp.J, variant="curve_only",
            n_trees=n_trees, min_leaf=_arm_min_leaf(int(np.sum(mask))),
            max_depth=8, random_state=arm_seed,
        ).fit(dgp.X[mask], U_obs[mask])
        predictions.append(model.predict(dgp.X_eval))
    return BaselineResult(
        name="wasserstein_random_forest",
        prediction=predictions[1] - predictions[0],
    )


def run_incumbent(
    dgp: DGPResult, method: str,
    n_trees: int = 200, n_folds: int = 5, seed: int = 0,
) -> Optional[BaselineResult]:
    if method == "causal_drf_port":
        return fit_causal_drf_port(dgp, n_trees, n_folds, seed)
    if method == "focal_dr_meta_learner":
        return fit_focal_dr_meta_learner(dgp, n_trees, n_folds, seed)
    if method == "wasserstein_random_forest":
        return fit_wasserstein_random_forest(dgp, n_trees, seed)
    raise ValueError(f"unsupported prior-art incumbent: {method}")
