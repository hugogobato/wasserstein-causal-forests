from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sim.config import (
    QUANTILE_GRID, FUNCTIONAL_GRID, L_FUNCTIONAL, PRIOR_ART_METHODS,
)
from sim.dgps import FW, trapezoidal_weights
from sim.dgps import DGPResult, u_matrix, u_vector, functional_vector_3
from wp3_odcf import ODCFEstimator


@dataclass
class BaselineResult:
    name: str
    prediction: np.ndarray


_LOGGER = None


def _log(msg: str) -> None:
    global _LOGGER
    if _LOGGER is None:
        import logging
        _LOGGER = logging.getLogger("sim.baselines")
    _LOGGER.warning(msg)


def _observed_U(dgp: DGPResult) -> np.ndarray:
    n = len(dgp.X)
    if dgp.observation_regime == "oracle_latent":
        z = dgp.Z
        q_obs = np.where(z[:, None] == 1, dgp.Q1_log, dgp.Q0_log)
        r_obs = np.where(z[:, None] == 1, dgp.Q1_raw_func, dgp.Q0_raw_func)
        return u_matrix(q_obs, r_obs, FW, FUNCTIONAL_GRID)
    from sim.dgps import empirical_u_vector
    return np.array([
        empirical_u_vector(samp, QUANTILE_GRID, FUNCTIONAL_GRID, FW)
        for samp in dgp.inner_samples
    ])


def _known_design_propensity(dgp: DGPResult) -> Optional[float]:
    """Return a known randomized propensity only for genuinely constant designs."""
    if len(dgp.true_propensity) and np.allclose(dgp.true_propensity, dgp.true_propensity[0]):
        return float(dgp.true_propensity[0])
    return None


def _cross_fit_nuisances(
    X: np.ndarray, Z: np.ndarray, U: np.ndarray,
    n_folds: int, seed: int, known_propensity: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    e, m0, m1 = np.empty(n), np.empty((n, U.shape[1])), np.empty((n, U.shape[1]))
    for fold in folds:
        train = np.setdiff1d(np.arange(n), fold)
        Xt, Zt, Ut = X[train], Z[train], U[train]
        Xv = X[fold]
        leaf = max(2, min(10, len(train) // 20))

        if known_propensity is None:
            clf = RandomForestClassifier(
                n_estimators=100, min_samples_leaf=leaf,
                random_state=seed, n_jobs=1,
            )
            clf.fit(Xt, Zt)
            e[fold] = np.clip(clf.predict_proba(Xv)[:, 1], 0.02, 0.98)
        else:
            if not 0 < known_propensity < 1:
                raise ValueError("known propensity must lie strictly inside (0, 1)")
            e[fold] = known_propensity

        m0_m = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=leaf, random_state=seed + 1, n_jobs=1)
        m1_m = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=leaf, random_state=seed + 2, n_jobs=1)
        m0_m.fit(Xt[Zt == 0], Ut[Zt == 0])
        m1_m.fit(Xt[Zt == 1], Ut[Zt == 1])
        m0[fold], m1[fold] = m0_m.predict(Xv), m1_m.predict(Xv)
    return e, m0, m1


_DR_INPUT_CACHE_ATTR = "_wp9_dr_input_cache"


def _dr_inputs(
    dgp: DGPResult, n_folds: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return observed U and nuisances, cached per simulation cell.

    Cross-fitting is deterministic given ``(n_folds, seed)``, but a dozen
    methods in one cell each used to refit the same 5-fold, 52-output nuisance
    forests.  Caching on the DGP instance is arithmetically identical and is
    the single largest cost saving in the feasible regime.
    """
    cache = getattr(dgp, _DR_INPUT_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(dgp, _DR_INPUT_CACHE_ATTR, cache)
    key = (int(n_folds), int(seed))
    if key in cache:
        U_obs, e, m0, m1 = cache[key]
        return U_obs.copy(), e.copy(), m0.copy(), m1.copy()

    U_obs = _observed_U(dgp)
    known = _known_design_propensity(dgp)
    if dgp.observation_regime == "oracle_latent":
        e, m0, m1 = _oracle_nuisances(dgp)
    else:
        e, m0, m1 = _cross_fit_nuisances(
            dgp.X, dgp.Z, U_obs, n_folds, seed, known_propensity=known
        )
    e, m0, m1 = np.asarray(e), np.asarray(m0), np.asarray(m1)
    cache[key] = (U_obs, e, m0, m1)
    return U_obs.copy(), e.copy(), m0.copy(), m1.copy()


def _oracle_nuisances(dgp: DGPResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return dgp.true_propensity, dgp.true_m0, dgp.true_m1


def _dr_scores(U: np.ndarray, Z: np.ndarray, e: np.ndarray,
               m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
    Z2 = Z[:, None]
    e2 = e[:, None]
    return m1 - m0 + (Z2 / e2) * (U - m1) - ((1.0 - Z2) / (1.0 - e2)) * (U - m0)


def fit_multi_output_dr_forest(
    dgp: DGPResult, n_trees: int = 200, n_folds: int = 5, seed: int = 0,
) -> BaselineResult:
    U_obs, e, m0, m1 = _dr_inputs(dgp, n_folds, seed)
    scores = _dr_scores(U_obs, dgp.Z, e, m0, m1)

    rf = RandomForestRegressor(
        n_estimators=n_trees, min_samples_leaf=5, max_depth=8,
        random_state=seed, n_jobs=1)
    rf.fit(dgp.X, scores)
    pred = rf.predict(dgp.X_eval)
    return BaselineResult(name="multi_output_dr_forest", prediction=pred)


def fit_pointwise_causal_forests(
    dgp: DGPResult, n_trees: int = 200, n_folds: int = 5, seed: int = 0,
) -> BaselineResult:
    U_obs, e, m0, m1 = _dr_inputs(dgp, n_folds, seed)
    scores = _dr_scores(U_obs, dgp.Z, e, m0, m1)

    K, J = dgp.K, dgp.J
    pred = np.full((len(dgp.X_eval), K + J), np.nan)
    for coord in range(K):
        rf = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=5, max_depth=8,
            random_state=seed + coord, n_jobs=1)
        rf.fit(dgp.X, scores[:, coord])
        pred[:, coord] = rf.predict(dgp.X_eval)
    return BaselineResult(name="pointwise_causal_forest", prediction=pred)


def fit_scalar_causal_forest(
    dgp: DGPResult, n_trees: int = 200, n_folds: int = 5, seed: int = 0,
) -> BaselineResult:
    U_obs, e, m0, m1 = _dr_inputs(dgp, n_folds, seed)
    scores = _dr_scores(U_obs, dgp.Z, e, m0, m1)

    K, J = dgp.K, dgp.J
    pred = np.full((len(dgp.X_eval), K + J), np.nan)
    for j in range(J):
        coord = K + j
        rf = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=5, max_depth=8,
            random_state=seed + j, n_jobs=1)
        rf.fit(dgp.X, scores[:, coord])
        pred[:, coord] = rf.predict(dgp.X_eval)
    return BaselineResult(name="scalar_causal_forest", prediction=pred)


def fit_two_arm_frechet_forest(
    dgp: DGPResult, n_trees: int = 200, seed: int = 0,
) -> BaselineResult:
    """Two-arm weighted direct-sum forest, not a causal DR forest."""
    U_obs = _observed_U(dgp)
    z = dgp.Z
    scale = np.r_[np.sqrt(trapezoidal_weights(dgp.K)), np.ones(dgp.J)]
    scaled = U_obs * scale[None, :]
    rf0 = RandomForestRegressor(
        n_estimators=n_trees, min_samples_leaf=5, max_depth=8,
        random_state=seed, n_jobs=1)
    rf1 = RandomForestRegressor(
        n_estimators=n_trees, min_samples_leaf=5, max_depth=8,
        random_state=seed + 1, n_jobs=1)
    rf0.fit(dgp.X[z == 0], scaled[z == 0])
    rf1.fit(dgp.X[z == 1], scaled[z == 1])
    pred = (rf1.predict(dgp.X_eval) - rf0.predict(dgp.X_eval)) / scale[None, :]
    return BaselineResult(name="two_arm_frechet_forest", prediction=pred)


def fit_drf_inspired_arm_mmd(
    dgp: DGPResult, n_trees: int = 200, seed: int = 0,
) -> BaselineResult:
    """Arm-specific MMD forest comparator inspired by DRF.

    This is deliberately named as a comparator.  It is not the official DRF
    implementation and makes no claim to reproduce its split or inference
    theory.
    """
    U_obs = _observed_U(dgp)
    z = dgp.Z
    models = []
    arm_sizes = [int(np.sum(z == arm)) for arm in (0, 1)]
    if min(arm_sizes) < 4:
        raise ValueError("MMD arm forest requires at least four regions per arm")
    arm_min_leaf = max(1, min(5, min(arm_sizes) // 4))
    for arm, arm_seed in ((0, seed), (1, seed + 1)):
        model = ODCFEstimator(
            K=dgp.K,
            J=dgp.J,
            variant="mmd_score",
            n_trees=n_trees,
            min_leaf=arm_min_leaf,
            max_depth=8,
            random_state=arm_seed,
        ).fit(dgp.X[z == arm], U_obs[z == arm])
        models.append(model)
    pred = models[1].predict(dgp.X_eval) - models[0].predict(dgp.X_eval)
    return BaselineResult(name="drf_inspired_arm_mmd", prediction=pred)


def fit_global_dr_estimator(
    dgp: DGPResult, n_trees: int = 200, seed: int = 0,
) -> BaselineResult:
    """Global/subgroup-DR baseline before any heterogeneous localization."""
    U_obs, e, m0, m1 = _dr_inputs(dgp, n_folds=5, seed=seed)
    scores = _dr_scores(U_obs, dgp.Z, e, m0, m1)
    average = np.mean(scores, axis=0)
    pred = np.tile(average, (len(dgp.X_eval), 1))
    return BaselineResult(name="global_dr_estimator", prediction=pred)


def run_baseline(
    dgp: DGPResult, method: str,
    n_trees: int = 200, n_folds: int = 5, seed: int = 0,
) -> Optional[BaselineResult]:
    try:
        if method == "multi_output_dr_forest":
            return fit_multi_output_dr_forest(dgp, n_trees, n_folds, seed)
        elif method == "pointwise_causal_forest":
            return fit_pointwise_causal_forests(dgp, n_trees, n_folds, seed)
        elif method == "scalar_causal_forest":
            return fit_scalar_causal_forest(dgp, n_trees, n_folds, seed)
        elif method == "two_arm_frechet_forest":
            return fit_two_arm_frechet_forest(dgp, n_trees, seed)
        elif method == "drf_inspired_arm_mmd":
            return fit_drf_inspired_arm_mmd(dgp, n_trees, seed)
        elif method == "global_dr_estimator":
            return fit_global_dr_estimator(dgp, n_trees, seed)
        elif method in PRIOR_ART_METHODS:
            # Imported lazily: sim.incumbents imports from this module.
            from sim.incumbents import run_incumbent
            return run_incumbent(dgp, method, n_trees, n_folds, seed)
        else:
            raise ValueError(f"unsupported simulation baseline: {method}")
    except Exception as exc:
        _log(f"baseline {method} failed: {exc}")
        raise
