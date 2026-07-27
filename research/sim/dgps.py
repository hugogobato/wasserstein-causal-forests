from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import stats as _stats

from sim.config import (
    DEFAULT_K,
    DEFAULT_J,
    FUNCTIONAL_GRID,
    L_FUNCTIONAL,
    QUANTILE_GRID,
)


norm_ppf = _stats.norm.ppf
FW = np.full(L_FUNCTIONAL, 1.0 / L_FUNCTIONAL)


def _inner_label(inner_samples, regime):
    if inner_samples is None or regime == "oracle_latent":
        return "oracle"
    ms = np.array([len(s) for s in inner_samples])
    if np.std(ms) / max(np.mean(ms), 1) > 0.3:
        return "heterogeneous"
    return f"m{int(np.mean(ms))}"


def trapezoidal_weights(k: int) -> np.ndarray:
    if k == 1:
        return np.ones(1, dtype=float)
    step = 0.9 / (k - 1)
    w = np.full(k, step, dtype=float)
    w[[0, -1]] = step / 2.0
    return w


def log1p_quantile(
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.log1p(np.exp(mu[..., None] + sigma[..., None] * norm_ppf(p)))


def raw_quantile(
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    return np.exp(
        np.asarray(mu, dtype=float)[..., None]
        + np.asarray(sigma, dtype=float)[..., None] * norm_ppf(p)
    )


def gini_from_quantiles(raw_q: np.ndarray, w: np.ndarray, p: np.ndarray) -> float:
    m = float(np.dot(w, raw_q))
    if m <= 1e-15:
        raise ValueError("Gini is undefined for a nonpositive mean")
    return float(1.0 - 2.0 * np.dot(w, (1.0 - p) * raw_q) / m)


def theil_from_quantiles(raw_q: np.ndarray, w: np.ndarray) -> float:
    m = float(np.dot(w, raw_q))
    if m <= 1e-15:
        raise ValueError("Theil is undefined for a nonpositive mean")
    r = raw_q / m
    return float(np.dot(w, np.where(r > 0, r * np.log(r), 0.0)))


def atkinson_from_quantiles(raw_q: np.ndarray, w: np.ndarray, eps: float = 0.5) -> float:
    m = float(np.dot(w, raw_q))
    if m <= 1e-15:
        raise ValueError("Atkinson is undefined for a nonpositive mean")
    if eps == 1.0:
        gm = float(np.exp(np.dot(w, np.log(np.maximum(raw_q, 1e-300)))))
        return 1.0 - gm / m
    beta = 1.0 - eps
    return float(1.0 - (np.dot(w, raw_q ** beta) / m) ** (1.0 / beta))


def functional_vector_3(raw_q: np.ndarray, w: np.ndarray, p: np.ndarray) -> np.ndarray:
    raw_q = np.asarray(raw_q, dtype=float)
    if raw_q.ndim != 1:
        raise ValueError(f"functional_vector_3 expects 1-D array, got shape {raw_q.shape}")
    return np.array([
        gini_from_quantiles(raw_q, w, p),
        theil_from_quantiles(raw_q, w),
        atkinson_from_quantiles(raw_q, w),
    ])


def u_vector(q_log: np.ndarray, q_raw: np.ndarray, w: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.r_[q_log, functional_vector_3(q_raw, w, p)]


def u_matrix(
    Q_log: np.ndarray,
    Q_raw: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    n = Q_log.shape[0]
    out = np.empty((n, Q_log.shape[1] + 3), dtype=float)
    out[:, :Q_log.shape[1]] = Q_log
    for i in range(n):
        out[i, Q_log.shape[1]:] = functional_vector_3(Q_raw[i], w, p)
    return out


def sample_inner_from_quantile(raw_quantile_func: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=m)
    idx = np.clip((u * L_FUNCTIONAL).astype(int), 0, L_FUNCTIONAL - 1)
    return raw_quantile_func[idx]


def empirical_u_vector(sample: np.ndarray, grid: np.ndarray, f_grid: np.ndarray, w: np.ndarray) -> np.ndarray:
    q_log = np.log1p(np.quantile(sample, grid, method="inverted_cdf"))
    q_raw = np.quantile(sample, f_grid, method="inverted_cdf")
    return u_vector(q_log, q_raw, w, f_grid)


def _draw_inner_sizes(
    rng: np.random.Generator,
    n: int,
    regime: str,
    X: Optional[np.ndarray] = None,
    d8: bool = False,
) -> np.ndarray:
    """Draw the declared inner-sample design for a simulation branch."""
    if regime == "oracle_latent":
        raise ValueError("oracle_latent has no finite inner-sample sizes")
    if regime == "identified_measurement_model":
        raise ValueError(
            "identified_measurement_model requires an explicit measurement model"
        )
    if regime == "feasible_growing_inner":
        m_min = max(25, int(np.sqrt(n) * 2))
        cap = 8 if d8 else 4
        return np.clip(rng.poisson(m_min, size=n), m_min, m_min * cap).astype(int)
    if regime == "empirical_proxy":
        upper = 1000 if d8 else 500
        raw = rng.integers(25, upper + 1, size=n).astype(float)
        if d8:
            if X is None:
                raise ValueError("D8 proxy sizes require covariates")
            raw *= 1.0 + 0.5 * X[:, 0]
            raw = np.clip(raw, 25, 2000)
        return raw.astype(int)
    raise ValueError(f"unknown observation regime: {regime}")


def _proxy_truth_deterministic(
    raw0_eval: np.ndarray,
    raw1_eval: np.ndarray,
    X_eval: np.ndarray,
    regime: str,
    seed: int,
    d8: bool = False,
    n_replicates: int = 8,
) -> Optional[np.ndarray]:
    """Monte Carlo truth for the empirical finite-inner-sample estimand.

    The latent and proxy targets are intentionally separate.  This target is
    frozen by ``seed`` and the declared inner-sample design, so it is suitable
    for comparing methods within one simulation cell.  It is not used for the
    latent-target regimes.
    """
    if regime == "oracle_latent":
        return None
    if regime == "identified_measurement_model":
        raise ValueError(
            "identified_measurement_model has no implemented truth generator"
        )
    raw0_eval = np.asarray(raw0_eval, dtype=float)
    raw1_eval = np.asarray(raw1_eval, dtype=float)
    if raw0_eval.shape != raw1_eval.shape:
        raise ValueError("proxy truth arm quantile arrays have different shapes")
    rng = np.random.default_rng(seed)
    output_dim = len(QUANTILE_GRID) + 3
    mean0 = np.zeros((raw0_eval.shape[0], output_dim), dtype=float)
    mean1 = np.zeros((raw1_eval.shape[0], output_dim), dtype=float)
    for _ in range(n_replicates):
        sizes = _draw_inner_sizes(rng, len(X_eval), regime, X_eval, d8=d8)
        for i, m in enumerate(sizes):
            sample0 = sample_inner_from_quantile(raw0_eval[i], int(m), rng)
            sample1 = sample_inner_from_quantile(raw1_eval[i], int(m), rng)
            mean0[i] += empirical_u_vector(
                sample0, QUANTILE_GRID, FUNCTIONAL_GRID, FW
            )
            mean1[i] += empirical_u_vector(
                sample1, QUANTILE_GRID, FUNCTIONAL_GRID, FW
            )
    mean0 /= n_replicates
    mean1 /= n_replicates
    return mean1 - mean0


def _proxy_truth_components(
    raw0_eval: np.ndarray,
    raw1_components_eval: np.ndarray,
    X_eval: np.ndarray,
    regime: str,
    seed: int,
    n_replicates: int = 8,
) -> Optional[np.ndarray]:
    """Proxy truth for D4's random treated latent distribution."""
    if regime == "oracle_latent":
        return None
    if regime == "identified_measurement_model":
        raise ValueError(
            "identified_measurement_model has no implemented truth generator"
        )
    raw0_eval = np.asarray(raw0_eval, dtype=float)
    components = np.asarray(raw1_components_eval, dtype=float)
    if components.ndim != 3 or components.shape[1] != 2:
        raise ValueError("D4 proxy components must have shape (n_eval, 2, L)")
    rng = np.random.default_rng(seed)
    output_dim = len(QUANTILE_GRID) + 3
    mean0 = np.zeros((raw0_eval.shape[0], output_dim), dtype=float)
    mean1 = np.zeros((raw0_eval.shape[0], output_dim), dtype=float)
    for _ in range(n_replicates):
        sizes = _draw_inner_sizes(rng, len(X_eval), regime, X_eval)
        for i, m in enumerate(sizes):
            sample0 = sample_inner_from_quantile(raw0_eval[i], int(m), rng)
            state = int(rng.integers(0, 2))
            sample1 = sample_inner_from_quantile(components[i, state], int(m), rng)
            mean0[i] += empirical_u_vector(
                sample0, QUANTILE_GRID, FUNCTIONAL_GRID, FW
            )
            mean1[i] += empirical_u_vector(
                sample1, QUANTILE_GRID, FUNCTIONAL_GRID, FW
            )
    mean0 /= n_replicates
    mean1 /= n_replicates
    return mean1 - mean0


def _assert_monotone_quantiles(values: np.ndarray, name: str) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or np.any(np.diff(values, axis=1) < -1e-12):
        raise ValueError(f"{name} contains a nonmonotone quantile curve")


@dataclass
class DGPResult:
    name: str
    seed: int
    observation_regime: str
    n_regions: int
    K: int
    J: int
    X: np.ndarray
    Z: np.ndarray
    true_propensity: np.ndarray
    Q0_log: np.ndarray
    Q1_log: np.ndarray
    Q0_raw_func: np.ndarray
    Q1_raw_func: np.ndarray
    true_m0: np.ndarray
    true_m1: np.ndarray
    X_eval: np.ndarray
    true_theta_eval: np.ndarray
    inner_samples: Optional[list[np.ndarray]] = None
    inner_n_label: str = "oracle"
    params: dict = field(default_factory=dict)
    proxy_theta_eval: Optional[np.ndarray] = None


@dataclass
class D0Null:
    K: int = DEFAULT_K
    J: int = DEFAULT_J
    d: int = 5
    n_eval: int = 200

    def sample(self, n: int, seed: int, regime: str = "oracle_latent") -> DGPResult:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, self.d))
        logit_e = X[:, 0] - 0.5 * X[:, 1]
        propensity = 1.0 / (1.0 + np.exp(-np.clip(logit_e, -10, 10)))
        Z = (rng.uniform(size=n) < propensity).astype(float)
        mu = 7.0 + 0.3 * X[:, 0]
        sigma = np.clip(0.8 + 0.1 * X[:, 1], 0.3, 1.5)

        Q0_log = log1p_quantile(mu, sigma, QUANTILE_GRID)
        Q1_log = Q0_log.copy()
        Q0_raw_func = raw_quantile(mu, sigma, FUNCTIONAL_GRID)
        Q1_raw_func = Q0_raw_func.copy()

        u0 = u_matrix(Q0_log, Q0_raw_func, FW, FUNCTIONAL_GRID)
        u1 = u0.copy()

        X_eval = rng.normal(0, 1, (self.n_eval, self.d))
        me = 7.0 + 0.3 * X_eval[:, 0]
        se = np.clip(0.8 + 0.1 * X_eval[:, 1], 0.3, 1.5)
        qe = log1p_quantile(me, se, QUANTILE_GRID)
        re = raw_quantile(me, se, FUNCTIONAL_GRID)
        ue = u_matrix(qe, re, FW, FUNCTIONAL_GRID)

        inner = self._make_inner(rng, n, regime, Q0_raw_func, Q1_raw_func, Z)
        proxy_theta_eval = _proxy_truth_deterministic(
            re, re, X_eval, regime, seed + 91001
        )

        inner_label = _inner_label(inner, regime)

        return DGPResult(
            name="D0", seed=seed, observation_regime=regime,
            n_regions=n, K=self.K, J=self.J,
            X=X, Z=Z, true_propensity=propensity,
            Q0_log=Q0_log, Q1_log=Q1_log,
            Q0_raw_func=Q0_raw_func, Q1_raw_func=Q1_raw_func,
            true_m0=u0, true_m1=u1,
            X_eval=X_eval, true_theta_eval=np.zeros_like(ue),
            inner_samples=inner, inner_n_label=inner_label, params={},
            proxy_theta_eval=proxy_theta_eval,
        )

    def _make_inner(self, rng, n, regime, raw0, raw1, Z):
        if regime == "oracle_latent":
            return None
        ms = _draw_inner_sizes(rng, n, regime)
        return [sample_inner_from_quantile(raw0[i], int(ms[i]), rng) for i in range(n)]


@dataclass
class D1LocationShift:
    K: int = DEFAULT_K
    J: int = DEFAULT_J
    d: int = 5
    n_eval: int = 200

    def sample(self, n: int, seed: int, regime: str = "oracle_latent") -> DGPResult:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, self.d))
        prop = 1.0 / (1.0 + np.exp(-np.clip(0.3 * X[:, 0] - 0.2 * X[:, 1], -10, 10)))
        Z = (rng.uniform(size=n) < prop).astype(float)
        mu = 7.0 + 0.3 * X[:, 0]
        sigma = np.clip(0.7 + 0.1 * X[:, 1], 0.3, 1.5)
        tau = 0.5 * np.tanh(0.5 * X[:, 0])
        mu1 = mu + tau

        Q0_log = log1p_quantile(mu, sigma, QUANTILE_GRID)
        Q1_log = log1p_quantile(mu1, sigma, QUANTILE_GRID)
        R0 = raw_quantile(mu, sigma, FUNCTIONAL_GRID)
        R1 = raw_quantile(mu1, sigma, FUNCTIONAL_GRID)

        u0 = u_matrix(Q0_log, R0, FW, FUNCTIONAL_GRID)
        u1 = u_matrix(Q1_log, R1, FW, FUNCTIONAL_GRID)

        Xe = rng.normal(0, 1, (self.n_eval, self.d))
        me = 7.0 + 0.3 * Xe[:, 0]
        se = np.clip(0.7 + 0.1 * Xe[:, 1], 0.3, 1.5)
        te = 0.5 * np.tanh(0.5 * Xe[:, 0])
        me1 = me + te
        q0e = log1p_quantile(me, se, QUANTILE_GRID)
        q1e = log1p_quantile(me1, se, QUANTILE_GRID)
        r0e = raw_quantile(me, se, FUNCTIONAL_GRID)
        r1e = raw_quantile(me1, se, FUNCTIONAL_GRID)
        u0e = u_matrix(q0e, r0e, FW, FUNCTIONAL_GRID)
        u1e = u_matrix(q1e, r1e, FW, FUNCTIONAL_GRID)

        inner = self._make_inner(rng, n, regime, R0, R1, Z) if regime != "oracle_latent" else None
        proxy_theta_eval = _proxy_truth_deterministic(
            r0e, r1e, Xe, regime, seed + 91002
        )

        return DGPResult(
            name="D1", seed=seed, observation_regime=regime,
            n_regions=n, K=self.K, J=self.J,
            X=X, Z=Z, true_propensity=prop,
            Q0_log=Q0_log, Q1_log=Q1_log,
            Q0_raw_func=R0, Q1_raw_func=R1,
            true_m0=u0, true_m1=u1,
            X_eval=Xe, true_theta_eval=u1e - u0e,
            inner_samples=inner, inner_n_label=_inner_label(inner, regime), params={},
            proxy_theta_eval=proxy_theta_eval,
        )

    def _make_inner(self, rng, n, regime, raw0, raw1, Z):
        if regime.startswith("feasible"):
            ms = _draw_inner_sizes(rng, n, regime)
        else:
            ms = _draw_inner_sizes(rng, n, regime)
        return [sample_inner_from_quantile(raw1[i] if Z[i] > 0.5 else raw0[i], int(ms[i]), rng)
                for i in range(n)]


@dataclass
class D2ScaleCrossing:
    K: int = DEFAULT_K; J: int = DEFAULT_J; d: int = 5; n_eval: int = 200

    def sample(self, n: int, seed: int, regime: str = "oracle_latent") -> DGPResult:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, self.d))
        prop = 1.0 / (1.0 + np.exp(-np.clip(X[:, 0] - 0.3 * X[:, 1], -10, 10)))
        Z = (rng.uniform(size=n) < prop).astype(float)
        mu = 7.0 + 0.2 * X[:, 0]
        s0, s1 = np.clip(0.6 + 0.1 * X[:, 1], 0.3, 1.5), np.clip(0.9 + 0.1 * X[:, 1], 0.3, 1.5)

        Q0_log = log1p_quantile(mu, s0, QUANTILE_GRID)
        Q1_log = log1p_quantile(mu, s1, QUANTILE_GRID)
        R0, R1 = raw_quantile(mu, s0, FUNCTIONAL_GRID), raw_quantile(mu, s1, FUNCTIONAL_GRID)
        u0, u1 = u_matrix(Q0_log, R0, FW, FUNCTIONAL_GRID), u_matrix(Q1_log, R1, FW, FUNCTIONAL_GRID)

        Xe = rng.normal(0, 1, (self.n_eval, self.d))
        me = 7.0 + 0.2 * Xe[:, 0]
        se0, se1 = np.clip(0.6 + 0.1 * Xe[:, 1], 0.3, 1.5), np.clip(0.9 + 0.1 * Xe[:, 1], 0.3, 1.5)
        q0e, q1e = log1p_quantile(me, se0, QUANTILE_GRID), log1p_quantile(me, se1, QUANTILE_GRID)
        r0e, r1e = raw_quantile(me, se0, FUNCTIONAL_GRID), raw_quantile(me, se1, FUNCTIONAL_GRID)
        u0e, u1e = u_matrix(q0e, r0e, FW, FUNCTIONAL_GRID), u_matrix(q1e, r1e, FW, FUNCTIONAL_GRID)

        inner = None if regime == "oracle_latent" else self._make_inner(rng, n, regime, R0, R1, Z)
        proxy_theta_eval = _proxy_truth_deterministic(
            r0e, r1e, Xe, regime, seed + 91003
        )
        return DGPResult("D2", seed, regime, n, self.K, self.J, X, Z, prop,
                         Q0_log, Q1_log, R0, R1, u0, u1, Xe, u1e - u0e, inner,
                         _inner_label(inner, regime), {}, proxy_theta_eval=proxy_theta_eval)

    def _make_inner(self, rng, n, regime, raw0, raw1, Z):
        ms = _draw_inner_sizes(rng, n, regime)
        return [sample_inner_from_quantile(raw1[i] if Z[i] > 0.5 else raw0[i], int(ms[i]), rng)
                for i in range(n)]


@dataclass
class D3TailLocalized:
    K: int = DEFAULT_K; J: int = DEFAULT_J; d: int = 5; n_eval: int = 200

    def sample(self, n: int, seed: int, regime: str = "oracle_latent") -> DGPResult:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, self.d))
        prop = 1.0 / (1.0 + np.exp(-np.clip(0.4 * X[:, 0] - 0.3 * X[:, 1], -10, 10)))
        Z = (rng.uniform(size=n) < prop).astype(float)
        mu, sigma = 7.0 + 0.3 * X[:, 0], np.clip(0.7 + 0.1 * X[:, 1], 0.3, 1.5)
        gX = np.tanh(0.5 * X[:, 0])
        # The amplitude/width pair is chosen so that the perturbation's
        # derivative is smaller than the minimum lognormal quantile slope
        # under sigma >= 0.3.  This is a valid monotone tail-localized DGP,
        # rather than a bump that can reverse the quantile ordering.
        tail = 0.03 * np.exp(-((QUANTILE_GRID - 0.85) ** 2) / (2 * 0.08 ** 2))
        tail_f = 0.03 * np.exp(-((FUNCTIONAL_GRID - 0.85) ** 2) / (2 * 0.08 ** 2))

        Q0_log = log1p_quantile(mu, sigma, QUANTILE_GRID)
        raw_shifted = np.exp(mu[:, None] + sigma[:, None] * norm_ppf(QUANTILE_GRID)[None, :] + gX[:, None] * tail[None, :])
        Q1_log = np.log1p(raw_shifted)
        R0 = raw_quantile(mu, sigma, FUNCTIONAL_GRID)
        R1 = np.exp(mu[:, None] + sigma[:, None] * norm_ppf(FUNCTIONAL_GRID)[None, :] + gX[:, None] * tail_f[None, :])
        _assert_monotone_quantiles(Q0_log, "D3 Q0")
        _assert_monotone_quantiles(Q1_log, "D3 Q1")
        _assert_monotone_quantiles(R0, "D3 raw Q0")
        _assert_monotone_quantiles(R1, "D3 raw Q1")

        u0, u1 = u_matrix(Q0_log, R0, FW, FUNCTIONAL_GRID), u_matrix(Q1_log, R1, FW, FUNCTIONAL_GRID)
        Xe = rng.normal(0, 1, (self.n_eval, self.d))
        me, se = 7.0 + 0.3 * Xe[:, 0], np.clip(0.7 + 0.1 * Xe[:, 1], 0.3, 1.5)
        ge = np.tanh(0.5 * Xe[:, 0])
        q0e = log1p_quantile(me, se, QUANTILE_GRID)
        q1e = np.log1p(np.exp(me[:, None] + se[:, None] * norm_ppf(QUANTILE_GRID)[None, :] + ge[:, None] * tail[None, :]))
        r0e = raw_quantile(me, se, FUNCTIONAL_GRID)
        r1e = np.exp(me[:, None] + se[:, None] * norm_ppf(FUNCTIONAL_GRID)[None, :] + ge[:, None] * tail_f[None, :])
        u0e, u1e = u_matrix(q0e, r0e, FW, FUNCTIONAL_GRID), u_matrix(q1e, r1e, FW, FUNCTIONAL_GRID)
        _assert_monotone_quantiles(q0e, "D3 evaluation Q0")
        _assert_monotone_quantiles(q1e, "D3 evaluation Q1")
        _assert_monotone_quantiles(r0e, "D3 evaluation raw Q0")
        _assert_monotone_quantiles(r1e, "D3 evaluation raw Q1")

        inner = None if regime == "oracle_latent" else self._make_inner(rng, n, regime, R0, R1, Z)
        proxy_theta_eval = _proxy_truth_deterministic(
            r0e, r1e, Xe, regime, seed + 91004
        )
        return DGPResult("D3", seed, regime, n, self.K, self.J, X, Z, prop,
                         Q0_log, Q1_log, R0, R1, u0, u1, Xe, u1e - u0e, inner,
                         _inner_label(inner, regime), {}, proxy_theta_eval=proxy_theta_eval)

    def _make_inner(self, rng, n, regime, raw0, raw1, Z):
        ms = _draw_inner_sizes(rng, n, regime)
        return [sample_inner_from_quantile(raw1[i] if Z[i] > 0.5 else raw0[i], int(ms[i]), rng) for i in range(n)]


@dataclass
class D4PureNonlinear:
    K: int = DEFAULT_K; J: int = DEFAULT_J; d: int = 5; n_eval: int = 200

    def sample(self, n: int, seed: int, regime: str = "oracle_latent") -> DGPResult:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, self.d))
        subgroup = (X[:, 0] > 0).astype(float)
        propensity = np.full(n, 0.5)
        Z = (rng.uniform(size=n) < 0.5).astype(float)

        def _tpf(p, lo, hi):
            return np.where(p <= 0.5, lo, hi)

        Q0_log = np.tile(np.log1p(_tpf(QUANTILE_GRID, 1.0, 3.0)), (n, 1))
        Q0_rf = np.tile(_tpf(FUNCTIONAL_GRID, 1.0, 3.0), (n, 1))

        Q1_log = np.empty((n, self.K))
        Q1_rf = np.empty((n, L_FUNCTIONAL))
        u1 = np.empty((n, self.K + 3))

        for i in range(n):
            if subgroup[i] > 0.5:
                if rng.binomial(1, 0.5):
                    Q1_log[i] = np.full(self.K, np.log1p(1.0))
                    Q1_rf[i] = np.full(L_FUNCTIONAL, 1.0)
                    u1[i, :self.K] = np.full(self.K, np.log1p(1.0))
                    raw_fine = np.full(L_FUNCTIONAL, 1.0)
                else:
                    Q1_log[i] = np.log1p(_tpf(QUANTILE_GRID, 1.0, 7.0))
                    Q1_rf[i] = _tpf(FUNCTIONAL_GRID, 1.0, 7.0)
                    u1[i, :self.K] = np.log1p(_tpf(QUANTILE_GRID, 1.0, 7.0))
                    raw_fine = _tpf(FUNCTIONAL_GRID, 1.0, 7.0)
                u1[i, self.K:] = functional_vector_3(raw_fine, FW, FUNCTIONAL_GRID)
            else:
                Q1_log[i] = Q0_log[i].copy()
                Q1_rf[i] = Q0_rf[i].copy()
                u1[i] = u_matrix(Q0_log[i:i+1], Q0_rf[i:i+1], FW, FUNCTIONAL_GRID)[0]

        u0 = u_matrix(Q0_log, Q0_rf, FW, FUNCTIONAL_GRID)

        u_true_m1 = np.empty((n, self.K + 3))
        for i in range(n):
            if subgroup[i] > 0.5:
                r_deg, r_spr = _tpf(FUNCTIONAL_GRID, 1.0, 1.0), _tpf(FUNCTIONAL_GRID, 1.0, 7.0)
                q_deg, q_spr = np.log1p(_tpf(QUANTILE_GRID, 1.0, 1.0)), np.log1p(_tpf(QUANTILE_GRID, 1.0, 7.0))
                u_deg = u_vector(q_deg, r_deg, FW, FUNCTIONAL_GRID)
                u_spr = u_vector(q_spr, r_spr, FW, FUNCTIONAL_GRID)
                u_true_m1[i] = 0.5 * u_deg + 0.5 * u_spr
            else:
                u_true_m1[i] = u0[i]

        Xe = rng.normal(0, 1, (self.n_eval, self.d))
        e_sub = (Xe[:, 0] > 0).astype(float)
        q0e = np.tile(np.log1p(_tpf(QUANTILE_GRID, 1.0, 3.0)), (self.n_eval, 1))
        r0e = np.tile(_tpf(FUNCTIONAL_GRID, 1.0, 3.0), (self.n_eval, 1))
        theta_e = np.zeros((self.n_eval, self.K + 3))
        proxy_components = np.empty((self.n_eval, 2, L_FUNCTIONAL))
        for i in range(self.n_eval):
            if e_sub[i] > 0.5:
                rd, rs = _tpf(FUNCTIONAL_GRID, 1.0, 1.0), _tpf(FUNCTIONAL_GRID, 1.0, 7.0)
                proxy_components[i] = np.stack([rd, rs])
                qd = np.log1p(_tpf(QUANTILE_GRID, 1.0, 1.0))
                qs = np.log1p(_tpf(QUANTILE_GRID, 1.0, 7.0))
                ud = u_vector(qd, rd, FW, FUNCTIONAL_GRID)
                us = u_vector(qs, rs, FW, FUNCTIONAL_GRID)
                u0e = u_vector(q0e[i], r0e[i], FW, FUNCTIONAL_GRID)
                theta_e[i] = 0.5 * ud + 0.5 * us - u0e
            else:
                proxy_components[i] = np.stack([r0e[i], r0e[i]])
            
        inner = None if regime == "oracle_latent" else self._make_inner(rng, n, regime, Q0_rf, Q1_rf, Z)
        proxy_theta_eval = _proxy_truth_components(
            r0e, proxy_components, Xe, regime, seed + 91005
        )
        return DGPResult("D4", seed, regime, n, self.K, self.J, X, Z, propensity,
                         Q0_log, Q1_log, Q0_rf, Q1_rf, u0, u_true_m1, Xe, theta_e, inner,
                         _inner_label(inner, regime), {"subgroup": subgroup},
                         proxy_theta_eval=proxy_theta_eval)

    def _make_inner(self, rng, n, regime, raw0, raw1, Z):
        ms = _draw_inner_sizes(rng, n, regime)
        return [sample_inner_from_quantile(raw1[i] if Z[i] > 0.5 else raw0[i], int(ms[i]), rng) for i in range(n)]


@dataclass
class D5MultipleConflicting:
    K: int = DEFAULT_K; J: int = DEFAULT_J; d: int = 5; n_eval: int = 200

    def sample(self, n: int, seed: int, regime: str = "oracle_latent") -> DGPResult:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, self.d))
        prop = 1.0 / (1.0 + np.exp(-np.clip(0.3 * X[:, 0] - 0.2 * X[:, 1], -10, 10)))
        Z = (rng.uniform(size=n) < prop).astype(float)
        mu, s0 = 7.0 + 0.4 * X[:, 0], np.clip(0.6 + 0.1 * X[:, 1], 0.3, 1.5)
        tm = 0.3 * np.tanh(0.5 * X[:, 0])
        s1 = np.clip(0.6 + 0.1 * X[:, 1] + 0.3 * (X[:, 2] > 0), 0.3, 1.5)
        mu1 = mu + tm

        Q0_log = log1p_quantile(mu, s0, QUANTILE_GRID)
        Q1_log = log1p_quantile(mu1, s1, QUANTILE_GRID)
        R0, R1 = raw_quantile(mu, s0, FUNCTIONAL_GRID), raw_quantile(mu1, s1, FUNCTIONAL_GRID)
        u0, u1 = u_matrix(Q0_log, R0, FW, FUNCTIONAL_GRID), u_matrix(Q1_log, R1, FW, FUNCTIONAL_GRID)

        Xe = rng.normal(0, 1, (self.n_eval, self.d))
        me = 7.0 + 0.4 * Xe[:, 0]
        s0e, te = np.clip(0.6 + 0.1 * Xe[:, 1], 0.3, 1.5), 0.3 * np.tanh(0.5 * Xe[:, 0])
        s1e = np.clip(0.6 + 0.1 * Xe[:, 1] + 0.3 * (Xe[:, 2] > 0), 0.3, 1.5)
        me1 = me + te
        q0e, q1e = log1p_quantile(me, s0e, QUANTILE_GRID), log1p_quantile(me1, s1e, QUANTILE_GRID)
        r0e, r1e = raw_quantile(me, s0e, FUNCTIONAL_GRID), raw_quantile(me1, s1e, FUNCTIONAL_GRID)
        u0e, u1e = u_matrix(q0e, r0e, FW, FUNCTIONAL_GRID), u_matrix(q1e, r1e, FW, FUNCTIONAL_GRID)

        inner = None if regime == "oracle_latent" else self._make_inner(rng, n, regime, R0, R1, Z)
        proxy_theta_eval = _proxy_truth_deterministic(
            r0e, r1e, Xe, regime, seed + 91007
        )
        return DGPResult("D5", seed, regime, n, self.K, self.J, X, Z, prop,
                         Q0_log, Q1_log, R0, R1, u0, u1, Xe, u1e - u0e, inner,
                         _inner_label(inner, regime), {}, proxy_theta_eval=proxy_theta_eval)

    def _make_inner(self, rng, n, regime, raw0, raw1, Z):
        ms = _draw_inner_sizes(rng, n, regime)
        return [sample_inner_from_quantile(raw1[i] if Z[i] > 0.5 else raw0[i], int(ms[i]), rng) for i in range(n)]


@dataclass
class D8UnequalInner:
    K: int = DEFAULT_K; J: int = DEFAULT_J; d: int = 5; n_eval: int = 200

    def sample(self, n: int, seed: int, regime: str = "feasible_growing_inner") -> DGPResult:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, self.d))
        prop = 1.0 / (1.0 + np.exp(-np.clip(0.3 * X[:, 0] - 0.2 * X[:, 1], -10, 10)))
        Z = (rng.uniform(size=n) < prop).astype(float)
        mu, sigma = 7.0 + 0.3 * X[:, 0], np.clip(0.7 + 0.1 * X[:, 1], 0.3, 1.5)
        tau = 0.4 * np.tanh(0.5 * X[:, 0])
        mu1 = mu + tau

        Q0_log = log1p_quantile(mu, sigma, QUANTILE_GRID)
        Q1_log = log1p_quantile(mu1, sigma, QUANTILE_GRID)
        R0, R1 = raw_quantile(mu, sigma, FUNCTIONAL_GRID), raw_quantile(mu1, sigma, FUNCTIONAL_GRID)
        u0, u1 = u_matrix(Q0_log, R0, FW, FUNCTIONAL_GRID), u_matrix(Q1_log, R1, FW, FUNCTIONAL_GRID)

        Xe = rng.normal(0, 1, (self.n_eval, self.d))
        me, se = 7.0 + 0.3 * Xe[:, 0], np.clip(0.7 + 0.1 * Xe[:, 1], 0.3, 1.5)
        te = 0.4 * np.tanh(0.5 * Xe[:, 0])
        me1 = me + te
        q0e, q1e = log1p_quantile(me, se, QUANTILE_GRID), log1p_quantile(me1, se, QUANTILE_GRID)
        r0e, r1e = raw_quantile(me, se, FUNCTIONAL_GRID), raw_quantile(me1, se, FUNCTIONAL_GRID)
        u0e, u1e = u_matrix(q0e, r0e, FW, FUNCTIONAL_GRID), u_matrix(q1e, r1e, FW, FUNCTIONAL_GRID)

        if regime == "oracle_latent":
            inner = None
        else:
            ms = _draw_inner_sizes(rng, n, regime, X, d8=True)
            inner = [
                sample_inner_from_quantile(
                    R1[i] if Z[i] > 0.5 else R0[i], int(ms[i]), rng
                )
                for i in range(n)
            ]
        proxy_theta_eval = _proxy_truth_deterministic(
            r0e, r1e, Xe, regime, seed + 91008, d8=True
        )

        return DGPResult("D8", seed, regime, n, self.K, self.J, X, Z, prop,
                         Q0_log, Q1_log, R0, R1, u0, u1, Xe, u1e - u0e, inner,
                         _inner_label(inner, regime), {}, proxy_theta_eval=proxy_theta_eval)


DGP_MAP: dict[str, type] = {
    "D0": D0Null,
    "D1": D1LocationShift,
    "D2": D2ScaleCrossing,
    "D3": D3TailLocalized,
    "D4": D4PureNonlinear,
    "D5": D5MultipleConflicting,
    "D8": D8UnequalInner,
}


def sample_dgp(name: str, n: int, seed: int, regime: str = "oracle_latent", **kw) -> DGPResult:
    if name not in DGP_MAP:
        raise KeyError(f"unknown DGP: {name}")
    requested_k = kw.get("K", DEFAULT_K)
    requested_j = kw.get("J", DEFAULT_J)
    if requested_k != DEFAULT_K or requested_j != DEFAULT_J:
        raise ValueError(
            "the frozen WP9 DGP library uses K=49 and J=3; "
            "use wp3_odcf directly for other finite-vector dimensions"
        )
    cls = DGP_MAP[name]
    dgp = cls(**{k: v for k, v in kw.items() if hasattr(cls, k)})
    return dgp.sample(n, seed, regime)
