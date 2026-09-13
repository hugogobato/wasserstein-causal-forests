"""Shared adapter from applied datasets to the WCF estimator.

Every applied study in this project uses this module so that the grid, the
declared functionals, the moderator transform, and the estimator settings are
identical across datasets. A study calls:

    ds = AppliedDataset(study=..., X=..., A=..., Q=..., moderator_raw=...,
                        q_star=..., feature_names=[...], meta={...})
    ds.save("results/applied_study_exploration/<study>/data")
    model, results = run_wcf(ds, random_state=0)

`X` must be a finite float matrix with the moderator transformed into (-1, 1)
in column 0 (use `ecdf_moderator`); the estimator bins column 0 at
(-0.5, 0, 0.5), so those four bins are the moderator quartiles. `A` is a binary
vector with both arms present, `Q` holds one monotone K-vector per unit on the
midpoint grid, and `q_star` is the benchmark on the same grid and scale.

`run_wcf` returns the fitted estimator and a JSON-ready results dict with
AIPW-calibrated marginals and moderator-bin contrasts for the five declared
functionals (mean, sd, skew, tail, reference distance), the plug-in contrast
counterpart, selection records, and propensity-overlap diagnostics. Nothing in
this module is a substitute for the study's own identification argument or its
trusted-scalar reproduction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

N_GRID = 25

FUNCTIONAL_NAMES = ("mean", "sd", "skew", "tail", "reference")


def midpoint_levels(n_grid: int = N_GRID) -> NDArray[np.float64]:
    """The uniform midpoint grid u_k = (k - 0.5) / K used by the manuscript."""

    return (np.arange(n_grid, dtype=float) + 0.5) / n_grid


def grid_weights(n_grid: int = N_GRID) -> NDArray[np.float64]:
    """Equal quadrature weights 1 / K."""

    return np.full(n_grid, 1.0 / n_grid)


def weighted_quantiles(
    values: ArrayLike,
    sample_weights: ArrayLike | None = None,
    levels: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Weighted quantiles of a scalar sample, evaluated at `levels`."""

    v = np.asarray(values, dtype=float)
    if levels is None:
        levels = midpoint_levels()
    u = np.asarray(levels, dtype=float)
    if sample_weights is None:
        sw = np.ones(v.shape[0], dtype=float)
    else:
        sw = np.asarray(sample_weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(sw) & (sw > 0)
    v, sw = v[mask], sw[mask]
    if v.size == 0:
        raise ValueError("no finite, positively weighted observations")
    order = np.argsort(v, kind="mergesort")
    v, sw = v[order], sw[order]
    cumulative = np.cumsum(sw)
    cumulative = cumulative / cumulative[-1]
    quantiles = np.interp(u, cumulative, v)
    return np.maximum.accumulate(quantiles)


def quantile_matrix(
    values: ArrayLike,
    sample_weights: ArrayLike,
    unit_ids: ArrayLike,
    levels: ArrayLike | None = None,
) -> tuple[NDArray, NDArray[np.float64]]:
    """One quantile vector per unit, grouped by `unit_ids`.

    Returns the sorted unique unit identifiers and the (n_units, K) matrix.
    Units with no finite positively weighted observation are dropped; the
    caller should compare the returned unit list against the design matrix.
    """

    v = np.asarray(values, dtype=float)
    sw = np.asarray(sample_weights, dtype=float)
    uid = np.asarray(unit_ids)
    unique_units = np.unique(uid)
    rows = []
    kept = []
    for unit in unique_units:
        mask = (uid == unit) & np.isfinite(v) & np.isfinite(sw) & (sw > 0)
        if not np.any(mask):
            continue
        kept.append(unit)
        rows.append(weighted_quantiles(v[mask], sw[mask], levels=levels))
    if not rows:
        raise ValueError("no unit has a usable outcome sample")
    return np.asarray(kept), np.vstack(rows)


def ecdf_moderator(v: ArrayLike) -> NDArray[np.float64]:
    """Rank transform into (-1, 1) so that the frozen bin edges are quartiles."""

    values = np.asarray(v, dtype=float)
    ranks = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return 2.0 * (ranks + 0.5) / values.size - 1.0


def make_functionals(
    q_star: ArrayLike,
    weights: ArrayLike | None = None,
) -> dict[str, object]:
    """The five declared grid functionals, including the reference distance."""

    star = np.asarray(q_star, dtype=float)
    if star.ndim != 1:
        raise ValueError("q_star must be a one-dimensional quantile vector")
    w = grid_weights(star.size) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()

    def h_mean(q: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(q, dtype=float) @ w

    def h_sd(q: ArrayLike) -> NDArray[np.float64]:
        block = np.asarray(q, dtype=float)
        centre = block @ w
        variance = ((block - centre[:, None]) ** 2) @ w
        return np.sqrt(np.maximum(variance, 0.0))

    def h_skew(q: ArrayLike) -> NDArray[np.float64]:
        block = np.asarray(q, dtype=float)
        centre = block @ w
        sd = np.sqrt(np.maximum(((block - centre[:, None]) ** 2) @ w, 0.0))
        third = ((block - centre[:, None]) ** 3) @ w
        ratio = np.divide(
            third,
            sd**3,
            out=np.zeros_like(third),
            where=sd > 1e-12,
        )
        return ratio

    half = np.zeros_like(w)
    half[int(np.floor(w.size / 2)) :] = w[int(np.floor(w.size / 2)) :]
    half = half / half.sum()

    def h_tail(q: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(q, dtype=float) @ half

    def h_ref(q: ArrayLike) -> NDArray[np.float64]:
        block = np.asarray(q, dtype=float)
        return np.sqrt(((block - star[None, :]) ** 2) @ w)

    return {
        "mean": h_mean,
        "sd": h_sd,
        "skew": h_skew,
        "tail": h_tail,
        "reference": h_ref,
    }


def _sha256(array: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class AppliedDataset:
    """One analysis-ready applied dataset on the frozen WCF grid."""

    study: str
    X: NDArray[np.float64]
    A: NDArray[np.int64]
    Q: NDArray[np.float64]
    moderator_raw: NDArray[np.float64]
    q_star: NDArray[np.float64]
    feature_names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def validate(self) -> "AppliedDataset":
        self.X = np.asarray(self.X, dtype=float)
        self.A = np.asarray(self.A, dtype=np.int64)
        self.Q = np.asarray(self.Q, dtype=float)
        self.moderator_raw = np.asarray(self.moderator_raw, dtype=float)
        self.q_star = np.asarray(self.q_star, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("X must be (n, p)")
        n = self.X.shape[0]
        if self.A.shape != (n,):
            raise ValueError("A must have shape (n,)")
        if self.Q.shape != (n, N_GRID):
            raise ValueError(f"Q must have shape (n, {N_GRID})")
        if self.moderator_raw.shape != (n,):
            raise ValueError("moderator_raw must have shape (n,)")
        if self.q_star.shape != (N_GRID,):
            raise ValueError(f"q_star must have shape ({N_GRID},)")
        if self.feature_names and len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names must have one entry per X column")
        if not np.all(np.isfinite(self.X)):
            raise ValueError("X has non-finite entries")
        if not np.all(np.isfinite(self.Q)):
            raise ValueError("Q has non-finite entries")
        if not np.all(np.isfinite(self.q_star)):
            raise ValueError("q_star has non-finite entries")
        if not np.all(np.diff(self.Q, axis=1) >= -1e-12):
            raise ValueError("Q rows must be nondecreasing")
        if not set(np.unique(self.A)).issubset({0, 1}):
            raise ValueError("A must be binary")
        if min(np.bincount(self.A, minlength=2)) < 5:
            raise ValueError("each arm needs at least five units")
        return self

    def save(self, out_dir: str | Path) -> Path:
        """Write dataset.npz plus a manifest.json with provenance and hashes."""

        ds = self.validate()
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        npz_path = path / "dataset.npz"
        np.savez_compressed(
            npz_path,
            X=ds.X,
            A=ds.A,
            Q=ds.Q,
            moderator_raw=ds.moderator_raw,
            q_star=ds.q_star,
        )
        from wasserstein_causal_forests.g3.dgps import moderator_bins

        bins = moderator_bins(ds.X)
        manifest = {
            "study": ds.study,
            "n": int(ds.X.shape[0]),
            "p": int(ds.X.shape[1]),
            "K": int(ds.Q.shape[1]),
            "n_treated": int(ds.A.sum()),
            "n_control": int((ds.A == 0).sum()),
            "moderator_bin_counts": np.bincount(bins, minlength=4).tolist(),
            "feature_names": list(ds.feature_names),
            "meta": ds.meta,
            "sha256": {
                "X": _sha256(ds.X),
                "A": _sha256(ds.A),
                "Q": _sha256(ds.Q),
                "moderator_raw": _sha256(ds.moderator_raw),
                "q_star": _sha256(ds.q_star),
            },
        }
        with open(path / "manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2, default=str)
        return npz_path

    @classmethod
    def load(cls, out_dir: str | Path) -> "AppliedDataset":
        path = Path(out_dir)
        with open(path / "manifest.json") as handle:
            manifest = json.load(handle)
        arrays = np.load(path / "dataset.npz")
        return cls(
            study=manifest["study"],
            X=arrays["X"],
            A=arrays["A"],
            Q=arrays["Q"],
            moderator_raw=arrays["moderator_raw"],
            q_star=arrays["q_star"],
            feature_names=list(manifest.get("feature_names", [])),
            meta=dict(manifest.get("meta", {})),
        ).validate()


def _propensity_summary(model: object, treatment: NDArray[np.int64]) -> dict:
    ehat = np.asarray(model.ehat_train_, dtype=float)
    return {
        "min": float(np.min(ehat)),
        "max": float(np.max(ehat)),
        "mean": float(np.mean(ehat)),
        "share_at_clip_low": float(np.mean(ehat <= 0.02 + 1e-12)),
        "share_at_clip_high": float(np.mean(ehat >= 0.98 - 1e-12)),
        "share_outside_0p1_0p9": float(np.mean((ehat < 0.1) | (ehat > 0.9))),
        "treated_mean": float(np.mean(ehat[treatment == 1])),
        "control_mean": float(np.mean(ehat[treatment == 0])),
    }


def run_wcf(
    dataset: AppliedDataset,
    *,
    random_state: int = 0,
    n_folds: int = 3,
    n_particles: int = 10,
    n_estimators: int = 100,
    learning_rate: float = 0.12,
    max_depth: int = 4,
    min_samples_leaf: int = 10,
    min_arm_leaf: int = 5,
    collision_epsilon: float = 1e-3,
    contrast_candidates: tuple[float, ...] = (0.0, 50.0, 500.0),
    arm_shrinkage: float = 5.0,
    propensity_factory: str | None = None,
    treatment_override: ArrayLike | None = None,
    extra_functionals: dict[str, Callable[[NDArray], NDArray]] | None = None,
) -> tuple[object, dict]:
    """Fit the frozen WCF specification and summarize every declared functional.

    ``extra_functionals`` maps additional names to callables (m, K) -> (m,),
    merged into the five declared functionals so that study-specific targets
    such as the lower-tail mean or a tenth-quantile coordinate are reported by
    the same AIPW layer. They must be declared before estimation.
    """

    from wasserstein_causal_forests.cwdb.dr_calibration import (
        DRCalibratedCWDB,
        hajek_bin_means,
    )
    from wasserstein_causal_forests.g3.dgps import moderator_bins

    ds = dataset.validate()
    a = (
        ds.A
        if treatment_override is None
        else np.asarray(treatment_override, dtype=np.int64)
    )
    if a.shape != ds.A.shape:
        raise ValueError("treatment_override must have shape (n,)")
    weights = grid_weights(ds.Q.shape[1])
    functionals = make_functionals(ds.q_star, weights)
    if extra_functionals:
        functionals.update(extra_functionals)
    model = DRCalibratedCWDB(
        functionals=functionals,
        contrast_candidates=contrast_candidates,
        n_folds=n_folds,
        n_particles=n_particles,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_arm_leaf=min_arm_leaf,
        collision_epsilon=collision_epsilon,
        arm_shrinkage=arm_shrinkage,
        architecture="v1",
        sharing="partial",
        init_sharing="pooled",
        random_state=random_state,
        propensity_factory=propensity_factory,
    )
    model.fit(ds.X, a, ds.Q, weights)

    bins_train = moderator_bins(ds.X)
    results = {
        "study": ds.study,
        "n": int(ds.X.shape[0]),
        "n_treated": int(a.sum()),
        "n_control": int((a == 0).sum()),
        "K": int(ds.Q.shape[1]),
        "random_state": int(random_state),
        "n_folds": int(n_folds),
        "selected_contrast_shrinkage": float(model.selected_contrast_shrinkage_),
        "selection_records": [
            {
                "contrast_shrinkage": float(record.contrast_shrinkage),
                "held_out_risk": float(record.held_out_risk),
                "n_scored": int(record.n_scored),
            }
            for record in model.selection_records_
        ],
        "marginal_dr": {name: float(model.dr_marginal(name)) for name in functionals},
        "if_se": {name: float(model.dr_if_se(name)) for name in functionals},
        "bin_contrasts_dr": {
            name: [
                None if np.isnan(value) else float(value)
                for value in model.aipw_.bin_contrasts_[name]
            ]
            for name in functionals
        },
        "moderator_bin_counts": np.bincount(bins_train, minlength=4).tolist(),
        "propensity": _propensity_summary(model, a),
    }
    plugin = plugin_contrasts(model, ds.X, functionals)
    results["marginal_plugin"] = {
        name: plugin[name]["marginal"] for name in functionals
    }
    results["bin_contrasts_plugin"] = {
        name: plugin[name]["bins"] for name in functionals
    }
    risk = getattr(model, "train_risk_", None)
    if risk is not None:
        results["train_energy_risk"] = float(risk)
    return model, results


def plugin_contrasts(
    model: object,
    X: ArrayLike,
    functionals: dict[str, object] | None = None,
) -> dict[str, dict]:
    """Particle-law plug-in contrast at each row and moderator bin of `X`."""

    from wasserstein_causal_forests.cwdb.dr_calibration import hajek_bin_means
    from wasserstein_causal_forests.g3.dgps import moderator_bins

    x = np.asarray(X, dtype=float)
    if functionals is None:
        raise ValueError("functionals are required")
    parts: dict[int, dict[str, NDArray[np.float64]]] = {}
    for arm in (0, 1):
        particles = model.predict_particles(x, arm)
        flat = particles.reshape(-1, particles.shape[-1])
        parts[arm] = {
            name: np.asarray(h(flat), dtype=float)
            .reshape(particles.shape[0], particles.shape[1])
            .mean(axis=1)
            for name, h in functionals.items()
        }
    bins = moderator_bins(x)
    out = {}
    for name in functionals:
        difference = parts[1][name] - parts[0][name]
        out[name] = {
            "marginal": float(np.mean(difference)),
            "bins": [
                None if np.isnan(value) else float(value)
                for value in hajek_bin_means(difference, bins, 4)
            ],
        }
    return out


def energy_score_on(
    model: object,
    X: ArrayLike,
    treatment: ArrayLike,
    Q: ArrayLike,
) -> float:
    """Mean collision-smoothed energy score on observed arms only."""

    x = np.asarray(X, dtype=float)
    a = np.asarray(treatment, dtype=np.int64)
    q = np.asarray(Q, dtype=float)
    scores = []
    for arm in (0, 1):
        rows = a == arm
        if np.any(rows):
            scores.append(model.score_samples(x[rows], arm, q[rows]))
    if not scores:
        raise ValueError("no arm present")
    return float(np.concatenate(scores).mean())


def run_wcf_replications(
    dataset: AppliedDataset,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    **kwargs: object,
) -> list[dict]:
    """Fit the frozen specification once per seed and return the result dicts."""

    outputs = []
    for seed in seeds:
        _, results = run_wcf(dataset, random_state=seed, **kwargs)
        outputs.append(results)
    return outputs


def aggregate_replications(results_list: list[dict]) -> dict:
    """Mean and spread of each reported quantity across replications."""

    names = list(results_list[0]["marginal_dr"])
    aggregated = {
        "n_replications": len(results_list),
        "marginal_dr": {},
        "bin_contrasts_dr": {},
        "selected_contrast_shrinkage": [],
    }
    for name in names:
        values = np.asarray([r["marginal_dr"][name] for r in results_list])
        aggregated["marginal_dr"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "values": values.tolist(),
        }
        bins = np.asarray(
            [
                [np.nan if v is None else v for v in r["bin_contrasts_dr"][name]]
                for r in results_list
            ],
            dtype=float,
        )
        aggregated["bin_contrasts_dr"][name] = {
            "mean": np.nanmean(bins, axis=0).tolist(),
            "std": np.nanstd(bins, axis=0, ddof=1).tolist()
            if bins.shape[0] > 1
            else np.zeros(bins.shape[1]).tolist(),
            "n_defined": int(np.sum(~np.isnan(bins), axis=0).max()) if bins.size else 0,
        }
    aggregated["selected_contrast_shrinkage"] = [
        r["selected_contrast_shrinkage"] for r in results_list
    ]
    return aggregated


def run_placebo(
    dataset: AppliedDataset,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    **kwargs: object,
) -> list[dict]:
    """Permute the treatment label and rerun; a null distribution for false effects."""

    outputs = []
    for seed in seeds:
        rng = np.random.default_rng(10_000 + seed)
        permuted = rng.permutation(dataset.A)
        _, results = run_wcf(
            dataset, random_state=seed, treatment_override=permuted, **kwargs
        )
        outputs.append(results)
    return outputs
