import numpy as np
from scipy.stats import norm

from wasserstein_causal_forests.applied.adapter import (
    FUNCTIONAL_NAMES,
    AppliedDataset,
    aggregate_replications,
    ecdf_moderator,
    energy_score_on,
    make_functionals,
    midpoint_levels,
    plugin_contrasts,
    run_placebo,
    run_wcf,
)


def _synthetic(n: int = 220, seed: int = 0) -> AppliedDataset:
    rng = np.random.default_rng(seed)
    levels = midpoint_levels()
    z = norm.ppf(levels)
    x = rng.uniform(-1.0, 1.0, size=(n, 4))
    raw_moderator = x[:, 0].copy()
    x[:, 0] = ecdf_moderator(x[:, 0])
    propensity = 1.0 / (1.0 + np.exp(-(0.8 * x[:, 1] - 0.6 * x[:, 2])))
    treatment = rng.binomial(1, propensity)
    location = 0.4 * x[:, 1] + 0.3 * x[:, 2] + treatment * (0.3 + 0.2 * x[:, 1])
    log_scale = 0.2 + 0.1 * x[:, 3]
    shape = 0.2 * ((z**2 - 1.0) / 2.0)
    quantiles = (
        location[:, None]
        + np.exp(log_scale)[:, None] * (z[None, :] + shape[None, :])
        + rng.normal(0.0, 0.02, size=(n, levels.size))
    )
    quantiles = np.sort(quantiles, axis=1)
    return AppliedDataset(
        study="smoke",
        X=x,
        A=treatment,
        Q=quantiles,
        moderator_raw=raw_moderator,
        q_star=z,
        feature_names=["moderator", "x2", "x3", "x4"],
        meta={"source": "synthetic smoke test"},
    )


def test_adapter_roundtrip_and_fit(tmp_path):
    dataset = _synthetic()
    dataset.save(tmp_path)
    loaded = AppliedDataset.load(tmp_path)
    assert loaded.X.shape == dataset.X.shape
    assert loaded.Q.shape == (220, 25)

    model, results = run_wcf(
        loaded,
        random_state=0,
        n_folds=2,
        n_estimators=10,
        n_particles=5,
        contrast_candidates=(0.0, 50.0),
    )
    assert set(results["marginal_dr"]) == set(FUNCTIONAL_NAMES)
    assert set(results["bin_contrasts_plugin"]) == set(FUNCTIONAL_NAMES)
    assert len(results["moderator_bin_counts"]) == 4
    assert results["n"] == 220

    functionals = make_functionals(loaded.q_star)
    plugin = plugin_contrasts(model, loaded.X, functionals)
    assert set(plugin) == set(FUNCTIONAL_NAMES)
    assert np.isfinite(plugin["mean"]["marginal"])

    score = energy_score_on(model, loaded.X, loaded.A, loaded.Q)
    assert np.isfinite(score)

    aggregated = aggregate_replications([results, results])
    assert aggregated["n_replications"] == 2
    for name in FUNCTIONAL_NAMES:
        assert np.isfinite(aggregated["marginal_dr"][name]["mean"])

    placebo = run_placebo(
        loaded,
        seeds=(0,),
        n_folds=2,
        n_estimators=10,
        n_particles=5,
        contrast_candidates=(0.0, 50.0),
    )
    assert len(placebo) == 1
