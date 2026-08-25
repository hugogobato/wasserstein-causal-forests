"""Structural invariants of the G3 tournament DGP suite.

Each regime exists to make one preregistered claim falsifiable, so each is
pinned to the property that gives it its name: D0 is deterministic, D2 has a
null effect, D5 matches barycenters while differing in law, D7 hides its whole
effect in a functional outside the training manifest, and D9 really does lose
overlap. The quadrature oracle truth is checked against Monte Carlo in
`research/checks/g3_dgp_truth_accuracy.py`; here it is checked against the
closed forms that exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from wasserstein_causal_forests.common.quantiles import validate_quantiles
from wasserstein_causal_forests.g3.dgps import (
    DGP_IDS,
    N_MODERATOR_BINS,
    GridSpec,
    OuterLaw,
    build_dgp,
    moderator_bins,
)


@pytest.mark.parametrize("dgp_id", DGP_IDS)
def test_samples_live_in_the_monotone_cone(dgp_id: str) -> None:
    for n_grid in (5, 25, 99):
        sample = build_dgp(dgp_id, n_grid).sample(400, seed=3)
        # Raises if any row leaves Q_K, which is the declared state space.
        validate_quantiles(sample.quantiles, n_grid)
        assert sample.quantiles.shape == (400, n_grid)
        assert set(np.unique(sample.treatment)) <= {0, 1}
        assert sample.treatment.sum() >= 2
        assert (1 - sample.treatment).sum() >= 2


@pytest.mark.parametrize("dgp_id", DGP_IDS)
def test_sampling_is_deterministic_given_the_seed(dgp_id: str) -> None:
    first = build_dgp(dgp_id, 9).sample(200, seed=11)
    second = build_dgp(dgp_id, 9).sample(200, seed=11)
    other = build_dgp(dgp_id, 9).sample(200, seed=12)
    assert np.array_equal(first.quantiles, second.quantiles)
    assert np.array_equal(first.treatment, second.treatment)
    assert not np.array_equal(first.quantiles, other.quantiles)


@pytest.mark.parametrize("dgp_id", DGP_IDS)
def test_quadrature_truth_reproduces_the_closed_form_mean(dgp_id: str) -> None:
    """E[q] has a closed form: xi is centred and E[exp(eta)] is lognormal."""

    dgp = build_dgp(dgp_id, 25)
    X = dgp.sample(300, seed=5).X
    z = dgp.grid.base_z
    hermite = (z**2 - 1.0) / 2.0 + (z**3 - 3.0 * z) / 6.0
    for arm in (0, 1):
        outer = dgp.spec.outer(arm)
        expected_scale_factor = np.exp(0.5 * outer.log_scale_sd**2)
        gamma = dgp.spec.shape(X, arm)
        psi = z + gamma[:, None] * hermite
        closed_form = (
            dgp.spec.location(X, arm)[:, None]
            + np.exp(dgp.spec.log_scale(X, arm))[:, None]
            * expected_scale_factor
            * psi
        )
        assert dgp.mean_quantiles(X, arm) == pytest.approx(closed_form, abs=1e-9)


def test_d0_conditional_laws_are_degenerate() -> None:
    """A deterministic regime: each unit's draw is its own conditional mean."""

    dgp = build_dgp("D0", 25)
    sample = dgp.sample(300, seed=7)
    for arm in (0, 1):
        rows = sample.treatment == arm
        truth = dgp.mean_quantiles(sample.X[rows], arm)
        assert sample.quantiles[rows] == pytest.approx(truth, abs=1e-12)


def test_d2_has_a_null_effect_on_every_target() -> None:
    dgp = build_dgp("D2", 25)
    X = dgp.sample(300, seed=7).X
    assert dgp.spec.null_effect
    assert dgp.mean_quantile_contrast(X) == pytest.approx(0.0, abs=1e-12)
    assert dgp.reference_contrast(X) == pytest.approx(0.0, abs=1e-12)
    for name in ("grid_mean", "grid_sd", "grid_skewness", "grid_upper_tail_mean"):
        assert dgp.functional_contrast(X, name) == pytest.approx(0.0, abs=1e-12)


def test_d5_matches_barycenters_while_the_laws_differ() -> None:
    """`MEANQ-A-K` agrees across arms; `LAW-A-K` does not."""

    dgp = build_dgp("D5", 25)
    X = dgp.sample(400, seed=7).X

    assert dgp.mean_quantile_contrast(X) == pytest.approx(0.0, abs=1e-9)

    # A law-level summary the two arms cannot share: the conditional variance
    # of the median coordinate.
    median = dgp.grid.n_grid // 2
    spread = []
    for arm in (0, 1):
        first = dgp.conditional_expectation(X, arm, lambda block: block[:, median])
        second = dgp.conditional_expectation(
            X, arm, lambda block: block[:, median] ** 2
        )
        spread.append(second - first**2)
    assert np.min(np.abs(spread[1] - spread[0])) > 1e-3


def test_d7_hides_its_effect_outside_the_mean() -> None:
    """The location and scale surfaces agree, so only the shape moves."""

    dgp = build_dgp("D7", 49)
    X = dgp.sample(400, seed=7).X

    mean_effect = np.mean(np.abs(dgp.functional_contrast(X, "grid_mean")))
    skewness_effect = np.mean(np.abs(dgp.functional_contrast(X, "grid_skewness")))
    tail_effect = np.mean(np.abs(dgp.functional_contrast(X, "grid_upper_tail_mean")))

    assert mean_effect < 0.02
    assert skewness_effect > 0.3
    assert tail_effect > 0.02
    # The functionals outside a {mean, sd} training manifest carry the effect,
    # by an order of magnitude over the one inside it.
    assert skewness_effect > 10.0 * mean_effect
    assert tail_effect > 2.0 * mean_effect
    # The quantile coordinates still move, so the mandatory common target is
    # informative even though the mean is not.
    assert np.max(np.abs(dgp.mean_quantile_contrast(X))) > 0.05


def test_d6_outer_law_is_multimodal_and_exposes_its_modes() -> None:
    dgp = build_dgp("D6", 25)
    sample = dgp.sample(2000, seed=7)
    for arm in (0, 1):
        assert dgp.location_modes(arm).size == 2
        centres = dgp.conditional_mode_centres(sample.X[:10], arm)
        assert centres.shape == (10, 2, 25)
        # The two centres are far apart relative to the within-mode spread.
        separation = np.abs(centres[:, 1] - centres[:, 0]).min()
        assert separation > 2.0

    unimodal = build_dgp("D1", 25)
    assert unimodal.location_modes(0).size == 1
    assert unimodal.conditional_mode_centres(sample.X[:10], 0).shape == (10, 1, 25)


def test_d3_and_d4_differ_in_which_covariates_drive_the_arms() -> None:
    """D3 rewards separate heads; D4 rewards one shared partition."""

    separate = build_dgp("D3", 25)
    shared = build_dgp("D4", 25)
    base = separate.sample(300, seed=7).X

    def sensitivity(dgp, arm: int, column: int) -> float:
        # Resample the column rather than negate it: several surfaces are even
        # in their covariate, so a sign flip would read as no dependence.
        moved = base.copy()
        moved[:, column] = np.random.default_rng(column).uniform(-1.0, 1.0, base.shape[0])
        reference = dgp.mean_quantiles(base, arm)
        return float(np.max(np.abs(dgp.mean_quantiles(moved, arm) - reference)))

    # D3 arm 0 ignores the covariates arm 1 uses, and conversely.
    assert sensitivity(separate, 0, 0) > 0.1
    assert sensitivity(separate, 0, 3) == pytest.approx(0.0, abs=1e-12)
    assert sensitivity(separate, 1, 3) > 0.1
    assert sensitivity(separate, 1, 0) == pytest.approx(0.0, abs=1e-12)

    # D4 puts every arm's structure on one axis.
    for arm in (0, 1):
        assert sensitivity(shared, arm, 0) > 0.1
        for column in (1, 2, 3, 4):
            assert sensitivity(shared, arm, column) == pytest.approx(0.0, abs=1e-12)


def test_overlap_degrades_from_d1_through_d8_to_d9() -> None:
    fractions = []
    for dgp_id in ("D1", "D8", "D9"):
        scores = build_dgp(dgp_id, 9).sample(4000, seed=7).propensity
        fractions.append(float(np.mean((scores < 0.1) | (scores > 0.9))))
    assert fractions[0] == 0.0
    assert 0.0 < fractions[1] < fractions[2]
    assert fractions[2] > 0.25


def test_moderator_is_pretreatment_and_covers_every_bin() -> None:
    sample = build_dgp("D1", 9).sample(2000, seed=7)
    bins = sample.moderator
    assert np.array_equal(bins, moderator_bins(sample.X))
    assert set(np.unique(bins)) == set(range(N_MODERATOR_BINS))
    assert np.min(np.bincount(bins, minlength=N_MODERATOR_BINS)) > 100


def test_reference_distance_is_nonnegative_and_uses_the_declared_reference() -> None:
    dgp = build_dgp("D1", 25)
    X = dgp.sample(200, seed=7).X
    for arm in (0, 1):
        assert np.all(dgp.reference_distance(X, arm) >= 0.0)
    assert dgp.grid.reference_quantiles() == pytest.approx(dgp.grid.base_z)


def test_tail_probability_is_a_probability() -> None:
    dgp = build_dgp("D1", 25)
    X = dgp.sample(200, seed=7).X
    upper = dgp.grid.n_grid - 1
    for arm in (0, 1):
        probability = dgp.tail_probability(X, arm, level_index=upper, threshold=1.5)
        assert np.all(probability >= 0.0) and np.all(probability <= 1.0)
        assert np.max(probability) > 0.05
    with pytest.raises(ValueError):
        dgp.tail_probability(X, 0, level_index=upper + 1, threshold=0.0)


@pytest.mark.parametrize("dgp_id", DGP_IDS)
def test_tail_probability_matches_a_direct_monte_carlo(dgp_id: str) -> None:
    """The indicator integrand needs the closed-form location, not quadrature.

    Gauss-Hermite has no polynomial exactness against a step function, so the
    generic quadrature path was wrong here by several percent. This pins the
    replacement to a direct simulation of the outer law.
    """

    dgp = build_dgp(dgp_id, 25)
    X = dgp.sample(60, seed=7).X
    upper = dgp.grid.n_grid - 1
    rng = np.random.default_rng(4)
    n_draws = 40_000
    for arm in (0, 1):
        hits = np.zeros(X.shape[0])
        for _ in range(n_draws):
            xi, eta = dgp.spec.outer(arm).sample(rng, X.shape[0])
            hits += dgp._grid_at_latent(X, arm, xi, eta)[:, upper] > 1.5
        empirical = hits / n_draws
        exact = dgp.tail_probability(X, arm, level_index=upper, threshold=1.5)
        # 40k draws give a standard error near 0.0025; 0.01 is a wide margin
        # on a quantity the old path missed by 0.07.
        assert np.max(np.abs(exact - empirical)) < 0.01


def test_location_survival_handles_every_outer_law_shape() -> None:
    cutoff = np.linspace(-3.0, 3.0, 25)

    smooth = OuterLaw(location_sd=0.5).location_survival(cutoff)
    assert np.all(np.diff(smooth) <= 0.0)
    assert smooth[0] == pytest.approx(1.0, abs=1e-6)
    assert smooth[-1] == pytest.approx(0.0, abs=1e-6)

    # A degenerate location makes the survival function a step at zero.
    degenerate = OuterLaw().location_survival(cutoff)
    assert np.array_equal(degenerate, (cutoff < 0.0).astype(float))

    # A mixture crosses one half at the midpoint between the two modes.
    mixture = OuterLaw(location_sd=0.25, mixture_shift=1.5)
    assert float(mixture.location_survival(np.zeros(1))[0]) == pytest.approx(0.5)


def test_quadrature_weights_are_a_probability_rule() -> None:
    for law in (
        OuterLaw(),
        OuterLaw(location_sd=0.35),
        OuterLaw(location_sd=0.35, log_scale_sd=0.2),
        OuterLaw(location_sd=0.25, log_scale_sd=0.15, mixture_shift=1.5),
    ):
        xi, eta, weights = law.quadrature(16)
        assert xi.shape == eta.shape == weights.shape
        assert np.all(weights > -1e-15)
        assert weights.sum() == pytest.approx(1.0)
        # Matches the first two moments of the outer location law.
        assert float(weights @ xi) == pytest.approx(0.0, abs=1e-9)
        expected_variance = law.location_sd**2 + law.mixture_shift**2
        assert float(weights @ (xi * xi)) == pytest.approx(expected_variance, abs=1e-9)


def test_degenerate_outer_law_collapses_the_rule_to_one_node() -> None:
    xi, eta, weights = OuterLaw().quadrature(16)
    assert xi.size == eta.size == weights.size == 1
    assert weights[0] == pytest.approx(1.0)


def test_grid_spec_rejects_a_degenerate_grid() -> None:
    with pytest.raises(ValueError):
        GridSpec(1)
    with pytest.raises(ValueError):
        build_dgp("D1", 25).functional(np.zeros((2, 5)), 0, "not_a_functional")
    with pytest.raises(ValueError):
        build_dgp("DX", 25)
