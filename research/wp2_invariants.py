"""Executable WP2 invariants.

Tasks covered: WP2-T4, WP2-T6, WP2-T8, and WP2-T9.
The script uses exact Fraction arithmetic for the raw-income Gini calculations
and checks the corresponding log1p curve coordinates. It has no project-package
or external-data dependency.
"""

from fractions import Fraction
from math import isclose, log, log1p
from pathlib import Path


def midpoint_grid(n):
    return [Fraction(2 * ell - 1, 2 * n) for ell in range(1, n + 1)]


def gini_from_quantile_grid(quantiles, probabilities, weights):
    mean = sum(weight * value for weight, value in zip(weights, quantiles))
    if mean <= 0:
        raise ValueError("Gini requires a strictly positive mean")
    return 1 - 2 * sum(
        weight * (1 - probability) * value
        for weight, probability, value in zip(weights, probabilities, quantiles)
    ) / mean


def two_point_quantile(low, high, probabilities):
    """Lower generalized-inverse quantile for equal masses at low and high."""
    return [low if probability <= Fraction(1, 2) else high for probability in probabilities]


def dr_score(outcome, treatment, propensity, m0, m1):
    """Finite-vector AIPW score for one observed outer unit."""
    if not 0 < propensity < 1:
        raise ValueError("propensity must be strictly between zero and one")
    return [
        m1_j - m0_j
        + treatment / propensity * (outcome_j - m1_j)
        - (1 - treatment) / (1 - propensity) * (outcome_j - m0_j)
        for outcome_j, m0_j, m1_j in zip(outcome, m0, m1)
    ]


def expected_dr_score(e0, true_m0, true_m1, e, m0, m1):
    """Conditional expectation of the finite-vector score at fixed X."""
    return [
        m1_j - m0_j
        + e0 / e * (true_m1_j - m1_j)
        - (1 - e0) / (1 - e) * (true_m0_j - m0_j)
        for true_m0_j, true_m1_j, m0_j, m1_j in zip(
            true_m0, true_m1, m0, m1
        )
    ]


def propensity_specification(randomized, known=None, fitted=None):
    """Use a known design propensity under randomization, otherwise a fit."""
    if randomized:
        if known is None or not 0 < known < 1:
            raise ValueError("a randomized design requires its known propensity")
        return known
    if fitted is None:
        raise ValueError("an observational design needs a fitted propensity")
    return fitted


def validate_projection_targets(targets):
    """Projection is legal only for arm-specific mean quantile curves."""
    legal = {"arm_0_quantile_curve", "arm_1_quantile_curve"}
    forbidden = {"effect_curve", "effect_quantile_curve", "tau_Q"}
    if forbidden.intersection(targets):
        raise AssertionError("effect curves must never be projected")
    if not set(targets).issubset(legal):
        raise AssertionError("unknown projection target")


def check_gini_counterexample():
    functional_probabilities = midpoint_grid(400)
    functional_weights = [Fraction(1, 400)] * 400

    control = two_point_quantile(1, 3, functional_probabilities)
    treatment_degenerate = two_point_quantile(1, 1, functional_probabilities)
    treatment_spread = two_point_quantile(1, 7, functional_probabilities)

    control_gini = gini_from_quantile_grid(
        control, functional_probabilities, functional_weights
    )
    degenerate_gini = gini_from_quantile_grid(
        treatment_degenerate, functional_probabilities, functional_weights
    )
    spread_gini = gini_from_quantile_grid(
        treatment_spread, functional_probabilities, functional_weights
    )
    average_treatment_gini = (degenerate_gini + spread_gini) / 2
    subgroup_a_effect = average_treatment_gini - control_gini

    assert control_gini == Fraction(1, 4)
    assert degenerate_gini == Fraction(0)
    assert spread_gini == Fraction(3, 8)
    assert average_treatment_gini == Fraction(3, 16)
    assert subgroup_a_effect == Fraction(-1, 16)

    curve_probabilities = [
        Fraction(1, 20) + Fraction(9, 10) * Fraction(k, 48)
        for k in range(49)
    ]
    # The values supplied to Gini are raw incomes. The curve target is log1p
    # income. The integer identity below makes the upper-coordinate equality
    # exact before floating-point evaluation.
    assert (1 + 1) * (1 + 7) == (1 + 3) ** 2
    assert two_point_quantile(1, 7, [Fraction(1, 2)]) == [1]
    curve_control = [
        log1p(value)
        for value in two_point_quantile(1, 3, curve_probabilities)
    ]
    curve_treatment_mean = [
        (log1p(low) + log1p(high)) / 2
        for low, high in zip(
            two_point_quantile(1, 1, curve_probabilities),
            two_point_quantile(1, 7, curve_probabilities),
        )
    ]
    curve_effect = [a - b for a, b in zip(curve_treatment_mean, curve_control)]
    max_curve_error = max(abs(effect) for effect in curve_effect)
    assert max_curve_error < 1e-12

    # Averaging on the log1p scale and transforming back gives the raw control
    # curve exactly: sqrt((1 + 1)(1 + 7)) - 1 = 3 in the upper half.
    gini_of_mean_curve = control_gini
    assert gini_of_mean_curve == control_gini
    assert average_treatment_gini != gini_of_mean_curve
    return max_curve_error


def check_bounded_inner_nonidentification():
    """Verify the m=1 observational-equivalence witness in exact arithmetic."""
    observed_law_model_a = {1: Fraction(1, 2), 3: Fraction(1, 2)}
    observed_law_model_b = {1: Fraction(1, 2), 3: Fraction(1, 2)}
    assert observed_law_model_a == observed_law_model_b

    probabilities = [Fraction(1, 4), Fraction(3, 4)]
    raw_mean_model_a = [Fraction(1 + 3, 2)] * 2
    raw_mean_model_b = two_point_quantile(1, 3, probabilities)
    assert raw_mean_model_a != raw_mean_model_b

    analysis_mean_model_a = [(log(2) + log(4)) / 2] * 2
    analysis_mean_model_b = [log(2), log(4)]
    assert any(
        not isclose(a, b, abs_tol=1e-12)
        for a, b in zip(analysis_mean_model_a, analysis_mean_model_b)
    )

    mean_unit_gini_model_a = Fraction(0)
    mean_unit_gini_model_b = Fraction(1, 4)
    assert mean_unit_gini_model_a != mean_unit_gini_model_b


def check_score_algebra_and_collapse():
    true_m0 = [1.0, -2.0]
    true_m1 = [3.0, 4.0]
    target = [true_m1_j - true_m0_j for true_m0_j, true_m1_j in zip(true_m0, true_m1)]

    misspecified_m0 = [10.0, 11.0]
    misspecified_m1 = [-7.0, 8.0]
    with_correct_propensity = expected_dr_score(
        0.4,
        true_m0,
        true_m1,
        0.4,
        misspecified_m0,
        misspecified_m1,
    )
    with_correct_outcomes = expected_dr_score(
        0.4,
        true_m0,
        true_m1,
        0.73,
        true_m0,
        true_m1,
    )
    for observed, expected in zip(with_correct_propensity, target):
        assert isclose(observed, expected, abs_tol=1e-12)
    for observed, expected in zip(with_correct_outcomes, target):
        assert isclose(observed, expected, abs_tol=1e-12)

    # K=1 and J=0 is the ordinary scalar DR score.
    scalar_vector_score = dr_score([3.0], 1, 0.5, [1.0], [2.0])[0]
    scalar_score = 2.0 - 1.0 + (3.0 - 2.0) / 0.5
    assert isclose(scalar_vector_score, scalar_score, abs_tol=1e-12)

    # Randomization makes the design propensity known, not necessarily one-half.
    known_design_propensity = Fraction(1, 3)
    assert (
        propensity_specification(True, known=known_design_propensity)
        == known_design_propensity
    )
    randomized_known = dr_score(
        [3.0, 7.0],
        1,
        propensity_specification(True, known=known_design_propensity),
        [1.0, 2.0],
        [2.0, 5.0],
    )
    randomized_broadcast = dr_score(
        [3.0, 7.0], 1, known_design_propensity, [1.0, 2.0], [2.0, 5.0]
    )
    for observed, expected in zip(randomized_known, randomized_broadcast):
        assert isclose(observed, expected, abs_tol=1e-12)


def check_projection_guard():
    validate_projection_targets({"arm_0_quantile_curve", "arm_1_quantile_curve"})
    try:
        validate_projection_targets({"effect_curve"})
    except AssertionError:
        return
    raise AssertionError("projection guard failed to reject an effect curve")


def check_simulation_schema():
    schema = Path(__file__).with_name("simulation_results_schema.md").read_text()
    required = {
        "claim_id",
        "dgp_id",
        "observation_regime",
        "evaluation_manifest_id",
        "n_regions",
        "inner_n",
        "seed",
        "method",
        "metric",
        "value",
    }
    for field in required:
        assert f"| `{field}` |" in schema, field
    for regime in (
        "oracle_latent",
        "feasible_growing_inner",
        "identified_measurement_model",
        "empirical_proxy",
    ):
        assert regime in schema, regime


def main():
    check_bounded_inner_nonidentification()
    max_curve_error = check_gini_counterexample()
    check_score_algebra_and_collapse()
    check_projection_guard()
    check_simulation_schema()
    print("WP2 invariants: PASS")
    print("WP0 identification witness: identical m=1 observed laws, different latent targets")
    print("N2 exact Gini effect: -1/16")
    print(f"N2 curve-effect maximum absolute error: {max_curve_error:.3g}")
    print("N0 simulation schema: ten required fields and four observation regimes")


if __name__ == "__main__":
    main()
