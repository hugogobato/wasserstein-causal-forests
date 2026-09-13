"""Exact finite-support checks of identities used in the revised manuscript.

Run from the repository root: python3 report/check_revision_identities.py
These checks illustrate algebra and counterexamples, not asymptotic guarantees.
"""

from fractions import Fraction as F
import json
from pathlib import Path


def score_expectation(e, ehat, m0, m1, fit0, fit1):
    return fit1 - fit0 + e / ehat * (m1 - fit1) - (1-e) / (1-ehat) * (m0-fit0)


def main():
    e, m0, m1 = F(1, 4), F(2), F(3)
    correct_propensity = score_expectation(e, e, m0, m1, F(0), F(0))
    correct_outcomes = score_expectation(e, F(1, 2), m0, m1, m0, m1)
    assert correct_propensity == correct_outcomes == m1-m0
    # A fixed clipping bound changes a genuinely smaller true propensity.
    clipped = score_expectation(F(1, 100), F(2, 100), F(0), F(1), F(0), F(0))
    assert clipped == F(1, 2)
    # Shrinkage must preserve the count-weighted, not unweighted, pooled mean.
    n0, n1, g0, g1, lam = F(8), F(2), F(1), F(5), F(3)
    pooled = (n0*g0+n1*g1)/(n0+n1)
    share = n1/(n0+n1)
    neff = n0*n1/(n0+n1)
    c = neff/(neff+lam)
    gap = g1-g0
    updated0, updated1 = pooled-share*c*gap, pooled+(1-share)*c*gap
    assert (n0*updated0+n1*updated1)/(n0+n1) == pooled
    old_average = (n0*(pooled-c*gap/2)+n1*(pooled+c*gap/2))/(n0+n1)
    assert old_average != pooled
    # A finite particle law CAN contain an exact zero atom. Its probabilities
    # are quantized in units of 1/M; it is not structurally unable to place mass.
    particles = [F(0), F(0), F(1), F(2)]
    zero_mass = F(sum(p == 0 for p in particles), len(particles))
    assert zero_mass == F(1, 2)
    # Expected distance to a reference differs from distance of the mean:
    # q = +/-1 with equal probability, q_star = 0.
    expected_distance = F(1)
    distance_of_mean = F(0)
    # X and Z independent Bernoulli(1/2); Y^0=Z, Y^1=Z+1.
    # Assignment depends strongly on X, which is unrelated to both outcomes.
    cells = [(x, z, F(1, 4)) for x in (0, 1) for z in (0, 1)]
    ep = {0: F(1, 5), 1: F(4, 5)}
    treated_mean = sum(p*ep[x]*(z+1) for x, z, p in cells) / sum(p*ep[x] for x, z, p in cells)
    control_mean = sum(p*(1-ep[x])*z for x, z, p in cells) / sum(p*(1-ep[x]) for x, z, p in cells)
    assert treated_mean-control_mean == F(1)
    result = {
        "arithmetic": "exact rational, Python standard library",
        "aipw_correct_propensity_expectation": str(correct_propensity),
        "aipw_correct_outcomes_expectation": str(correct_outcomes),
        "clipped_propensity_expectation_true_effect_1": str(clipped),
        "weighted_leaf_mean": str(pooled),
        "incorrect_symmetric_leaf_formula_weighted_mean": str(old_average),
        "finite_particle_zero_mass": str(zero_mass),
        "expected_reference_distance": str(expected_distance),
        "distance_of_mean_to_reference": str(distance_of_mean),
        "nonconstant_propensity_unconfounded_raw_difference": str(treated_mean-control_mean),
        "status": "all exact checks passed",
    }
    destination = Path(__file__).with_name("revision_identity_checks.json")
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
