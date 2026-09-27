"""Bounded, reproducible numerical checks for the supplement audit.

Run: OPENBLAS_NUM_THREADS=1 python3 research/audits/supplement_20260924/check_math.py
Numerical successes are diagnostic evidence, not proofs of universal claims.
"""
from itertools import product
from pathlib import Path
import json
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import linprog

SEED = 20260924
rng = np.random.default_rng(SEED)
out = {"seed": SEED, "python": platform.python_version(),
       "numpy": np.__version__, "scipy": scipy.__version__}


def dist(p, q, w, eps):
    return np.sqrt(np.sum((p[:, None] - q[None, :]) ** 2 * w, axis=-1) + eps**2) - eps


def energy(p, a, q, b, w, eps):
    return float(2 * a @ dist(p, q, w, eps) @ b
                 - a @ dist(p, p, w, eps) @ a - b @ dist(q, q, w, eps) @ b)


def transport(p, a, q, b, w):
    m, n = len(a), len(b)
    rows = np.zeros((m + n, m * n))
    for i in range(m):
        rows[i, i*n:(i+1)*n] = 1
    for j in range(n):
        rows[m+j, j::n] = 1
    fit = linprog(dist(p, q, w, 0).ravel(), A_eq=rows,
                  b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    assert fit.success
    return fit.fun


# Unequal weights and coincident particles test the factor two and diagonal terms.
w = np.array([0.1, 0.3, 0.6])
p = np.sort(rng.normal(size=(4, 3)), axis=1)
p[1] = p[0]
q = np.sort(rng.normal(size=(1, 3)), axis=1)
eps = 0.17
def score(z):
    return dist(z, q, w, eps).mean() - 0.5 * dist(z, z, w, eps).mean()
r1 = dist(p, q, w, eps) + eps
r2 = dist(p, p, w, eps) + eps
grad = w / len(p) * ((p-q) / r1 - ((p[:, None]-p[None, :])/r2[..., None]).mean(axis=1))
numeric = np.zeros_like(p)
for i, k in product(range(len(p)), range(3)):
    shift = np.zeros_like(p)
    shift[i, k] = 1e-6
    numeric[i, k] = (score(p+shift)-score(p-shift))/(2e-6)
out["gradient_max_abs_error"] = float(np.max(np.abs(grad-numeric)))
assert out["gradient_max_abs_error"] < 1e-8

# Leaf shrinkage solves the two normal equations, including lambda=0.
errors = []
for n0, n1, lam in product([1, 5, 100], [1, 7, 50], [0, 0.5, 500]):
    u0, u1 = rng.normal(size=(2, 5))
    mean = (n0*u0+n1*u1)/(n0+n1)
    neff = n0*n1/(n0+n1)
    contrast = neff/(neff+lam)*(u1-u0)
    fitted = np.array([mean-n1/(n0+n1)*contrast, mean+n0/(n0+n1)*contrast])
    exact = np.linalg.solve([[n0+lam, -lam], [-lam, n1+lam]], [n0*u0, n1*u1])
    errors.append(np.max(np.abs(fitted-exact)))
out["shrinkage_max_abs_error"] = float(max(errors))
assert max(errors) < 1e-10

# Substitute t=u^2 in the Bernstein integral to remove the square-root endpoint.
errors = []
for r, smooth in product([0.01, 0.5, 3], [0, 0.1, 2]):
    value = quad(lambda u: -np.expm1(-r*r*u*u)*np.exp(-smooth*smooth*u*u)
                 / (np.sqrt(np.pi)*u*u), 0, np.inf, epsabs=1e-10)[0]
    errors.append(abs(value-(np.sqrt(r*r+smooth*smooth)-smooth)))
out["bernstein_integral_max_abs_error"] = float(max(errors))
assert max(errors) < 1e-8

# Enumerate all 3^3 empirical clouds: no Monte Carlo error in the expectation.
support = np.array([[-1.], [0.], [2.]])
prob = np.array([0.2, 0.3, 0.5])
errors = []
for smooth in [0, 0.01, 2]:
    lhs = sum(np.prod(prob[list(ix)]) * energy(support[list(ix)], np.ones(3)/3,
              support, prob, np.ones(1), smooth) for ix in product(range(3), repeat=3))
    rhs = float(prob @ dist(support, support, np.ones(1), smooth) @ prob / 3)
    errors.append(abs(lhs-rhs))
out["enumerated_particle_identity_max_abs_error"] = float(max(errors))
assert max(errors) < 1e-12

# Scalar DR remainder, including either nuisance correct and both incorrect.
errors = []
for e, g, mu0, mu1, m0, m1 in product([0.02, 0.5, 0.98], [0.02, 0.5, 0.98],
                                     [-1, 1], [-1, 1], [-1, 2], [-1, 2]):
    mean = m1-m0 + e/g*(mu1-m1) - (1-e)/(1-g)*(mu0-m0)
    bias = (g-e)*((m1-mu1)/g + (m0-mu0)/(1-g))
    errors.append(abs(mean-(mu1-mu0)-bias))
out["dr_remainder_max_abs_error"] = float(max(errors))
assert max(errors) < 1e-11

# Mixture W1 bound checked by independent linear programming, pi=0 and pi=1 included.
slacks = []
for pi, phat in product([0, 0.2, 1], [0, 0.7, 1]):
    plus = np.array([[-2.], [1.]])
    pred = np.array([[-1.], [3.]])
    a, b = np.array([0.4, 0.6]), np.array([0.7, 0.3])
    lhs = transport(np.r_[np.zeros((1,1)), pred], np.r_[1-phat, phat*b],
                    np.r_[np.zeros((1,1)), plus], np.r_[1-pi, pi*a], np.ones(1))
    rhs = 5*abs(phat-pi) + pi*transport(pred, b, plus, a, np.ones(1))
    slacks.append(rhs-lhs)
out["mixture_min_slack"] = float(min(slacks))
assert min(slacks) >= -1e-10

# Analytic tail example: energy -> 0, W1 remains one.
out["tail_escape"] = [{"j": j, "D_eps_1": 2/j**2*(np.sqrt(j*j+1)-1), "W1": 1}
                      for j in [10, 100, 1000, 10000]]
tail_errors = []
for row in out["tail_escape"]:
    j = row["j"]
    direct = energy(np.array([[0., 0., 0.], [float(j)]*3]), np.array([1-1/j, 1/j]),
                    np.zeros((1, 3)), np.ones(1), w, 1.)
    tail_errors.append(abs(direct-row["D_eps_1"]))
out["tail_formula_direct_energy_max_error"] = float(max(tail_errors))
assert max(tail_errors) < 1e-12

# Analytic integrability witness: X uniform (0,1), Q=+/-1/X.
# Conditional first moment 1/x is finite at each x>0; R(P|x)=1/(2x).
out["risk_integrability_witness"] = [
    {"lower_cutoff": t, "integral_R_P_from_cutoff_to_1": -0.5*np.log(t)}
    for t in [1e-2, 1e-4, 1e-8, 1e-16]]

# Constant oracle mean 2, variance 1; X uniform [-1,1]; box kernel L=1/2 on [-1,1].
# Exact scaled numerator variance tends to f * (variance+mean^2) * int L^2=1.25.
# Residual numerator tends to f * variance * int L^2=0.25; ratio variance=1.
out["local_ratio_variance"] = [{"bandwidth": b,
    "raw_centered_numerator_variance_over_b": 1.25-b,
    "residual_numerator_variance_over_b": 0.25,
    "limiting_ratio_variance": 1.0} for b in [0.1, 0.01, 0.001]]

# DGP monotonicity and equal expected scales use exact algebra, checked numerically.
z = np.linspace(-10, 10, 2001)
out["dgp_min_derivative_gamma_085"] = float(np.min(1-0.85+0.85*(z+1)**2/2))
out["S3_equal_scale_max_error"] = float(np.max(np.abs(
    np.exp(0.2*np.linspace(-1, 1, 101)+0.45**2/2)
    - np.exp(0.2*np.linspace(-1, 1, 101))*np.exp(0.45**2/2))))
v = 0.45**2
out["S3_coordinate_variance_gap_at_z_1"] = [
    {"x4": x4, "control_minus_treated": 0.4**2-0.15**2+np.exp(0.4*x4+v)*np.expm1(v)}
    for x4 in [-1, 0, 1]]
assert out["S3_coordinate_variance_gap_at_z_1"][0]["control_minus_treated"] < out["S3_coordinate_variance_gap_at_z_1"][-1]["control_minus_treated"]

target = Path(__file__).with_name("numerical_results.json")
target.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
