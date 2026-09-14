#!/usr/bin/env python3
"""External normative benchmark for the Kenya cash-transfer study.

Builds the national Kenyan per-capita consumption distribution from the World
Bank PIP summary for the KIHBS 2015/16 survey (decile shares, mean, median,
Gini) and converts it to the units of the study outcome (annual nominal KSH
per capita nondurable consumption).

The reconstruction follows a monotone piecewise-cubic Lorenz curve through the
decile points. The derivative of the Lorenz curve at the median is fixed so
that the implied median matches the reported PIP median; without that
constraint the decile-only curve understates the median by roughly nine
percent. The resulting quantile function is the external benchmark vector.

The KSH-per-PPP-dollar scalar is read from the study microdata: the replication
package's PPP variable is a single constant rescaling of the nominal outcome,
so the conversion is exact and documented, mirroring the Nordic scaling in the
minimum-wage application.

Also computes the unadjusted reference contrast to the benchmark and a
permutation p-value, so the benchmark can be evaluated before any refits.

Usage:
    PYTHONPATH=src python research/applied/egger_kenya/build_external_benchmark.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyreadstat

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/applied_study_exploration" / "egger_kenya"
BENCH = OUT / "benchmark_external"
PKG = Path("/tmp/opencode/wcf_scout/egger/replication_package")
N_LEVELS = 25
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 12_345

LEVELS = (np.arange(N_LEVELS) + 0.5) / N_LEVELS


def load_pip() -> dict:
    with open(BENCH / "pip_kenya_2015_national.json") as handle:
        return json.load(handle)


def study_scalar() -> tuple[float, dict]:
    """KSH per 2017-PPP-dollar implied by the study's own PPP variable."""

    df, _ = pyreadstat.read_dta(
        str(PKG / "code" / "data" / "GE_HHLevel_ECMA.dta"),
        usecols=["nondurables_exp_pc", "nondurables_exp_pc_PPP"],
    )
    block = df.dropna()
    block = block[(block.nondurables_exp_pc > 0) & (block.nondurables_exp_pc_PPP > 0)]
    ratio = (block.nondurables_exp_pc / block.nondurables_exp_pc_PPP).to_numpy()
    scalar = float(np.median(ratio))
    spread = float(np.max(np.abs(ratio - scalar)))
    if spread > 1e-6 * scalar:
        raise SystemExit(f"PPP conversion is not a single scalar (max deviation {spread})")
    return scalar, {"n_households": int(ratio.size), "max_abs_deviation": spread}


def pchip_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Shape-preserving derivative estimates (Fritsch-Carlson)."""

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)
    for index in range(1, len(y) - 1):
        if delta[index - 1] * delta[index] <= 0:
            d[index] = 0.0
        else:
            w1 = 2.0 * h[index] + h[index - 1]
            w2 = h[index] + 2.0 * h[index - 1]
            d[index] = (w1 + w2) / (w1 / delta[index - 1] + w2 / delta[index])
    d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d[0]) != np.sign(delta[0]):
        d[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d[0]) > abs(3.0 * delta[0]):
        d[0] = 3.0 * delta[0]
    d[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(d[-1]) != np.sign(delta[-1]):
        d[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(d[-1]) > abs(3.0 * delta[-1]):
        d[-1] = 3.0 * delta[-1]
    return d


def hermite_eval(x: np.ndarray, y: np.ndarray, d: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate a cubic Hermite spline with knot derivatives d at points u."""

    index = np.clip(np.searchsorted(x, u, side="right") - 1, 0, len(x) - 2)
    h = x[index + 1] - x[index]
    t = (u - x[index]) / h
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (
        h00 * y[index]
        + h10 * h * d[index]
        + h01 * y[index + 1]
        + h11 * h * d[index + 1]
    )


def hermite_derivative(
    x: np.ndarray, y: np.ndarray, d: np.ndarray, u: np.ndarray
) -> np.ndarray:
    index = np.clip(np.searchsorted(x, u, side="right") - 1, 0, len(x) - 2)
    h = x[index + 1] - x[index]
    t = (u - x[index]) / h
    dh00 = 6 * t**2 - 6 * t
    dh10 = 3 * t**2 - 4 * t + 1
    dh01 = -6 * t**2 + 6 * t
    dh11 = 3 * t**2 - 2 * t
    return (
        dh00 * y[index] / h
        + dh10 * d[index]
        + dh01 * y[index + 1] / h
        + dh11 * d[index + 1]
    )


def lorenz_and_quantiles(pip: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    fields = pip["fields"]
    shares = np.array([fields[f"decile{i}"] for i in range(1, 11)])
    mean = float(fields["mean"])
    median = float(fields["median"])
    gini_reported = float(fields["gini"])

    p = np.linspace(0.0, 1.0, 11)
    cumulative = np.concatenate([[0.0], np.cumsum(shares)])
    d = pchip_derivatives(p, cumulative)
    d[5] = median / mean  # impose the reported median at u = 0.5

    fine = np.linspace(0.0, 1.0, 2001)
    derivative = hermite_derivative(p, cumulative, d, fine)
    if np.any(derivative <= 1e-6):
        raise SystemExit("Lorenz reconstruction lost monotonicity")

    quantile_fine = mean * derivative
    quantile = mean * hermite_derivative(p, cumulative, d, LEVELS)
    gini_implied = 1.0 - 2.0 * np.trapezoid(
        hermite_eval(p, cumulative, d, fine), fine
    )
    checks = {
        "mean_integral": float(np.trapezoid(quantile_fine, fine)),
        "median_implied": float(np.interp(0.5, fine, quantile_fine)),
        "median_reported": median,
        "gini_implied": float(gini_implied),
        "gini_reported": gini_reported,
        "decile_shares_sum": float(shares.sum()),
    }
    return quantile, quantile_fine, checks


def study_units() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(OUT / "data" / "dataset.npz")
    return data["Q"], data["A"], data["q_star"]


def raw_reference(g: np.ndarray, a: np.ndarray, q_star: np.ndarray) -> dict:
    w = np.full(g.shape[1], 1.0 / g.shape[1])
    distance = np.sqrt(((g - q_star[None, :]) ** 2) @ w)
    d0 = float(distance[a == 0].mean())
    d1 = float(distance[a == 1].mean())
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.empty(N_PERMUTATIONS)
    for draw in range(N_PERMUTATIONS):
        permuted = rng.permutation(a)
        null[draw] = distance[permuted == 1].mean() - distance[permuted == 0].mean()
    p_value = float((np.abs(null) >= abs(d1 - d0)).mean())
    return {"d_control": d0, "d_treated": d1, "contrast": d1 - d0, "permutation_p": p_value}


def main() -> None:
    pip = load_pip()
    scalar, scalar_info = study_scalar()
    quantile_ppp, fine_ppp, checks = lorenz_and_quantiles(pip)
    annual_factor = 365.0 * scalar
    quantile_ksh = quantile_ppp * annual_factor

    g, a, q_control = study_units()
    results = {
        "benchmark": "Kenya KIHBS 2015/16 national per-capita consumption distribution (World Bank PIP)",
        "source_file": "benchmark_external/pip_kenya_2015_national.json",
        "pip_fields": pip["fields"],
        "ksh_per_ppp_dollar": scalar,
        "ksh_scalar_info": scalar_info,
        "annualisation": "PIP welfare is per capita per day; multiplied by 365 and by the study KSH/PPP scalar.",
        "levels": LEVELS.tolist(),
        "q_star_ppp_per_day": quantile_ppp.tolist(),
        "q_star_ksh_per_year": quantile_ksh.tolist(),
        "fine_levels": np.linspace(0.0, 1.0, 2001).tolist(),
        "fine_ppp_per_day": fine_ppp.tolist(),
        "checks": checks,
        "raw_contrast_to_external_benchmark": raw_reference(g, a, quantile_ksh),
        "raw_contrast_to_control_benchmark": raw_reference(g, a, q_control),
    }
    (BENCH / "kihbs_2015_national_benchmark.json").write_text(
        json.dumps(results, indent=2)
    )
    np.savez(
        BENCH / "kihbs_2015_national_benchmark.npz",
        levels=LEVELS,
        q_star_ppp_per_day=quantile_ppp,
        q_star_ksh_per_year=quantile_ksh,
        fine_levels=np.linspace(0.0, 1.0, 2001),
        fine_ppp_per_day=fine_ppp,
    )

    print("KSH per PPP dollar:", round(scalar, 6))
    print("checks:", json.dumps(checks, indent=2))
    print("benchmark quantiles (KSH/year):")
    for level, value in zip(LEVELS, quantile_ksh):
        print(f"  u={level:.2f}: {value:,.0f}")
    print("raw external contrast:", json.dumps(results["raw_contrast_to_external_benchmark"], indent=2))
    print("raw control contrast:", json.dumps(results["raw_contrast_to_control_benchmark"], indent=2))


if __name__ == "__main__":
    main()
