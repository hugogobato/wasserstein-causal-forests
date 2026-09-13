"""Construct a Nordic-average benchmark quantile vector on the WCF K=25 midpoint grid.

All inputs were downloaded from official sources into this directory:
  - ilc_di01.tsv        Eurostat "Distribution of income by quantiles" (statinfo=TC:
                        quantile threshold values, in EUR / NAC / PPS).
                        TIME_PERIOD is the EU-SILC SURVEY year; the income reference
                        period is the previous calendar year (Eurostat ESMS ilc_sieusilc).
  - demo_pjan_*.json    Eurostat total population, 1 January, geo DK FI SE IS NO.
  - pip_percentiles_2021ppp.csv  World Bank PIP 100-percentile file (2021 PPP $/day).
  - owid_lis_decile_thresholds.csv  OWID/LIS decile thresholds (free cross-check).

Outputs: nordic_qstar_k25_table.csv, nordic_qstar_k25_grid.csv, nordic_benchmark.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
K = 25
U_GRID = (np.arange(K) + 0.5) / K

LEVEL_TO_U = {}
for i in range(1, 6):
    LEVEL_TO_U[f"P{i}"] = i / 100
for i in range(1, 10):
    LEVEL_TO_U[f"D{i}"] = i / 10
for i in range(1, 5):
    LEVEL_TO_U[f"QU{i}"] = i / 5
for i in range(1, 4):
    LEVEL_TO_U[f"Q{i}"] = i / 4
for i in range(95, 100):
    LEVEL_TO_U[f"P{i}"] = i / 100

COUNTRIES = ("DK", "FI", "NO", "SE", "IS")
# Population on 1 January (Eurostat demo_pjan, age=TOTAL, sex=T).
POP = {
    2020: {"DK": 5822763, "FI": 5525292, "SE": 10327589, "IS": 364134, "NO": 5367580},
    2023: {"DK": 5932654, "FI": 5563970, "SE": 10521556, "IS": 387758, "NO": 5488984},
    2024: {"DK": 5961249, "FI": 5603851, "SE": 10551707, "IS": 383567, "NO": 5550203},
    2025: {"DK": 5992734, "FI": 5635971, "SE": 10587710, "IS": 389444, "NO": 5594340},
}
EU2PIP = {"DK": "DNK", "FI": "FIN", "NO": "NOR", "SE": "SWE", "IS": "ISL"}


def load_eurostat_tc(unit: str = "PPS") -> pd.DataFrame:
    rows = []
    with open(HERE / "ilc_di01.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        years = [h.strip() for h in header[1:]]
        for row in reader:
            meta = row[0].split(",")
            if len(meta) < 5:
                continue
            _freq, quant, statinfo, uom, geo = meta[:5]
            if geo not in COUNTRIES or statinfo != "TC" or uom != unit:
                continue
            if quant not in LEVEL_TO_U:
                continue
            for i, year in enumerate(years):
                idx = i + 1
                if idx >= len(row):
                    continue
                raw = row[idx].strip()
                if raw in (":", ""):
                    continue
                flag = ""
                while raw and raw[-1].isalpha():
                    flag = raw[-1] + flag
                    raw = raw[:-1].strip()
                try:
                    value = float(raw)
                except ValueError:
                    continue
                rows.append({"geo": geo, "year": int(year), "u": LEVEL_TO_U[quant],
                             "value": value, "flag": flag})
    return pd.DataFrame(rows)


def collapse_us(us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    agg = pd.DataFrame({"u": us, "v": vs}).groupby("u", as_index=False)["v"].mean()
    return agg.u.to_numpy(), agg.v.to_numpy()


def country_curve(df: pd.DataFrame, geo: str, year: int):
    sub = df[(df.geo == geo) & (df.year == year)].sort_values("u")
    if sub.empty:
        return None
    us, vs = collapse_us(sub.u.to_numpy(), sub.value.to_numpy())
    return PchipInterpolator(us, vs, extrapolate=True)


def barycenter(df, survey_year, countries, weights):
    curves = {}
    for c in countries:
        f = country_curve(df, c, survey_year)
        if f is None:
            raise ValueError(f"no data for {c} survey year {survey_year}")
        curves[c] = f
    w = np.array([weights[c] for c in countries], dtype=float)
    w = w / w.sum()
    fine = np.linspace(0.01, 0.99, 9801)
    q_fine = np.zeros_like(fine)
    for i, c in enumerate(countries):
        q_fine += w[i] * curves[c](fine)
    vec = np.array([np.interp(u, fine, q_fine) for u in U_GRID])
    return vec, curves, q_fine, fine


def summarise(vec):
    w = np.full(K, 1.0 / K)
    mean = float(np.sum(w * vec))
    sd = float(np.sqrt(np.sum(w * (vec - mean) ** 2)))
    q50 = float(np.interp(0.5, U_GRID, vec))
    return {
        "mean": mean, "median": q50, "sd": sd,
        "skew": float(np.sum(w * (vec - mean) ** 3) / sd**3),
        "q10_q50": float(np.interp(0.1, U_GRID, vec) / q50),
        "q90_q50": float(np.interp(0.9, U_GRID, vec) / q50),
        "q98_q50": float(np.interp(0.98, U_GRID, vec) / q50),
    }


def pip_barycenter(survey_year_map, weights):
    df = pd.read_csv(HERE / "pip_percentiles_2021ppp.csv")
    curves, used_weights = {}, {}
    for eu, yr in survey_year_map.items():
        sub = df[(df.country_code == EU2PIP[eu]) & (df.year == yr)
                 & (df.reporting_level == "national") & (df.welfare_type == "income")
                 ].sort_values("percentile")
        sub = sub[sub["quantile"].notna()]
        if sub.empty:
            continue
        us, vs = collapse_us(sub.percentile.to_numpy() / 100.0, sub["quantile"].to_numpy())
        curves[eu] = PchipInterpolator(us, vs, extrapolate=True)
        used_weights[eu] = weights[eu]
    wsum = sum(used_weights.values())
    fine = np.linspace(0.01, 0.99, 9801)
    q = np.zeros_like(fine)
    for eu, f in curves.items():
        q += (used_weights[eu] / wsum) * f(fine)
    return np.array([np.interp(u, fine, q) for u in U_GRID]), curves


def main() -> None:
    tc = load_eurostat_tc("PPS")

    out = {}
    # Default: four mainland Nordics, latest available income year (2024 => survey 2025)
    vec_b, curves_b, fine_b, _ = barycenter(tc, 2025, ("DK", "FI", "NO", "SE"), POP[2025])
    out["B_incY2024_4c_popw"] = vec_b
    out["B_incY2024_4c_simple"] = np.mean([curves_b[c](U_GRID) for c in curves_b], axis=0)

    # Alternative: all five Nordics at the latest common income year (2019 => survey 2020)
    vec_a, curves_a, fine_a, _ = barycenter(tc, 2020, COUNTRIES, POP[2020])
    out["A_incY2019_5c_popw"] = vec_a
    out["A_incY2019_5c_simple"] = np.mean([curves_a[c](U_GRID) for c in curves_a], axis=0)

    # Robustness: income year 2023 (survey 2024), four countries
    vec_c, curves_c, _, _ = barycenter(tc, 2024, ("DK", "FI", "NO", "SE"), POP[2024])
    out["C_incY2023_4c_popw"] = vec_c

    # PIP cross-check (survey year 2023), four countries, population weighted
    vec_pip, _ = pip_barycenter({"DK": 2023, "FI": 2023, "NO": 2023, "SE": 2023}, POP[2024])
    out["PIP_survey2023_4c_popw"] = vec_pip

    grid_df = pd.DataFrame({"u": U_GRID, **out})
    grid_df.to_csv(HERE / "nordic_qstar_k25_grid.csv", index=False)

    print("Nordic benchmark on the WCF K=25 midpoint grid u_k=(k-0.5)/25")
    print(grid_df.round(2).to_string(index=False))

    print("\n--- Summary ---")
    for name, v in out.items():
        s = summarise(v)
        print(f"{name:26s} mean={s['mean']:9.0f} median={s['median']:9.0f} sd={s['sd']:8.0f} "
              f"skew={s['skew']:5.3f} q90/q50={s['q90_q50']:5.3f} q98/q50={s['q98_q50']:5.3f}")

    # exact decile thresholds of the default benchmark (from the fine barycenter)
    print("\nDefault benchmark (incY2024, 4c, pop-wtd), exact quantile values (PPS):")
    for u in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        print(f"  Q({u:.2f}) = {np.interp(u, np.linspace(0.01,0.99,9801), fine_b):9.0f}")

    # Benchmark-choice sensitivity: d_W of a representative country to variants
    w = np.full(K, 1.0 / K)

    def dw(q, ref):
        return float(np.sqrt(np.sum(w * (q - ref) ** 2)))

    print("\nW2 distance (PPS) of each country distribution to the default benchmark:")
    for c in ("DK", "FI", "NO", "SE"):
        print(f"  {c}: {dw(curves_b[c](U_GRID), vec_b):7.0f}")
    print("\nEffect of benchmark variant on distances (DK / FI / NO / SE):")
    variants = {
        "5c incY2019": vec_a,
        "simple avg": out["B_incY2024_4c_simple"],
        "incY2023": vec_c,
    }
    for vn, ref in variants.items():
        ds = [dw(curves_b[c](U_GRID), ref) for c in ("DK", "FI", "NO", "SE")]
        print(f"  {vn:12s}: " + " ".join(f"{d:7.0f}" for d in ds))

    # linear vs PCHIP interpolation sensitivity
    fine = np.linspace(0.01, 0.99, 9801)
    q_lin = np.zeros_like(fine)
    wts = np.array([POP[2025][c] for c in ("DK", "FI", "NO", "SE")], float)
    wts = wts / wts.sum()
    for i, c in enumerate(("DK", "FI", "NO", "SE")):
        sub = tc[(tc.geo == c) & (tc.year == 2025)].sort_values("u")
        agg = sub.groupby("u", as_index=False)["value"].mean()
        q_lin += wts[i] * np.interp(fine, agg.u.to_numpy(), agg.value.to_numpy())
    vec_lin = np.array([np.interp(u, fine, q_lin) for u in U_GRID])
    print(f"\nPCHIP vs linear: max |diff| = {np.max(np.abs(vec_b-vec_lin)):.0f} PPS "
          f"({100*np.max(np.abs(vec_b-vec_lin))/np.interp(0.5,U_GRID,vec_b):.2f}% of median)")
    print(f"W2(PCHIP, linear) = {dw(vec_b, vec_lin):.0f} PPS")

    dec = pd.DataFrame({"u": U_GRID, **out})
    dec.to_csv(HERE / "nordic_qstar_k25_table.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ug = np.linspace(0.01, 0.99, 400)
    for c, f in curves_b.items():
        axes[0].plot(ug, f(ug) / 1000, lw=1, label=c)
    axes[0].plot(U_GRID, vec_b / 1000, "k--o", ms=3, lw=1.5, label="B: 4c pop-wtd")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("1000 PPS (income year 2024)")
    axes[0].set_title("Eurostat ilc_di01 TC, Nordic quantile functions")
    axes[0].legend(fontsize=8)

    q50b = np.interp(0.5, U_GRID, vec_b)
    q50p = np.interp(0.5, U_GRID, vec_pip)
    axes[1].plot(U_GRID, vec_b / q50b, "k--o", ms=3, label="Eurostat 4c (incY2024)")
    axes[1].plot(U_GRID, vec_pip / q50p, "r-s", ms=3, label="PIP 4c (survey 2023)")
    axes[1].plot(U_GRID, vec_a / np.interp(0.5, U_GRID, vec_a), "g-^", ms=3, label="Eurostat 5c (incY2019)")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("quantile / median")
    axes[1].set_title("Normalised benchmark shapes")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "nordic_benchmark.png", dpi=140)
    print("\nSaved:", HERE / "nordic_qstar_k25_table.csv", "and nordic_benchmark.png")


if __name__ == "__main__":
    main()
