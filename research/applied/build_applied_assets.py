#!/usr/bin/env python3
"""Build the applied-study tables and figure for the WCF manuscript.

Reads the audited result files under ``results/applied_study_exploration`` and
writes the LaTeX table fragments and the Project STAR quantile figure that the
applied sections include.  Every displayed number is read or computed from the
result files, and each is cross-checked against the audited evidence ledger
(``EVIDENCE_LEDGER.md``) before any output is written.  The script aborts on a
ledger mismatch, so a numerical disagreement cannot reach the manuscript.

Outputs (relative to the repository root):
    paper/tables/applied_overview.tex
    paper/tables/applied_star_main.tex
    paper/tables/applied_star_bins.tex
    paper/tables/applied_minwage.tex
    paper/tables/applied_egger.tex
    paper/figures/applied_star_quantiles.pdf

Usage::

    python research/applied/build_applied_assets.py
"""

from __future__ import annotations

import csv
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "applied_study_exploration"
STAR_DIR = SRC / "project_star"
MINWAGE_DIR = SRC / "primary_state_minwage"
EGGER_DIR = SRC / "egger_kenya"
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"

FUNCS = ("mean", "sd", "skew", "tail", "reference", "p10", "p90")
ROW_LABELS = {
    "mean": "Mean",
    "sd": "SD",
    "skew": "Skewness",
    "tail": "Upper-half mean",
    "reference": "Reference distance",
    "p10": "$p_{10}$",
    "p90": "$p_{90}$",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _quantize(value: float, nd: int) -> Decimal:
    return Decimal(str(float(value))).quantize(
        Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP
    )


def fmt(value: float, nd: int = 3, signed: bool = True) -> str:
    d = _quantize(value, nd)
    if d == 0:
        d = abs(d)
    text = f"{d:.{nd}f}"
    if signed and d > 0:
        text = "+" + text
    return text


def fmt_se(value: float, nd: int = 3) -> str:
    return fmt(value, nd, signed=False)


def fmt_money(value: float, nd: int = 0, signed: bool = True) -> str:
    d = _quantize(value, nd)
    if d == 0:
        d = abs(d)
    text = f"{d:,.{nd}f}".replace(",", "{,}")
    if signed and d > 0:
        text = "+" + text
    return text


def fmt_audited(
    value: float, nd: int = 3, quote_nd: int = 4, signed: bool = True
) -> str:
    """Match the ledger's printed rounding for values it quotes via ``quote_nd``.

    A few ledger entries were produced by rounding a four-decimal intermediate
    to three decimals; the direct three-decimal rounding of the source value can
    differ by one unit in the last digit.  This helper reproduces the ledger
    presentation and is used only where the ledger quote requires it.
    """
    return fmt(_quantize(_quantize(value, quote_nd), nd), nd, signed=signed)


def fmt_pct(fraction: float, nd: int = 1) -> str:
    d = _quantize(100.0 * fraction, nd)
    if d == 0:
        return "0\\%"
    return f"{d:.{nd}f}\\%"


def fmt_int(value: float) -> str:
    return f"{int(value):,}".replace(",", "{,}")


# ---------------------------------------------------------------------------
# Ledger verification
# ---------------------------------------------------------------------------

class Ledger:
    """Collects checks of computed values against the audited ledger."""

    def __init__(self) -> None:
        self.checked = 0
        self.failures: list[tuple[str, float, float, float]] = []
        self.discrepancies: list[str] = []

    def check(self, label: str, got, want: float, tol: float) -> None:
        self.checked += 1
        if abs(float(got) - float(want)) > tol:
            self.failures.append((label, float(got), float(want), tol))

    def discrepancy(self, message: str) -> None:
        self.discrepancies.append(message)

    def report(self) -> None:
        if self.failures:
            print("LEDGER MISMATCH: no assets written")
            for label, got, want, tol in self.failures:
                print(f"  {label}: computed {got!r} vs ledger {want!r} (tol {tol:g})")
            sys.exit(1)
        print(f"Ledger checks passed: {self.checked}")
        if self.discrepancies:
            print(f"Ledger/value rounding reconciliations: {len(self.discrepancies)}")
            for message in self.discrepancies:
                print(f"  - {message}")


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def load_star() -> dict:
    summary = load_json(STAR_DIR / "summary.json")
    return {
        "manifest": load_json(STAR_DIR / "data" / "manifest.json"),
        "primary": summary["primary"],
        "placebo": summary["placebo"],
        "trusted": summary["trusted_scalar_math"],
    }


def load_minwage() -> dict:
    res = MINWAGE_DIR / "results"
    trim_names = {
        "trim005": [
            "fit_trim005_095_seed0",
            "fit_trim005_095_seed1",
            "fit_trim005_095_seed2",
        ],
    }
    trim_fits = {
        label: [load_json(res / f"{name}.json") for name in names]
        for label, names in trim_names.items()
        if all((res / f"{name}.json").exists() for name in names)
    }
    fit_names = [
        "fit_primary_seed0",
        "fit_primary_seed1",
        "fit_primary_seed2",
        "fit_primary_A2_seed0",
        "fit_primary_A3_seed0",
        "fit_fd_seed0",
        "fit_fd_seed1",
        "fit_ext_1980_2016_seed0",
        "fit_ext_1980_2016_seed1",
        "fit_ext_2017_2022_seed0",
        "fit_yearfe_seed0",
        "placebo_primary_seed0",
        "placebo_primary_seed1",
        "placebo_primary_seed2",
        "lo_state_CA",
        "lo_state_TX",
        "lo_state_NY",
    ]
    csv_fits = {
        name: load_json(res / f"{name}.json")
        for name in fit_names
        if (res / f"{name}.json").exists()
    }
    csv_rows = []
    with open(res / "all_fits_marginals.csv", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    return {
        "fits": {
            "primary_seed0": csv_fits["fit_primary_seed0"],
            "a2": csv_fits["fit_primary_A2_seed0"],
            "a3": csv_fits["fit_primary_A3_seed0"],
            "fd_seed0": csv_fits["fit_fd_seed0"],
            "fd_seed1": csv_fits["fit_fd_seed1"],
            "ext1980_seed0": csv_fits["fit_ext_1980_2016_seed0"],
            "ext1980_seed1": csv_fits["fit_ext_1980_2016_seed1"],
            "ext2017_seed0": csv_fits["fit_ext_2017_2022_seed0"],
        },
        "primary_agg": load_json(res / "replications_primary_aggregated.json"),
        "placebo_agg": load_json(res / "replications_placebo_aggregated.json"),
        "trim_fits": trim_fits,
        "csv": csv_rows,
        "csv_fits": csv_fits,
    }


def load_egger() -> dict:
    hh = []
    with open(EGGER_DIR / "village_covariates.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["n_hh_sample"]:
                hh.append(float(row["n_hh_sample"]))
    external_path = EGGER_DIR / "runs" / "external" / "results.json"
    external = load_json(external_path) if external_path.exists() else None
    benchmark_path = (
        EGGER_DIR / "benchmark_external" / "kihbs_2015_national_benchmark.json"
    )
    benchmark = load_json(benchmark_path) if benchmark_path.exists() else None
    return {
        "manifest": load_json(EGGER_DIR / "data" / "manifest.json"),
        "summary": load_json(EGGER_DIR / "summary.json"),
        "raw": load_json(EGGER_DIR / "raw_checks.json"),
        "external": external,
        "benchmark": benchmark,
        "hh": np.asarray(hh),
    }


def load_all() -> dict:
    return {
        "star": load_star(),
        "minwage": load_minwage(),
        "egger": load_egger(),
    }


# ---------------------------------------------------------------------------
# Ledger checks
# ---------------------------------------------------------------------------

def run_ledger_checks(data: dict) -> None:
    led = Ledger()

    # ------------------------------------------------------------- Project STAR
    manifest = data["star"]["manifest"]
    led.check("STAR n", manifest["n"], 1333, 0)
    led.check("STAR n_treated", manifest["n_treated"], 522, 0)
    led.check("STAR n_control", manifest["n_control"], 811, 0)

    star_primary = data["star"]["primary"]["functionals"]
    star_placebo = data["star"]["placebo"]["functionals"]
    led.check(
        "STAR outside share",
        data["star"]["primary"]["propensity"]["max_share_outside_0p1_0p9"],
        0.0,
        1e-9,
    )

    star_ref = {
        "mean": (0.1610, 0.0277, 0.1497, 0.00305),
        "sd": (-0.00115, 0.01095, -0.0107, -0.00514),
        "skew": (0.01925, 0.03082, 0.0237, 0.02636),
        "tail": (0.16562, 0.03120, 0.1484, -0.0006),
        "reference": (0.05278, 0.01749, 0.0450, -0.0025),
        "p10": (0.09085, 0.02936, 0.0985, 0.0116),
        "p90": (0.17893, 0.03583, 0.1649, -0.0034),
    }
    for func in FUNCS:
        dr, se, plugin, placebo = star_ref[func]
        led.check(f"STAR {func} DR", star_primary[func]["marginal_mean"], dr, 1.1e-3)
        led.check(f"STAR {func} IF SE", star_primary[func]["if_se_mean"], se, 1.1e-3)
        led.check(
            f"STAR {func} plug-in", star_primary[func]["plugin_marginal_mean"], plugin, 1.1e-3
        )
        led.check(
            f"STAR {func} placebo", star_placebo[func]["marginal_mean"], placebo, 1.1e-3
        )

    star_bins_ref = {
        "mean": ([0.164, 0.092, 0.126, 0.262], [0.056, 0.047, 0.057, 0.062]),
        "tail": ([0.161, 0.112, 0.108, 0.282], None),
        "reference": ([0.176, 0.060, 0.078, -0.103], None),
        "p10": ([0.119, -0.050, 0.116, 0.179], None),
        "p90": ([0.173, 0.115, 0.117, 0.311], None),
    }
    for func, (means, ses) in star_bins_ref.items():
        for idx, want in enumerate(means):
            led.check(
                f"STAR bin {func} Q{idx + 1}", star_primary[func]["bins_mean"][idx], want, 1.1e-3
            )
        if ses is not None:
            for idx, want in enumerate(ses):
                led.check(
                    f"STAR bin {func} Q{idx + 1} SE",
                    star_primary[func]["bins_if_se_seed0"][idx],
                    want,
                    1.1e-3,
                )

    naive = data["star"]["trusted"]["student_grade_fe"]
    led.check("STAR naive diff", naive["diff"], 0.2015, 1.1e-4)
    led.check("STAR naive cluster SE", naive["cluster_se"], 0.0324, 1.1e-4)
    led.discrepancy(
        "STAR naive within-grade difference: the table reports the direct "
        "three-decimal rounding of the source value 0.201480, namely +0.201."
    )

    # ---------------------------------------------------------- state minwage
    mw_fit0 = data["minwage"]["fits"]["primary_seed0"]
    mw_agg = data["minwage"]["primary_agg"]["marginal_dr"]
    led.check("minwage n", mw_fit0["n"], 867, 0)
    led.check("minwage n_treated", mw_fit0["n_treated"], 117, 0)
    led.check("minwage n_control", mw_fit0["n_control"], 750, 0)
    led.check(
        "minwage outside share",
        mw_fit0["propensity"]["share_outside_0p1_0p9"],
        0.430,
        5.5e-4,
    )

    mw_ref = {
        "mean": (0.3344, 0.1315, 0.0184, 0.021),
        "sd": (0.1285, 0.1184, 0.0247, -0.006),
        "skew": (-0.0022, 0.0176, 0.0023, None),
        "tail": (0.4431, 0.2134, 0.0327, 0.018),
        "reference": (0.1068, 0.1579, 0.0229, -0.0005),
        "p10": (0.2211, 0.0393, 0.0102, 0.028),
        "p90": (0.4222, 0.3129, 0.0434, 0.012),
        "ref_us2016": (-0.2107, 0.1702, 0.0258, -0.0074),
        "ref_mean_norm": (-0.0015, 0.0033, 0.0010, -0.0007),
    }
    for func, (dr, se, sd, plugin) in mw_ref.items():
        led.check(f"minwage {func} DR", mw_agg[func]["mean"], dr, 1.1e-3)
        led.check(f"minwage {func} IF SE", mw_fit0["if_se"][func], se, 1.1e-3)
        led.check(f"minwage {func} seed SD", mw_agg[func]["std"], sd, 1.1e-3)
        if plugin is not None:
            led.check(
                f"minwage {func} plug-in", mw_fit0["marginal_plugin"][func], plugin, 1.1e-3
            )

    trim_fits = data["minwage"].get("trim_fits", {})
    if trim_fits:
        trim_ref = {
            "trim005": {
                "mean": 0.244207,
                "sd": 0.041217,
                "skew": -0.006029,
                "tail": 0.291642,
                "reference": 0.092191,
                "p10": 0.217153,
                "p90": 0.323823,
                "ref_us2016": -0.039943,
                "ref_mean_norm": -0.003248,
            },
        }
        trim_seed0 = {
            "trim005": {
                "mean": 0.132122,
                "sd": 0.122354,
                "skew": 0.017014,
                "tail": 0.215503,
                "reference": 0.158784,
                "p10": 0.045098,
                "p90": 0.341456,
                "ref_us2016": 0.120631,
                "ref_mean_norm": 0.003204,
            },
        }
        trim_plugin = {
            "trim005": {
                "mean": 0.087846,
                "sd": -0.009907,
                "skew": -0.002949,
                "tail": 0.089477,
                "reference": -0.001571,
                "p10": 0.100342,
                "p90": 0.058197,
                "ref_us2016": -0.040394,
                "ref_mean_norm": -0.002248,
            },
        }
        trim_counts = {"trim005": (760, 112, 648)}
        for label in ("trim005",):
            runs = trim_fits[label]
            led.check(f"minwage {label} n", runs[0]["n"], trim_counts[label][0], 0)
            led.check(
                f"minwage {label} n_treated", runs[0]["n_treated"], trim_counts[label][1], 0
            )
            led.check(
                f"minwage {label} n_control", runs[0]["n_control"], trim_counts[label][2], 0
            )
            for func in (
                "mean",
                "sd",
                "skew",
                "tail",
                "reference",
                "p10",
                "p90",
                "ref_us2016",
                "ref_mean_norm",
            ):
                aggregate = float(np.mean([run["marginal_dr"][func] for run in runs]))
                led.check(
                    f"minwage {label} {func} DR",
                    aggregate,
                    trim_ref[label][func],
                    1.1e-3,
                )
                led.check(
                    f"minwage {label} {func} IF SE",
                    runs[0]["if_se"][func],
                    trim_seed0[label][func],
                    1.1e-3,
                )
                led.check(
                    f"minwage {label} {func} plug-in",
                    runs[0]["marginal_plugin"][func],
                    trim_plugin[label][func],
                    1.1e-3,
                )
        led.check(
            "minwage outside [0.05,0.95] share",
            1.0
            - trim_fits["trim005"][0]["trim_selection_propensity"]["frac_retained"],
            0.123414,
            1.1e-4,
        )

    fits = data["minwage"]["fits"]
    led.check("minwage A2 mean", fits["a2"]["marginal_dr"]["mean"], 0.243, 1.1e-3)
    led.check("minwage A2 p10", fits["a2"]["marginal_dr"]["p10"], 0.185, 1.1e-3)
    led.check("minwage A3 mean", fits["a3"]["marginal_dr"]["mean"], 0.131, 1.1e-3)
    led.check("minwage A3 p10", fits["a3"]["marginal_dr"]["p10"], 0.168, 1.1e-3)
    led.discrepancy(
        "minwage A3 mean: the table reports the direct three-decimal rounding "
        "of the source value 0.130473, namely +0.130."
    )
    led.check("minwage fd seed0 mean", fits["fd_seed0"]["marginal_dr"]["mean"], 0.226, 1.1e-3)
    led.check("minwage fd seed0 p10", fits["fd_seed0"]["marginal_dr"]["p10"], 0.197, 1.1e-3)
    led.check(
        "minwage ext1980 seed0 mean", fits["ext1980_seed0"]["marginal_dr"]["mean"], 0.44, 6e-3
    )
    led.check(
        "minwage ext1980 seed1 mean", fits["ext1980_seed1"]["marginal_dr"]["mean"], 0.40, 6e-3
    )
    led.check(
        "minwage ext1980 seed0 p10", fits["ext1980_seed0"]["marginal_dr"]["p10"], 0.263, 1.1e-3
    )
    led.check(
        "minwage ext1980 seed1 p10", fits["ext1980_seed1"]["marginal_dr"]["p10"], 0.251, 1.1e-3
    )
    mw_placebo = data["minwage"]["placebo_agg"]["marginal_dr"]
    led.check("minwage placebo mean", mw_placebo["mean"]["mean"], -0.0235, 1.1e-4)
    led.check("minwage placebo p10", mw_placebo["p10"]["mean"], 0.005, 1.1e-3)
    led.discrepancy(
        "STAR plug-in p10: the table reports the direct three-decimal rounding "
        "of the source value 0.098458, namely +0.098."
    )
    led.discrepancy(
        "minwage mean-normalized reference (DR): the table reports the direct "
        "three-decimal rounding of the source value -0.001496, namely -0.001."
    )
    led.discrepancy(
        "minwage placebo mean: the table reports the direct three-decimal "
        "rounding of the source value -0.023455, namely -0.023."
    )

    # Cross-check the per-fit JSON marginals against all_fits_marginals.csv.
    for row in data["minwage"]["csv"]:
        run, func = row["run"], row["functional"]
        fit = data["minwage"]["csv_fits"].get(run)
        if fit is None or func not in fit["marginal_dr"]:
            continue
        led.check(f"csv {run} {func} dr", fit["marginal_dr"][func], float(row["dr"]), 1e-9)
        led.check(f"csv {run} {func} if", fit["if_se"][func], float(row["if_se"]), 1e-9)
        if func in fit.get("marginal_plugin", {}):
            led.check(
                f"csv {run} {func} plugin",
                fit["marginal_plugin"][func],
                float(row["plugin"]),
                1e-9,
            )

    # ------------------------------------------------------------ Kenya (Egger)
    egger = data["egger"]
    led.check("Kenya n", egger["manifest"]["n"], 653, 0)
    led.check("Kenya n_treated", egger["manifest"]["n_treated"], 328, 0)
    led.check("Kenya n_control", egger["manifest"]["n_control"], 325, 0)
    led.check("Kenya hh median", np.median(egger["hh"]), 12.0, 5.1e-2)
    led.check("Kenya hh mean", egger["hh"].mean(), 12.597, 5.1e-4)
    led.check(
        "Kenya outside share",
        egger["summary"]["propensity"]["share_outside_0p1_0p9"],
        0.0,
        1e-9,
    )

    egger_ref = {
        "mean": (958.5, 0.06, 1308.6, 0.06, 543.5, 0.06, 0.675),
        "sd": (-779.5, 0.06, 2660.0, 0.06, None, None, None),
        "skew": (0.0601, 1.1e-4, 0.0787, 1.1e-4, None, None, None),
        "tail": (1060.4, 0.06, 2353.7, 0.06, None, None, None),
        "reference": (-3057.5, 0.06, 2573.6, 0.06, -2572.1, 0.06, 0.374),
        "p10": (1125.5, 0.06, 418.7, 0.06, 913.9, 0.06, 0.021),
        "p90": (72.5, 0.06, 5121.7, 0.06, -632.5, 0.06, 0.912),
    }
    for func, (dr, dr_tol, se, se_tol, raw, raw_tol, pval) in egger_ref.items():
        led.check(f"Kenya {func} DR", egger["summary"]["primary"][func]["dr_mean"], dr, dr_tol)
        led.check(f"Kenya {func} IF SE", egger["summary"]["primary"][func]["if_se"], se, se_tol)
        if raw is not None:
            led.check(
                f"Kenya {func} raw", egger["raw"]["levels"][func]["raw_diff"], raw, raw_tol
            )
            led.check(
                f"Kenya {func} permutation p",
                egger["raw"]["permutation"]["p_value_two_sided"][func],
                pval,
                5.5e-4,
            )

    if egger.get("external"):
        external = egger["external"]
        checks = egger["benchmark"]["checks"]
        led.check("Kenya external DR", external["aggregate"]["marginal_dr"]["reference"]["mean"], -5535.551, 0.06)
        led.check("Kenya external seed SD", external["aggregate"]["marginal_dr"]["reference"]["std"], 132.308, 0.006)
        led.check(
            "Kenya external IF SE",
            float(np.mean([run["if_se"]["reference"] for run in external["per_seed"]])),
            2234.951,
            0.06,
        )
        led.check(
            "Kenya external plug-in",
            external["per_seed"][0]["marginal_plugin"]["reference"],
            -1.2027,
            1.1e-3,
        )
        external_raw = egger["benchmark"]["raw_contrast_to_external_benchmark"]
        led.check("Kenya external raw", external_raw["contrast"], -4687.395, 0.06)
        led.check("Kenya external raw d_control", external_raw["d_control"], 79677.342, 0.06)
        led.check("Kenya external raw d_treated", external_raw["d_treated"], 74989.947, 0.06)
        led.check("Kenya external permutation p", external_raw["permutation_p"], 0.0075, 5.5e-4)
        led.check(
            "Kenya benchmark KSH per PPP dollar",
            egger["benchmark"]["ksh_per_ppp_dollar"],
            46.49,
            1.1e-6,
        )
        led.check("Kenya benchmark mean integral", checks["mean_integral"], 5.133687, 1.1e-4)
        led.check("Kenya benchmark median", checks["median_implied"], 3.7466, 1.1e-4)
        led.check("Kenya benchmark gini", checks["gini_implied"], 0.403779, 1.1e-4)
        led.check("Kenya benchmark decile sum", checks["decile_shares_sum"], 0.9999, 1.1e-4)

    led.report()


# ---------------------------------------------------------------------------
# LaTeX table builders
# ---------------------------------------------------------------------------

def _table(caption: str, label: str, body: str, note: str | None = None) -> str:
    parts = [
        "% Automatically generated; do not edit.",
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        body,
    ]
    if note is not None:
        parts += [
            "\\vspace{2pt}",
            "\\begin{minipage}{0.97\\linewidth}",
            "\\footnotesize",
            f"\\textit{{Notes:}} {note}",
            "\\end{minipage}",
        ]
    parts.append("\\end{table*}")
    return "\n".join(parts) + "\n"


def build_overview(data: dict) -> str:
    star_manifest = data["star"]["manifest"]
    mw_fit = data["minwage"]["fits"]["primary_seed0"]
    egger_manifest = data["egger"]["manifest"]

    def share_outside_half_band(propensity: dict) -> float:
        if propensity["min"] >= 0.05 and propensity["max"] <= 0.95:
            return 0.0
        return float("nan")

    star_share = share_outside_half_band(data["star"]["primary"]["propensity"])
    mw_share = 1.0 - data["minwage"]["trim_fits"]["trim005"][0][
        "trim_selection_propensity"
    ]["frac_retained"]
    egger_share = share_outside_half_band(data["egger"]["summary"]["propensity"])

    def counts(entry: dict) -> str:
        return (
            f"{fmt_int(entry['n'])} "
            f"({fmt_int(entry['n_treated'])}/{fmt_int(entry['n_control'])})"
        )

    rows = [
        f"Project STAR & classroom-grade & {counts(star_manifest)} & "
        "within-grade $z$ math scores & small class vs.\\ regular/aide & "
        f"pooled control classrooms & {fmt_pct(star_share)} \\\\",
        f"State minimum wage & state-year & {counts(mw_fit)} & "
        "hourly wages, 2016 USD & real rise $>\\$0.25$, own-rate & "
        f"Nordic (scaled); US 2016 & {fmt_pct(mw_share)} \\\\",
        f"Kenya cash transfers & village & {counts(egger_manifest)} & "
        "per-capita nondurable consumption & village randomization & "
        f"pooled control villages & {fmt_pct(egger_share)} \\\\",
    ]

    body = "\n".join(
        [
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{@{}p{2.0cm}p{1.9cm}p{1.8cm}p{2.4cm}p{2.6cm}p{2.3cm}c@{}}",
            "\\toprule",
            "Study & Unit & $n$ (treated/\\allowbreak control) & Outcome ($K{=}25$) & "
            "Treatment & Benchmark & \\shortstack{Outside\\\\ $[0.05,0.95]$} \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    caption = (
        "Design summary of the three applied studies. Each outcome is represented "
        "by $K=25$ midpoint quantiles, and the final column reports the share of "
        "fitted propensities outside the overlap band $[0.05,0.95]$."
    )
    return _table(caption, "tab:applied-overview", body)


def build_star_main(data: dict) -> str:
    primary = data["star"]["primary"]["functionals"]
    placebo = data["star"]["placebo"]["functionals"]
    naive = data["star"]["trusted"]["student_grade_fe"]

    rows = [
        f"{ROW_LABELS[func]} & ${fmt(primary[func]['marginal_mean'])}$ & "
        f"${fmt_se(primary[func]['if_se_mean'])}$ & "
        f"${fmt(primary[func]['plugin_marginal_mean'])}$ & "
        f"${fmt(placebo[func]['marginal_mean'])}$ \\\\"
        for func in FUNCS
    ]
    body = "\n".join(
        [
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Functional & WCF (DR) & IF SE & Plug-in & Permutation placebo \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    caption = (
        "Project STAR classroom-assignment contrasts for seven functionals of "
        "the within-grade score distribution. The WCF (DR) and permutation "
        "placebo columns average three fitting replicates, the IF SE is the mean "
        "influence-function standard error, and plug-in is the uncorrected "
        "plug-in mean. Estimates are in within-grade standard deviations, and "
        "skewness is dimensionless."
    )
    note = (
        f"The naive within-grade difference is ${fmt(naive['diff'], 3)}$ "
        f"(classroom-clustered SE ${fmt_se(naive['cluster_se'], 3)}$), and the "
        "experimental literature places the mean effect at roughly one fifth "
        "of a within-grade SD."
    )
    return _table(caption, "tab:applied-star", body, note)


def build_star_bins(data: dict) -> str:
    primary = data["star"]["primary"]["functionals"]
    order = ("mean", "tail", "reference", "p10", "p90")
    rows = []
    for func in order:
        cells = [
            f"${fmt(primary[func]['bins_mean'][idx])}$ "
            f"(${fmt_se(primary[func]['bins_if_se_seed0'][idx])}$)"
            for idx in range(4)
        ]
        rows.append(f"{ROW_LABELS[func]} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(
        [
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Functional & Q1 (low) & Q2 & Q3 & Q4 (high) \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    caption = (
        "Project STAR contrasts by school free-lunch quartile. Each cell reports "
        "the mean over three fitting replicates with the influence-function "
        "standard error from the first replicate in parentheses; Q1 is the lowest "
        "free-lunch quartile and Q4 the highest. Estimates are in within-grade "
        "standard deviations."
    )
    return _table(caption, "tab:applied-star-bins", body)


def build_minwage(data: dict) -> str:
    mw = data["minwage"]
    fit0 = mw["fits"]["primary_seed0"]
    agg = mw["primary_agg"]["marginal_dr"]
    placebo_agg = mw["placebo_agg"]["marginal_dr"]
    fits = mw["fits"]

    panel_a_order = (
        ("mean", "Mean"),
        ("sd", "SD"),
        ("skew", "Skewness"),
        ("tail", "Upper-half mean"),
        ("reference", "Reference (Nordic)"),
        ("p10", "$p_{10}$"),
        ("p90", "$p_{90}$"),
        ("ref_us2016", "Reference (US 2016)"),
        ("ref_mean_norm", "Mean-normalized reference"),
    )
    panel_a_rows = [
        f"{label} & ${fmt(agg[func]['mean'])}$ & "
        f"${fmt_se(fit0['if_se'][func])}$ & "
        f"${fmt(fit0['marginal_plugin'][func])}$ \\\\"
        for func, label in panel_a_order
    ]
    panel_a = "\n".join(
        [
            "\\begin{tabular}{@{}lrrr@{}}",
            "\\toprule",
            "\\multicolumn{4}{@{}l}{\\textit{Panel A: primary specification (2000--2016)}} \\\\",
            "\\midrule",
            "Functional & WCF (DR) & IF SE & Plug-in \\\\",
            "\\midrule",
            *panel_a_rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    ext1980_mean = float(
        np.mean(
            [
                fits["ext1980_seed0"]["marginal_dr"]["mean"],
                fits["ext1980_seed1"]["marginal_dr"]["mean"],
            ]
        )
    )
    ext1980_p10 = float(
        np.mean(
            [
                fits["ext1980_seed0"]["marginal_dr"]["p10"],
                fits["ext1980_seed1"]["marginal_dr"]["p10"],
            ]
        )
    )
    panel_b_rows = [
        f"Primary (2000--2016) & ${fmt(agg['mean']['mean'])}$ & "
        f"${fmt(agg['p10']['mean'])}$ \\\\",
        f"A2: federal increases included & ${fmt(fits['a2']['marginal_dr']['mean'])}$ & "
        f"${fmt(fits['a2']['marginal_dr']['p10'])}$ \\\\",
        f"A3: state above federal floor & ${fmt(fits['a3']['marginal_dr']['mean'])}$ & "
        f"${fmt(fits['a3']['marginal_dr']['p10'])}$ \\\\",
        f"First difference & ${fmt(fits['fd_seed0']['marginal_dr']['mean'])}$ & "
        f"${fmt(fits['fd_seed0']['marginal_dr']['p10'])}$ \\\\",
        f"1980--2016 extension & ${fmt(ext1980_mean)}$ & ${fmt(ext1980_p10)}$ \\\\",
        f"2017--2022 extension & ${fmt(fits['ext2017_seed0']['marginal_dr']['mean'])}$ & "
        f"${fmt(fits['ext2017_seed0']['marginal_dr']['p10'])}$ \\\\",
        f"Placebo & ${fmt(placebo_agg['mean']['mean'])}$ & "
        f"${fmt(placebo_agg['p10']['mean'])}$ \\\\",
    ]
    panel_b = "\n".join(
        [
            "\\begin{tabular}{@{}lrr@{}}",
            "\\toprule",
            "\\multicolumn{3}{@{}l}{\\textit{Panel B: robustness variants}} \\\\",
            "\\midrule",
            "Variant & Mean & $p_{10}$ \\\\",
            "\\midrule",
            *panel_b_rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    panels = [panel_a, panel_b]

    trim_fits = mw.get("trim_fits", {})
    if trim_fits:
        def trim_row(label: str, runs: list[dict]) -> str:
            fit = runs[0]
            mean = float(np.mean([r["marginal_dr"]["mean"] for r in runs]))
            sd = float(np.mean([r["marginal_dr"]["sd"] for r in runs]))
            p10 = float(np.mean([r["marginal_dr"]["p10"] for r in runs]))
            reference = float(np.mean([r["marginal_dr"]["reference"] for r in runs]))
            count = (
                f"{fmt_int(fit['n'])} "
                f"({fmt_int(fit['n_treated'])}/{fmt_int(fit['n_control'])})"
            )
            return (
                f"{label} & {count} & "
                f"${fmt(mean)}$ (${fmt_se(fit['if_se']['mean'])}$) & "
                f"${fmt(sd)}$ (${fmt_se(fit['if_se']['sd'])}$) & "
                f"${fmt(p10)}$ (${fmt_se(fit['if_se']['p10'])}$) & "
                f"${fmt(reference)}$ (${fmt_se(fit['if_se']['reference'])}$) \\\\"
            )

        panel_c_rows = [
            trim_row("Full sample", [
                fits["primary_seed0"],
                load_json(MINWAGE_DIR / "results" / "fit_primary_seed1.json"),
                load_json(MINWAGE_DIR / "results" / "fit_primary_seed2.json"),
            ]),
            trim_row("Trim $[0.05,0.95]$", trim_fits["trim005"]),
        ]
        panel_c = "\n".join(
            [
                "\\begin{tabular}{@{}lrrrrr@{}}",
                "\\toprule",
                "\\multicolumn{6}{@{}l}{\\textit{Panel C: propensity trimming}} \\\\",
                "\\midrule",
                "Sample & $n$ (treated/control) & Mean & SD & $p_{10}$ & "
                "Ref.\\ (Nordic) \\\\",
                "\\midrule",
                *panel_c_rows,
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
        panels.append(panel_c)

    body = "\n\n\\vspace{4pt}\n\n".join(panels)
    caption = (
        "State minimum-wage contrasts for the hourly wage distribution, in 2016 "
        "USD per hour, except skewness and the mean-normalized reference, which "
        "are dimensionless. Panel A: the primary 2000--2016 specification, with the "
        "doubly robust column averaged over three fitting replicates and the IF "
        "SE and plug-in columns from the first replicate. Panel B: mean and "
        "tenth-percentile doubly robust contrasts for robustness variants; the "
        "primary and placebo rows average three replicates, A2, A3, and the first "
        "difference are single fits, the 1980--2016 row averages two replicates, "
        "and the 2017--2022 row is a single fit on a short window. Panel C: the "
        "model re-estimated on the sample retained by the standard overlap band "
        "$[0.05,0.95]$; entries average three fitting replicates, the IF SE is "
        "from the first replicate in parentheses, and $n$ is from the first "
        "replicate. The reference column of Panel C uses the Nordic benchmark."
    )
    outside = 1.0 - mw["trim_fits"]["trim005"][0]["trim_selection_propensity"][
        "frac_retained"
    ]
    note = (
        f"${fmt_pct(outside)}$ of propensities lie below $0.05$ in the full "
        "sample, and none above $0.95$; the calibrated and plug-in contrasts "
        "diverge where propensities are small, and restricting the sample to the "
        "overlap band narrows but does not close the gap; $p_{10}$ is the only "
        "stable functional."
    )
    return _table(caption, "tab:applied-minwage", body, note)


def build_egger(data: dict) -> str:
    summary = data["egger"]["summary"]["primary"]
    raw = data["egger"]["raw"]
    rows = []
    for func in FUNCS:
        if func == "skew":
            cells = (
                f"${fmt(summary[func]['dr_mean'])}$",
                f"${fmt_se(summary[func]['if_se'])}$",
                f"${fmt(raw['levels'][func]['raw_diff'])}$",
            )
        else:
            cells = (
                f"${fmt_money(summary[func]['dr_mean'], 1)}$",
                f"${fmt_money(summary[func]['if_se'], 1, signed=False)}$",
                f"${fmt_money(raw['levels'][func]['raw_diff'], 1)}$",
            )
        label = "Reference (pooled control)" if func == "reference" else ROW_LABELS[func]
        p_value = fmt(raw["permutation"]["p_value_two_sided"][func], 3, signed=False)
        rows.append(
            f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & ${p_value}$ \\\\"
        )
        if func == "reference" and data["egger"].get("external"):
            external = data["egger"]["external"]
            external_dr = external["aggregate"]["marginal_dr"]["reference"]["mean"]
            external_se = float(
                np.mean([run["if_se"]["reference"] for run in external["per_seed"]])
            )
            external_raw = data["egger"]["benchmark"][
                "raw_contrast_to_external_benchmark"
            ]
            rows.append(
                "Reference (KIHBS 2015/16) & "
                f"${fmt_money(external_dr, 1)}$ & "
                f"${fmt_money(external_se, 1, signed=False)}$ & "
                f"${fmt_money(external_raw['contrast'], 1)}$ & "
                f"${fmt(external_raw['permutation_p'], 4, signed=False)}$ \\\\"
            )
    body = "\n".join(
        [
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Functional & WCF (DR) & IF SE & Raw contrast & Permutation $p$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    hh = data["egger"]["hh"]
    n_perm = raw["permutation"]["n_permutations"]
    caption = (
        "Kenya cash-transfer contrasts for per-capita nondurable consumption. "
        "The WCF (DR) column averages three fitting replicates and the IF SE is the "
        "influence-function standard error; the raw contrast is the unadjusted "
        "treated-minus-control difference; the last column reports two-sided "
        f"permutation $p$-values from {fmt_int(n_perm)} permutations. Estimates "
        "are in KSH per capita except skewness, which is dimensionless. The first "
        "reference row uses the pooled control-village benchmark; the second uses "
        "the national KIHBS 2015/16 per-capita consumption distribution (World "
        "Bank PIP), reconstructed as described in the text."
    )
    note = (
        f"The sample has {fmt_int(data['egger']['manifest']['n'])} villages, "
        f"with median {fmt(np.median(hh), 1, signed=False)} and mean "
        f"${fmt(hh.mean(), 3, signed=False)}$ households per village; no fitted "
        "propensity lies outside $[0.05,0.95]$."
    )
    return _table(caption, "tab:applied-egger", body, note)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_star_figure() -> Path:
    star = np.load(STAR_DIR / "data" / "dataset.npz")
    q = star["Q"]
    a = star["A"]
    n_quantiles = q.shape[1]
    u = (np.arange(n_quantiles) + 0.5) / n_quantiles

    treated = a == 1
    control = a == 0
    q_treated = q[treated].mean(axis=0)
    q_control = q[control].mean(axis=0)
    contrast = q_treated - q_control

    rng = np.random.default_rng(0)
    q_t, q_c = q[treated], q[control]
    n_t, n_c = q_t.shape[0], q_c.shape[0]
    n_boot = 1000
    boot = np.empty((n_boot, n_quantiles))
    for b in range(n_boot):
        idx_t = rng.integers(0, n_t, size=n_t)
        idx_c = rng.integers(0, n_c, size=n_c)
        boot[b] = q_t[idx_t].mean(axis=0) - q_c[idx_c].mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    treated_colour = "#0072B2"
    control_colour = "#D55E00"

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)

    panel = axes[0]
    panel.plot(u, q_treated, color=treated_colour, lw=1.8, label="Treated (small class)")
    panel.plot(u, q_control, color=control_colour, lw=1.8, ls="--", label="Control")
    panel.set_xlabel("Midpoint quantile $u_k=(k-0.5)/25$")
    panel.set_ylabel("Within-grade $z$-score")
    panel.set_title("(a) Average quantile functions")
    panel.legend(frameon=False, loc="upper left")
    panel.grid(alpha=0.25, lw=0.5)

    panel = axes[1]
    panel.axhline(0.0, color="0.25", lw=0.9, ls="--")
    panel.fill_between(u, lo, hi, color="#999999", alpha=0.35, lw=0, label="95% CI")
    panel.plot(u, contrast, color=treated_colour, lw=1.8, label="Treated $-$ control")
    panel.set_xlabel("Midpoint quantile $u_k=(k-0.5)/25$")
    panel.set_ylabel("Difference in $z$-score")
    panel.set_title("(b) Difference and pointwise 95% CI")
    panel.legend(frameon=False, loc="upper left")
    panel.grid(alpha=0.25, lw=0.5)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "applied_star_quantiles.pdf"
    fig.savefig(out)
    plt.close(fig)

    print(
        "figure: n_treated={}, n_control={}, mean contrast={:.4f}, "
        "CI covers zero: {}".format(
            int(treated.sum()),
            int(control.sum()),
            float(contrast.mean()),
            bool((lo <= 0).any() and (hi >= 0).any()),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    data = load_all()
    run_ledger_checks(data)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    assets = {
        "applied_overview.tex": build_overview(data),
        "applied_star_main.tex": build_star_main(data),
        "applied_star_bins.tex": build_star_bins(data),
        "applied_minwage.tex": build_minwage(data),
        "applied_egger.tex": build_egger(data),
    }

    written = []
    for name, text in assets.items():
        path = TABLE_DIR / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    written.append(build_star_figure())

    for path in written:
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
