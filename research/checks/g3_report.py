#!/usr/bin/env python3
"""WP3-B3: build the tournament tables, figures, and analysis payload.

Run from the repository root, after the merge:

    python research/checks/g3_report.py

Writes `tables/simulation/`, `figures/simulation/`, and
`results/merged/analysis_payload.json`. Every artefact records the merged
table's checksum and the git revision that produced it, so a figure can always
be traced back to the rows behind it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.analysis import (  # noqa: E402
    CLAIMANT,
    compute_gate_flags,
    cost_summary,
    failure_rates,
    load_rows,
    mechanism_ablations,
    method_means,
    paired_comparison,
    worst_regime,
)
from wasserstein_causal_forests.g3.dgps import DGP_IDS  # noqa: E402
from wasserstein_causal_forests.g3.manifest import PRIMARY_LAW_METRIC  # noqa: E402

MERGED = ROOT / "results" / "merged" / "main_results.parquet"
TABLES = ROOT / "tables" / "simulation"
FIGURES = ROOT / "figures" / "simulation"
PAYLOAD = ROOT / "results" / "merged" / "analysis_payload.json"

REPORTED_METRICS = (
    "mean_quantile_rmse",
    "kernel_law_error",
    "arm_energy_risk",
    "tcate_functional_rmse",
    "reference_tcate_rmse",
    "reference_effect_rmse",
    "tail_calibration",
    "mode_coverage",
    "barycenter_rmse",
)
LAW_METHODS = ("cwdb_v1", "cwdb_v0", "wdrft", "causal_drf")


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not MERGED.exists():
        raise SystemExit(f"no merged results at {MERGED}; run the merge first")
    rows = load_rows(MERGED)
    checksum = hashlib.sha256(MERGED.read_bytes()).hexdigest()
    revision = git_revision()
    provenance = {"merged_checksum": checksum, "git_revision": revision}
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ main table
    summary_rows = []
    for grid in sorted({row["grid"] for row in rows}):
        for dgp in sorted({r["dgp"] for r in rows if r["grid"] == grid}):
            for metric in REPORTED_METRICS:
                for method, statistics in method_means(
                    rows, metric, grid=grid, dgp=dgp
                ).items():
                    summary_rows.append(
                        {
                            "grid": grid, "dgp": dgp, "metric": metric,
                            "method": method, **statistics, **provenance,
                        }
                    )
    write_csv(
        TABLES / "method_means.csv",
        summary_rows,
        ["grid", "dgp", "metric", "method", "mean", "standard_error", "n",
         "merged_checksum", "git_revision"],
    )

    # ----------------------------------------------------- paired comparisons
    paired_rows = []
    for comparator in ("causal_drf", "wdrft", "pta_s", "cwdb_v0"):
        for dgp in DGP_IDS:
            for metric in REPORTED_METRICS:
                comparison = paired_comparison(
                    rows, metric, comparator=comparator, grid="main", dgp=dgp
                )
                if comparison is not None:
                    paired_rows.append({**comparison, **provenance})
    write_csv(
        TABLES / "paired_comparisons.csv",
        paired_rows,
        ["grid", "dgp", "metric", "claimant", "comparator", "n_seeds",
         "claimant_mean", "comparator_mean", "paired_mean_difference",
         "paired_standard_error", "seed_win_fraction", "claimant_wins",
         "comparator_wins", "merged_checksum", "git_revision"],
    )

    # -------------------------------------------------- cost and failure table
    costs = cost_summary(rows)
    failures = failure_rates(rows)
    write_csv(
        TABLES / "cost_and_failures.csv",
        [
            {"method": method, **costs.get(method, {}), **failures.get(method, {}),
             **provenance}
            for method in sorted(set(costs) | set(failures))
        ],
        ["method", "median_runtime_seconds", "max_runtime_seconds",
         "median_peak_ram_mb", "n_cells", "n_failed", "failure_rate",
         "merged_checksum", "git_revision"],
    )

    gate_flags = compute_gate_flags(rows)
    ablations = mechanism_ablations(rows)
    worst = {
        metric: worst_regime(rows, metric, comparator="causal_drf")
        for metric in (PRIMARY_LAW_METRIC, "mean_quantile_rmse")
    }

    # ------------------------------------------------------------- figures
    _crossover_figure(rows, FIGURES / "crossover_primary_law_metric.png", provenance)
    _worst_regime_figure(paired_rows, FIGURES / "worst_regime.png", provenance)
    _particle_figure(ablations, FIGURES / "particle_sensitivity.png", provenance)
    _collapse_figure(rows, FIGURES / "mode_coverage_d6.png", provenance)

    PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD.write_text(
        json.dumps(
            {
                **provenance,
                "gate_flags": gate_flags,
                "mechanism_ablations": ablations,
                "worst_regime": worst,
                "cost": costs,
                "failures": failures,
                "n_rows": len(rows),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(json.dumps(gate_flags["summary"], indent=2))
    print(f"wrote tables to {TABLES} and figures to {FIGURES}")
    return 0


def _stamp(figure, provenance: dict) -> None:
    figure.text(
        0.005, 0.005,
        f"merged {provenance['merged_checksum'][:12]}  rev {provenance['git_revision'][:12]}",
        fontsize=5, color="0.45",
    )


def _crossover_figure(rows, path: Path, provenance: dict) -> None:
    """Primary law metric by regime, one line per method: where do they cross?"""

    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    positions = np.arange(len(DGP_IDS))
    for method in LAW_METHODS:
        means, errors = [], []
        for dgp in DGP_IDS:
            statistics = method_means(
                rows, PRIMARY_LAW_METRIC, grid="main", dgp=dgp
            ).get(method)
            means.append(statistics["mean"] if statistics else np.nan)
            errors.append(statistics["standard_error"] if statistics else np.nan)
        axis.errorbar(
            positions, means, yerr=np.array(errors) * 2.0, marker="o",
            capsize=3, label=method, linewidth=1.4, markersize=4,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(DGP_IDS)
    axis.set_ylabel(f"{PRIMARY_LAW_METRIC} (lower is better)")
    axis.set_xlabel("regime")
    axis.set_yscale("log")
    axis.set_title("Primary law metric across regimes, main grid, 2 standard errors")
    axis.legend(fontsize=8, ncol=4)
    axis.grid(alpha=0.25, linewidth=0.5)
    _stamp(figure, provenance)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _worst_regime_figure(paired_rows, path: Path, provenance: dict) -> None:
    """Paired difference against Causal-DRF, sorted worst first."""

    selected = [
        row for row in paired_rows
        if row["comparator"] == "causal_drf" and row["metric"] == PRIMARY_LAW_METRIC
    ]
    if not selected:
        return
    selected.sort(key=lambda r: -r["paired_mean_difference"])
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    positions = np.arange(len(selected))
    values = [row["paired_mean_difference"] for row in selected]
    errors = [2.0 * row["paired_standard_error"] for row in selected]
    colours = ["#c0392b" if v > 0 else "#2471a3" for v in values]
    axis.barh(positions, values, xerr=errors, color=colours, capsize=3, height=0.65)
    axis.axvline(0.0, color="0.3", linewidth=1.0)
    axis.set_yticks(positions)
    axis.set_yticklabels([row["dgp"] for row in selected])
    axis.set_xlabel(
        f"paired {PRIMARY_LAW_METRIC}: {CLAIMANT} minus causal_drf "
        "(negative favours the claimant)"
    )
    axis.set_title("Worst regime first, 2 paired standard errors")
    axis.grid(axis="x", alpha=0.25, linewidth=0.5)
    _stamp(figure, provenance)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _particle_figure(ablations, path: Path, provenance: dict) -> None:
    series = ablations.get("particles", {})
    if not any(series.values()):
        return
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for dgp, entries in series.items():
        if not entries:
            continue
        counts = sorted(int(m) for m in entries)
        means = [entries[str(m)]["mean_excess_energy_risk"] for m in counts]
        errors = [2.0 * (entries[str(m)]["standard_error"] or 0.0) for m in counts]
        axis.errorbar(counts, means, yerr=errors, marker="o", capsize=3, label=dgp)
    axis.set_xlabel("particles M")
    axis.set_ylabel("excess energy risk over the true law")
    axis.set_xscale("log", base=2)
    axis.set_title("Finite-particle approximation, C-WDB-v1")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, linewidth=0.5)
    _stamp(figure, provenance)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _collapse_figure(rows, path: Path, provenance: dict) -> None:
    """The repulsion claim: mode coverage on the multimodal regime."""

    coverage = method_means(rows, "mode_coverage", grid="smallk", dgp="D6")
    if not coverage:
        return
    methods = sorted(coverage)
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    means = [coverage[m]["mean"] for m in methods]
    errors = [2.0 * (coverage[m]["standard_error"] or 0.0) for m in methods]
    axis.bar(methods, means, yerr=errors, capsize=3, color="#2471a3")
    axis.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0)
    axis.text(
        0.02, 0.52, "one mode of two", transform=axis.get_yaxis_transform(),
        fontsize=7, color="0.35",
    )
    axis.set_ylabel("mode coverage on D6")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Proper-score repulsion against collapse, smallk grid")
    axis.tick_params(axis="x", labelrotation=20, labelsize=8)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    _stamp(figure, provenance)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
