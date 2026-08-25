"""WP2-C2 published-cell reproduction driver.

Runs `research/baselines/causal_drf_r/reproduction.R`, converts its CSV to the
project's parquet result schema, and applies the reproduction rule fixed before
the run: every reproduced cell mean must sit within a stated relative tolerance
of the published value, judged against the Monte Carlo standard error.

The rule is deliberately two-sided. A reimplementation that is much *better*
than the published method is as much a fidelity failure as one that is worse,
because the tournament would then compare against something the literature does
not contain.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

#: Relative tolerance on a reproduced cell mean against its published value.
REPRODUCTION_TOLERANCE = 0.20

#: Number of Monte Carlo standard errors a gap may span before it counts as a
#: real discrepancy rather than simulation noise.
REPRODUCTION_SE_TOLERANCE = 3.0

#: Witness-function results from Näf, Park, Susmann (2026), Appendix B, Table 3.
CAUSAL_DRF_PUBLISHED = pd.DataFrame(
    {
        "regime": [1, 1, 3, 3, 4, 4],
        "n": [250, 1000, 250, 1000, 250, 1000],
        "published_causal_drf_mae": [0.041, 0.029, 0.065, 0.053, 0.070, 0.053],
        "published_causal_drf_coverage": [1.000, 1.000, 0.970, 0.974, 0.974, 0.968],
        "published_drf_mae": [0.035, 0.027, 0.066, 0.054, 0.072, 0.055],
        "published_drf_coverage": [1.000, 1.000, 0.782, 0.844, 0.776, 0.918],
    }
)

_METHOD_TO_PUBLISHED = {
    "CAUSAL-DRF": ("published_causal_drf_mae", "published_causal_drf_coverage"),
    "W-DRF-T": ("published_drf_mae", "published_drf_coverage"),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def driver_path() -> Path:
    return repository_root() / "research" / "baselines" / "causal_drf_r" / "reproduction.R"


def rscript_executable() -> str | None:
    return shutil.which("Rscript")


def run_reproduction(
    output_csv: str | Path,
    *,
    regimes: tuple[int, ...] = (1, 3, 4),
    sizes: tuple[int, ...] = (250, 1000),
    replications: int = 200,
    trees: int = 2500,
    groups: int = 50,
    workers: int = 6,
    timeout_seconds: float = 21600.0,
) -> pd.DataFrame:
    """Drive the R reproduction and return its replicate-level rows."""

    executable = rscript_executable()
    if executable is None:
        raise RuntimeError("Rscript is not available on PATH")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            executable,
            str(driver_path()),
            "--regimes", ",".join(str(value) for value in regimes),
            "--sizes", ",".join(str(value) for value in sizes),
            "--replications", str(replications),
            "--trees", str(trees),
            "--groups", str(groups),
            "--workers", str(workers),
            "--output", str(output_csv),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=repository_root(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Causal-DRF reproduction failed: {completed.stderr.strip()[-2000:]}"
        )
    return pd.read_csv(output_csv)


def cell_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicate rows to per-cell means with Monte Carlo error."""

    successful = results[results["status"] == "ok"]
    grouped = successful.groupby(["regime", "n", "method"], as_index=False).agg(
        replications=("witness_mae", "size"),
        mae=("witness_mae", "mean"),
        mae_se=("witness_mae", lambda values: values.std(ddof=1) / np.sqrt(len(values))),
        coverage=("band_covers", "mean"),
        rejection_rate=("reject", "mean"),
        band_half_width=("band_half_width", "mean"),
        runtime_seconds=("runtime_seconds", "mean"),
        bandwidth=("bandwidth", "mean"),
    )
    grouped["coverage_se"] = np.sqrt(
        grouped["coverage"] * (1.0 - grouped["coverage"]) / grouped["replications"]
    )

    merged = grouped.merge(CAUSAL_DRF_PUBLISHED, on=["regime", "n"], how="left")
    merged["published_mae"] = [
        row[_METHOD_TO_PUBLISHED[row["method"]][0]] if row["method"] in _METHOD_TO_PUBLISHED else np.nan
        for _, row in merged.iterrows()
    ]
    merged["published_coverage"] = [
        row[_METHOD_TO_PUBLISHED[row["method"]][1]] if row["method"] in _METHOD_TO_PUBLISHED else np.nan
        for _, row in merged.iterrows()
    ]
    merged = merged.drop(columns=list(CAUSAL_DRF_PUBLISHED.columns.difference(["regime", "n"])))

    merged["mae_relative_gap"] = (
        merged["mae"] - merged["published_mae"]
    ) / merged["published_mae"]
    merged["mae_gap_in_se"] = np.where(
        merged["mae_se"] > 0,
        (merged["mae"] - merged["published_mae"]).abs() / merged["mae_se"],
        np.nan,
    )
    merged["mae_within_tolerance"] = (
        merged["mae_relative_gap"].abs() <= REPRODUCTION_TOLERANCE
    ) | (merged["mae_gap_in_se"] <= REPRODUCTION_SE_TOLERANCE)
    return merged.sort_values(["regime", "n", "method"]).reset_index(drop=True)


def summarize_reproduction(results: pd.DataFrame) -> dict[str, object]:
    """Apply the reproduction rule and return the machine-readable verdict."""

    summary = cell_summary(results)
    causal = summary[summary["method"] == "CAUSAL-DRF"]
    separate = summary[summary["method"] == "W-DRF-T"]

    null_cells = causal[causal["regime"] == 1]
    effect_cells = causal[causal["regime"].isin([3, 4])]

    verdict = {
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "reproduction_se_tolerance": REPRODUCTION_SE_TOLERANCE,
        "failed_replications": int((results["status"] != "ok").sum()),
        "cells": json.loads(summary.to_json(orient="records")),
        "causal_drf_mae_within_tolerance": bool(causal["mae_within_tolerance"].all()),
        "separate_mae_within_tolerance": bool(separate["mae_within_tolerance"].all()),
        "worst_causal_drf_relative_gap": float(
            causal["mae_relative_gap"].abs().max()
        ),
        "worst_separate_relative_gap": float(
            separate["mae_relative_gap"].abs().max()
        ),
        # A valid test must not reject under the null regime beyond its level,
        # and must reject when there is an effect.
        "null_rejection_rate": float(null_cells["rejection_rate"].max()),
        "null_coverage": float(null_cells["coverage"].min()),
        "effect_rejection_rate": float(effect_cells["rejection_rate"].min()),
        "effect_coverage": float(effect_cells["coverage"].min()),
    }
    verdict["type_one_error_controlled"] = bool(verdict["null_rejection_rate"] <= 0.05)
    verdict["power_under_effect"] = bool(verdict["effect_rejection_rate"] >= 0.90)
    verdict["coverage_at_or_above_nominal"] = bool(verdict["effect_coverage"] >= 0.95)
    verdict["decision"] = (
        "CAUSAL-DRF-FIDELITY-ESTABLISHED"
        if (
            verdict["causal_drf_mae_within_tolerance"]
            and verdict["type_one_error_controlled"]
            and verdict["power_under_effect"]
            and verdict["coverage_at_or_above_nominal"]
        )
        else "CAUSAL-DRF-FIDELITY-UNRESOLVED"
    )
    return verdict


def write_outputs(results: pd.DataFrame, output: str | Path) -> tuple[Path, Path]:
    """Write the parquet result table and its machine-readable summary."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output, index=False)
    reloaded = pd.read_parquet(output)
    if not results.equals(reloaded):
        raise RuntimeError(f"Parquet round-trip changed the rows in {output}")

    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summarize_reproduction(results), indent=2, sort_keys=True) + "\n"
    )
    return output, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="results/smoke/causal_drf_reproduction.parquet"
    )
    parser.add_argument("--csv", default=None, help="Reuse an existing CSV instead of rerunning R")
    parser.add_argument("--regimes", default="1,3,4")
    parser.add_argument("--sizes", default="250,1000")
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--trees", type=int, default=2500)
    parser.add_argument("--groups", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()

    output = Path(arguments.output)
    if arguments.csv is not None:
        frame = pd.read_csv(arguments.csv)
    else:
        frame = run_reproduction(
            output.with_suffix(".csv"),
            regimes=tuple(int(v) for v in arguments.regimes.split(",")),
            sizes=tuple(int(v) for v in arguments.sizes.split(",")),
            replications=arguments.replications,
            trees=arguments.trees,
            groups=arguments.groups,
            workers=arguments.workers,
        )

    parquet_path, summary_path = write_outputs(frame, output)
    summary = json.loads(summary_path.read_text())
    print(f"wrote {len(frame)} rows to {parquet_path}")
    print(f"decision: {summary['decision']}")
    print(pd.DataFrame(summary["cells"])[
        ["regime", "n", "method", "mae", "mae_se", "published_mae",
         "mae_relative_gap", "coverage", "published_coverage", "rejection_rate"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
