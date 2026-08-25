"""Reproducible G2 smoke artifacts for the PTA-BCF phase.

Subcommands:

* `published`  summarize the McJames et al. reproduction produced by the R
  bridge and apply the preregistered reproduction rule;
* `scaling`    measure the forced-shared sampler cost at D = K + J + 1 in
  {2, 4, 8}, because its local updates use dense D x D inverses;
* `crossover`  run the D3/D4/null crossover cells that decide whether a joint
  partially pooled sampler is worth writing.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import dgps
from .mvbcf import (
    MVBCFBudget,
    MVBCFForcedShared,
    bridge_path,
    bridge_available,
    repository_root,
    rscript_executable,
)

OBSERVATION_REGIME = "ORACLE-V1"

# --- WP2-B2 published-cell reproduction ------------------------------------

PUBLISHED_MANIFEST_ID = "PTA-F-PUBLISHED-CELL-v1"
#: McJames, Parnell, Goh and O'Shea, arXiv:2303.04874, Table 1, homogeneous
#: treatment-effect column, averaged over the two reported outcomes.
PUBLISHED_MVBCF = {"rmse_mu": 1.58, "pehe_tau": 0.34, "rmse_y": 3.97}
PUBLISHED_BCF = {"rmse_mu": 1.80, "pehe_tau": 0.405, "rmse_y": 4.07}
#: Preregistered reproduction band. It is wider than Monte Carlo error alone
#: because the licensed package accepts a single shared treatment vector while
#: the published cell assigned one treatment per outcome.
REPRODUCTION_BAND = 0.30

# --- WP2-B2 dimension scaling ----------------------------------------------

SCALING_MANIFEST_ID = "PTA-F-DIMENSION-SCALING-v1"
SCALING_DIMENSIONS = (2, 4, 8)
#: Viability rule from the plan: no smoke cell may exceed thirty minutes.
SCALING_RUNTIME_LIMIT_SECONDS = 1800.0

# --- WP2-B3 crossover -------------------------------------------------------

CROSSOVER_MANIFEST_ID = "PTA-CROSSOVER-v1"
#: The diagnostic must beat the losing endpoint in each favorable regime by at
#: least this relative margin, and may lose at most this much under the null.
CROSSOVER_WIN_MARGIN = 0.02
CROSSOVER_NULL_TOLERANCE = 0.10


def _peak_ram_mb() -> float:
    # Linux reports ru_maxrss in KiB for the process and its reaped children.
    usage_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    usage_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return float(max(usage_self, usage_children) / 1024.0)


def _write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    reloaded = pd.read_parquet(output)
    if not frame.equals(reloaded):
        raise RuntimeError(f"Parquet round-trip changed the rows in {output}")
    return output


# ---------------------------------------------------------------------------
# WP2-B2: published-cell reproduction
# ---------------------------------------------------------------------------


def run_published_cell(
    output_csv: str | Path,
    *,
    replicates: int = 30,
    workers: int = 6,
    timeout_seconds: float = 7200.0,
) -> pd.DataFrame:
    """Drive the R reproduction of the published homogeneous-effect cell."""

    executable = rscript_executable()
    if executable is None:
        raise RuntimeError("Rscript is not available on PATH")
    completed = subprocess.run(
        [
            executable,
            str(bridge_path()),
            "reproduce",
            str(output_csv),
            str(replicates),
            str(workers),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=repository_root(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"published-cell reproduction failed: {completed.stderr.strip()[-2000:]}"
        )
    return pd.read_csv(output_csv)


def summarize_published_cell(results: pd.DataFrame) -> dict[str, object]:
    """Apply the preregistered reproduction rule to the replicate metrics."""

    metrics = ["rmse_mu", "pehe_tau", "rmse_y"]
    failed = int((results["method"] == "FAILED").sum())
    usable = results[results["method"] != "FAILED"]
    observed = usable.groupby("method")[metrics].mean()
    if "MVBCF" not in observed.index:
        return {"decision": "INDETERMINATE", "reason": "no successful MVBCF replicate"}

    mvbcf = observed.loc["MVBCF"]
    within_band = {
        metric: bool(
            abs(float(mvbcf[metric]) - PUBLISHED_MVBCF[metric])
            <= REPRODUCTION_BAND * PUBLISHED_MVBCF[metric]
        )
        for metric in metrics
    }
    comparison: dict[str, object] = {}
    ordering_holds = None
    if "BCF-univariate" in observed.index:
        univariate = observed.loc["BCF-univariate"]
        comparison = {
            "bcf_univariate": {
                metric: float(univariate[metric]) for metric in metrics
            }
        }
        ordering_holds = bool(
            float(mvbcf["rmse_mu"]) < float(univariate["rmse_mu"])
            and float(mvbcf["pehe_tau"]) < float(univariate["pehe_tau"])
        )

    passed = all(within_band.values()) and (ordering_holds is not False)
    return {
        "decision": "REPRODUCED" if passed else "NOT-REPRODUCED",
        "evaluation_manifest_id": PUBLISHED_MANIFEST_ID,
        "replicates": int(usable["seed"].nunique()),
        "failed_replicates": failed,
        "reproduction_band": REPRODUCTION_BAND,
        "published_mvbcf": PUBLISHED_MVBCF,
        "published_bcf_univariate": PUBLISHED_BCF,
        "observed_mvbcf": {metric: float(mvbcf[metric]) for metric in metrics},
        "within_band": within_band,
        "published_ordering_reproduced": ordering_holds,
        **comparison,
    }


# ---------------------------------------------------------------------------
# WP2-B2: dimension scaling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingConfiguration:
    n_train: int = 500
    n_test: int = 500
    n_iter: int = 1000
    n_burn: int = 500
    n_tree: int = 50
    n_tree_tau: int = 20
    regime: str = "shared"


#: (K, J, reference) choices giving D = K + J + 1 at each benchmarked width.
SCALING_LAYOUTS = {
    2: (1, ()),
    4: (2, ("grid_sd",)),
    8: (5, ("grid_mean", "grid_sd")),
}


def run_dimension_scaling(
    *,
    dimensions: tuple[int, ...] = SCALING_DIMENSIONS,
    seeds: tuple[int, ...] = (0, 1, 2),
    configuration: ScalingConfiguration = ScalingConfiguration(),
) -> pd.DataFrame:
    """Measure forced-shared sampler cost as the target dimension grows."""

    rows: list[dict[str, object]] = []
    budget = MVBCFBudget(
        n_iter=configuration.n_iter,
        n_burn=configuration.n_burn,
        n_tree=configuration.n_tree,
        n_tree_tau=configuration.n_tree_tau,
    )
    hyperparameters = json.dumps(
        {**asdict(configuration), **asdict(budget)}, sort_keys=True
    )
    for dimension in dimensions:
        if dimension not in SCALING_LAYOUTS:
            raise ValueError(f"no declared layout for D={dimension}")
        n_grid, functionals = SCALING_LAYOUTS[dimension]
        manifest = dgps.pta_manifest(
            n_grid, functionals=functionals, with_reference=True
        )
        if manifest.dimension != dimension:
            raise ValueError(
                f"layout for D={dimension} builds {manifest.dimension} coordinates"
            )
        for seed in seeds:
            train = dgps.sample_dataset(
                configuration.n_train, configuration.regime, seed, n_grid=n_grid
            )
            test = dgps.sample_dataset(
                configuration.n_test,
                configuration.regime,
                1000 + seed,
                n_grid=n_grid,
            )
            targets = manifest.build(train["quantiles"])
            X_control = np.column_stack([train["X"], train["propensity"]])
            X_control_test = np.column_stack([test["X"], test["propensity"]])

            status = "ok"
            failure_reason = ""
            runtime = float("nan")
            sampler_seconds = float("nan")
            contrast_rmse = float("nan")
            try:
                started = time.perf_counter()
                result = MVBCFForcedShared(
                    budget=budget, random_state=seed + 1
                ).fit_predict(
                    X_control,
                    targets,
                    train["treatment"],
                    train["X"],
                    X_control_test=X_control_test,
                    X_moderator_test=test["X"],
                )
                runtime = time.perf_counter() - started
                sampler_seconds = float(result.meta["elapsed_seconds"])
                truth = dgps.true_target_contrast(
                    test["X"], configuration.regime, manifest
                )
                contrast_rmse = float(
                    np.sqrt(np.mean((result.contrast_mean("test") - truth) ** 2))
                )
            except Exception as error:  # Preserve failed cells in the artifact.
                status = "failed"
                failure_reason = f"{type(error).__name__}: {error}"

            rows.append(
                {
                    "claim_id": "WP2-B2",
                    "dgp": f"D-{configuration.regime}",
                    "observation_regime": OBSERVATION_REGIME,
                    "evaluation_manifest_id": SCALING_MANIFEST_ID,
                    "target_id": "MEANQ-A-K",
                    "n": configuration.n_train,
                    "n_test": configuration.n_test,
                    "K": n_grid,
                    "J": len(functionals),
                    "D": dimension,
                    "M": 0,
                    "seed": seed,
                    "method": "PTA-F",
                    "hyperparameter_manifest_id": SCALING_MANIFEST_ID,
                    "hyperparameters": hyperparameters,
                    "metric": "contrast_rmse",
                    "value": contrast_rmse,
                    "runtime_seconds": runtime,
                    "sampler_seconds": sampler_seconds,
                    "peak_ram_mb": _peak_ram_mb(),
                    "n_draws": budget.n_draws,
                    "status": status,
                    "failure_reason": failure_reason,
                }
            )
    return pd.DataFrame(rows)


def summarize_dimension_scaling(results: pd.DataFrame) -> dict[str, object]:
    """Report the measured cost curve against the preregistered viability rule."""

    successful = results[results["status"] == "ok"]
    if successful.empty:
        return {"decision": "INDETERMINATE", "reason": "every scaling cell failed"}
    by_dimension = successful.groupby("D").agg(
        mean_runtime_seconds=("runtime_seconds", "mean"),
        max_runtime_seconds=("runtime_seconds", "max"),
        mean_peak_ram_mb=("peak_ram_mb", "mean"),
        mean_contrast_rmse=("value", "mean"),
        cells=("seed", "count"),
    )
    slowest = float(by_dimension["max_runtime_seconds"].max())
    dimensions = sorted(by_dimension.index.tolist())
    growth = {}
    for smaller, larger in zip(dimensions, dimensions[1:], strict=False):
        growth[f"{larger}_over_{smaller}"] = float(
            by_dimension.loc[larger, "mean_runtime_seconds"]
            / by_dimension.loc[smaller, "mean_runtime_seconds"]
        )
    return {
        "decision": (
            "VIABLE" if slowest <= SCALING_RUNTIME_LIMIT_SECONDS else "PROHIBITIVE"
        ),
        "evaluation_manifest_id": SCALING_MANIFEST_ID,
        "runtime_limit_seconds": SCALING_RUNTIME_LIMIT_SECONDS,
        "slowest_cell_seconds": slowest,
        "failed_cells": int((results["status"] != "ok").sum()),
        "by_dimension": {
            int(dimension): {
                key: float(value)
                for key, value in by_dimension.loc[dimension].to_dict().items()
            }
            for dimension in dimensions
        },
        "runtime_growth_ratios": growth,
    }


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------


def _emit(summary: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    published = subparsers.add_parser("published")
    published.add_argument("--output", default="results/smoke/mvbcf_published_cell.csv")
    published.add_argument("--replicates", type=int, default=30)
    published.add_argument("--workers", type=int, default=6)
    published.add_argument("--reuse", action="store_true")

    scaling = subparsers.add_parser("scaling")
    scaling.add_argument(
        "--output", default="results/smoke/mvbcf_dimension_scaling.parquet"
    )
    scaling.add_argument("--seeds", type=int, default=3)

    crossover = subparsers.add_parser("crossover")
    crossover.add_argument(
        "--output", default="results/smoke/pta_diagnostic_crossover.parquet"
    )
    crossover.add_argument("--seeds", type=int, default=5)

    arguments = parser.parse_args()

    if arguments.command == "published":
        path = Path(arguments.output)
        if arguments.reuse and path.exists():
            frame = pd.read_csv(path)
        else:
            frame = run_published_cell(
                path, replicates=arguments.replicates, workers=arguments.workers
            )
        summary = summarize_published_cell(frame)
        _write_parquet(frame, path.with_suffix(".parquet"))
        _emit(summary, path.with_suffix(".summary.json"))
        return

    if arguments.command == "scaling":
        if not bridge_available():
            raise SystemExit("the pinned mvbcf R bridge is unavailable")
        frame = run_dimension_scaling(seeds=tuple(range(arguments.seeds)))
        output = _write_parquet(frame, arguments.output)
        _emit(summarize_dimension_scaling(frame), output.with_suffix(".summary.json"))
        return

    if arguments.command == "crossover":
        from .diagnostic_partial import run_crossover, summarize_crossover

        if not bridge_available():
            raise SystemExit("the pinned mvbcf R bridge is unavailable")
        frame = run_crossover(seeds=tuple(range(arguments.seeds)))
        output = _write_parquet(frame, arguments.output)
        _emit(summarize_crossover(frame), output.with_suffix(".summary.json"))
        return


if __name__ == "__main__":
    main()
