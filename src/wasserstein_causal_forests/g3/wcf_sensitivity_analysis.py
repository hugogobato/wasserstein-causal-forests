"""Analysis layer for the WCF sensitivity and assignment-DGP study.

The study contract lives in `report/sensitivity_and_dgp_experiments.md` and the
frozen manifest in `g3.wcf_sensitivity`. This module turns the merged cell rows
into the preregistered tables, the fixed K/M decision, and the figures.

Three conventions are load-bearing.

1. Arm-specific law metrics are averaged within a cell before replication
   aggregation, while functional-specific TATE and TCATE rows stay separate by
   ``target_id``. The report warns that averaging across functionals can conceal
   opposing changes, so every table keeps one row per functional.
2. A replication mean carries a Monte Carlo standard error ``sd / sqrt(R)``.
   Nothing here is called a standard error of a test-row RMSE.
3. Sensitivity comparisons are paired by seed. The Monte Carlo SE of a paired
   difference is computed from the seed-level differences, and the decision
   rule's "two standard errors" always means that paired quantity.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .wcf_sensitivity import (
    ALIGN_DGPS,
    ALIGN_GRID,
    FLEX_GRID,
    INCOME_DGPS,
    INCOME_GRID,
    PROPENSITY_DGPS,
    SYM_DGPS,
    SYM_GRID,
)

# ---------------------------------------------------------------- roster

CELL_COORDINATES: tuple[str, ...] = (
    "grid",
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "method",
    "seed",
)
BLOCK_COORDINATES: tuple[str, ...] = (
    "grid",
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "method",
)
MATCH_ON: tuple[str, ...] = (
    "grid",
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "seed",
    "metric",
    "target_id",
)
#: Matching for two slices of one cell, e.g. primary versus alternative pair.
SAME_CELL_ON: tuple[str, ...] = (
    "grid",
    "dgp",
    "n_train",
    "method",
    "seed",
    "metric",
    "target_id",
)
#: Matching two methods inside one replication (method is the comparison axis,
#: so it is deliberately absent from the key).
CROSS_METHOD_ON: tuple[str, ...] = MATCH_ON
#: Matching two evaluations of one replication with different target ids.
NATIVE_COMMON_ON: tuple[str, ...] = (
    "grid",
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "method",
    "seed",
    "metric",
)

WCF_METHOD = "cwdb_dr"
BASELINE_METHODS: tuple[str, ...] = ("causal_drf", "drf")
SYM_METHODS: tuple[str, ...] = (WCF_METHOD, *BASELINE_METHODS)

PRIMARY_PAIR: tuple[int, int] = (25, 10)
ONE_FACTOR_K_PAIRS: tuple[tuple[int, int], ...] = ((5, 10), (25, 10), (49, 10))
ONE_FACTOR_M_PAIRS: tuple[tuple[int, int], ...] = ((25, 5), (25, 10), (25, 25))
DECISION_ALTERNATIVES: tuple[tuple[int, int], ...] = (
    (5, 10),
    (49, 10),
    (25, 5),
    (25, 25),
)
LARGER_RESOLUTION_PAIRS: tuple[tuple[int, int], ...] = (
    (49, 10),
    (25, 25),
    (49, 25),
)
FACTORIAL_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (k, m) for k in (5, 25, 49) for m in (5, 10, 25)
)
DECISION_DGPS: tuple[str, ...] = ("IC1", "IC3")

#: Metrics whose evaluator rows carry one value per arm. Every other metric has
#: a single row per cell and target; grouping by ``target_id`` averages nothing
#: across functionals.
ARM_METRICS: tuple[str, ...] = (
    "barycenter_rmse",
    "arm_energy_risk",
    "kernel_law_error",
    "tail_calibration",
    "mode_coverage",
    "zero_mass_abs_error",
)

REFERENCE_TCATE: tuple[str, str] = ("reference_tcate_rmse", "REF-TCATE-K")
REFERENCE_ATE: tuple[str, str] = ("reference_effect_rmse", "REF-ATE-K")
KERNEL_LAW: tuple[str, str] = ("kernel_law_error", "LAW-A-K")
MEAN_QUANTILE: tuple[str, str] = ("mean_quantile_rmse", "MEANQ-A-K")

ONEFACTOR_FIXED_TARGETS: tuple[tuple[str, str], ...] = (
    REFERENCE_TCATE,
    REFERENCE_ATE,
    KERNEL_LAW,
    MEAN_QUANTILE,
)
FUNCTIONAL_TARGET_PREFIXES: tuple[str, ...] = ("TATE-K-", "TCATE-K-")
SYM_TARGETS: tuple[tuple[str, str], ...] = (
    REFERENCE_TCATE,
    REFERENCE_ATE,
    KERNEL_LAW,
    MEAN_QUANTILE,
    ("reference_tcate_rmse", "REF-TCATE-COMMON199"),
)
DECISION_TARGETS: tuple[tuple[str, str], ...] = (REFERENCE_TCATE, KERNEL_LAW)

NATIVE_COMMON_COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "reference_tcate_rmse",
        "REF-TCATE-K",
        "REF-TCATE-COMMON199",
        "REF-TCATE native vs COMMON199",
    ),
    (
        "reference_tcate_rmse",
        "REF-TCATE-K",
        "REF-TCATE-INTERIOR",
        "REF-TCATE native vs INTERIOR",
    ),
    (
        "kernel_law_error",
        "LAW-A-K",
        "LAW-A-COMMON199",
        "LAW-A native vs COMMON199",
    ),
)

CELL_METRIC_COLUMNS: tuple[str, ...] = (
    *CELL_COORDINATES,
    "metric",
    "target_id",
    "value",
    "n_arm_rows",
    "wall_seconds",
)
REPLICATION_COLUMNS: tuple[str, ...] = (
    *BLOCK_COORDINATES,
    "metric",
    "target_id",
    "n_seeds",
    "mean",
    "sd",
    "mc_se",
    "median",
)
ONEFACTOR_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "metric",
    "target_id",
    "functional",
    "mean",
    "mc_se",
    "n_seeds",
    "primary_mean",
    "absolute_delta",
    "relative_delta",
    "paired_delta",
    "paired_mc_se",
    "n_pairs",
    "evaluated",
)
FACTORIAL_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "metric",
    "target_id",
    "functional",
    "mean",
    "mc_se",
    "n_seeds",
)
NATIVE_COMMON_COLUMNS: tuple[str, ...] = (
    "grid",
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "method",
    "metric",
    "comparison",
    "native_target_id",
    "common_target_id",
    "n_pairs",
    "native_mean",
    "common_mean",
    "paired_delta",
    "paired_mc_se",
    "p_value",
)
METHOD_VIEW_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n_train",
    "n_grid",
    "n_particles",
    "method",
    "comparator",
    "metric",
    "target_id",
    "n_seeds",
    "mean",
    "mc_se",
    "comparator_mean",
    "paired_delta",
    "paired_mc_se",
    "n_pairs",
)
ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "n_train",
    "n_grid",
    "n_particles",
    "metric",
    "target_id",
    "n_pairs",
    "align_mean",
    "irrelevant_mean",
    "paired_delta",
    "paired_mc_se",
    "p_value",
)
RUNTIME_COLUMNS: tuple[str, ...] = (
    "method",
    "n_grid",
    "n_particles",
    "n_runs",
    "mean_runtime_seconds",
    "sum_runtime_seconds",
    "mean_wall_seconds",
    "mean_process_peak_ram_mb",
    "max_process_peak_ram_mb",
)

# ---------------------------------------------------------------- loading


def load_results(path: str | Path) -> pd.DataFrame:
    """Read the merged WCF sensitivity rows.

    Raises ``FileNotFoundError`` with the launcher command in the message when
    the merged parquet has not been produced yet. The full frame is returned,
    failure rows included, so :func:`failure_summary` can count them; the metric
    tables filter them out inside :func:`cell_metrics`.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no merged WCF sensitivity results at {path}; run "
            "`python research/run_wcf_sensitivity.py merge` (or merge --partial) "
            "first"
        )
    if path.suffix == ".jsonl":
        frame = pd.read_json(path, lines=True)
    else:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:  # pragma: no cover - dependency error path
            raise ImportError(
                "pyarrow is required to read the merged WCF sensitivity parquet"
            ) from error
        frame = pq.read_table(path).to_pandas()
    for column in ("value", "wall_seconds"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def failure_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Failed cells with their recorded reasons, one row per block coordinate."""

    columns = [
        *BLOCK_COORDINATES,
        "n_failed_cells",
        "failure_reasons",
    ]
    if df.empty or "metric" not in df.columns:
        return pd.DataFrame(columns=columns)
    failures = df.loc[df["metric"] == "cell_failure"]
    if failures.empty:
        return pd.DataFrame(columns=columns)

    def _reasons(values: pd.Series) -> str:
        unique = sorted({str(value) for value in values.dropna()})
        return "; ".join(unique)[:600]

    grouped = failures.groupby(list(BLOCK_COORDINATES), dropna=False).agg(
        n_failed_cells=("cell_key", "nunique"),
        failure_reasons=("failure_reason", _reasons),
    )
    return grouped.reset_index()[columns]


# ---------------------------------------------------------------- aggregation


def cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One value per (cell coordinates, metric, target id).

    Rows with ``status != "ok"`` and ``cell_failure`` rows are excluded. The
    arm-specific metrics in :data:`ARM_METRICS` contribute one row per arm, so
    the cell value is their arm average; the report calls for exactly that.
    Functional-specific TATE and TCATE rows are keyed by ``target_id`` and are
    therefore never averaged together.
    """

    if df.empty or "value" not in df.columns:
        return pd.DataFrame(columns=list(CELL_METRIC_COLUMNS))
    keep = (
        (df["status"] == "ok")
        & (df["metric"] != "cell_failure")
        & df["value"].notna()
    )
    ok = df.loc[keep]
    if ok.empty:
        return pd.DataFrame(columns=list(CELL_METRIC_COLUMNS))
    grouped = ok.groupby(
        [*CELL_COORDINATES, "metric", "target_id"], dropna=False
    ).agg(
        value=("value", "mean"),
        n_arm_rows=("value", "size"),
        wall_seconds=("wall_seconds", "mean"),
    )
    return grouped.reset_index()[list(CELL_METRIC_COLUMNS)]


def replication_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Per replication group: ``n_seeds``, ``mean``, ``sd``, ``mc_se``, ``median``.

    ``mc_se`` is the Monte Carlo standard error of the replication mean,
    ``sd / sqrt(n_seeds)``, with ``sd`` the sample standard deviation
    (``ddof=1``). Groups with one seed have ``sd`` and ``mc_se`` undefined and
    report NaN rather than a fabricated zero.
    """

    if cells.empty:
        return pd.DataFrame(columns=list(REPLICATION_COLUMNS))
    keys = [*BLOCK_COORDINATES, "metric", "target_id"]
    grouped = cells.groupby(keys, dropna=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        mean=("value", "mean"),
        sd=("value", lambda values: float(values.std(ddof=1))),
        median=("value", "median"),
    ).reset_index()
    summary["mc_se"] = summary["sd"] / np.sqrt(summary["n_seeds"])
    return summary[list(REPLICATION_COLUMNS)]


def paired_difference(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    on: Sequence[str] = MATCH_ON,
) -> dict[str, Any]:
    """Seed-paired difference ``a - b`` over the keys in ``on``.

    Both frames must carry a numeric ``value`` column. Duplicate keys are
    collapsed to their mean before matching. The Monte Carlo standard error is
    ``sd(differences) / sqrt(R)`` with the sample standard deviation, and a
    normal-approximation two-sided p-value is included when scipy is available.
    A paired difference with zero spread is the strongest evidence, not the
    weakest, so a positive mean with a zero SE is not discarded.
    """

    keys = list(on)
    result: dict[str, Any] = {
        "n_pairs": 0,
        "mean_a": float("nan"),
        "mean_b": float("nan"),
        "paired_delta": float("nan"),
        "paired_sd": float("nan"),
        "paired_mc_se": float("nan"),
        "p_value": None,
    }
    missing = [
        key for key in keys if key not in df_a.columns or key not in df_b.columns
    ]
    if missing or "value" not in df_a.columns or "value" not in df_b.columns:
        return result
    left = _collapse_duplicates(df_a, keys)
    right = _collapse_duplicates(df_b, keys)
    merged = left.merge(right, on=keys, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return result
    difference = (
        merged["value_a"].to_numpy(dtype=float)
        - merged["value_b"].to_numpy(dtype=float)
    )
    n_pairs = int(difference.size)
    mean = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1)) if n_pairs > 1 else float("nan")
    mc_se = sd / np.sqrt(n_pairs) if n_pairs > 1 else float("nan")
    result.update(
        n_pairs=n_pairs,
        mean_a=float(merged["value_a"].mean()),
        mean_b=float(merged["value_b"].mean()),
        paired_delta=mean,
        paired_sd=sd,
        paired_mc_se=mc_se,
        p_value=_normal_p_value(mean, mc_se),
    )
    return result


def _collapse_duplicates(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, dropna=False, as_index=False)["value"]
        .mean()
        .dropna(subset=["value"])
    )


def _normal_p_value(mean: float, mc_se: float) -> float | None:
    if not np.isfinite(mean) or not np.isfinite(mc_se):
        return None
    try:
        from scipy.stats import norm
    except ImportError:  # pragma: no cover - scipy is an optional convenience
        return None
    if mc_se == 0.0:
        return 0.0 if mean != 0.0 else 1.0
    return float(2.0 * norm.sf(abs(mean) / mc_se))


# ---------------------------------------------------------------- view helpers


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _warn_absent(view: str, detail: str) -> None:
    warnings.warn(
        f"{view}: {detail}; returning an empty table", RuntimeWarning, stacklevel=2
    )


def _slice_cell(cells: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    mask = np.ones(len(cells), dtype=bool)
    for column, value in filters.items():
        if column not in cells.columns:
            return cells.iloc[0:0]
        mask &= cells[column].to_numpy() == value
    return cells.loc[mask]


def _lookup(summary: pd.DataFrame, **filters: Any) -> pd.Series | None:
    if summary.empty:
        return None
    mask = np.ones(len(summary), dtype=bool)
    for column, value in filters.items():
        mask &= summary[column].to_numpy() == value
    hit = summary.loc[mask]
    if hit.empty:
        return None
    return hit.iloc[0]


def _target_keys(
    summary: pd.DataFrame,
    fixed: Iterable[tuple[str, str]],
    prefixes: Iterable[str] = (),
) -> list[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for metric, target in fixed:
        rows = summary[
            (summary["metric"] == metric) & (summary["target_id"] == target)
        ]
        keys.update(zip(rows["metric"].astype(str), rows["target_id"].astype(str)))
    for prefix in prefixes:
        mask = summary["target_id"].astype(str).str.startswith(prefix)
        rows = summary.loc[mask]
        keys.update(zip(rows["metric"].astype(str), rows["target_id"].astype(str)))
    return sorted(keys)


def _functional_name(target_id: str) -> str:
    for prefix in FUNCTIONAL_TARGET_PREFIXES:
        if target_id.startswith(prefix):
            return target_id[len(prefix):]
    return ""


def _relative_delta(mean: float, primary_mean: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(primary_mean) or primary_mean <= 0:
        return float("nan")
    return abs(mean - primary_mean) / primary_mean


# ---------------------------------------------------------------- views


def _one_factor(
    df: pd.DataFrame, *, varying: str, pairs: Sequence[tuple[int, int]]
) -> pd.DataFrame:
    view = f"one-factor {varying} view"
    cells = cell_metrics(df)
    if cells.empty or INCOME_GRID not in set(cells["grid"]):
        _warn_absent(view, f"grid {INCOME_GRID!r} is absent")
        return _empty(ONEFACTOR_COLUMNS)
    subset = cells[
        (cells["grid"] == INCOME_GRID)
        & (cells["method"] == WCF_METHOD)
        & (cells["dgp"].isin(INCOME_DGPS))
    ]
    if subset.empty:
        _warn_absent(view, f"no {WCF_METHOD} rows on {INCOME_GRID!r}")
        return _empty(ONEFACTOR_COLUMNS)
    summary = replication_summary(subset)
    keys = _target_keys(summary, ONEFACTOR_FIXED_TARGETS, FUNCTIONAL_TARGET_PREFIXES)
    if not keys:
        _warn_absent(view, "none of the preregistered metrics are present")
        return _empty(ONEFACTOR_COLUMNS)

    primary_grid, primary_particles = PRIMARY_PAIR
    records: list[dict[str, Any]] = []
    for dgp in INCOME_DGPS:
        for metric, target in keys:
            primary = _lookup(
                summary,
                grid=INCOME_GRID,
                dgp=dgp,
                method=WCF_METHOD,
                n_grid=primary_grid,
                n_particles=primary_particles,
                metric=metric,
                target_id=target,
            )
            primary_mean = (
                float(primary["mean"]) if primary is not None else float("nan")
            )
            if primary is not None:
                n_train = int(primary["n_train"])
            elif not subset.empty:
                n_train = int(subset["n_train"].iloc[0])
            else:
                n_train = 0
            for n_grid, n_particles in pairs:
                alternative = _lookup(
                    summary,
                    grid=INCOME_GRID,
                    dgp=dgp,
                    method=WCF_METHOD,
                    n_grid=n_grid,
                    n_particles=n_particles,
                    metric=metric,
                    target_id=target,
                )
                evaluated = alternative is not None and np.isfinite(
                    float(alternative["mean"])
                )
                mean = float(alternative["mean"]) if evaluated else float("nan")
                comparison = paired_difference(
                    _slice_cell(
                        subset,
                        dgp=dgp,
                        metric=metric,
                        target_id=target,
                        n_grid=primary_grid,
                        n_particles=primary_particles,
                    ),
                    _slice_cell(
                        subset,
                        dgp=dgp,
                        metric=metric,
                        target_id=target,
                        n_grid=n_grid,
                        n_particles=n_particles,
                    ),
                    on=SAME_CELL_ON,
                )
                records.append(
                    {
                        "dgp": dgp,
                        "n_train": n_train,
                        "n_grid": n_grid,
                        "n_particles": n_particles,
                        "metric": metric,
                        "target_id": target,
                        "functional": _functional_name(target),
                        "mean": mean,
                        "mc_se": (
                            float(alternative["mc_se"]) if evaluated else float("nan")
                        ),
                        "n_seeds": int(alternative["n_seeds"]) if evaluated else 0,
                        "primary_mean": primary_mean,
                        "absolute_delta": (
                            abs(mean - primary_mean)
                            if evaluated and np.isfinite(primary_mean)
                            else float("nan")
                        ),
                        "relative_delta": _relative_delta(mean, primary_mean),
                        "paired_delta": comparison["paired_delta"],
                        "paired_mc_se": comparison["paired_mc_se"],
                        "n_pairs": comparison["n_pairs"],
                        "evaluated": bool(evaluated),
                    }
                )
    frame = pd.DataFrame.from_records(records, columns=list(ONEFACTOR_COLUMNS))
    return frame.sort_values(
        ["dgp", "metric", "target_id", varying], kind="stable"
    ).reset_index(drop=True)


def one_factor_k(df: pd.DataFrame) -> pd.DataFrame:
    """Grid-resolution slice at ``M = 10`` for the income regimes."""

    return _one_factor(df, varying="n_grid", pairs=ONE_FACTOR_K_PAIRS)


def one_factor_m(df: pd.DataFrame) -> pd.DataFrame:
    """Particle-resolution slice at ``K = 25`` for the income regimes."""

    return _one_factor(df, varying="n_particles", pairs=ONE_FACTOR_M_PAIRS)


def factorial(df: pd.DataFrame) -> pd.DataFrame:
    """The complete 3x3 (K, M) block on IC1 and IC3, when it exists.

    A regime enters only when all nine pairs are present; if neither regime is
    complete the view warns and returns empty rather than showing a misleading
    partial surface.
    """

    view = "factorial view"
    cells = cell_metrics(df)
    if cells.empty or INCOME_GRID not in set(cells["grid"]):
        _warn_absent(view, f"grid {INCOME_GRID!r} is absent")
        return _empty(FACTORIAL_COLUMNS)
    subset = cells[
        (cells["grid"] == INCOME_GRID)
        & (cells["method"] == WCF_METHOD)
        & (cells["dgp"].isin(("IC1", "IC3")))
    ]
    if subset.empty:
        _warn_absent(view, f"no {WCF_METHOD} factorial rows")
        return _empty(FACTORIAL_COLUMNS)
    summary = replication_summary(subset)
    expected = set(FACTORIAL_PAIRS)
    records: list[dict[str, Any]] = []
    for dgp in ("IC1", "IC3"):
        regime = subset.loc[subset["dgp"] == dgp]
        present = set(
            zip(regime["n_grid"].astype(int), regime["n_particles"].astype(int))
        )
        if not expected.issubset(present):
            continue
        for metric, target in _target_keys(
            summary.loc[summary["dgp"] == dgp],
            ONEFACTOR_FIXED_TARGETS,
            FUNCTIONAL_TARGET_PREFIXES,
        ):
            for n_grid, n_particles in FACTORIAL_PAIRS:
                hit = _lookup(
                    summary,
                    grid=INCOME_GRID,
                    dgp=dgp,
                    method=WCF_METHOD,
                    n_grid=n_grid,
                    n_particles=n_particles,
                    metric=metric,
                    target_id=target,
                )
                if hit is None or not np.isfinite(float(hit["mean"])):
                    continue
                records.append(
                    {
                        "dgp": dgp,
                        "n_train": int(hit["n_train"]),
                        "n_grid": n_grid,
                        "n_particles": n_particles,
                        "metric": metric,
                        "target_id": target,
                        "functional": _functional_name(target),
                        "mean": float(hit["mean"]),
                        "mc_se": float(hit["mc_se"]),
                        "n_seeds": int(hit["n_seeds"]),
                    }
                )
    if not records:
        _warn_absent(view, "no regime has the full nine-pair factorial block")
        return _empty(FACTORIAL_COLUMNS)
    return pd.DataFrame.from_records(records, columns=list(FACTORIAL_COLUMNS))


def native_vs_common(df: pd.DataFrame) -> pd.DataFrame:
    """Native-grid errors against the dense common-grid and interior targets.

    ``paired_delta`` is the paired native minus common value, so a positive
    number means the native-grid error is larger than the dense-target error.
    Native and dense target ids are never pooled: each output row names both and
    keeps the cell's K and M visible.
    """

    view = "native-vs-common view"
    cells = cell_metrics(df)
    if cells.empty:
        _warn_absent(view, "no successful metric rows")
        return _empty(NATIVE_COMMON_COLUMNS)
    common_targets = {common for _, _, common, _ in NATIVE_COMMON_COMPARISONS} | {
        "MEANQ-A-COMMON199",
        "MEANQ-A-INTERIOR",
        "REF-ATE-COMMON199",
    }
    if not common_targets & set(cells["target_id"]):
        _warn_absent(view, "no COMMON199 or INTERIOR target rows")
        return _empty(NATIVE_COMMON_COLUMNS)
    records: list[dict[str, Any]] = []
    group_keys = ["grid", "dgp", "n_train", "n_grid", "n_particles", "method"]
    for group_key, _group in cells.groupby(group_keys, dropna=False):
        grid, dgp, n_train, n_grid, n_particles, method = group_key
        for metric, native_target, common_target, label in NATIVE_COMMON_COMPARISONS:
            native = _slice_cell(
                cells,
                grid=grid,
                dgp=dgp,
                n_train=n_train,
                n_grid=n_grid,
                n_particles=n_particles,
                method=method,
                metric=metric,
                target_id=native_target,
            )
            common = _slice_cell(
                cells,
                grid=grid,
                dgp=dgp,
                n_train=n_train,
                n_grid=n_grid,
                n_particles=n_particles,
                method=method,
                metric=metric,
                target_id=common_target,
            )
            if native.empty or common.empty:
                continue
            comparison = paired_difference(native, common, on=NATIVE_COMMON_ON)
            records.append(
                {
                    "grid": grid,
                    "dgp": dgp,
                    "n_train": int(n_train),
                    "n_grid": int(n_grid),
                    "n_particles": int(n_particles),
                    "method": method,
                    "metric": metric,
                    "comparison": label,
                    "native_target_id": native_target,
                    "common_target_id": common_target,
                    "n_pairs": comparison["n_pairs"],
                    "native_mean": comparison["mean_a"],
                    "common_mean": comparison["mean_b"],
                    "paired_delta": comparison["paired_delta"],
                    "paired_mc_se": comparison["paired_mc_se"],
                    "p_value": comparison["p_value"],
                }
            )
    if not records:
        _warn_absent(view, "no native/dense target pair is complete")
        return _empty(NATIVE_COMMON_COLUMNS)
    return pd.DataFrame.from_records(records, columns=list(NATIVE_COMMON_COLUMNS))


def _method_view(
    df: pd.DataFrame,
    *,
    grid: str,
    dgps: Sequence[str],
    methods: Sequence[str],
    comparators: Sequence[str],
    view: str,
) -> pd.DataFrame:
    cells = cell_metrics(df)
    if cells.empty or grid not in set(cells["grid"]):
        _warn_absent(view, f"grid {grid!r} is absent")
        return _empty(METHOD_VIEW_COLUMNS)
    subset = cells[
        (cells["grid"] == grid)
        & (cells["dgp"].isin(dgps))
        & (cells["method"].isin(methods))
    ]
    if subset.empty:
        _warn_absent(view, f"no rows for methods {list(methods)} on {grid!r}")
        return _empty(METHOD_VIEW_COLUMNS)
    summary = replication_summary(subset)
    records: list[dict[str, Any]] = []
    for (dgp, n_train, n_grid, n_particles, method), _group in subset.groupby(
        ["dgp", "n_train", "n_grid", "n_particles", "method"], dropna=False
    ):
        for metric, target in SYM_TARGETS:
            hit = _lookup(
                summary,
                grid=grid,
                dgp=dgp,
                n_train=n_train,
                n_grid=n_grid,
                n_particles=n_particles,
                method=method,
                metric=metric,
                target_id=target,
            )
            if hit is None:
                continue
            records.append(
                {
                    "dgp": dgp,
                    "n_train": int(n_train),
                    "n_grid": int(n_grid),
                    "n_particles": int(n_particles),
                    "method": method,
                    "comparator": "",
                    "metric": metric,
                    "target_id": target,
                    "n_seeds": int(hit["n_seeds"]),
                    "mean": float(hit["mean"]),
                    "mc_se": float(hit["mc_se"]),
                    "comparator_mean": float("nan"),
                    "paired_delta": float("nan"),
                    "paired_mc_se": float("nan"),
                    "n_pairs": 0,
                }
            )
    for comparator in comparators:
        for (dgp, n_train, n_grid, n_particles), _group in subset.groupby(
            ["dgp", "n_train", "n_grid", "n_particles"], dropna=False
        ):
            for metric, target in SYM_TARGETS:
                base = _slice_cell(
                    subset,
                    dgp=dgp,
                    n_train=n_train,
                    n_grid=n_grid,
                    n_particles=n_particles,
                    method=WCF_METHOD,
                    metric=metric,
                    target_id=target,
                )
                against = _slice_cell(
                    subset,
                    dgp=dgp,
                    n_train=n_train,
                    n_grid=n_grid,
                    n_particles=n_particles,
                    method=comparator,
                    metric=metric,
                    target_id=target,
                )
                if base.empty or against.empty:
                    continue
                comparison = paired_difference(base, against, on=CROSS_METHOD_ON)
                records.append(
                    {
                        "dgp": dgp,
                        "n_train": int(n_train),
                        "n_grid": int(n_grid),
                        "n_particles": int(n_particles),
                        "method": WCF_METHOD,
                        "comparator": comparator,
                        "metric": metric,
                        "target_id": target,
                        "n_seeds": int(comparison["n_pairs"]),
                        "mean": comparison["mean_a"],
                        "mc_se": float("nan"),
                        "comparator_mean": comparison["mean_b"],
                        "paired_delta": comparison["paired_delta"],
                        "paired_mc_se": comparison["paired_mc_se"],
                        "n_pairs": comparison["n_pairs"],
                    }
                )
    if not records:
        _warn_absent(view, "no preregistered target rows are present")
        return _empty(METHOD_VIEW_COLUMNS)
    return pd.DataFrame.from_records(records, columns=list(METHOD_VIEW_COLUMNS))


def sym_methods(df: pd.DataFrame) -> pd.DataFrame:
    """WCF against Causal-DRF and DRF across the symmetric assignment regimes.

    Summary rows carry ``comparator == ""``; comparison rows carry the paired
    ``WCF minus baseline`` difference, so a positive ``paired_delta`` means WCF
    has the larger error.
    """

    return _method_view(
        df,
        grid=SYM_GRID,
        dgps=SYM_DGPS,
        methods=SYM_METHODS,
        comparators=BASELINE_METHODS,
        view="symmetric-method view",
    )



def propensity_models(df: pd.DataFrame) -> pd.DataFrame:
    """Logistic against flexible and oracle propensity on SYM-NL and SYM-MU.

    The oracle row is a diagnostic, not a feasible competitor, and is marked
    with ``diagnostic_only``. Comparison rows pair the logistic specification
    from the symmetric grid with the variants on the flexible grid by seed.
    """

    view = "propensity-model view"
    columns = (*METHOD_VIEW_COLUMNS, "diagnostic_only")
    cells = cell_metrics(df)
    if (
        cells.empty
        or SYM_GRID not in set(cells["grid"])
        or FLEX_GRID not in set(cells["grid"])
    ):
        _warn_absent(view, f"grids {SYM_GRID!r} and {FLEX_GRID!r} are required")
        return _empty(columns)
    logistic = cells[
        (cells["grid"] == SYM_GRID)
        & (cells["dgp"].isin(PROPENSITY_DGPS))
        & (cells["method"] == WCF_METHOD)
    ]
    variants = cells[
        (cells["grid"] == FLEX_GRID)
        & (cells["dgp"].isin(PROPENSITY_DGPS))
        & (cells["method"].isin(("cwdb_dr_flex", "cwdb_dr_oracle")))
    ]
    if logistic.empty or variants.empty:
        _warn_absent(view, "logistic or variant propensity rows are absent")
        return _empty(columns)
    summary = replication_summary(
        cells[
            (cells["dgp"].isin(PROPENSITY_DGPS))
            & (cells["method"].isin((WCF_METHOD, "cwdb_dr_flex", "cwdb_dr_oracle")))
        ]
    )
    records: list[dict[str, Any]] = []
    regimes = sorted(set(zip(logistic["dgp"], logistic["n_train"].astype(int))))
    for dgp, n_train in regimes:
        for metric, target in SYM_TARGETS:
            for method in (WCF_METHOD, "cwdb_dr_flex", "cwdb_dr_oracle"):
                for grid in (SYM_GRID, FLEX_GRID):
                    hit = _lookup(
                        summary,
                        grid=grid,
                        dgp=dgp,
                        n_train=n_train,
                        method=method,
                        metric=metric,
                        target_id=target,
                    )
                    if hit is None:
                        continue
                    records.append(
                        {
                            "dgp": dgp,
                            "n_train": n_train,
                            "n_grid": int(hit["n_grid"]),
                            "n_particles": int(hit["n_particles"]),
                            "method": method,
                            "comparator": "",
                            "metric": metric,
                            "target_id": target,
                            "n_seeds": int(hit["n_seeds"]),
                            "mean": float(hit["mean"]),
                            "mc_se": float(hit["mc_se"]),
                            "comparator_mean": float("nan"),
                            "paired_delta": float("nan"),
                            "paired_mc_se": float("nan"),
                            "n_pairs": 0,
                            "diagnostic_only": method == "cwdb_dr_oracle",
                        }
                    )
                    break
            on = (
                "dgp",
                "n_train",
                "n_grid",
                "n_particles",
                "seed",
                "metric",
                "target_id",
            )
            for variant in ("cwdb_dr_flex", "cwdb_dr_oracle"):
                base = _slice_cell(
                    logistic, dgp=dgp, n_train=n_train, metric=metric, target_id=target
                )
                other = _slice_cell(
                    variants,
                    dgp=dgp,
                    n_train=n_train,
                    method=variant,
                    metric=metric,
                    target_id=target,
                )
                if base.empty or other.empty:
                    continue
                comparison = paired_difference(base, other, on=on)
                records.append(
                    {
                        "dgp": dgp,
                        "n_train": n_train,
                        "n_grid": int(base["n_grid"].iloc[0]),
                        "n_particles": int(base["n_particles"].iloc[0]),
                        "method": WCF_METHOD,
                        "comparator": variant,
                        "metric": metric,
                        "target_id": target,
                        "n_seeds": int(comparison["n_pairs"]),
                        "mean": comparison["mean_a"],
                        "mc_se": float("nan"),
                        "comparator_mean": comparison["mean_b"],
                        "paired_delta": comparison["paired_delta"],
                        "paired_mc_se": comparison["paired_mc_se"],
                        "n_pairs": comparison["n_pairs"],
                        "diagnostic_only": variant == "cwdb_dr_oracle",
                    }
                )
    if not records:
        _warn_absent(view, "no preregistered target rows are present")
        return _empty(columns)
    return pd.DataFrame.from_records(records, columns=list(columns))


def default_design_checks_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "results" / "wcf_sensitivity" / "design_checks.json"


def _balance_columns() -> tuple[str, ...]:
    return (
        "dgp",
        "description",
        "propensity_mean",
        "propensity_min",
        "propensity_max",
        "treated_fraction",
        "largest_abs_smd_feature",
        "largest_abs_smd_value",
        "alignment_marginal_max_abs_diff",
        "checks_passed",
        "seed",
        "n_rows",
    )


def alignment_balance(design_checks_path: str | Path) -> pd.DataFrame:
    """Balance and overlap numbers for the two alignment regimes.

    Returns empty when the design-check file is absent, so callers can treat
    alignment as "block missing" rather than as an error.
    """

    path = Path(design_checks_path)
    if not path.is_file():
        return _empty(_balance_columns())
    document = json.loads(path.read_text(encoding="utf-8"))
    regimes = document.get("regimes", {})
    records: list[dict[str, Any]] = []
    for dgp in ALIGN_DGPS:
        entry = regimes.get(dgp)
        if not isinstance(entry, dict):
            continue
        largest = entry.get("largest_abs_smd", {})
        records.append(
            {
                "dgp": dgp,
                "description": entry.get("description", ""),
                "propensity_mean": entry.get("propensity_mean"),
                "propensity_min": entry.get("propensity_min"),
                "propensity_max": entry.get("propensity_max"),
                "treated_fraction": entry.get("treated_fraction"),
                "largest_abs_smd_feature": largest.get("feature"),
                "largest_abs_smd_value": largest.get("value"),
                "alignment_marginal_max_abs_diff": document.get(
                    "alignment_marginal_max_abs_diff"
                ),
                "checks_passed": document.get("checks_passed"),
                "seed": document.get("seed"),
                "n_rows": document.get("n_rows"),
            }
        )
    if not records:
        return _empty(_balance_columns())
    return pd.DataFrame.from_records(records, columns=list(_balance_columns()))


def alignment(
    df: pd.DataFrame, *, design_checks_path: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligned versus irrelevant assignment, plus the balance diagnostics.

    Returns ``(paired, balance)``. ``paired`` holds seed-paired differences
    ``SYM-ALIGN minus SYM-IRREL``, one row per metric and target. ``balance``
    reads the propensity and standardized-difference numbers for both regimes
    from ``design_checks.json`` when that file exists, because the report
    requires overlap and balance reporting for each alignment pair.
    """

    view = "alignment view"
    cells = cell_metrics(df)
    if cells.empty or ALIGN_GRID not in set(cells["grid"]):
        _warn_absent(view, f"grid {ALIGN_GRID!r} is absent")
        return _empty(ALIGNMENT_COLUMNS), _empty(_balance_columns())
    subset = cells[
        (cells["grid"] == ALIGN_GRID)
        & (cells["dgp"].isin(ALIGN_DGPS))
        & (cells["method"] == WCF_METHOD)
    ]
    if subset.empty:
        _warn_absent(view, f"no {WCF_METHOD} rows on {ALIGN_GRID!r}")
        return _empty(ALIGNMENT_COLUMNS), _empty(_balance_columns())
    records: list[dict[str, Any]] = []
    align, irrelevant = ALIGN_DGPS
    for (n_train, n_grid, n_particles), _group in subset.groupby(
        ["n_train", "n_grid", "n_particles"], dropna=False
    ):
        for metric, target in SYM_TARGETS:
            base = _slice_cell(
                subset,
                dgp=align,
                n_train=n_train,
                n_grid=n_grid,
                n_particles=n_particles,
                metric=metric,
                target_id=target,
            )
            other = _slice_cell(
                subset,
                dgp=irrelevant,
                n_train=n_train,
                n_grid=n_grid,
                n_particles=n_particles,
                metric=metric,
                target_id=target,
            )
            if base.empty or other.empty:
                continue
            comparison = paired_difference(
                base,
                other,
                on=(
                    "n_train",
                    "n_grid",
                    "n_particles",
                    "seed",
                    "metric",
                    "target_id",
                ),
            )
            records.append(
                {
                    "n_train": int(n_train),
                    "n_grid": int(n_grid),
                    "n_particles": int(n_particles),
                    "metric": metric,
                    "target_id": target,
                    "n_pairs": comparison["n_pairs"],
                    "align_mean": comparison["mean_a"],
                    "irrelevant_mean": comparison["mean_b"],
                    "paired_delta": comparison["paired_delta"],
                    "paired_mc_se": comparison["paired_mc_se"],
                    "p_value": comparison["p_value"],
                }
            )
    if not records:
        _warn_absent(view, "no aligned/irrelevant target pair is complete")
        paired = _empty(ALIGNMENT_COLUMNS)
    else:
        paired = pd.DataFrame.from_records(records, columns=list(ALIGNMENT_COLUMNS))
    path = (
        Path(design_checks_path) if design_checks_path else default_design_checks_path()
    )
    return paired, alignment_balance(path)


def runtime_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Runtime and process memory per method, grid size, and particle count."""

    if df.empty or "metric" not in df.columns:
        return _empty(RUNTIME_COLUMNS)
    ok = df[(df["status"] == "ok") & (df["metric"] != "cell_failure")]
    groups = ["method", "n_grid", "n_particles"]
    runtime = ok.loc[ok["metric"] == "runtime"]
    memory = ok.loc[ok["metric"] == "process_peak_ram"]
    if runtime.empty and memory.empty:
        return _empty(RUNTIME_COLUMNS)
    parts: list[pd.DataFrame] = []
    if not runtime.empty:
        block = runtime.groupby(groups, dropna=False).agg(
            n_runs=("value", "size"),
            mean_runtime_seconds=("value", "mean"),
            sum_runtime_seconds=("value", "sum"),
            mean_wall_seconds=("wall_seconds", "mean"),
        )
        parts.append(block)
    if not memory.empty:
        block = memory.groupby(groups, dropna=False).agg(
            mean_process_peak_ram_mb=("value", "mean"),
            max_process_peak_ram_mb=("value", "max"),
        )
        parts.append(block)
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.join(part, how="outer")
    combined = combined.reset_index()
    for column in RUNTIME_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan
    return combined[list(RUNTIME_COLUMNS)]



# ---------------------------------------------------------------- decision


def decide_k_m(
    df: pd.DataFrame,
    *,
    relative_tolerance: float = 0.10,
    se_multiple: float = 2.0,
) -> dict[str, Any]:
    """Evaluate the pre-fixed K/M rule on the income regimes.

    Condition A holds when every one-factor alternative (5, 10), (49, 10),
    (25, 5), (25, 25) keeps both decision targets within ``relative_tolerance``
    of the primary (25, 10) in both IC1 and IC3. Condition B holds when some
    larger-resolution pair (49, 10), (25, 25), or (49, 25) improves both targets
    in both regimes by more than ``se_multiple`` paired Monte Carlo standard
    errors. ``retain_primary`` is condition A and not condition B. An absent
    larger pair is recorded with ``evaluated=False`` and never triggers B; an
    absent one-factor or primary record leaves the rule not evaluable, because
    the precondition cannot be checked.
    """

    cells = cell_metrics(df)
    if not cells.empty:
        subset = cells[
            (cells["grid"] == INCOME_GRID)
            & (cells["method"] == WCF_METHOD)
            & (cells["dgp"].isin(DECISION_DGPS))
        ]
    else:
        subset = cells
    result: dict[str, Any] = {
        "rule": (
            "retain (K, M) = (25, 10) when every one-factor alternative stays "
            "within relative_tolerance of the primary on REF-TCATE-K and "
            "LAW-A-K in IC1 and IC3, and no larger-resolution pair improves "
            "both targets by more than se_multiple paired Monte Carlo standard "
            "errors in both regimes"
        ),
        "retain_primary": False,
        "rule_failed": True,
        "evaluable": False,
        "primary_pair": {"n_grid": PRIMARY_PAIR[0], "n_particles": PRIMARY_PAIR[1]},
        "source_targets": [
            {"metric": metric, "target_id": target}
            for metric, target in DECISION_TARGETS
        ],
        "tolerance": {
            "relative_tolerance": float(relative_tolerance),
            "se_multiple": float(se_multiple),
        },
        "condition_a": {
            "passed": False,
            "n_records": 0,
            "n_evaluated": 0,
            "n_within": 0,
            "records": [],
        },
        "condition_b": {
            "any_improves": False,
            "se_multiple": float(se_multiple),
            "combinations": [],
            "improving_pairs": [],
        },
        "recommended_pair": None,
        "missing": [],
        "reason": "",
    }
    if subset.empty:
        result["missing"] = [
            {
                "dgp": dgp,
                "n_grid": n_grid,
                "n_particles": n_particles,
                "metric": metric,
                "target_id": target,
            }
            for dgp in DECISION_DGPS
            for n_grid, n_particles in (PRIMARY_PAIR, *DECISION_ALTERNATIVES)
            for metric, target in DECISION_TARGETS
        ]
        result["reason"] = (
            f"no {WCF_METHOD} cells on grid {INCOME_GRID!r} for "
            f"{list(DECISION_DGPS)}"
        )
        return result

    summary = replication_summary(subset)
    missing: list[dict[str, Any]] = []
    a_records: list[dict[str, Any]] = []
    for dgp in DECISION_DGPS:
        for metric, target in DECISION_TARGETS:
            primary = _lookup(
                summary,
                grid=INCOME_GRID,
                dgp=dgp,
                method=WCF_METHOD,
                n_grid=PRIMARY_PAIR[0],
                n_particles=PRIMARY_PAIR[1],
                metric=metric,
                target_id=target,
            )
            primary_mean = (
                float(primary["mean"]) if primary is not None else float("nan")
            )
            for n_grid, n_particles in DECISION_ALTERNATIVES:
                alternative = _lookup(
                    summary,
                    grid=INCOME_GRID,
                    dgp=dgp,
                    method=WCF_METHOD,
                    n_grid=n_grid,
                    n_particles=n_particles,
                    metric=metric,
                    target_id=target,
                )
                record = {
                    "dgp": dgp,
                    "n_grid": n_grid,
                    "n_particles": n_particles,
                    "metric": metric,
                    "target_id": target,
                    "primary_mean": primary_mean,
                    "alternative_mean": float("nan"),
                    "absolute_delta": float("nan"),
                    "relative_delta": float("nan"),
                    "within_tolerance": False,
                    "evaluated": False,
                }
                available = (
                    primary is not None
                    and alternative is not None
                    and np.isfinite(primary_mean)
                    and np.isfinite(float(alternative["mean"]))
                )
                if available:
                    alternative_mean = float(alternative["mean"])
                    delta = abs(alternative_mean - primary_mean)
                    relative = (
                        delta / primary_mean
                        if primary_mean > 0
                        else float("inf")
                    )
                    record.update(
                        alternative_mean=alternative_mean,
                        absolute_delta=delta,
                        relative_delta=relative,
                        within_tolerance=bool(relative <= relative_tolerance),
                        evaluated=True,
                    )
                else:
                    missing.append(
                        {
                            "dgp": dgp,
                            "n_grid": n_grid,
                            "n_particles": n_particles,
                            "metric": metric,
                            "target_id": target,
                        }
                    )
                a_records.append(record)

    condition_a_passed = bool(a_records) and all(
        record["evaluated"] and record["within_tolerance"] for record in a_records
    )
    result["condition_a"] = {
        "passed": condition_a_passed,
        "n_records": len(a_records),
        "n_evaluated": sum(record["evaluated"] for record in a_records),
        "n_within": sum(
            record["evaluated"] and record["within_tolerance"]
            for record in a_records
        ),
        "records": a_records,
    }

    present = set(
        zip(subset["n_grid"].astype(int), subset["n_particles"].astype(int))
    )
    b_combinations: list[dict[str, Any]] = []
    improving: list[dict[str, Any]] = []
    for n_grid, n_particles in LARGER_RESOLUTION_PAIRS:
        records: list[dict[str, Any]] = []
        for dgp in DECISION_DGPS:
            for metric, target in DECISION_TARGETS:
                base = _slice_cell(
                    subset,
                    dgp=dgp,
                    metric=metric,
                    target_id=target,
                    n_grid=PRIMARY_PAIR[0],
                    n_particles=PRIMARY_PAIR[1],
                )
                larger = _slice_cell(
                    subset,
                    dgp=dgp,
                    metric=metric,
                    target_id=target,
                    n_grid=n_grid,
                    n_particles=n_particles,
                )
                comparison = paired_difference(base, larger, on=SAME_CELL_ON)
                evaluated = bool(
                    comparison["n_pairs"] >= 2
                    and np.isfinite(comparison["paired_mc_se"])
                )
                improves = bool(
                    evaluated
                    and comparison["paired_delta"]
                    > se_multiple * comparison["paired_mc_se"]
                )
                records.append(
                    {
                        "dgp": dgp,
                        "n_grid": n_grid,
                        "n_particles": n_particles,
                        "metric": metric,
                        "target_id": target,
                        "n_pairs": comparison["n_pairs"],
                        "paired_delta": comparison["paired_delta"],
                        "paired_mc_se": comparison["paired_mc_se"],
                        "p_value": comparison["p_value"],
                        "evaluated": evaluated,
                        "improves": improves,
                    }
                )
        evaluated_combo = bool(
            (n_grid, n_particles) in present
            and records
            and all(record["evaluated"] for record in records)
        )
        improves_combo = bool(
            evaluated_combo and all(record["improves"] for record in records)
        )
        mean_improvement = float(
            np.mean([record["paired_delta"] for record in records])
        )
        combination = {
            "n_grid": n_grid,
            "n_particles": n_particles,
            "evaluated": evaluated_combo,
            "improves_both_targets_both_regimes": improves_combo,
            "mean_paired_improvement": mean_improvement,
            "records": records,
        }
        b_combinations.append(combination)
        if improves_combo:
            improving.append(combination)

    any_improves = bool(improving)
    ranked = sorted(
        improving,
        key=lambda item: (
            -item["mean_paired_improvement"],
            -item["n_grid"],
            -item["n_particles"],
        ),
    )
    recommended_pair = (
        [ranked[0]["n_grid"], ranked[0]["n_particles"]] if ranked else None
    )
    result["condition_b"] = {
        "any_improves": any_improves,
        "se_multiple": float(se_multiple),
        "combinations": b_combinations,
        "improving_pairs": [
            [item["n_grid"], item["n_particles"]] for item in improving
        ],
    }
    result["missing"] = missing
    result["recommended_pair"] = recommended_pair
    result["evaluable"] = not missing
    result["retain_primary"] = bool(
        result["evaluable"] and condition_a_passed and not any_improves
    )
    result["rule_failed"] = not result["retain_primary"]
    if not result["evaluable"]:
        result["reason"] = (
            f"{len(missing)} required primary or one-factor records are absent"
        )
    elif result["retain_primary"]:
        result["reason"] = (
            "primary retained: condition A holds and no larger pair improves "
            "both targets in both regimes"
        )
    elif any_improves:
        result["reason"] = (
            "a larger-resolution pair improves both targets in both regimes by "
            "more than se_multiple paired Monte Carlo standard errors"
        )
    else:
        result["reason"] = (
            "at least one one-factor alternative leaves the relative-tolerance "
            "band and no larger pair proves an improvement"
        )
    return result



# ---------------------------------------------------------------- outputs


def write_tables(
    tables: dict[str, pd.DataFrame], output_dir: str | Path
) -> list[Path]:
    """Write one CSV per table name; return the written paths in name order."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(tables):
        frame = tables[name]
        if frame is None:
            continue
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def plot_sensitivity(
    tables: dict[str, pd.DataFrame], output_dir: str | Path
) -> list[Path]:
    """Draw the preregistered figures; skip quietly when matplotlib is absent."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping sensitivity plots")
        return []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(_plot_sensitivity_curves(tables, output, plt))
    written.extend(_plot_paired_differences(tables, output, plt))
    written.extend(_plot_native_vs_common(tables, output, plt))
    return written


_DECISION_PLOT_TARGETS = (
    ("reference_tcate_rmse", "REF-TCATE-K"),
    ("kernel_law_error", "LAW-A-K"),
)


def _plot_sensitivity_curves(
    tables: dict[str, pd.DataFrame], output: Path, plt
) -> list[Path]:
    frames = {
        "K (M = 10)": (tables.get("one_factor_k"), "n_grid"),
        "M (K = 25)": (tables.get("one_factor_m"), "n_particles"),
    }
    usable = {
        label: (frame, varying)
        for label, (frame, varying) in frames.items()
        if frame is not None and not frame.empty
    }
    if not usable:
        print("sensitivity_curves: no one-factor table; skipping")
        return []
    regimes = sorted(
        {
            dgp
            for frame, _ in usable.values()
            for dgp in frame["dgp"].unique()
            if dgp in DECISION_DGPS
        }
    )
    if not regimes:
        return []
    figure, axes = plt.subplots(
        len(regimes), len(usable), figsize=(5.5 * len(usable), 4.0 * len(regimes)),
        squeeze=False,
    )
    for row, dgp in enumerate(regimes):
        for column, (label, (frame, varying)) in enumerate(usable.items()):
            axis = axes[row][column]
            for metric, target in _DECISION_PLOT_TARGETS:
                subset = frame[
                    (frame["dgp"] == dgp)
                    & (frame["metric"] == metric)
                    & (frame["target_id"] == target)
                    & frame["evaluated"]
                    & frame["primary_mean"].gt(0)
                ].sort_values(varying)
                if subset.empty:
                    continue
                axis.errorbar(
                    subset[varying],
                    subset["mean"] / subset["primary_mean"],
                    yerr=subset["mc_se"] / subset["primary_mean"],
                    marker="o",
                    capsize=3,
                    label=target,
                )
            axis.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
            axis.set_title(f"{dgp}: error vs {label}")
            axis.set_xlabel(varying)
            axis.set_ylabel("replication mean / primary mean")
            axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "sensitivity_curves.pdf"
    figure.savefig(path)
    plt.close(figure)
    return [path]


def _plot_paired_differences(
    tables: dict[str, pd.DataFrame], output: Path, plt
) -> list[Path]:
    frames = [
        ("K", tables.get("one_factor_k"), "n_grid"),
        ("M", tables.get("one_factor_m"), "n_particles"),
    ]
    usable = [
        (label, frame, varying)
        for label, frame, varying in frames
        if frame is not None and not frame.empty
    ]
    if not usable:
        print("paired_differences: no one-factor table; skipping")
        return []
    regimes = sorted(
        {
            dgp
            for _, frame, _ in usable
            for dgp in frame["dgp"].unique()
            if dgp in DECISION_DGPS
        }
    )
    if not regimes:
        return []
    figure, axes = plt.subplots(
        len(_DECISION_PLOT_TARGETS), len(regimes),
        figsize=(5.0 * len(regimes), 3.8 * len(_DECISION_PLOT_TARGETS)),
        squeeze=False,
    )
    for row, (metric, target) in enumerate(_DECISION_PLOT_TARGETS):
        for column, dgp in enumerate(regimes):
            axis = axes[row][column]
            positions = 0
            ticks: list[int] = []
            labels: list[str] = []
            for label, frame, varying in usable:
                subset = frame[
                    (frame["dgp"] == dgp)
                    & (frame["metric"] == metric)
                    & (frame["target_id"] == target)
                    & frame["evaluated"]
                    & ~(
                        (frame["n_grid"] == PRIMARY_PAIR[0])
                        & (frame["n_particles"] == PRIMARY_PAIR[1])
                    )
                ].sort_values(varying)
                if subset.empty:
                    continue
                coordinates = list(range(positions, positions + len(subset)))
                axis.errorbar(
                    coordinates,
                    subset["paired_delta"],
                    yerr=subset["paired_mc_se"],
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    label=label,
                )
                ticks.extend(coordinates)
                labels.extend(
                    f"{label}={int(value)}" for value in subset[varying]
                )
                positions += len(subset) + 1
            axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
            axis.set_xticks(ticks)
            axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            axis.set_title(f"{dgp}: paired improvement vs primary ({target})")
            axis.set_ylabel("primary minus alternative")
            if ticks:
                axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "paired_differences.pdf"
    figure.savefig(path)
    plt.close(figure)
    return [path]


def _plot_native_vs_common(
    tables: dict[str, pd.DataFrame], output: Path, plt
) -> list[Path]:
    frame = tables.get("native_vs_common")
    if frame is None or frame.empty:
        print("native_vs_common: table is empty; skipping")
        return []
    comparisons = list(dict.fromkeys(frame["comparison"]))
    n_trains = sorted(int(value) for value in frame["n_train"].unique())
    figure, axes = plt.subplots(
        len(comparisons), len(n_trains),
        figsize=(5.5 * len(n_trains), 3.8 * len(comparisons)),
        squeeze=False,
    )
    for row, comparison in enumerate(comparisons):
        for column, n_train in enumerate(n_trains):
            axis = axes[row][column]
            subset = frame[
                (frame["comparison"] == comparison) & (frame["n_train"] == n_train)
            ]
            if subset.empty:
                continue
            combos = sorted(
                set(
                    zip(
                        subset["dgp"],
                        subset["n_grid"].astype(int),
                        subset["n_particles"].astype(int),
                    )
                )
            )
            positions = np.arange(len(combos))
            methods = sorted(subset["method"].unique())
            for index, method in enumerate(methods):
                block = subset[subset["method"] == method]
                offset = (index - (len(methods) - 1) / 2.0) * 0.25
                x_values = [
                    positions[combos.index(
                        (record.dgp, int(record.n_grid), int(record.n_particles))
                    )]
                    + offset
                    for record in block.itertuples(index=False)
                ]
                axis.errorbar(
                    x_values,
                    block["paired_delta"],
                    yerr=block["paired_mc_se"],
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    label=method,
                )
            axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
            axis.set_xticks(positions)
            axis.set_xticklabels(
                [f"{dgp}\nK={k}, M={m}" for dgp, k, m in combos],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            axis.set_title(f"{comparison}, n={n_train}")
            axis.set_ylabel("native minus dense")
            if len(methods) > 1:
                axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "native_vs_common.pdf"
    figure.savefig(path)
    plt.close(figure)
    return [path]





