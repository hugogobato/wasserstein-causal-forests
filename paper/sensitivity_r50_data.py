"""Fifty-replication sensitivity ledger for the paper assets.

The r50 sensitivity rerun evaluates every frozen sensitivity cell with the
confirmatory protocol: native-grid law and mean-quantile metrics plus paired
plug-in and cross-fitted AIPW functional views.  To keep continuity with the
primary comparison, WCF's reported functional columns use its cross-fitted
AIPW layer while the forest columns use plug-in contrasts; law and
mean-quantile entries are direct metrics.  Both views are present in the
frozen result file, so the convention is a reporting choice.

Aggregation follows the confirmatory ledger: law entries are Monte Carlo
means over replications, functional-effect entries are Monte Carlo RMSEs, and
parentheses are Monte Carlo standard errors over the fifty paired seeds.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT.parent
    / "results/wcf_sensitivity_r50/merged/wcf_sensitivity_r50_results.parquet"
)

N_REPLICATIONS = 50

GRID_PRIMARY = "wcf_sensitivity_v1_logit_f3"
GRID_SYM = "wcf_sensitivity_v1_sym"
GRID_SYM_NULL = "wcf_sensitivity_v1_sym_null"
GRID_ALIGN_NULL = "wcf_sensitivity_v1_align_null"
GRID_FLEX = "wcf_sensitivity_v1_flex"
GRID_FLEX_RF = "wcf_sensitivity_v1_flex_rf"

DESIGN_DGP = {"LS0": "IC0", "LS1": "IC1", "LS2": "IC2", "LS3": "IC3"}
DESIGNS = tuple(DESIGN_DGP)
SYM_DGPS = ("SYM-RANDOM", "SYM-LIN", "SYM-NL", "SYM-MU")
PROPENSITY_DGPS = ("SYM-NL", "SYM-MU")
NULL_COORDINATES = (
    (GRID_SYM_NULL, "SYM-RANDOM-NULL"),
    (GRID_SYM_NULL, "SYM-LIN-NULL"),
    (GRID_SYM_NULL, "SYM-NL-NULL"),
    (GRID_SYM_NULL, "SYM-MU-NULL"),
    (GRID_ALIGN_NULL, "SYM-IRREL-NULL"),
    (GRID_ALIGN_NULL, "SYM-ALIGN-NULL"),
)

METHOD_PREFIX = {
    "cwdb_dr": "aipw",
    "cwdb_dr_flex": "aipw",
    "cwdb_dr_flex_rf": "aipw",
    "cwdb_dr_oracle": "aipw",
    "causal_drf": "plugin",
    "drf": "plugin",
}

FUNCTIONAL_BASE = {
    "tate": "tate_abs_error",
    "tcate": "tcate_rmse",
    "ref_tate": "reference_tate_abs_error",
    "ref_tcate": "reference_tcate_rmse",
}
FUNCTIONAL_TARGET = {
    "tate": "TATE-K-grid_mean",
    "tcate": "TCATE-K-grid_mean",
    "ref_tate": "REF-ATE-K",
    "ref_tcate": "REF-TCATE-K",
}
LAW_METRIC = ("kernel_law_error", "LAW-A-K")
MEANQ_METRIC = ("mean_quantile_rmse", "MEANQ-A-K")

COORDINATE_COLUMNS = (
    "grid", "dgp", "n_train", "n_grid", "n_particles", "method", "seed",
    "metric", "target_id",
)


def _mean_se(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    if len(array) < 2:
        raise ValueError("cannot summarise a single replication")
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def _jackknife_rmse_se(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    leave_one_out = [
        math.sqrt((array.sum() - value) / (len(array) - 1)) for value in array
    ]
    center = sum(leave_one_out) / len(leave_one_out)
    return math.sqrt(
        (len(array) - 1) / len(array)
        * sum((x - center) ** 2 for x in leave_one_out)
    )


def load_per_seed() -> pd.DataFrame:
    """One row per cell coordinate, replication, metric, and target."""

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    frame = pd.read_parquet(SOURCE)
    frame = frame[(frame["status"] == "ok") & frame["value"].notna()].copy()
    # Law and barycenter metrics are stored per treatment arm; average the
    # arms within each replication before any seed-level aggregation.
    frame = frame.groupby(list(COORDINATE_COLUMNS), as_index=False)["value"].mean()
    return frame


def seed_values(
    frame: pd.DataFrame,
    metric: str,
    target_id: str,
    *,
    n_train: int | None = None,
    n_grid: int | None = None,
    n_particles: int | None = None,
    **filters: object,
) -> pd.Series:
    """Seed-indexed per-replication values for one metric cell."""

    selected = frame[(frame["metric"] == metric) & (frame["target_id"] == target_id)]
    for column, value in (
        ("n_train", n_train), ("n_grid", n_grid), ("n_particles", n_particles),
    ):
        if value is not None:
            selected = selected[selected[column] == value]
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    series = selected.set_index("seed")["value"].sort_index()
    if series.index.duplicated().any():
        raise ValueError(f"duplicate seeds for {metric}/{target_id}")
    return series


def end_to_end_metric(method: str, functional: str) -> str:
    prefix = METHOD_PREFIX[method]
    return f"{prefix}_{FUNCTIONAL_BASE[functional]}"


def functional_values(
    frame: pd.DataFrame,
    functional: str,
    method: str,
    **filters: object,
) -> pd.Series:
    metric = end_to_end_metric(method, functional)
    target = FUNCTIONAL_TARGET[functional]
    return seed_values(frame, metric, target, method=method, **filters)


def summarize(
    values: pd.Series, *, law: bool = False
) -> tuple[float, float, int]:
    """Monte Carlo summary: mean for laws, RMSE for error metrics."""

    if len(values) != N_REPLICATIONS:
        raise ValueError(
            f"expected {N_REPLICATIONS} replications, found {len(values)}"
        )
    if law:
        mean, se = _mean_se(values)
        return mean, se, len(values)
    rmse = math.sqrt(float((values.to_numpy(dtype=float) ** 2).mean()))
    return rmse, _jackknife_rmse_se(values), len(values)


def pooled_summary(values: pd.Series) -> tuple[float, float, int]:
    """Pooled Monte Carlo RMSE over designs and replications."""

    rmse = math.sqrt(float((values.to_numpy(dtype=float) ** 2).mean()))
    return rmse, _jackknife_rmse_se(values), len(values)


def paired_percentage(
    frame: pd.DataFrame,
    *,
    metric: str,
    target_id: str,
    method: str,
    low: tuple[int, int],
    high: tuple[int, int],
    **filters: object,
) -> tuple[float, float]:
    """Seed-paired percentage change of ``high`` over ``low`` coordinates."""

    low_values = seed_values(
        frame, metric, target_id, method=method,
        n_grid=low[0], n_particles=low[1], **filters,
    )
    high_values = seed_values(
        frame, metric, target_id, method=method,
        n_grid=high[0], n_particles=high[1], **filters,
    )
    joined = pd.concat({"low": low_values, "high": high_values}, axis=1).dropna()
    if len(joined) != N_REPLICATIONS:
        raise ValueError(f"expected {N_REPLICATIONS} pairs, found {len(joined)}")
    delta = joined["high"] - joined["low"]
    baseline = float(joined["low"].mean())
    if baseline <= 0:
        raise ValueError("non-positive baseline for paired percentage")
    mean_delta, se_delta = _mean_se(delta)
    return 100.0 * mean_delta / baseline, 100.0 * se_delta / baseline
