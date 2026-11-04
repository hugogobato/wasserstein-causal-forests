"""Fifty-replication confirmatory ledger for the paper simulation assets.

The confirmatory run evaluates all fourteen designs at n=1000 with fifty
paired seeds.  To keep continuity with the earlier ten-replication tables,
WCF's functional columns use its cross-fitted AIPW layer while the forest
columns use plug-in contrasts from their fitted laws; law and mass entries
are direct metrics.  Both views are present in the frozen result file, so the
convention is a reporting choice, not a data-availability limit.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT.parent
    / "results/wcf_confirmatory_r50/merged/wcf_confirmatory_results.parquet"
)
LEDGER_PATH = ROOT / "tables" / "simulation_metrics_rmse_50.csv"

N_TRAIN = 1000
N_REPLICATIONS = 50

DGP_LABEL = {
    "IC0": "LS0", "IC1": "LS1", "IC2": "LS2", "IC3": "LS3",
    "D0": "S1", "D2": "S2", "D5": "S3", "D6": "S4", "D7": "S5", "D8": "S6",
    "ZI0": "Z0", "ZI1": "Z1", "ZI2": "Z2", "ZI3": "Z3",
}
FAMILY = {label: label.rstrip("0123456789") for label in DGP_LABEL.values()}
FUNCTIONAL_PREFIX = {
    "cwdb_dr": "aipw",
    "cwdb_zipt": "aipw",
    "causal_drf": "plugin",
    "drf": "plugin",
}
FUNCTIONAL_METRICS = {
    "tate_functional_rmse": "{prefix}_tate_abs_error",
    "tcate_functional_rmse": "{prefix}_tcate_rmse",
    "reference_effect_rmse": "{prefix}_reference_tate_abs_error",
    "reference_tcate_rmse": "{prefix}_reference_tcate_rmse",
}
DIRECT_METRICS = ("kernel_law_error", "zero_mass_abs_error", "mass_contrast_rmse")


def _mean_se(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def _jackknife_rmse_se(seed_mse: pd.Series) -> float:
    values = seed_mse.to_numpy(dtype=float)
    leave_one_out = [
        math.sqrt((values.sum() - value) / (len(values) - 1)) for value in values
    ]
    center = sum(leave_one_out) / len(leave_one_out)
    return math.sqrt(
        (len(values) - 1) / len(values)
        * sum((x - center) ** 2 for x in leave_one_out)
    )


def build_ledger() -> pd.DataFrame:
    """One row per design, method, and reported metric, with Monte Carlo SEs."""

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    frame = pd.read_parquet(SOURCE)
    frame = frame[frame["status"] == "ok"]
    rows: list[dict[str, object]] = []
    for (dgp, method), group in frame.groupby(["dgp", "method"], sort=False):
        label = DGP_LABEL[dgp]
        family = FAMILY[label]
        base = {
            "suite": "zi" if family == "Z" else ("income" if family == "LS" else "abstract"),
            "design": label,
            "family": family,
            "dgp": dgp,
            "method": method,
            "n_train": N_TRAIN,
            "n_replications": N_REPLICATIONS,
        }
        direct = group[group["metric"].isin(DIRECT_METRICS)]
        if not direct.empty:
            by_seed = direct.groupby(["metric", "seed"])["value"].mean()
            present = set(by_seed.index.get_level_values(0))
            for metric in DIRECT_METRICS:
                if metric not in present:
                    continue
                values = by_seed.loc[metric]
                if len(values) != N_REPLICATIONS:
                    raise RuntimeError(
                        f"{dgp}/{method}/{metric}: {len(values)} seeds"
                    )
                mean, se = _mean_se(values)
                rows.append({**base, "metric": metric, "mean": mean, "se": se})
        prefix = FUNCTIONAL_PREFIX[method]
        for name, template in FUNCTIONAL_METRICS.items():
            subset = group[group["metric"] == template.format(prefix=prefix)]
            if subset.empty:
                continue
            seed_mse = (
                subset.assign(squared=subset["value"].astype(float) ** 2)
                .groupby("seed")["squared"]
                .mean()
            )
            if len(seed_mse) != N_REPLICATIONS:
                raise RuntimeError(f"{dgp}/{method}/{name}: {len(seed_mse)} seeds")
            rows.append({
                **base,
                "metric": name,
                "mean": math.sqrt(float(seed_mse.mean())),
                "se": _jackknife_rmse_se(seed_mse),
            })
    ledger = pd.DataFrame(rows)
    ordinary = sum(1 for label in DGP_LABEL.values() if FAMILY[label] != "Z")
    zero = len(DGP_LABEL) - ordinary
    expected = 3 * (ordinary * (1 + len(FUNCTIONAL_METRICS)) + zero * (3 + len(FUNCTIONAL_METRICS)))
    if len(ledger) != expected:
        raise RuntimeError(f"Expected {expected} ledger rows, found {len(ledger)}")
    return ledger


def write_ledger() -> pd.DataFrame:
    ledger = build_ledger()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    return ledger
