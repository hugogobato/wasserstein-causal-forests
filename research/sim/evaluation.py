from typing import Optional

import numpy as np

from sim.config import QUANTILE_GRID, METHOD_NAMES
from sim.dgps import DGPResult


def integrated_squared_error(
    pred: np.ndarray,
    true: np.ndarray,
    K: int,
) -> float:
    curve_pred = pred[:, :K]
    curve_true = true[:, :K]
    step = 0.9 / (K - 1) if K > 1 else 1.0
    weights = np.full(K, step)
    if K > 1:
        weights[[0, -1]] = step / 2.0
    se = (curve_pred - curve_true) ** 2
    return float(np.mean(np.dot(se, weights)))


def rmse(
    pred: np.ndarray,
    true: np.ndarray,
    coord: int,
) -> float:
    return float(np.sqrt(np.mean((pred[:, coord] - true[:, coord]) ** 2)))


def worst_coordinate_standardized_error(
    pred: np.ndarray,
    true: np.ndarray,
    scales: Optional[np.ndarray] = None,
) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    available = np.all(np.isfinite(pred), axis=0) & np.all(np.isfinite(true), axis=0)
    if not np.any(available):
        raise ValueError("no finite coordinates are available for standardization")
    if scales is None:
        scales = np.std(true, axis=0, ddof=1 if len(true) > 1 else 0)
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (available.shape[0],):
        raise ValueError("standardizer has the wrong number of coordinates")
    pred = pred[:, available]
    true = true[:, available]
    mse_per_coord = np.mean((pred - true) ** 2, axis=0)
    scales = scales[available]
    scales = np.maximum(scales, 1e-8)
    standardized = np.sqrt(mse_per_coord) / scales
    return float(np.max(standardized))


def monotonicity_violations(curves: np.ndarray) -> float:
    diffs = np.diff(curves, axis=1)
    return float(np.mean(diffs < -1e-12))


def integrated_squared_error_curve_only(
    pred: np.ndarray,
    true: np.ndarray,
    K: int,
) -> float:
    return integrated_squared_error(pred, true, K)


def rmse_functional(
    pred: np.ndarray,
    true: np.ndarray,
    K: int,
    j: int,
) -> float:
    return rmse(pred, true, K + j)


METRIC_NAMES = ("ise_curve", "worst_standardized_error", "rmse_functional")
REQUIRED_RESULT_FIELDS = {
    "claim_id", "dgp_id", "observation_regime", "evaluation_manifest_id",
    "n_regions", "inner_n", "seed", "method", "metric", "value",
}


def validate_result_rows(rows: list[dict]) -> None:
    """Fail fast if a simulation cell would silently produce an incomplete table."""
    for row in rows:
        missing = REQUIRED_RESULT_FIELDS.difference(row)
        if missing:
            raise ValueError(f"simulation row is missing fields: {sorted(missing)}")
        if not np.isfinite(float(row["value"])):
            raise ValueError(f"simulation row has a nonfinite value: {row}")


def compute_metrics(
    prediction: np.ndarray,
    dgp: DGPResult,
    method_name: str,
    claim_id: str = "exploratory",
    evaluation_manifest_id: str = "default",
) -> list[dict]:
    if dgp.observation_regime == "empirical_proxy":
        if dgp.proxy_theta_eval is None:
            raise ValueError("empirical_proxy has no proxy truth for evaluation")
        true_theta = dgp.proxy_theta_eval
    else:
        true_theta = dgp.true_theta_eval
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != true_theta.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match truth {true_theta.shape}"
        )
    K, J = dgp.K, dgp.J

    truth_stack = np.vstack([dgp.true_m0, dgp.true_m1])
    truth_scales = np.std(
        truth_stack, axis=0, ddof=1 if len(truth_stack) > 1 else 0
    )

    base = dict(
        claim_id=claim_id,
        dgp_id=dgp.name,
        observation_regime=dgp.observation_regime,
        evaluation_manifest_id=evaluation_manifest_id,
        n_regions=dgp.n_regions,
        inner_n=dgp.inner_n_label,
        seed=dgp.seed,
        method=method_name,
    )
    if dgp.inner_samples is not None:
        sizes = np.asarray([len(sample) for sample in dgp.inner_samples], dtype=int)
        base.update(inner_n_min=int(np.min(sizes)), inner_n_max=int(np.max(sizes)))

    rows = []

    curve_available = np.all(np.isfinite(prediction[:, :K]))
    if curve_available:
        rows.append(dict(
            **base,
            metric="ise_curve",
            value=integrated_squared_error(prediction, true_theta, K),
        ))

    available = np.all(np.isfinite(prediction), axis=0)
    if np.any(available):
        rows.append(dict(
            **base,
            metric="worst_standardized_error",
            value=worst_coordinate_standardized_error(
                prediction[:, available], true_theta[:, available], truth_scales[available]
            ),
        ))

    for j in range(J):
        coord = K + j
        if np.all(np.isfinite(prediction[:, coord])):
            v = rmse_functional(prediction, true_theta, K, j)
            rows.append(dict(**base, metric=f"rmse_functional_{j}", value=v))

    return rows
