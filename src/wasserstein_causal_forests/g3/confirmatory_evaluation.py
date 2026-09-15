"""Symmetric functional evaluation for the paper's confirmatory rerun.

The historical tournament mixed two functional conventions: ``cwdb_dr`` used
its AIPW-calibrated scalar layer, while the forest comparators used integrals
of their fitted laws.  This module keeps those scientifically distinct views
side by side.  Every law-producing method is evaluated both by plug-in law
integration and by the same cross-fitted AIPW construction.  The fitted laws
and all law-level metrics are unchanged.

Income designs use the covariate that actually modifies treatment response:
IC1 and IC3 use X4, and IC2 uses X3 (one-based paper notation).  IC0 is a null
design and retains X1 as a stability diagnostic.  The mechanism and two-part
designs retain their prespecified X1 strata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..cwdb.cross_fitted import stratified_folds
from ..cwdb.dr_calibration import FunctionalAIPW
from ..pta_bcf.targets import GRID_FUNCTIONALS
from .dgps import DGPSample, MODERATOR_EDGES, DistributionalDGP
from .evaluation import _bin_means, _rmse, evaluate
from .manifest import BOOSTING_BUDGET, TRAINING_FUNCTIONALS, Cell
from .methods import CWDBAdapter
from .methods import MethodOutput
from .phase6_methods import _declared_functionals

CONFIRMATORY_PROTOCOL_ID = "WCF-CONFIRMATORY-EVAL-v1"
NUISANCE_FOLDS = 3

# Zero-based columns.  The descriptions exposed in result rows use the
# corresponding one-based paper notation.
ACTIVE_MODERATOR_COLUMN: dict[str, int] = {
    "IC0": 0,
    "IC1": 3,
    "IC2": 2,
    "IC3": 3,
    "D5": 3,
    "ZI1": 3,
    "ZI2": 1,
    "ZI3": 3,
}

_REPLACED_NATIVE_METRICS = {
    "tate_functional_rmse",
    "tcate_functional_rmse",
    "reference_effect_rmse",
    "reference_tcate_rmse",
}


def moderator_column(dgp_id: str) -> int:
    """Return the frozen evaluation moderator for a source DGP identifier."""

    return ACTIVE_MODERATOR_COLUMN.get(dgp_id, 0)


def moderator_bins_for_dgp(
    dgp_id: str, X: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Four fixed strata along the DGP's declared moderator coordinate."""

    column = moderator_column(dgp_id)
    if column >= X.shape[1]:
        raise ValueError(
            f"{dgp_id}: moderator column {column} unavailable for X with "
            f"{X.shape[1]} columns"
        )
    return np.searchsorted(
        np.asarray(MODERATOR_EDGES), np.asarray(X)[:, column], side="right"
    )


def _functional_values(
    output: MethodOutput, dgp: DistributionalDGP
) -> dict[str, dict[int, NDArray[np.float64]]]:
    """Integrate every declared scalar functional against ``output.law``."""

    if output.law is None:
        raise ValueError("confirmatory functional evaluation requires a fitted law")
    functions = _declared_functionals(
        dgp.grid.weights, dgp.grid.reference_quantiles()
    )
    return {
        name: {
            arm: law.scalar_expectation(function)
            for arm, law in output.law.items()
        }
        for name, function in functions.items()
    }


def _subset(sample: DGPSample, rows: NDArray[np.bool_]) -> DGPSample:
    return DGPSample(
        X=sample.X[rows],
        treatment=sample.treatment[rows],
        quantiles=sample.quantiles[rows],
        propensity=sample.propensity[rows],
        dgp_id=sample.dgp_id,
        seed=sample.seed,
    )


def _common_aipw_scores(
    cell: Cell,
    output: MethodOutput,
    train: DGPSample,
    dgp: DistributionalDGP,
    cache_directory: Path | None,
) -> dict[str, NDArray[np.float64]]:
    """Return scores from strict outer-fold nuisance fits for every method."""

    # Imported lazily to avoid a module cycle during runner registration.
    from .runner import build_adapter

    folds = stratified_folds(train.treatment, NUISANCE_FOLDS, cell.seed)
    functions = _declared_functionals(
        dgp.grid.weights, dgp.grid.reference_quantiles()
    )
    oof = {
        name: {arm: np.full(train.n_rows, np.nan) for arm in (0, 1)}
        for name in functions
    }
    for fold in range(NUISANCE_FOLDS):
        held_out = folds == fold
        if not np.any(held_out):
            continue
        fold_train = _subset(train, ~held_out)
        # The method and all frozen hyperparameters are identical to the full
        # fit.  Keeping the base seed matches WCF's internal OOF convention.
        if cell.method == "cwdb_dr":
            # Law-only form of the paper WCF.  Selection is repeated wholly
            # within each outer training fold, avoiding leakage from a penalty
            # selected with the held-out fold's outcomes.
            adapter = CWDBAdapter(
                n_particles=cell.n_particles,
                architecture="v1",
                sharing="partial",
                init_sharing="pooled",
                arm_shrinkage=5.0,
                contrast_candidates=(0.0, 50.0, 500.0),
                n_folds=NUISANCE_FOLDS,
                **BOOSTING_BUDGET,
            )
        else:
            adapter = build_adapter(cell, cache_directory)
        fold_output = adapter.fit_predict(
            fold_train,
            train.X[held_out],
            dgp,
            TRAINING_FUNCTIONALS,
            seed=cell.seed,
        )
        values = _functional_values(fold_output, dgp)
        for name in functions:
            for arm in (0, 1):
                oof[name][arm][held_out] = values[name][arm]

    missing = [
        f"{name}/arm{arm}"
        for name, arm_values in oof.items()
        for arm, values in arm_values.items()
        if not np.all(np.isfinite(values))
    ]
    if missing:
        raise RuntimeError(f"nonfinite OOF functional predictions: {missing}")
    observed = {
        name: np.asarray(function(train.quantiles), dtype=float)
        for name, function in functions.items()
    }
    calibration = FunctionalAIPW(n_bins=4).fit(
        observed=observed,
        oof_arm_means=oof,
        X=train.X,
        treatment=train.treatment,
        random_state=cell.seed + 31,
    )
    return {
        name: np.asarray(values, dtype=float).copy()
        for name, values in calibration.scores_.items()
    }


def _row(metric: str, target: str, value: float, detail: str) -> dict[str, object]:
    return {
        "metric": metric,
        "target_id": target,
        "arm": None,
        "detail": detail,
        "value": float(value),
        "status": "ok",
        "failure_reason": "",
    }


def _view_rows(
    *,
    prefix: str,
    estimates: dict[str, NDArray[np.float64]],
    marginal_estimates: dict[str, float] | None,
    dgp: DistributionalDGP,
    X_test: NDArray[np.float64],
    bins: NDArray[np.int64],
    moderator_detail: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in GRID_FUNCTIONALS:
        truth = dgp.functional_contrast(X_test, name)
        estimated = estimates[name]
        marginal_estimate = (
            float(np.mean(estimated))
            if marginal_estimates is None
            else float(marginal_estimates[name])
        )
        marginal_truth = float(np.mean(truth))
        rows.append(
            _row(
                f"{prefix}_tate_abs_error",
                f"TATE-K-{name}",
                abs(marginal_estimate - marginal_truth),
                f"{prefix} functional contrast; marginal absolute error",
            )
        )
        rows.extend([
            _row(f"{prefix}_tate_estimate", f"TATE-K-{name}", marginal_estimate,
                 f"{prefix} functional contrast; marginal estimate"),
            _row(f"{prefix}_tate_oracle", f"TATE-K-{name}", marginal_truth,
                 f"{prefix} functional contrast; test-design oracle"),
        ])
        estimated_bins = _bin_means(estimated, bins)
        truth_bins = _bin_means(truth, bins)
        rows.append(
            _row(
                f"{prefix}_tcate_rmse",
                f"TCATE-K-{name}",
                _rmse(estimated_bins, truth_bins),
                f"{prefix} functional contrast; {moderator_detail}",
            )
        )
        for bin_index, (estimate, oracle) in enumerate(
            zip(estimated_bins, truth_bins, strict=True)
        ):
            rows.extend([
                _row(f"{prefix}_tcate_estimate", f"TCATE-K-{name}", estimate,
                     f"{prefix}; moderator bin {bin_index}; {moderator_detail}"),
                _row(f"{prefix}_tcate_oracle", f"TCATE-K-{name}", oracle,
                     f"{prefix}; oracle moderator bin {bin_index}; {moderator_detail}"),
            ])
    truth = dgp.reference_contrast(X_test)
    estimated = estimates["reference"]
    marginal_estimate = (
        float(np.mean(estimated))
        if marginal_estimates is None
        else float(marginal_estimates["reference"])
    )
    marginal_truth = float(np.mean(truth))
    rows.append(
        _row(
            f"{prefix}_reference_tate_abs_error",
            "REF-ATE-K",
            abs(marginal_estimate - marginal_truth),
            f"{prefix} reference contrast; marginal absolute error",
        )
    )
    rows.extend([
        _row(f"{prefix}_reference_tate_estimate", "REF-ATE-K", marginal_estimate,
             f"{prefix} reference contrast; marginal estimate"),
        _row(f"{prefix}_reference_tate_oracle", "REF-ATE-K", marginal_truth,
             f"{prefix} reference contrast; test-design oracle"),
    ])
    estimated_bins = _bin_means(estimated, bins)
    truth_bins = _bin_means(truth, bins)
    rows.append(
        _row(
            f"{prefix}_reference_tcate_rmse",
            "REF-TCATE-K",
            _rmse(estimated_bins, truth_bins),
            f"{prefix} reference contrast; {moderator_detail}",
        )
    )
    for bin_index, (estimate, oracle) in enumerate(
        zip(estimated_bins, truth_bins, strict=True)
    ):
        rows.extend([
            _row(f"{prefix}_reference_tcate_estimate", "REF-TCATE-K", estimate,
                 f"{prefix}; moderator bin {bin_index}; {moderator_detail}"),
            _row(f"{prefix}_reference_tcate_oracle", "REF-TCATE-K", oracle,
                 f"{prefix}; oracle moderator bin {bin_index}; {moderator_detail}"),
        ])
    return rows


def evaluate_confirmatory(
    cell: Cell,
    output: MethodOutput,
    train: DGPSample,
    dgp: DistributionalDGP,
    X_test: NDArray[np.float64],
    evaluation_manifest,
    *,
    cache_directory: Path | None = None,
) -> list[dict[str, object]]:
    """Evaluate laws plus symmetric plug-in and AIPW functional views."""

    native = evaluate(
        output,
        dgp,
        X_test,
        evaluation_manifest,
        cache_key=(cell.test_seed,),
    )
    rows = [row for row in native if row["metric"] not in _REPLACED_NATIVE_METRICS]
    bins_test = moderator_bins_for_dgp(cell.dgp, X_test)
    column = moderator_column(cell.dgp)
    detail = (
        f"RMSE over four strata of X{column + 1} with edges "
        f"{tuple(float(value) for value in MODERATOR_EDGES)}"
    )

    plugin = _functional_values(output, dgp)
    plugin_contrasts = {
        name: values[1] - values[0] for name, values in plugin.items()
    }
    rows.extend(
        _view_rows(
            prefix="plugin",
            estimates=plugin_contrasts,
            marginal_estimates=None,
            dgp=dgp,
            X_test=X_test,
            bins=bins_test,
            moderator_detail=detail,
        )
    )

    scores = _common_aipw_scores(cell, output, train, dgp, cache_directory)
    bins_train = moderator_bins_for_dgp(cell.dgp, train.X)
    aipw_estimates = {
        name: np.full(X_test.shape[0], np.nan) for name in scores
    }
    for name, values in scores.items():
        bin_means = _bin_means(values, bins_train)
        aipw_estimates[name] = bin_means[bins_test]
    rows.extend(
        _view_rows(
            prefix="aipw",
            estimates=aipw_estimates,
            marginal_estimates={name: float(np.mean(values)) for name, values in scores.items()},
            dgp=dgp,
            X_test=X_test,
            bins=bins_test,
            moderator_detail=detail,
        )
    )
    rows.append(
        _row(
            "diagnostic_confirmatory_nuisance_folds",
            "NONE_OPERATIONAL",
            NUISANCE_FOLDS,
            CONFIRMATORY_PROTOCOL_ID,
        )
    )
    rows.append(
        _row(
            "diagnostic_confirmatory_moderator_column",
            "NONE_OPERATIONAL",
            column + 1,
            CONFIRMATORY_PROTOCOL_ID,
        )
    )
    return rows
