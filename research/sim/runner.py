from __future__ import annotations

import logging
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import DEFAULT_N_TREES, DEFAULT_N_FOLDS, METHOD_NAMES
from sim.dgps import (
    DGPResult, sample_dgp, u_matrix, u_vector,
    FW, FUNCTIONAL_GRID, QUANTILE_GRID,
    empirical_u_vector, functional_vector_3,
)
from sim.evaluation import compute_metrics, validate_result_rows

from wp3_odcf import (
    ODCFEstimator,
    cross_fitted_dr_scores,
    oracle_dr_scores,
    fit_arm_curve_forests,
    fit_specialized_forests,
    fit_odcf_from_inner_samples,
)
from sim.baselines import _known_design_propensity, run_baseline

_logger = logging.getLogger("sim.runner")


def _run_cell_task(args: tuple) -> list[dict]:
    """Top-level worker entry point for cell-level multiprocessing."""
    return run_simulation_cell(*args)


def _write_checkpoint(path: str, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, default=str))


def build_simulation_tasks(
    dgp_names: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4", "D5", "D8"),
    n_regions_list: tuple[int, ...] = (500, 1000),
    regimes: Optional[tuple[str, ...]] = None,
    n_seeds: int = 30,
    n_trees: int = DEFAULT_N_TREES,
    n_folds: int = DEFAULT_N_FOLDS,
    methods: tuple[str, ...] = METHOD_NAMES,
    d: int = 5,
    n_eval: int = 200,
    claim_id: str = "exploratory",
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> list[tuple]:
    """Build the deterministic cell list used by the serial and Colab runs.

    A shard is selected by its position in this frozen task list.  This makes
    27 independent Colab sessions reproducible and prevents a session from
    silently receiving a different collection of DGPs after a code change.
    """
    if (shard_index is None) != (num_shards is None):
        raise ValueError("shard_index and num_shards must be supplied together")
    if num_shards is not None:
        if not isinstance(num_shards, int) or not 1 <= num_shards <= 27:
            raise ValueError("num_shards must be an integer between 1 and 27")
        if not isinstance(shard_index, int) or not 0 <= shard_index < num_shards:
            raise ValueError("shard_index must lie in [0, num_shards)")

    tasks = []
    for dgp_name in dgp_names:
        for n_regions in n_regions_list:
            dgp_regimes = regimes
            if dgp_regimes is None:
                dgp_regimes = (
                    ("feasible_growing_inner", "empirical_proxy")
                    if dgp_name == "D8" else ("oracle_latent",)
                )
            for regime in dgp_regimes:
                for seed in range(n_seeds):
                    tasks.append((
                        dgp_name, n_regions, regime, seed,
                        n_trees, n_folds, methods, d, n_eval, claim_id,
                    ))

    if shard_index is not None:
        tasks = [task for index, task in enumerate(tasks) if index % num_shards == shard_index]
    return tasks


def _make_evaluation_manifest_id(dgp: DGPResult) -> str:
    proxy = "latent" if dgp.observation_regime != "empirical_proxy" else "proxyMC8"
    return f"eval-v2-{dgp.name}-seed{dgp.seed}-K{dgp.K}-J{dgp.J}-n{len(dgp.X_eval)}-{proxy}"


def _observed_score_inputs(
    dgp: DGPResult,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Return observed U, cross-fitted/oracle scores, and known propensity."""
    n = len(dgp.X)
    known_propensity = _known_design_propensity(dgp)
    if dgp.observation_regime == "oracle_latent":
        z = dgp.Z
        observed = np.where(z[:, None] == 1, dgp.Q1_log, dgp.Q0_log)
        raw_func = np.where(z[:, None] == 1, dgp.Q1_raw_func, dgp.Q0_raw_func)
        U_obs = np.array([
            np.r_[observed[i], functional_vector_3(raw_func[i], FW, FUNCTIONAL_GRID)]
            for i in range(n)
        ])
        scores = oracle_dr_scores(
            U_obs, z, dgp.true_propensity, dgp.true_m0, dgp.true_m1
        )
        return U_obs, scores, known_propensity

    U_obs = np.array([
        empirical_u_vector(samp, QUANTILE_GRID, FUNCTIONAL_GRID, FW)
        for samp in dgp.inner_samples
    ])
    cf = cross_fitted_dr_scores(
        dgp.X,
        dgp.Z,
        U_obs,
        n_folds=n_folds,
        random_state=seed,
        known_propensity=known_propensity,
    )
    return U_obs, cf.scores, known_propensity


def run_odcf_variant(
    dgp: DGPResult,
    variant: str,
    n_trees: int = DEFAULT_N_TREES,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
) -> Optional[np.ndarray]:
    K, J = dgp.K, dgp.J
    bootstrap = variant.endswith("_bootstrap")
    base_variant = variant.removesuffix("_bootstrap")
    if base_variant not in {"composite", "curve_only", "mmd_score"}:
        raise ValueError(f"unsupported ODCF variant: {variant}")

    if bootstrap and dgp.observation_regime != "oracle_latent":
        estimator = ODCFEstimator(
            K=K, J=J, variant=base_variant,
            n_trees=n_trees, random_state=seed,
        )
        known_propensity = _known_design_propensity(dgp)
        fitted, _, _ = fit_odcf_from_inner_samples(
            dgp.X,
            dgp.Z,
            dgp.inner_samples,
            QUANTILE_GRID,
            estimator,
            nuisance_folds=n_folds,
            random_state=seed,
            known_propensity=known_propensity,
        )
        return fitted.predict(dgp.X_eval)

    _, scores, _ = _observed_score_inputs(dgp, n_folds, seed)
    estimator = ODCFEstimator(
        K=K, J=J, variant=base_variant,
        n_trees=n_trees, random_state=seed,
    )
    estimator.fit(dgp.X, scores)
    return estimator.predict(dgp.X_eval)


def run_specialized_forest(
    dgp: DGPResult,
    n_trees: int = DEFAULT_N_TREES,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
) -> np.ndarray:
    """Fit separate curve/function-coordinate forests under one API."""
    _, scores, _ = _observed_score_inputs(dgp, n_folds, seed)
    groups = {"curve": np.arange(dgp.K)}
    groups.update({f"functional_{j}": [dgp.K + j] for j in range(dgp.J)})
    model = fit_specialized_forests(
        dgp.X,
        scores,
        dgp.K,
        dgp.J,
        groups,
        n_trees=n_trees,
        random_state=seed,
    )
    return model.predict(dgp.X_eval)


def run_simulation_cell(
    dgp_name: str,
    n_regions: int,
    regime: str,
    seed: int,
    n_trees: int = DEFAULT_N_TREES,
    n_folds: int = DEFAULT_N_FOLDS,
    methods: tuple[str, ...] = METHOD_NAMES,
    d: int = 5,
    n_eval: int = 200,
    claim_id: str = "exploratory",
) -> list[dict]:
    dgp = sample_dgp(dgp_name, n_regions, seed, regime, d=d, n_eval=n_eval)
    eval_manifest = _make_evaluation_manifest_id(dgp)
    results = []

    for method in methods:
        started = time.perf_counter()
        if method.startswith("odcf_"):
            variant = method.replace("odcf_", "")
            pred = run_odcf_variant(dgp, variant, n_trees, n_folds, seed)
        elif method == "specialized_forest":
            pred = run_specialized_forest(dgp, n_trees, n_folds, seed)
        else:
            baseline_result = run_baseline(dgp, method, n_trees, n_folds, seed)
            if baseline_result is None:
                raise RuntimeError(f"baseline {method} returned no result")
            pred = baseline_result.prediction

        rows = compute_metrics(
            prediction=pred,
            dgp=dgp,
            method_name=method,
            claim_id=claim_id,
            evaluation_manifest_id=eval_manifest,
        )
        validate_result_rows(rows)
        runtime_row = dict(
            claim_id=claim_id,
            dgp_id=dgp.name,
            observation_regime=dgp.observation_regime,
            evaluation_manifest_id=eval_manifest,
            n_regions=dgp.n_regions,
            inner_n=dgp.inner_n_label,
            seed=dgp.seed,
            method=method,
            metric="runtime_seconds",
            value=time.perf_counter() - started,
        )
        if dgp.inner_samples is not None:
            sizes = [len(sample) for sample in dgp.inner_samples]
            runtime_row.update(inner_n_min=min(sizes), inner_n_max=max(sizes))
        validate_result_rows([runtime_row])
        rows.append(runtime_row)
        results.extend(rows)

    return results


def run_simulation(
    dgp_names: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4", "D5", "D8"),
    n_regions_list: tuple[int, ...] = (500, 1000),
    regimes: Optional[tuple[str, ...]] = None,
    n_seeds: int = 30,
    n_trees: int = DEFAULT_N_TREES,
    n_folds: int = DEFAULT_N_FOLDS,
    methods: tuple[str, ...] = METHOD_NAMES,
    d: int = 5,
    n_eval: int = 200,
    claim_id: str = "exploratory",
    workers: int = 1,
    output_path: Optional[str] = None,
    resume: bool = False,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> list[dict]:
    if not isinstance(workers, int) or workers < 1 or workers > 10:
        raise ValueError("workers must be an integer between 1 and 10")
    tasks = build_simulation_tasks(
        dgp_names=dgp_names,
        n_regions_list=n_regions_list,
        regimes=regimes,
        n_seeds=n_seeds,
        n_trees=n_trees,
        n_folds=n_folds,
        methods=methods,
        d=d,
        n_eval=n_eval,
        claim_id=claim_id,
        shard_index=shard_index,
        num_shards=num_shards,
    )

    all_results: list[dict] = []
    completed_cells: set[tuple] = set()
    if resume and output_path and Path(output_path).exists():
        all_results = json.loads(Path(output_path).read_text())
        validate_result_rows(all_results)
        for row in all_results:
            completed_cells.add(
                (row["dgp_id"], row["n_regions"], row["observation_regime"], row["seed"])
            )
        tasks = [
            task for task in tasks
            if (task[0], task[1], task[2], task[3]) not in completed_cells
        ]
    if workers == 1:
        cells = (_run_cell_task(task) for task in tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        cells = pool.map(_run_cell_task, tasks)
    try:
        for task, cell in zip(tasks, cells):
            print(f"  {task[0]} n={task[1]} {task[2]} seed={task[3]}")
            all_results.extend(cell)
            if output_path:
                _write_checkpoint(output_path, all_results)
    finally:
        if workers > 1:
            pool.shutdown(wait=True)
    if output_path:
        _write_checkpoint(output_path, all_results)
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgps", nargs="+", default=["D0", "D1", "D4"])
    parser.add_argument("--n", nargs="+", type=int, default=[200])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--regime", default="auto")
    parser.add_argument("--n_trees", type=int, default=50)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--claim", default="cli-run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(
        f"Running {args.dgps} n={args.n} regime={args.regime} seeds={args.seeds} "
        f"shard={args.shard_index}/{args.num_shards}"
    )
    requested_regimes = None if args.regime == "auto" else (args.regime,)
    results = run_simulation(
        dgp_names=tuple(args.dgps),
        n_regions_list=tuple(args.n),
        regimes=requested_regimes,
        n_seeds=args.seeds,
        n_trees=args.n_trees,
        n_eval=args.n_eval,
        workers=args.workers,
        output_path=args.out,
        resume=args.resume,
        claim_id=args.claim,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(f"Completed {len(results)} evaluation rows")
    if args.out:
        print(f"Wrote {args.out}")
