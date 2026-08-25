"""Reproducible G2 shared-partition ablation for C-WDB."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import CWDBRegressor


SHARED_IMPROVEMENT_THRESHOLD = 0.02
SEPARATE_LOSS_TOLERANCE = 0.05
EVALUATION_MANIFEST_ID = "CWDB-G2-SHARED-ABLATION-v2"
HYPERPARAMETER_MANIFEST_ID = "CWDB-SMOKE-HYPER-v2"


@dataclass(frozen=True)
class SmokeConfiguration:
    n_train: int = 160
    n_test: int = 1000
    n_particles: int = 5
    total_tree_budget: int = 40
    learning_rate: float = 0.12
    max_depth: int = 2
    min_samples_leaf: int = 8
    min_arm_leaf: int = 3
    arm_shrinkage: float = 2.0
    collision_epsilon: float = 1e-3
    random_state: int = 100


def generate_structure_dgp(
    seed: int, structure: str, n_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate oracle quantile vectors with shared or arm-specific structure."""

    if structure not in {"shared", "separate"}:
        raise ValueError("structure must be 'shared' or 'separate'")
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_rows, 3))
    treatment = rng.binomial(1, 0.5, size=n_rows)
    treatment[:2] = [0, 1]
    if structure == "shared":
        active_covariate = X[:, 0]
    else:
        active_covariate = np.where(treatment == 0, X[:, 0], X[:, 1])
    region = np.where(active_covariate > 0.0, 1.0, -1.0)
    outer_noise = rng.normal(size=n_rows)
    location = 0.9 * region + 0.25 * treatment + 0.45 * outer_noise
    scale = np.exp(
        0.12 * region + 0.08 * treatment + 0.08 * rng.normal(size=n_rows)
    )
    template = np.array([-1.3, -0.55, 0.0, 0.55, 1.3])
    quantiles = location[:, None] + scale[:, None] * template
    weights = np.ones(template.size) / template.size
    return X, treatment, quantiles, weights


def _peak_ram_mb() -> float:
    # Linux reports ru_maxrss in KiB. The phase environment is Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _model_parameters(
    configuration: SmokeConfiguration, architecture: str
) -> dict[str, object]:
    if configuration.total_tree_budget < 2:
        raise ValueError("total_tree_budget must be at least two")
    if architecture == "v0":
        if configuration.total_tree_budget % 2:
            raise ValueError("v0 requires an even total_tree_budget")
        n_estimators = configuration.total_tree_budget // 2
    else:
        n_estimators = configuration.total_tree_budget
    return {
        "architecture": architecture,
        "n_particles": configuration.n_particles,
        "n_estimators": n_estimators,
        "learning_rate": configuration.learning_rate,
        "max_depth": configuration.max_depth,
        "min_samples_leaf": configuration.min_samples_leaf,
        "min_arm_leaf": configuration.min_arm_leaf,
        "arm_shrinkage": configuration.arm_shrinkage,
        "collision_epsilon": configuration.collision_epsilon,
        "random_state": configuration.random_state,
    }


def run_shared_ablation(
    seeds: tuple[int, ...] = tuple(range(10)),
    configuration: SmokeConfiguration = SmokeConfiguration(),
) -> pd.DataFrame:
    """Run identical-budget v0/v1 comparisons in both structure regimes."""

    rows: list[dict[str, object]] = []
    hyperparameters = json.dumps(asdict(configuration), sort_keys=True)
    for structure in ("shared", "separate"):
        for seed in seeds:
            X, treatment, quantiles, weights = generate_structure_dgp(
                seed, structure, configuration.n_train
            )
            X_test, treatment_test, Q_test, _ = generate_structure_dgp(
                1000 + seed, structure, configuration.n_test
            )
            for architecture in ("v0", "v1"):
                started = time.perf_counter()
                failure_reason = ""
                status = "ok"
                risk = np.nan
                accepted_trees = 0
                try:
                    model = CWDBRegressor(
                        **_model_parameters(configuration, architecture)
                    ).fit(X, treatment, quantiles, weights)
                    arm_scores = []
                    for arm in (0, 1):
                        mask = treatment_test == arm
                        arm_scores.append(
                            model.score_samples(
                                X_test[mask], arm, Q_test[mask]
                            )
                        )
                    risk = float(np.mean(np.concatenate(arm_scores)))
                    if model.fitted_architecture_ == "v0":
                        accepted_trees = int(
                            sum(
                                len(arm_model.estimators_)
                                for arm_model in model.arm_models_.values()
                            )
                        )
                    else:
                        accepted_trees = len(model.estimators_)
                except Exception as error:  # Preserve failed runs in the artifact.
                    status = "failed"
                    failure_reason = f"{type(error).__name__}: {error}"
                runtime = time.perf_counter() - started
                rows.append(
                    {
                        "claim_id": "WP2-A3",
                        "dgp": f"D-{structure}",
                        "observation_regime": "ORACLE-V1",
                        "evaluation_manifest_id": EVALUATION_MANIFEST_ID,
                        "n": configuration.n_train,
                        "n_test": configuration.n_test,
                        "K": quantiles.shape[1],
                        "M": configuration.n_particles,
                        "seed": seed,
                        "method": f"C-WDB-{architecture}",
                        "hyperparameter_manifest_id": HYPERPARAMETER_MANIFEST_ID,
                        "hyperparameters": hyperparameters,
                        "metric": "heldout_energy_score",
                        "value": risk,
                        "runtime_seconds": runtime,
                        "peak_ram_mb": _peak_ram_mb(),
                        "accepted_trees": accepted_trees,
                        "tree_budget": configuration.total_tree_budget,
                        "status": status,
                        "failure_reason": failure_reason,
                    }
                )
    return pd.DataFrame(rows)


def summarize_shared_gate(results: pd.DataFrame) -> dict[str, object]:
    """Apply the preregistered G2 shared-architecture rule."""

    successful = results[
        (results["status"] == "ok")
        & (results["metric"] == "heldout_energy_score")
    ]
    expected = {
        ("D-shared", "C-WDB-v0"),
        ("D-shared", "C-WDB-v1"),
        ("D-separate", "C-WDB-v0"),
        ("D-separate", "C-WDB-v1"),
    }
    observed = set(zip(successful["dgp"], successful["method"]))
    if observed != expected:
        return {
            "decision": "INDETERMINATE",
            "reason": "one or more required method-by-DGP cells failed",
        }
    means = successful.groupby(["dgp", "method"])["value"].mean()
    shared_v0 = float(means.loc[("D-shared", "C-WDB-v0")])
    shared_v1 = float(means.loc[("D-shared", "C-WDB-v1")])
    separate_v0 = float(means.loc[("D-separate", "C-WDB-v0")])
    separate_v1 = float(means.loc[("D-separate", "C-WDB-v1")])
    shared_improvement = (shared_v0 - shared_v1) / shared_v0
    separate_loss = (separate_v1 - separate_v0) / separate_v0
    passed = (
        shared_improvement >= SHARED_IMPROVEMENT_THRESHOLD
        and separate_loss <= SEPARATE_LOSS_TOLERANCE
    )
    return {
        "decision": "v1" if passed else "v0-only",
        "shared_improvement": shared_improvement,
        "required_shared_improvement": SHARED_IMPROVEMENT_THRESHOLD,
        "separate_relative_loss": separate_loss,
        "allowed_separate_relative_loss": SEPARATE_LOSS_TOLERANCE,
        "mean_risks": {
            "shared_v0": shared_v0,
            "shared_v1": shared_v1,
            "separate_v0": separate_v0,
            "separate_v1": separate_v1,
        },
    }


def write_ablation(
    output: str | Path,
    *,
    seeds: tuple[int, ...] = tuple(range(10)),
    configuration: SmokeConfiguration = SmokeConfiguration(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run, validate, and write the required Parquet artifact."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = run_shared_ablation(seeds=seeds, configuration=configuration)
    results.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)
    if not results.equals(reloaded):
        raise RuntimeError("Parquet round-trip changed the ablation rows")
    summary = summarize_shared_gate(results)
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/smoke/cwdb_shared_ablation.parquet",
    )
    parser.add_argument("--seeds", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.seeds < 1:
        parser.error("--seeds must be positive")
    _, summary = write_ablation(
        arguments.output, seeds=tuple(range(arguments.seeds))
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
