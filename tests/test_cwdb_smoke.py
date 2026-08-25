from __future__ import annotations

import numpy as np

from wasserstein_causal_forests.cwdb.smoke import (
    SmokeConfiguration,
    generate_structure_dgp,
    run_shared_ablation,
    summarize_shared_gate,
)


def test_smoke_dgps_are_deterministic_and_monotone() -> None:
    first = generate_structure_dgp(3, "shared", 30)
    second = generate_structure_dgp(3, "shared", 30)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
    assert np.all(np.diff(first[2], axis=1) >= 0.0)
    assert set(first[1]) == {0, 1}


def test_tiny_ablation_has_complete_schema_and_gate_summary() -> None:
    configuration = SmokeConfiguration(
        n_train=50,
        n_test=80,
        n_particles=2,
        total_tree_budget=2,
        max_depth=1,
        min_samples_leaf=5,
        min_arm_leaf=2,
    )
    results = run_shared_ablation(seeds=(0,), configuration=configuration)
    required = {
        "claim_id",
        "dgp",
        "observation_regime",
        "evaluation_manifest_id",
        "n",
        "K",
        "M",
        "seed",
        "method",
        "hyperparameter_manifest_id",
        "metric",
        "value",
        "runtime_seconds",
        "peak_ram_mb",
        "tree_budget",
        "status",
        "failure_reason",
    }
    assert required.issubset(results.columns)
    assert results.shape[0] == 4
    assert set(results["status"]) == {"ok"}
    summary = summarize_shared_gate(results)
    assert summary["decision"] in {"v1", "v0-only"}
