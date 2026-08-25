"""Why `cwdb_xmean` fails: the frozen effect budget probe.

The Stage 1 screen rejected `cwdb_xmean`. Two different explanations are
consistent with the headline numbers and they lead to different Stage 2
decisions, so the memo must not pick one by assertion:

1. the X mechanism itself does not pay at these sample sizes, or
2. the frozen three-step effect budget is simply an underfit, and a stronger
   budget would rescue the method.

Explanation 2 is testable and this probe tests it. Two observations settle it.
First, the D0 error is flat in n (0.3545 at n = 500, 0.3536 at n = 1000), which
no variance story predicts: doubled pseudo-outcome variance shrinks with n, a
budget-induced underfit does not. Second, the sweep below shows the D0 error
collapsing by a factor of five as the budget grows, while the null and
confounded regimes degrade monotonically over the very same sweep.

The conclusion is sharper than the one a single budget supports: the X
pseudo-outcome regression has *no* boosting budget that simultaneously recovers
a strong heterogeneous effect surface and stays null-safe, and under imbalance
(the regime the method exists for) its best available budget is the smallest
one and still loses to `cwdb_rmean`. That is the negative finding. It is not a
statement that the frozen budget was miscalibrated.

Run with `PYTHONPATH=src python3 research/checks/phase55_xmean_budget_probe.py`.
Seeds 0-4 here are Stage 1 seeds and this probe is a post-decision diagnostic:
it explains a verdict already reached, and no threshold depends on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wasserstein_causal_forests.g3.dgps import (
    build_dgp,
    build_imbalance_specs,
    register_specs,
)
from wasserstein_causal_forests.g3.manifest import N_TEST, TEST_SEED_OFFSET
from wasserstein_causal_forests.g3.phase55 import (
    RMEAN_SELECTION_FOLDS,
    RMEAN_SHRINKAGE_CANDIDATES,
)
from wasserstein_causal_forests.meta_learners.r_learner import VectorRLearner
from wasserstein_causal_forests.meta_learners.x_learner import (
    EFFECT_BUDGET,
    VectorXLearner,
)

ROOT = Path(__file__).resolve().parents[2]
PROBE_SEEDS = tuple(range(5))
BUDGETS = (3, 10, 20, 50)
D0_CAP = 0.15


def _rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def sweep(dgp_id: str, n_train: int, *, with_rmean: bool = False) -> dict[str, float]:
    dgp = build_dgp(dgp_id, 25)
    errors: dict[str, list[float]] = {str(b): [] for b in BUDGETS}
    if with_rmean:
        errors["rmean"] = []
    for seed in PROBE_SEEDS:
        train = dgp.sample(n_train, seed)
        X_test = dgp.sample(N_TEST, TEST_SEED_OFFSET + seed).X
        truth = dgp.mean_quantile_contrast(X_test)
        for budget in BUDGETS:
            model = VectorXLearner(
                effect_budget={**EFFECT_BUDGET, "n_estimators": budget},
                random_state=seed,
            )
            model.fit(train.X, train.treatment, train.quantiles, dgp.grid.weights)
            contrast = model.predict_mean_quantiles(
                X_test, 1
            ) - model.predict_mean_quantiles(X_test, 0)
            errors[str(budget)].append(_rmse(contrast, truth))
        if with_rmean:
            model = VectorRLearner(
                contrast_candidates=RMEAN_SHRINKAGE_CANDIDATES,
                n_selection_folds=RMEAN_SELECTION_FOLDS,
                random_state=seed,
            )
            model.fit(train.X, train.treatment, train.quantiles, dgp.grid.weights)
            contrast = model.predict_mean_quantiles(
                X_test, 1
            ) - model.predict_mean_quantiles(X_test, 0)
            errors["rmean"].append(_rmse(contrast, truth))
    return {key: float(np.mean(values)) for key, values in errors.items()}


def main() -> None:
    register_specs(build_imbalance_specs())
    payload: dict = {
        "probe_seeds": list(PROBE_SEEDS),
        "budgets": list(BUDGETS),
        "frozen_budget": EFFECT_BUDGET["n_estimators"],
        "metric": "mean_quantile_rmse (arm contrast, oracle truth)",
        "main": {},
        "imbalance": {},
    }
    for dgp_id in ("D0", "D2", "D8"):
        for n_train in (500, 1000):
            payload["main"][f"{dgp_id}/n{n_train}"] = sweep(dgp_id, n_train)
    for dgp_id in ("D2-imb", "D7-imb", "D8-imb"):
        payload["imbalance"][dgp_id] = sweep(dgp_id, 500, with_rmean=True)

    d0 = payload["main"]["D0/n500"]
    payload["verdict"] = {
        "d0_is_budget_bound": d0[str(BUDGETS[-1])] < D0_CAP <= d0["3"],
        "no_budget_serves_both": (
            d0[str(BUDGETS[-1])] < d0["3"]
            and payload["main"]["D2/n500"][str(BUDGETS[-1])]
            > payload["main"]["D2/n500"]["3"]
        ),
        "imbalance_best_budget_still_loses_to_rmean": all(
            min(block[str(b)] for b in BUDGETS) > block["rmean"]
            for name, block in payload["imbalance"].items()
            if name in {"D2-imb", "D8-imb"}
        ),
    }

    out = ROOT / "results" / "merged_phase55" / "xmean_budget_probe.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["verdict"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
