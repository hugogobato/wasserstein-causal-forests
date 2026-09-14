"""Is the positive reference-distance effect mechanical sampling noise?

Small classes have 7-20 tested students (median 14), control classes 13-29
(median 21). Noisier unit quantiles are farther from any fixed benchmark, so
E[d(q_hat, q_star)] can be larger for the treated arm purely from sampling
noise. This script estimates each unit's quantile-vector variance by repeated
random splits into two halves and reports the implied distance inflation using
the delta approximation E[d] ~ d + Var/(2 d).

    PYTHONPATH=src python research/applied/project_star/08_reference_noise_check.py

Writes results/applied_study_exploration/project_star/misc/reference_noise.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_checks import student_panel  # noqa: E402
from star_common import RESULTS_DIR, midpoint_levels  # noqa: E402

from wasserstein_causal_forests.applied.adapter import (  # noqa: E402
    AppliedDataset,
    weighted_quantiles,
)
from wasserstein_causal_forests.g3.dgps import moderator_bins  # noqa: E402

N_SPLITS = 40
OUT = RESULTS_DIR / "misc" / "reference_noise.json"


def main() -> None:
    rng = np.random.default_rng(7)
    panel = student_panel("math")
    ds = AppliedDataset.load(RESULTS_DIR / "data")
    units = pd.read_csv(RESULTS_DIR / "data" / "units_math_primary.csv")
    q_star = ds.q_star

    noise = {}
    rows = []
    for (unit, a), grp in panel.groupby(["unit_id", "A"]):
        z = grp["z"].to_numpy()
        n = z.size
        if n < 4:
            continue
        sq = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(n)
            half = n // 2
            qa = weighted_quantiles(z[perm[:half]], None, midpoint_levels())
            qb = weighted_quantiles(z[perm[half:]], None, midpoint_levels())
            sq.append(float(np.mean((qa - qb) ** 2)))
        # E||q_A - q_B||^2 = 4 Var(q_hat) for half-samples.
        rows.append({"unit_id": unit, "A": int(a), "n": n, "var_hat": np.mean(sq) / 4.0})
    var_df = pd.DataFrame(rows)
    var_df["dist_to_star"] = np.sqrt(
        ((ds.Q - q_star[None, :]) ** 2).mean(axis=1)
    )
    unit_dist = units.set_index("unit_id").join(var_df.set_index("unit_id"))

    out = {}
    for arm, label in ((1, "small"), (0, "control")):
        grp = unit_dist[unit_dist["A"] == arm]
        var = float(grp["var_hat"].mean())
        dist = float(grp["dist_to_star"].mean())
        out[label] = {
            "n_units": int(len(grp)),
            "mean_tested_n": float(grp["n"].mean()),
            "mean_quantile_variance_W2sq": var,
            "mean_distance_to_qstar": dist,
            "implied_inflation_var_over_2d": var / (2.0 * dist),
        }
    mech = (
        out["small"]["implied_inflation_var_over_2d"]
        - out["control"]["implied_inflation_var_over_2d"]
    )
    out["mechanical_component_of_reference_contrast"] = float(mech)
    out["observed_reference_marginal_seed0"] = float(
        json.load(open(RESULTS_DIR / "primary" / "primary_seed0.json"))["marginal_dr"][
            "reference"
        ]
    )
    # Same summary for the highest-free-lunch quartile only.
    bins = moderator_bins(ds.X)
    unit_dist["bin"] = bins
    q4 = unit_dist[unit_dist["bin"] == 3]
    out["high_fl_quartile"] = {
        label: {
            "n_units": int((q4["A"] == arm).sum()),
            "mean_quantile_variance_W2sq": float(
                q4.loc[q4["A"] == arm, "var_hat"].mean()
            ),
            "mean_distance_to_qstar": float(
                q4.loc[q4["A"] == arm, "dist_to_star"].mean()
            ),
        }
        for arm, label in ((1, "small"), (0, "control"))
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
