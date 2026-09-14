"""Summarize and verify the Egger Kenya WCF runs.

Produces:
  - summary.json: marginals, IF SEs, bin contrasts, placebo/noise spreads
  - summary_tables.csv: one row per functional x stage
  - raw_checks.json: unadjusted contrasts and exact 10k-permutation p-values
  - figures: quantile curves, marginal effects, moderator-bin contrasts

Run from the repo root:
    PYTHONPATH=src /usr/bin/python3 research/applied/egger_kenya/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.applied.adapter import (  # noqa: E402
    AppliedDataset,
    make_functionals,
    midpoint_levels,
)
from wasserstein_causal_forests.g3.dgps import moderator_bins  # noqa: E402

OUT = ROOT / "results/applied_study_exploration/egger_kenya"
RUNS = OUT / "runs"
FUNCTIONALS = ("mean", "sd", "skew", "tail", "reference", "p10", "p90")
EXTRA_FUNCTIONALS = {"p10": lambda q: q[:, 2], "p90": lambda q: q[:, 22]}
N_PERM = 10_000


def all_functionals(q_star: np.ndarray) -> dict:
    funcs = make_functionals(q_star)
    funcs.update(EXTRA_FUNCTIONALS)
    return funcs


def load(stage: str) -> dict:
    with open(RUNS / stage / "results.json") as fh:
        return json.load(fh)


def raw_functionals(ds: AppliedDataset) -> dict[str, float]:
    funcs = all_functionals(ds.q_star)
    values = {name: np.asarray(h(ds.Q), dtype=float) for name, h in funcs.items()}
    return {
        name: float(values[name][ds.A == 1].mean() - values[name][ds.A == 0].mean())
        for name in funcs
    }


def raw_levels(ds: AppliedDataset) -> dict[str, dict[str, float]]:
    funcs = all_functionals(ds.q_star)
    out = {}
    for name, h in funcs.items():
        v = np.asarray(h(ds.Q), dtype=float)
        m1, m0 = float(v[ds.A == 1].mean()), float(v[ds.A == 0].mean())
        out[name] = {
            "treated_mean": m1,
            "control_mean": m0,
            "raw_diff": m1 - m0,
            "raw_pct_of_control": 100.0 * (m1 - m0) / m0 if abs(m0) > 1e-12 else None,
        }
    return out


def raw_bins(ds: AppliedDataset) -> dict[str, list[float]]:
    funcs = all_functionals(ds.q_star)
    bins = moderator_bins(ds.X)
    out = {}
    for name, h in funcs.items():
        v = np.asarray(h(ds.Q), dtype=float)
        vals = []
        for b in range(4):
            rows = bins == b
            vals.append(float(v[rows & (ds.A == 1)].mean() - v[rows & (ds.A == 0)].mean()))
        out[name] = vals
    return out


def permutation_test(ds: AppliedDataset, n_perm: int = N_PERM, seed: int = 7) -> dict:
    funcs = all_functionals(ds.q_star)
    values = {name: np.asarray(h(ds.Q), dtype=float) for name, h in funcs.items()}
    bins = moderator_bins(ds.X)
    rng = np.random.default_rng(seed)
    obs = raw_functionals(ds)
    obs_bins = raw_bins(ds)
    counts = {name: 0 for name in funcs}
    counts_bins = {name: np.zeros(4, dtype=int) for name in funcs}
    for _ in range(n_perm):
        perm = rng.permutation(ds.A)
        for name, v in values.items():
            diff = v[perm == 1].mean() - v[perm == 0].mean()
            if abs(diff) >= abs(obs[name]) - 1e-12:
                counts[name] += 1
            for b in range(4):
                rows = bins == b
                bd = v[rows & (perm == 1)].mean() - v[rows & (perm == 0)].mean()
                if abs(bd) >= abs(obs_bins[name][b]) - 1e-12:
                    counts_bins[name][b] += 1
    return {
        "n_permutations": n_perm,
        "observed_raw": obs,
        "p_value_two_sided": {name: (counts[name] + 1) / (n_perm + 1) for name in funcs},
        "observed_raw_bins": obs_bins,
        "p_value_two_sided_bins": {
            name: ((counts_bins[name] + 1) / (n_perm + 1)).tolist() for name in funcs
        },
    }


def coordinate_raw(ds: AppliedDataset, n_perm: int = N_PERM, seed: int = 11) -> dict:
    """Raw difference of mean quantile coordinates by arm + permutation p-values."""

    obs = ds.Q[ds.A == 1].mean(axis=0) - ds.Q[ds.A == 0].mean(axis=0)
    rng = np.random.default_rng(seed)
    counts = np.zeros(ds.Q.shape[1], dtype=int)
    for _ in range(n_perm):
        perm = rng.permutation(ds.A)
        d = ds.Q[perm == 1].mean(axis=0) - ds.Q[perm == 0].mean(axis=0)
        counts += np.abs(d) >= np.abs(obs) - 1e-12
    return {
        "u": midpoint_levels().tolist(),
        "treated_mean_q": ds.Q[ds.A == 1].mean(axis=0).tolist(),
        "control_mean_q": ds.Q[ds.A == 0].mean(axis=0).tolist(),
        "raw_diff": obs.tolist(),
        "p_value_two_sided": ((counts + 1) / (n_perm + 1)).tolist(),
        "n_permutations": n_perm,
    }


def stage_table(stage: str) -> list[dict]:
    payload = load(stage)
    first = payload["per_seed"][0]
    rows = []
    for name in FUNCTIONALS:
        if name not in payload["aggregate"]["marginal_dr"]:
            continue
        agg = payload["aggregate"]["marginal_dr"][name]
        ifse = np.mean([s["if_se"][name] for s in payload["per_seed"]])
        plugin = np.mean([s["marginal_plugin"][name] for s in payload["per_seed"]])
        rows.append(
            {
                "stage": stage,
                "functional": name,
                "dr_mean": agg["mean"],
                "dr_seed_sd": agg["std"],
                "if_se_mean": float(ifse),
                "plugin_mean": float(plugin),
                "n": first["n"],
                "n_treated": first["n_treated"],
                "n_control": first["n_control"],
                "selected_shrinkage": payload["aggregate"]["selected_contrast_shrinkage"],
            }
        )
    return rows


def main() -> None:
    ds = AppliedDataset.load(OUT / "data")

    stages = ["primary", "primary_extras", "capped", "placebo", "noise", "sens15", "lowsat", "hisat", "total", "logcons"]
    rows = []
    for stage in stages:
        if (RUNS / stage / "results.json").exists():
            rows.append(pd.DataFrame(stage_table(stage)))
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "summary_tables.csv", index=False)

    raw = {
        "levels": raw_levels(ds),
        "unadjusted_marginals": raw_functionals(ds),
        "unadjusted_bins": raw_bins(ds),
        "permutation": permutation_test(ds),
        "coordinate_raw": coordinate_raw(ds),
    }
    with open(OUT / "raw_checks.json", "w") as fh:
        json.dump(raw, fh, indent=2)

    primary = load("primary")
    primary_extra = load("primary_extras")
    for name in ("mean", "sd", "skew", "tail", "reference"):
        for key in ("marginal_dr", "dr_marginal") if False else ("marginal_dr",):
            assert (
                primary["aggregate"][key][name]["mean"]
                == primary_extra["aggregate"][key][name]["mean"]
            ), f"{name} differs between primary and primary_extras"
    capped = load("capped")
    placebo = load("placebo")
    noise = load("noise")
    summary = {
        "primary": {
            name: {
                "dr_mean": primary_extra["aggregate"]["marginal_dr"][name]["mean"],
                "dr_seed_sd": primary_extra["aggregate"]["marginal_dr"][name]["std"],
                "if_se": float(
                    np.mean([s["if_se"][name] for s in primary_extra["per_seed"]])
                ),
                "plugin_mean": float(
                    np.mean([s["marginal_plugin"][name] for s in primary_extra["per_seed"]])
                ),
                "bins_mean": primary_extra["aggregate"]["bin_contrasts_dr"][name]["mean"],
                "bins_seed_sd": primary_extra["aggregate"]["bin_contrasts_dr"][name]["std"],
            }
            for name in FUNCTIONALS
            if name in primary_extra["aggregate"]["marginal_dr"]
        },
        "capped_reference": {
            "dr_mean": capped["aggregate"]["marginal_dr"]["reference"]["mean"],
            "dr_seed_sd": capped["aggregate"]["marginal_dr"]["reference"]["std"],
            "if_se": float(np.mean([s["if_se"]["reference"] for s in capped["per_seed"]])),
        },
        "placebo": {
            name: {
                "mean": placebo["aggregate"]["marginal_dr"][name]["mean"],
                "seed_sd": placebo["aggregate"]["marginal_dr"][name]["std"],
                "values": placebo["aggregate"]["marginal_dr"][name]["values"],
            }
            for name in FUNCTIONALS
            if name in placebo["aggregate"]["marginal_dr"]
        },
        "noise": {
            name: {
                "mean": noise["aggregate"]["marginal_dr"][name]["mean"],
                "seed_sd": noise["aggregate"]["marginal_dr"][name]["std"],
                "values": noise["aggregate"]["marginal_dr"][name]["values"],
            }
            for name in FUNCTIONALS
            if name in noise["aggregate"]["marginal_dr"]
        },
        "raw_checks": raw,
        "propensity": primary["per_seed"][0]["propensity"],
        "selection_records_seed0": primary["per_seed"][0]["selection_records"],
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(table.to_string(index=False))
    print("\nunadjusted marginals:", {k: round(v, 1) for k, v in raw["unadjusted_marginals"].items()})
    print("permutation p:", {k: round(v, 4) for k, v in raw["permutation"]["p_value_two_sided"].items()})
    print("unadjusted bins:", {k: [round(x, 1) for x in v] for k, v in raw["unadjusted_bins"].items()})
    print(
        "permutation p bins:",
        {k: [round(x, 4) for x in v] for k, v in raw["permutation"]["p_value_two_sided_bins"].items()},
    )


if __name__ == "__main__":
    main()
