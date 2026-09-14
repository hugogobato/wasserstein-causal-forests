"""Run the frozen WCF specification on the Egger Kenya village dataset.

Stages (run one per process; see --help):
  smoke      quick fit (50 trees, 2 folds) to validate the pipeline and sign
  primary    frozen spec, seeds (0,1,2), primary q_star (pooled control)
  capped     frozen spec, seeds (0,1,2), compressed q_star
  placebo    frozen spec with permuted treatment, seeds (0,1,2)
  sens15     frozen spec restricted to villages with >=15 sampled households
  sens20     frozen spec restricted to villages with >=20 sampled households
  noise      frozen spec on within-village bootstrap-resampled Q, seeds (0,1,2)
  total      robustness outcome: total (nondurable+durable) consumption, seed 0
  logcons    robustness outcome: log(1+nondurable) per-capita consumption, seed 0

Example (from the repo root):
    PYTHONPATH=src OMP_NUM_THREADS=1 python research/applied/egger_kenya/run_wcf.py primary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.applied.adapter import (  # noqa: E402
    AppliedDataset,
    aggregate_replications,
    midpoint_levels,
    quantile_matrix,
    run_placebo,
    run_wcf,
    run_wcf_replications,
    weighted_quantiles,
)

BASE = Path("/tmp/opencode/wcf_scout/egger/replication_package")
OUT = ROOT / "results/applied_study_exploration/egger_kenya"
DATA = OUT / "data"
RUNS = OUT / "runs"
LEVELS = midpoint_levels()
WEIGHT = "hhweight_EL"


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, cls=Encoder)


def load_ds() -> AppliedDataset:
    ds = AppliedDataset.load(DATA)
    order = pd.read_csv(OUT / "village_quantiles.csv").village_code.to_numpy()
    cov = pd.read_csv(OUT / "village_covariates.csv")
    assert np.array_equal(order, cov.village_code.to_numpy())
    return ds


def hh_frame(cols: list[str]) -> pd.DataFrame:
    df, _ = pyreadstat.read_dta(str(BASE / "code/data/GE_HHLevel_ECMA.dta"), usecols=cols)
    return df


def subset(ds: AppliedDataset, mask: np.ndarray) -> AppliedDataset:
    return AppliedDataset(
        study=ds.study,
        X=ds.X[mask],
        A=ds.A[mask],
        Q=ds.Q[mask],
        moderator_raw=ds.moderator_raw[mask],
        q_star=ds.q_star,
        feature_names=list(ds.feature_names),
        meta=dict(ds.meta),
    ).validate()


def variant_outcome(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (units, Q, q_star, A-aligned treat) for a robustness outcome."""

    hh = hh_frame(["village_code", "treat", WEIGHT, "nondurables_exp_pc", "durables_exp_pc"])
    if name == "total":
        hh["value"] = hh.nondurables_exp_pc + hh.durables_exp_pc
    elif name == "logcons":
        hh["value"] = np.log1p(hh.nondurables_exp_pc)
    else:
        raise ValueError(name)
    mask = hh["value"].notna() & hh[WEIGHT].notna() & (hh[WEIGHT] > 0) & hh.village_code.notna()
    hh = hh[mask]
    units, q = quantile_matrix(
        hh["value"].to_numpy(), hh[WEIGHT].to_numpy(), hh.village_code.to_numpy(), levels=LEVELS
    )
    ctrl = hh[hh.treat == 0]
    q_star = weighted_quantiles(ctrl["value"].to_numpy(), ctrl[WEIGHT].to_numpy(), levels=LEVELS)
    return units, q, q_star, hh


def build_variant_ds(name: str, base: AppliedDataset) -> AppliedDataset:
    units, q, q_star, _ = variant_outcome(name)
    order = pd.read_csv(OUT / "village_covariates.csv").village_code.to_numpy()
    pos = {v: i for i, v in enumerate(order)}
    idx = np.array([pos[v] for v in units])
    ds = AppliedDataset(
        study=f"egger_kenya_{name}",
        X=base.X[idx],
        A=base.A[idx],
        Q=q,
        moderator_raw=base.moderator_raw[idx],
        q_star=q_star,
        feature_names=list(base.feature_names),
        meta=dict(base.meta),
    ).validate()
    ds.save(OUT / "variants" / name / "data")
    return ds


def noisy_q(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Within-village nonparametric bootstrap of household outcomes -> Q_noisy."""

    hh = hh_frame(["village_code", WEIGHT, "nondurables_exp_pc"])
    hh = hh[hh.nondurables_exp_pc.notna() & hh[WEIGHT].notna() & (hh[WEIGHT] > 0)]
    rng = np.random.default_rng(20_000 + seed)
    rows = []
    for _, block in hh.groupby("village_code", sort=True):
        n = len(block)
        pick = rng.integers(0, n, size=n)
        rows.append(block.iloc[pick])
    boot = pd.concat(rows, ignore_index=True)
    units, q = quantile_matrix(
        boot.nondurables_exp_pc.to_numpy(),
        boot[WEIGHT].to_numpy(),
        boot.village_code.to_numpy(),
        levels=LEVELS,
    )
    return units, q


def jsonable(value: object) -> object:
    if callable(value):
        return getattr(value, "__name__", None)
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def run_and_save(
    stage: str,
    ds: AppliedDataset,
    *,
    seeds: tuple[int, ...],
    extra: dict | None = None,
    **kwargs,
) -> dict:
    t0 = time.time()
    per_seed = []
    for seed in seeds:
        model, results = run_wcf(ds, random_state=seed, **kwargs)
        per_seed.append(results)
    aggregate = aggregate_replications(per_seed)
    payload = {
        "stage": stage,
        "seeds": list(seeds),
        "kwargs": {k: jsonable(v) for k, v in kwargs.items()},
        "n": int(ds.X.shape[0]),
        "n_treated": int(ds.A.sum()),
        "n_control": int((ds.A == 0).sum()),
        "elapsed_seconds": time.time() - t0,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    if extra:
        payload.update(extra)
    save_json(RUNS / stage / "results.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["smoke", "primary", "primary_extras", "capped", "placebo", "sens15", "sens20", "noise", "total", "logcons", "lowsat", "hisat"],
    )
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    ds = load_ds()
    n_hh = pd.read_csv(OUT / "village_covariates.csv").n_hh_sample.to_numpy()

    if args.stage == "smoke":
        payload = run_and_save("smoke", ds, seeds=(0,), n_folds=2, n_estimators=50, n_particles=5)
        print(json.dumps(payload["per_seed"][0]["marginal_dr"], indent=2))
        return

    if args.stage == "primary":
        payload = run_and_save("primary", ds, seeds=(0, 1, 2))
        print(json.dumps(payload["aggregate"]["marginal_dr"], indent=2))

    elif args.stage == "primary_extras":
        extras = {"p10": lambda q: q[:, 2], "p90": lambda q: q[:, 22]}
        payload = run_and_save("primary_extras", ds, seeds=(0, 1, 2), extra_functionals=extras)
        print(json.dumps(payload["aggregate"]["marginal_dr"], indent=2))

    elif args.stage == "capped":
        capped = np.asarray(ds.meta["q_star_capped"], dtype=float)
        ds_capped = replace(ds, q_star=capped).validate()
        payload = run_and_save(
            "capped", ds_capped, seeds=(0, 1, 2), extra={"primary_q_star": ds.q_star.tolist()}
        )
        print(json.dumps(payload["aggregate"]["marginal_dr"], indent=2))

    elif args.stage == "placebo":
        t0 = time.time()
        per_seed = run_placebo(ds, seeds=(0, 1, 2))
        aggregate = aggregate_replications(per_seed)
        save_json(
            RUNS / "placebo/results.json",
            {
                "stage": "placebo",
                "seeds": [0, 1, 2],
                "elapsed_seconds": time.time() - t0,
                "per_seed": per_seed,
                "aggregate": aggregate,
            },
        )
        print(json.dumps(aggregate["marginal_dr"], indent=2))

    elif args.stage in ("sens15", "sens20"):
        cut = 15 if args.stage == "sens15" else 20
        mask = n_hh >= cut
        runs = RUNS / args.stage
        a_sub = ds.A[mask]
        if min(np.bincount(a_sub, minlength=2)) < 5:
            save_json(
                runs / "results.json",
                {
                    "stage": args.stage,
                    "cut": cut,
                    "n": int(mask.sum()),
                    "n_treated": int(a_sub.sum()),
                    "n_control": int((a_sub == 0).sum()),
                    "status": "skipped: fewer than five units in an arm (adapter guard)",
                },
            )
            print("skipped: arm too small", np.bincount(a_sub, minlength=2).tolist())
            return
        ds_sub = subset(ds, mask)
        payload = run_and_save(
            args.stage, ds_sub, seeds=(0, 1, 2), extra={"sensitivity_cut": cut}
        )
        print(json.dumps(payload["aggregate"]["marginal_dr"], indent=2))

    elif args.stage == "noise":
        # one noisy Q per seed, refit with the primary benchmark
        t0 = time.time()
        per_seed = []
        for seed in (0, 1, 2):
            units, q = noisy_q(seed)
            cov = pd.read_csv(OUT / "village_covariates.csv").village_code.to_numpy()
            assert np.array_equal(units, cov), "village order mismatch in noisy Q"
            ds_noisy = replace(ds, Q=q).validate()
            _, results = run_wcf(ds_noisy, random_state=seed)
            per_seed.append(results)
        aggregate = aggregate_replications(per_seed)
        save_json(
            RUNS / "noise/results.json",
            {
                "stage": "noise",
                "seeds": [0, 1, 2],
                "elapsed_seconds": time.time() - t0,
                "per_seed": per_seed,
                "aggregate": aggregate,
            },
        )
        print(json.dumps(aggregate["marginal_dr"], indent=2))

    elif args.stage in ("lowsat", "hisat"):
        cov = pd.read_csv(OUT / "village_covariates.csv")
        want = 0 if args.stage == "lowsat" else 1
        mask = cov.hi_sat.to_numpy() == want
        a_sub = ds.A[mask]
        if min(np.bincount(a_sub, minlength=2)) < 5:
            print("skipped: arm too small", np.bincount(a_sub, minlength=2).tolist())
            return
        ds_sub = subset(ds, mask)
        payload = run_and_save(
            args.stage, ds_sub, seeds=(0, 1, 2), extra={"hi_sat": want}
        )
        print(json.dumps(payload["aggregate"]["marginal_dr"], indent=2))

    elif args.stage in ("total", "logcons"):
        ds_var = build_variant_ds(args.stage, ds)
        payload = run_and_save(args.stage, ds_var, seeds=(0,))
        print(json.dumps(payload["per_seed"][0]["marginal_dr"], indent=2))


if __name__ == "__main__":
    main()
