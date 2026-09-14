#!/usr/bin/env python3
"""Fit the frozen WCF specification on one pre-registered dataset window.

Run from the repository root:

    PYTHONPATH=src python research/applied/primary_state_minwage/fit_wcf.py \
        --data-dir results/applied_study_exploration/primary_state_minwage/data \
        --out results/applied_study_exploration/primary_state_minwage/results/fit_primary_seed0.json

The frozen defaults live inside `run_wcf`; `--reduced` exists only for smoke
tests and is never used for reported numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from wasserstein_causal_forests.applied.adapter import (  # noqa: E402
    AppliedDataset,
    energy_score_on,
    grid_weights,
    run_wcf,
)


def study_extras(bench_path: Path) -> dict:
    """Study-specific functionals: p10, p90, mean-normalized and US-2016 refs.

    For the first-difference variant the benchmark file carries a compression
    ramp instead of the level benchmarks, so only p10, p90 and the ramp
    reference are returned there.
    """

    b = np.load(bench_path)
    files = set(b.files)
    w = grid_weights(25)

    def p10(q):
        return np.asarray(q)[:, 2]

    def p90(q):
        return np.asarray(q)[:, 22]

    extras: dict = {"p10": p10, "p90": p90}

    if "qstar_ramp" in files:
        ramp = b["qstar_ramp"].astype(float)

        def ref_ramp(q):
            block = np.asarray(q, dtype=float)
            return np.sqrt(((block - ramp[None, :]) ** 2) @ w)

        extras["ref_ramp"] = ref_ramp
        return extras

    nordic = b["nordic_raw"].astype(float)
    us2016 = b["us2016"].astype(float)

    def ref_us2016(q):
        block = np.asarray(q, dtype=float)
        return np.sqrt(((block - us2016[None, :]) ** 2) @ w)

    def ref_mean_norm(q):
        block = np.asarray(q, dtype=float)
        mean = block @ w
        star_mean = float(nordic @ w)
        return np.sqrt((((block / mean[:, None]) - (nordic / star_mean)[None, :]) ** 2) @ w)

    extras.update(ref_us2016=ref_us2016, ref_mean_norm=ref_mean_norm)
    return extras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--treatment", choices=["A1", "A2", "A3"], default="A1")
    parser.add_argument("--placebo", action="store_true",
                        help="permute the treatment label with rng 10000+seed")
    parser.add_argument("--drop-states", type=str, default="",
                        help="comma-separated FIPS codes to drop (leave-one-state-out)")
    parser.add_argument("--trim-lo", type=float, default=None,
                        help="drop units whose full-sample fitted propensity is below this value")
    parser.add_argument("--trim-hi", type=float, default=None,
                        help="drop units whose full-sample fitted propensity is above this value")
    parser.add_argument("--reduced", action="store_true",
                        help="smoke-test settings only (50 trees, 2 folds, 5 particles)")
    args = parser.parse_args()

    ds = AppliedDataset.load(args.data_dir)
    extras = study_extras(args.data_dir / "benchmarks.npz")

    if args.drop_states:
        import pandas as pd

        fips = [int(x) for x in args.drop_states.split(",")]
        panel = pd.read_csv(args.data_dir / "panel_with_lags.csv")
        keep = ~panel["statefips"].isin(fips)
        assert keep.sum() == ds.X.shape[0] - int((~keep).sum())
        ds = AppliedDataset(
            study=ds.study + "_drop" + "_".join(str(f) for f in fips),
            X=ds.X[keep.to_numpy()],
            A=ds.A[keep.to_numpy()],
            Q=ds.Q[keep.to_numpy()],
            moderator_raw=ds.moderator_raw[keep.to_numpy()],
            q_star=ds.q_star,
            feature_names=list(ds.feature_names),
            meta={**ds.meta, "dropped_states": fips},
        ).validate()

    trim_mask = None
    n_dropped = 0
    trim_probe = None
    if args.trim_lo is not None or args.trim_hi is not None:
        from wasserstein_causal_forests.cwdb.dr_calibration import FunctionalAIPW

        lo = 0.0 if args.trim_lo is None else float(args.trim_lo)
        hi = 1.0 if args.trim_hi is None else float(args.trim_hi)
        if not 0.0 <= lo < hi <= 1.0:
            parser.error("--trim-lo and --trim-hi must satisfy 0 <= lo < hi <= 1")
        trim_probe = FunctionalAIPW(n_bins=4).fit(
            observed={},
            oof_arm_means={},
            X=ds.X,
            treatment=ds.A,
            random_state=args.seed + 31,
        )
        ehat = np.asarray(trim_probe.ehat_train_, dtype=float)
        trim_mask = (ehat >= lo) & (ehat <= hi)
        n_dropped = int((~trim_mask).sum())
        trim_stats = {
            "min": float(np.min(ehat)),
            "max": float(np.max(ehat)),
            "mean": float(np.mean(ehat)),
            "treated_mean": float(np.mean(ehat[ds.A == 1])),
            "control_mean": float(np.mean(ehat[ds.A == 0])),
            "share_outside_0p1_0p9": float(np.mean((ehat < 0.1) | (ehat > 0.9))),
            "frac_retained": float(np.mean(trim_mask)),
        }
        if trim_mask.sum() < 20:
            raise SystemExit(f"trim [{lo}, {hi}] leaves only {int(trim_mask.sum())} units")
        ds = AppliedDataset(
            study=ds.study + f"_trim{int(round(lo * 100))}_{int(round(hi * 100))}",
            X=ds.X[trim_mask],
            A=ds.A[trim_mask],
            Q=ds.Q[trim_mask],
            moderator_raw=ds.moderator_raw[trim_mask],
            q_star=ds.q_star,
            feature_names=list(ds.feature_names),
            meta={**ds.meta, "trim_lo": lo, "trim_hi": hi, "n_dropped": n_dropped},
        ).validate()

    treatment_override = None
    if args.placebo:
        rng = np.random.default_rng(10_000 + args.seed)
        treatment_override = rng.permutation(ds.A)
    elif args.treatment != "A1":
        import pandas as pd

        panel = pd.read_csv(args.data_dir / "panel_with_lags.csv")
        treatment_override = panel[args.treatment].to_numpy(dtype=np.int64)
        if trim_mask is not None:
            treatment_override = treatment_override[trim_mask]
        assert len(treatment_override) == ds.X.shape[0], "treatment rows do not match dataset"

    kwargs = {}
    if args.reduced:
        kwargs = dict(n_estimators=50, n_folds=2, n_particles=5)

    started = time.time()
    model, results = run_wcf(
        ds,
        random_state=args.seed,
        extra_functionals=extras,
        treatment_override=treatment_override,
        **kwargs,
    )
    elapsed = time.time() - started
    results["elapsed_seconds"] = elapsed
    results["reduced_settings"] = bool(args.reduced)
    results["treatment_variant"] = args.treatment
    results["placebo"] = bool(args.placebo)
    results["data_dir"] = str(args.data_dir)
    results["energy_score_train"] = energy_score_on(model, ds.X, ds.A, ds.Q)
    if trim_probe is not None:
        results["trim_lo"] = float(0.0 if args.trim_lo is None else args.trim_lo)
        results["trim_hi"] = float(1.0 if args.trim_hi is None else args.trim_hi)
        results["n_dropped"] = int(n_dropped)
        results["trim_selection_propensity"] = trim_stats
    results["n_features"] = int(ds.X.shape[1])
    results["feature_names"] = list(ds.feature_names)
    results["selected_contrast_shrinkage"] = float(model.selected_contrast_shrinkage_)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"wrote {args.out} in {elapsed:.1f}s | n={results['n']} "
          f"A1={results['n_treated']} A0={results['n_control']} "
          f"shrinkage={results['selected_contrast_shrinkage']}")


if __name__ == "__main__":
    main()
