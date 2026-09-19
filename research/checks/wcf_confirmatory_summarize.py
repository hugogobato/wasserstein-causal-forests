#!/usr/bin/env python3
"""Summarize the merged confirmatory results with arm-aware identities.

The frozen runner's ``summarize`` subcommand predates arm-specific metrics and
pivots on a key that omits ``arm``, so the paired comparison fails on the
fifty-replication results.  The run itself is frozen, so this companion script
reproduces the same summaries with ``arm`` added to both the aggregation and
pairing identities instead of editing the frozen runner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.run_wcf_confirmatory import (  # noqa: E402
    ERROR_METRIC_SUFFIXES,
    MERGED_DIRECTORY,
    MERGED_PATH,
    _read_rows,
    _rmse_and_se,
    load_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(MERGED_PATH))
    parser.add_argument("--output-directory", default=str(MERGED_DIRECTORY))
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    output = Path(args.output_directory)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    document = load_manifest()
    frame = pd.DataFrame(_read_rows(source))
    frame = frame[(frame.status == "ok") & frame.value.notna()].copy()
    reverse = {value: key for key, value in document["paper_dgp_map"].items()}
    frame["paper_dgp"] = frame.dgp.map(reverse)

    keys = ["paper_dgp", "dgp", "method", "metric", "target_id", "detail", "arm"]
    records = []
    for key, group in frame.groupby(keys, dropna=False):
        values = group.value.to_numpy(float)
        if str(key[3]).endswith(ERROR_METRIC_SUFFIXES):
            estimate, mc_se = _rmse_and_se(values)
            aggregation = "root_mean_square_over_replications"
        else:
            estimate = float(np.mean(values))
            mc_se = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0
            )
            aggregation = "mean_over_replications"
        records.append({
            **dict(zip(keys, key)),
            "estimate": estimate,
            "mc_se": mc_se,
            "n_replications": len(values),
            "aggregation": aggregation,
        })
    summary = pd.DataFrame(records).sort_values(keys)
    summary.to_csv(output / "confirmatory_summary.csv", index=False)

    paired = []
    error = frame[frame.metric.str.endswith(ERROR_METRIC_SUFFIXES)].copy()
    pair_keys = ["paper_dgp", "dgp", "metric", "target_id", "detail", "arm"]
    for key, group in error.groupby(pair_keys, dropna=False):
        wide = group.pivot(index="seed", columns="method", values="value")
        methods = sorted(wide.columns)
        for index, left in enumerate(methods):
            for right in methods[index + 1:]:
                block = wide[[left, right]].dropna()
                if len(block) < 2:
                    continue
                x = np.square(block[left].to_numpy(float))
                y = np.square(block[right].to_numpy(float))
                rx, ry = float(np.sqrt(x.mean())), float(np.sqrt(y.mean()))
                gradient = np.array([
                    0.0 if rx == 0 else 1 / (2 * rx),
                    0.0 if ry == 0 else -1 / (2 * ry),
                ])
                covariance = (
                    np.cov(np.column_stack([x, y]), rowvar=False, ddof=1)
                    / len(block)
                )
                se = float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))
                paired.append({
                    **dict(zip(pair_keys, key)),
                    "method_left": left,
                    "method_right": right,
                    "rmse_difference": rx - ry,
                    "paired_mc_se": se,
                    "ci95_low": rx - ry - 1.96 * se,
                    "ci95_high": rx - ry + 1.96 * se,
                    "n_pairs": len(block),
                })
    pd.DataFrame(paired).to_csv(output / "paired_rmse_differences.csv", index=False)
    print(f"wrote {len(summary)} summary rows and {len(paired)} paired comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
