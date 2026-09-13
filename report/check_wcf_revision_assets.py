"""Independent recalculation of manuscript summary values from result rows."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "merged_phase6": "results/merged_phase6/phase6_results.parquet",
    "merged_phase65": "results/merged_phase65/phase65_results.parquet",
    "merged_original_drf": "results/merged_original_drf/main_results.parquet",
    "merged_original_causal_drf": "results/merged_original_causal_drf/main_results.parquet",
}


def main():
    frames = {name: pd.read_parquet(ROOT / path) for name, path in SOURCES.items()}
    counts = {}
    for name in ("evidence", "functional_detail"):
        table = pd.read_csv(ROOT / f"report/tables_generated/wcf_revision_{name}.csv")
        for row in table.itertuples(index=False):
            source = frames[row.source]
            keep = (
                (source.grid == row.grid) & (source.dgp == row.dgp)
                & (source.method == row.method) & (source.n_train == row.n_train)
                & (source.n_grid == 25) & (source.n_particles == 10)
                & (source.metric == row.metric) & (source.status == "ok")
                & source.seed.isin(range(10))
            )
            if name == "functional_detail":
                keep &= source.target_id == row.target_id
            values = source.loc[keep].groupby("seed").value.mean().sort_index()
            assert values.index.tolist() == list(range(10))
            mean = sum(values) / 10
            se = np.sqrt(sum((values - mean) ** 2) / (9 * 10))
            assert abs(mean - row.mean) < 5e-10, (name, row.dgp, row.metric)
            assert abs(se - row.se) < 5e-10, (name, row.dgp, row.metric)
        counts[name] = len(table)
    result = {"status": "passed", "summary_rows_recomputed": counts,
              "tolerance": 5e-10, "replications_per_summary": 10,
              "source_files": SOURCES}
    (ROOT / "report/revision_asset_checks.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
