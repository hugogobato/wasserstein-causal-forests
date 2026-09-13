#!/usr/bin/env python3
"""Write the WCF sensitivity tables, decision, and figures.

    python research/analyze_wcf_sensitivity.py
    python research/analyze_wcf_sensitivity.py --results PATH --output-dir PATH

Reads the merged parquet plus the frozen manifest, writes every preregistered
table and figure under ``results/wcf_sensitivity/analysis/``, and records the
fixed K/M decision in ``decision.json`` and ``summary.md``. The command never
runs simulation cells: it only aggregates rows that a prior ``merge`` produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.wcf_sensitivity_analysis import (  # noqa: E402
    alignment,
    cell_metrics,
    decide_k_m,
    failure_summary,
    factorial,
    load_results,
    native_vs_common,
    one_factor_k,
    one_factor_m,
    plot_sensitivity,
    propensity_models,
    replication_summary,
    runtime_memory,
    sym_methods,
    write_tables,
)

DEFAULT_RESULTS = (
    ROOT / "results" / "wcf_sensitivity" / "merged" / "wcf_sensitivity_results.parquet"
)
DEFAULT_MANIFEST = ROOT / "results" / "wcf_sensitivity" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "results" / "wcf_sensitivity" / "analysis"

DECISION_A_COLUMNS = (
    "dgp",
    "n_grid",
    "n_particles",
    "metric",
    "target_id",
    "primary_mean",
    "alternative_mean",
    "absolute_delta",
    "relative_delta",
    "within_tolerance",
    "evaluated",
)
DECISION_B_COLUMNS = (
    "dgp",
    "n_grid",
    "n_particles",
    "metric",
    "target_id",
    "n_pairs",
    "paired_delta",
    "paired_mc_se",
    "improves",
    "evaluated",
)
ONEFACTOR_HEADLINE_COLUMNS = (
    "dgp",
    "n_grid",
    "n_particles",
    "metric",
    "target_id",
    "mean",
    "mc_se",
    "primary_mean",
    "absolute_delta",
    "relative_delta",
    "paired_delta",
    "paired_mc_se",
    "n_pairs",
)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    raise TypeError(f"cannot serialise {type(value)!r}")


def _sanitize(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, dict):
            return {key: _sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_sanitize(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not np.isfinite(float(value)) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    return value


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.4g}"
    return str(value).replace("|", "\\|")


def _markdown_table(
    frame: Any, columns: tuple[str, ...], max_rows: int = 60
) -> str:
    if frame is None or len(frame) == 0:
        return "_No rows._"
    present = [column for column in columns if column in frame.columns]
    if not present:
        return "_No matching columns._"
    head = frame[present].head(max_rows)
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join("---" for _ in present) + " |",
    ]
    for record in head.to_dict("records"):
        lines.append(
            "| " + " | ".join(_format_cell(record[column]) for column in present) + " |"
        )
    if len(frame) > max_rows:
        lines.append("")
        lines.append(f"_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def _decision_section(decision: dict[str, Any]) -> str:
    condition_a = decision["condition_a"]
    condition_b = decision["condition_b"]
    lines = [
        "## Decision rule",
        "",
        (
            "The rule was fixed before outcomes were inspected. It retains the "
            "primary (K, M) = (25, 10) when condition A holds and condition B "
            "does not. Condition A requires every one-factor alternative "
            "(5, 10), (49, 10), (25, 5), (25, 25) to stay within 10 percent of "
            "the primary on REF-TCATE-K and LAW-A-K in IC1 and IC3. Condition B "
            "holds when some larger-resolution pair (49, 10), (25, 25), or "
            "(49, 25) improves both targets in both regimes by more than two "
            "paired Monte Carlo standard errors."
        ),
        "",
        (
            f"**Verdict:** `retain_primary = {decision['retain_primary']}`, "
            f"`rule_failed = {decision['rule_failed']}`, "
            f"`evaluable = {decision['evaluable']}`. "
            f"Condition A passed: {condition_a['passed']} "
            f"({condition_a['n_within']} of {condition_a['n_records']} records "
            f"within tolerance). Condition B any improvement: "
            f"{condition_b['any_improves']}. "
            f"Recommended pair: {decision['recommended_pair']}. "
            f"Reason: {decision['reason']}."
        ),
        "",
        "### Condition A details",
        "",
        _markdown_table(
            _records_frame(condition_a.get("records", [])), DECISION_A_COLUMNS
        ),
        "",
        "### Condition B details",
        "",
        _markdown_table(
            _records_frame(
                [
                    record
                    for combination in condition_b.get("combinations", [])
                    for record in combination.get("records", [])
                ]
            ),
            DECISION_B_COLUMNS,
        ),
        "",
    ]
    return "\n".join(lines)


def _records_frame(records: list[dict[str, Any]]) -> Any:
    import pandas as pd

    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def _render_summary(
    *,
    results_path: Path,
    merged_checksum: str | None,
    manifest: dict[str, Any] | None,
    frame: Any,
    cells: Any,
    failures: Any,
    decision: dict[str, Any],
    one_factor_k_table: Any,
    one_factor_m_table: Any,
    runtime_table: Any,
) -> str:
    manifest_line = "manifest not found beside the results"
    if manifest is not None:
        manifest_line = (
            f"manifest `{manifest.get('manifest_contract_id')}` with checksum "
            f"`{manifest.get('manifest_checksum')}`, "
            f"{manifest.get('n_cells')} declared cells"
        )
    checksum_line = merged_checksum or "unavailable"
    n_failed = int(failures["n_failed_cells"].sum()) if len(failures) else 0
    if n_failed == 0:
        failure_phrase = "no failed cells are recorded"
    elif n_failed == 1:
        failure_phrase = "1 failed cell is recorded with its reason"
    else:
        failure_phrase = f"{n_failed} failed cells are recorded with their reasons"
    lines = [
        "# WCF sensitivity analysis",
        "",
        (
            f"Input rows come from `{results_path}` (SHA-256 `{checksum_line}`) "
            f"and the frozen {manifest_line}. The merged frame holds "
            f"{len(frame)} rows covering {frame['cell_key'].nunique() if 'cell_key' in frame.columns else 0} "
            f"cells; the failure table shows that {failure_phrase}. Metric "
            "tables exclude failed and inapplicable rows."
        ),
        "",
        (
            "Aggregation first averages arm-specific law metrics inside each "
            "cell and keeps every functional TATE and TCATE target separate. "
            "Replication tables report the mean with its Monte Carlo standard "
            "error `sd / sqrt(n_seeds)`, and every sensitivity comparison is a "
            "seed-paired difference with a Monte Carlo SE computed from the "
            "differences themselves."
        ),
        "",
        _decision_section(decision),
        "## One-factor sensitivity, grid resolution (M = 10)",
        "",
        _markdown_table(one_factor_k_table, ONEFACTOR_HEADLINE_COLUMNS, max_rows=120),
        "",
        "## One-factor sensitivity, particle resolution (K = 25)",
        "",
        _markdown_table(one_factor_m_table, ONEFACTOR_HEADLINE_COLUMNS, max_rows=120),
        "",
        "## Runtime and memory",
        "",
        _markdown_table(runtime_table, tuple(runtime_table.columns), max_rows=40),
        "",
        "The complete tables, including paired method comparisons, the factorial "
        "surface, native versus common-grid diagnostics, propensity variants, "
        "the alignment block with its balance numbers, and the decision payload, "
        "are written beside this file as CSV and `decision.json`.",
        "",
    ]
    return "\n".join(lines)


def _checksum(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--design-checks", type=Path, default=None)
    arguments = parser.parse_args()

    try:
        frame = load_results(arguments.results)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    manifest: dict[str, Any] | None = None
    if arguments.manifest.is_file():
        manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))

    cells = cell_metrics(frame)
    replication = replication_summary(cells)
    failures = failure_summary(frame)
    one_factor_k_table = one_factor_k(frame)
    one_factor_m_table = one_factor_m(frame)
    factorial_table = factorial(frame)
    native_common_table = native_vs_common(frame)
    sym_table = sym_methods(frame)
    propensity_table = propensity_models(frame)
    align_paired, align_balance_table = alignment(
        frame, design_checks_path=arguments.design_checks
    )
    runtime_table = runtime_memory(frame)
    decision = decide_k_m(frame)

    tables = {
        "cell_metrics": cells,
        "replication_summary": replication,
        "failure_summary": failures,
        "one_factor_k": one_factor_k_table,
        "one_factor_m": one_factor_m_table,
        "factorial": factorial_table,
        "native_vs_common": native_common_table,
        "sym_methods": sym_table,
        "propensity_models": propensity_table,
        "alignment_paired": align_paired,
        "alignment_balance": align_balance_table,
        "runtime_memory": runtime_table,
    }

    output_dir = arguments.output_dir
    written = write_tables(tables, output_dir)
    figures = plot_sensitivity(tables, output_dir)

    if manifest is not None:
        decision["manifest_checksum"] = manifest.get("manifest_checksum")
        decision["manifest_contract_id"] = manifest.get("manifest_contract_id")
    decision["results_path"] = str(arguments.results)
    decision_path = output_dir / "decision.json"
    decision_path.write_text(
        json.dumps(_sanitize(decision), indent=2, default=_json_default),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(
        _render_summary(
            results_path=arguments.results,
            merged_checksum=_checksum(arguments.results),
            manifest=manifest,
            frame=frame,
            cells=cells,
            failures=failures,
            decision=decision,
            one_factor_k_table=one_factor_k_table,
            one_factor_m_table=one_factor_m_table,
            runtime_table=runtime_table,
        ),
        encoding="utf-8",
    )

    print(
        "decision: retain_primary="
        f"{decision['retain_primary']} rule_failed={decision['rule_failed']} "
        f"evaluable={decision['evaluable']} "
        f"recommended_pair={decision['recommended_pair']}"
    )
    print(f"reason: {decision['reason']}")
    print(f"wrote {len(written)} tables and {len(figures)} figures to {output_dir}")
    print(f"wrote {decision_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
