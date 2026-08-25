#!/usr/bin/env python3
"""Assemble the G3 gate memo from the analysis payload.

Run last:

    python research/checks/g3_write_memo.py

Writes `research/gates/G3_simulation_memo.md`. The evidence sections are
generated from `results/merged/analysis_payload.json` so no number in the memo
is transcribed by hand, and the adversarial reading is assembled first and from
the same data as the favourable one: every regime where the claimant loses by
more than two paired standard errors is listed before any regime where it wins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PAYLOAD = ROOT / "results" / "merged" / "analysis_payload.json"
AUDIT = ROOT / "results" / "merged" / "merge_audit.json"
INDEPENDENT = ROOT / "results" / "merged" / "gate_flags_independent.json"
OUTPUT = ROOT / "research" / "gates" / "G3_simulation_memo.md"


def comparison_table(comparisons: list[dict]) -> list[str]:
    """Render paired comparisons, and capability records, in one table.

    A capability record has no paired statistics because the comparator
    produced no estimate at all. It is shown with its cells blank rather than
    omitted, so a reader sees on which targets the comparison was an accuracy
    test and on which it was not.
    """

    if not comparisons:
        return ["_No comparison had enough paired seeds to report._", ""]
    lines = [
        "| Regime | Target | Comparator | Claimant | Comparator value | "
        "Paired difference | Paired SE | Seeds won | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in comparisons:
        if item.get("kind") == "capability":
            claimant = item.get("claimant_mean")
            lines.append(
                f"| {item['dgp']} | `{item.get('target_id') or item['metric']}` | "
                f"`{item['comparator']}` | "
                f"{claimant:.4g} | _no estimate_ | | | | "
                f"claimant, on capability |"
            )
            continue
        if item["claimant_wins"]:
            verdict = "claimant"
        elif item["comparator_wins"]:
            verdict = "**comparator**"
        else:
            verdict = "tie"
        label = item.get("target_id") or item["metric"]
        lines.append(
            f"| {item['dgp']} | `{label}` | `{item['comparator']}` | "
            f"{item['claimant_mean']:.4g} | {item['comparator_mean']:.4g} | "
            f"{item['paired_mean_difference']:+.4g} | "
            f"{item['paired_standard_error']:.3g} | "
            f"{item['seed_win_fraction']:.0%} | {verdict} |"
        )
    lines.append("")
    return lines


def main() -> int:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    independent = (
        json.loads(INDEPENDENT.read_text(encoding="utf-8"))
        if INDEPENDENT.exists()
        else None
    )
    flags = payload["gate_flags"]
    summary = flags["summary"]

    lines: list[str] = []
    add = lines.append

    add("# Phase G3 gate memo: preregistered simulation tournament")
    add("")
    add(f"**Verdict:** `{summary['verdict']}`")
    add(f"**Rules passed:** {summary['n_passed']} of {summary['n_rules']}")
    add(f"**Merged results checksum:** `{payload['merged_checksum']}`")
    add(f"**Code revision:** `{payload['git_revision']}`")
    add(f"**Manifest checksum:** `{audit['manifest_checksum']}`")
    add(f"**Preregistration:** `research/simulation_preregistration.md`")
    add("")
    add(
        "Every threshold applied below was frozen in the preregistration before "
        "the first decisive seed. Nothing in this memo re-chooses a metric, a "
        "regime, or a cutoff."
    )
    add("")

    # ------------------------------------------------------------ reconciliation
    add("## 1. Execution and reconciliation")
    add("")
    add(
        f"The manifest declares {audit['n_manifest_cells']} cells; "
        f"{audit['n_observed_cells']} produced rows, for a total of "
        f"{audit['n_rows']} result rows. Failed cells: "
        f"{audit['n_failed_cells']}. Merge audit status: `{audit['status']}`."
    )
    add("")
    if audit["problems"]:
        add("Reconciliation problems, reported rather than cleaned:")
        add("")
        for problem in audit["problems"]:
            add(f"- {problem}")
        add("")
    add("Failure rate by method:")
    add("")
    add("| Method | Cells | Failed | Rate |")
    add("|---|---|---|---|")
    for method, entry in payload["failures"].items():
        add(
            f"| `{method}` | {entry['n_cells']} | {entry['n_failed']} | "
            f"{entry['failure_rate']:.1%} |"
        )
    add("")

    # -------------------------------------------------------------- adversarial
    add("## 2. The adversarial reading, stated first")
    add("")
    losses = [
        item
        for item in flags["rule_2_law_advantage"]["comparisons"]
        + flags["rule_3_transfer"]["comparisons"]
        + flags["rule_4_beats_direct_learner"]["comparisons"]
        if item.get("comparator_wins")
    ]
    if losses:
        add(
            "C-WDB-v1 is beaten by more than two paired standard errors in the "
            "following regimes and targets. These are the cells a sceptical "
            "reader should weigh first, and averaging them into an overall "
            "score would hide them."
        )
        add("")
        lines.extend(comparison_table(losses))
    else:
        add(
            "On the preregistered comparisons, no comparator beats C-WDB-v1 by "
            "more than two paired standard errors. That is a weaker statement "
            "than it sounds: it covers only the frozen metric and regime list, "
            "and the ties below carry no evidence in either direction."
        )
        add("")

    ties = [
        item
        for item in flags["rule_2_law_advantage"]["comparisons"]
        if not item["claimant_wins"] and not item["comparator_wins"]
    ]
    if ties:
        add(
            "Regimes where the primary law metric cannot separate C-WDB-v1 from "
            "Causal-DRF at the frozen decision multiple:"
        )
        add("")
        lines.extend(comparison_table(ties))

    add("Structural limits of this tournament, restated from the preregistration:")
    add("")
    add(
        "1. PTA-F runs only at $K=5$, because its cost accelerates in the target "
        "dimension; no conclusion about it at the working resolution is "
        "available. 2. The `Uncertainty usable` claim row is **not evaluated**: "
        "C-WDB has no interval construction, and contract Section 4 forbids "
        "substituting a posterior-draw quantity. 3. No claim about Causal-DRF's "
        "band coverage relative to the published two-forest benchmark is made, "
        "per the Phase 4 limitation. 4. Grid-resolution conclusions hold for "
        "$K\\in\\{5,25,49\\}$ only."
    )
    add("")

    # ------------------------------------------------------------- rule by rule
    add("## 3. Gate rules")
    add("")
    add("| Rule | Statement | Result |")
    add("|---|---|---|")
    for name in flags:
        if name == "summary":
            continue
        entry = flags[name]
        mark = "PASS" if entry["passed"] else "**FAIL**"
        add(f"| `{name}` | {entry.get('statement', '')} | {mark} |")
    add("")

    rule1 = flags["rule_1_correctness"]
    add("### Rule 1, correctness and nulls")
    add("")
    add(
        f"D0 `mean_quantile_rmse` = {rule1['d0_mean_quantile_rmse']:.4g}; "
        f"D2 = {rule1['d2_mean_quantile_rmse']:.4g} against a best baseline of "
        f"{rule1['d2_best_baseline']:.4g}, a false-effect ratio of "
        f"{rule1['d2_false_effect_ratio']:.3g}."
    )
    add("")

    add("### Rule 2, primary law metric against Causal-DRF")
    add("")
    add(
        f"Wins: {flags['rule_2_law_advantage']['n_wins']} of "
        f"{flags['rule_2_law_advantage']['min_wins']} required, in "
        f"{flags['rule_2_law_advantage']['winning_dgps'] or 'no regime'}."
    )
    add("")
    lines.extend(comparison_table(flags["rule_2_law_advantage"]["comparisons"]))

    add("### Rules 3 and 4, transfer to a causal functional")
    add("")
    add(
        f"Against Causal-DRF: {flags['rule_3_transfer']['n_wins']} winning "
        f"targets ({flags['rule_3_transfer']['winning_targets'] or 'none'}). "
        f"Against PTA-S on those same targets: "
        f"{flags['rule_4_beats_direct_learner']['n_wins']} wins, of which "
        f"{flags['rule_4_beats_direct_learner'].get('n_accuracy_wins', 0)} are "
        f"accuracy wins on a target PTA-S also estimates and "
        f"{flags['rule_4_beats_direct_learner'].get('n_capability_wins', 0)} are "
        f"capability wins on a target it cannot estimate at all."
    )
    add("")
    lines.extend(comparison_table(flags["rule_3_transfer"]["comparisons"]))
    lines.extend(comparison_table(flags["rule_4_beats_direct_learner"]["comparisons"]))

    rule5 = flags["rule_5_no_collapse"]
    add("### Rule 5, particle collapse")
    add("")
    add(
        f"D6 mode coverage {rule5['d6_mode_coverage']:.3g}; effective particle "
        f"support {rule5['effective_support']:.3g} of $M$, a fraction of "
        f"{rule5['effective_support_fraction']:.3g}. The squared-$W_2$ "
        f"comparator, which removes the repulsion term, reaches mode coverage "
        f"{rule5['comparator_sqw2_mode_coverage']}."
    )
    add("")

    rule6 = flags["rule_6_cost"]
    add("### Rule 6, cost")
    add("")
    add(
        f"Median runtime {rule6['claimant_median_runtime_seconds']:.4g} s against "
        f"Causal-DRF's {rule6['causal_drf_median_runtime_seconds']:.4g} s, a "
        f"ratio of {rule6['runtime_ratio']:.3g} against a ceiling of "
        f"{rule6['max_allowed']:.3g}."
    )
    add("")
    add("| Method | Median runtime (s) | Max runtime (s) | Median peak RSS (MB) |")
    add("|---|---|---|---|")
    for method, entry in payload["cost"].items():
        add(
            f"| `{method}` | {entry['median_runtime_seconds']:.3g} | "
            f"{entry['max_runtime_seconds']:.3g} | "
            f"{entry['median_peak_ram_mb']:.3g} |"
        )
    add("")

    # -------------------------------------------------------------- ablations
    add("## 4. Mechanism ablations")
    add("")
    ablations = payload["mechanism_ablations"]
    for name, title in (
        ("repulsion", "Repulsion, against the squared-$W_2$ booster"),
        ("sharing", "Arm-shared localisation, against C-WDB-v0"),
        ("shrinkage", "Causal regularisation, against `arm_shrinkage = 0`"),
    ):
        add(f"### {title}")
        add("")
        lines.extend(comparison_table(ablations.get(name, [])))

    add("### Finite-particle approximation")
    add("")
    add("| Regime | $M$ | Excess energy risk | SE | Replications |")
    add("|---|---|---|---|---|")
    for dgp, series in ablations.get("particles", {}).items():
        for count in sorted(series, key=int):
            entry = series[count]
            add(
                f"| {dgp} | {count} | "
                f"{entry['mean_excess_energy_risk']:.4g} | "
                f"{entry['standard_error']:.3g} | {entry['n']} |"
            )
    add("")

    # ------------------------------------------------------ favourable reading
    add("## 5. The favourable reading")
    add("")
    wins = [
        item
        for item in flags["rule_2_law_advantage"]["comparisons"]
        + flags["rule_3_transfer"]["comparisons"]
        if item.get("claimant_wins")
    ]
    if wins:
        add(
            "Where C-WDB-v1 does win on a preregistered comparison, it wins "
            "here:"
        )
        add("")
        lines.extend(comparison_table(wins))
    else:
        add(
            "There is no preregistered comparison on which C-WDB-v1 wins at the "
            "frozen decision multiple. The favourable reading is therefore "
            "empty, and the verdict below follows from that rather than from "
            "any threshold choice."
        )
        add("")

    # ---------------------------------------------------------------- verdict
    add("## 6. Verdict")
    add("")
    add(f"**`{summary['verdict']}`**")
    add("")
    if summary["rules_failed"]:
        add("Rules not met:")
        add("")
        for name in summary["rules_failed"]:
            add(f"- `{name}`: {flags[name].get('statement', '')}")
        add("")
        add(
            "Under the Phase G3 decision list, C-WDB does not return `GO`. The "
            "choice among `PIVOT`, `INCREMENTAL-ONLY`, and `KILL` is argued in "
            "Section 7."
        )
    else:
        add(
            "Every frozen rule is met, so C-WDB returns `GO` and proceeds to the "
            "applied phase."
        )
    add("")

    if independent is not None:
        add(
            f"The independent recomputation in "
            f"`research/checks/g3_gate_flags.py` returns "
            f"`{independent['independent']['summary']['verdict']}` with "
            f"{len(independent['disagreements'])} disagreements against the "
            f"analysis code."
        )
        add("")

    add("## 7. Interpretation and next step")
    add("")
    add(
        "_This section is written by hand against the tables above and is the "
        "only part of this memo not generated from the payload._"
    )
    add("")

    add("## 8. Artefacts")
    add("")
    add("- `results/merged/main_results.parquet` (merged rows)")
    add("- `results/merged/merge_audit.json` (reconciliation)")
    add("- `results/merged/analysis_payload.json` (every number above)")
    add("- `results/manifests/main_manifest.json` (frozen manifest)")
    add("- `results/manifests/cost_pilot.json` (measured cost basis)")
    add("- `tables/simulation/`, `figures/simulation/`")
    add("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"verdict: {summary['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
