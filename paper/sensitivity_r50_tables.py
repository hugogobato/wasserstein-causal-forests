"""Build the fifty-replication sensitivity LaTeX tables for the paper.

Every entry is computed from the frozen r50 sensitivity result file through
``sensitivity_r50_data``; nothing is hardcoded.  Law entries are Monte Carlo
means and functional entries are Monte Carlo RMSEs over the fifty paired
seeds, following the confirmatory convention.  WCF's functional columns use
its cross-fitted AIPW layer and the forest columns use plug-in contrasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

import sensitivity_r50_data as S

ROOT = Path(__file__).resolve().parent

METHOD_HEAD = {"cwdb_dr": "WCF", "causal_drf": "Causal-DRF", "drf": "DRF"}
K_VALUES = (5, 25, 49)
M_VALUES = (5, 10, 25)
FACTORIES = (
    ("Logistic", S.GRID_SYM, "cwdb_dr"),
    ("Random forest", S.GRID_FLEX_RF, "cwdb_dr_flex_rf"),
    ("Gradient boosting", S.GRID_FLEX, "cwdb_dr_flex"),
    ("Oracle", S.GRID_FLEX, "cwdb_dr_oracle"),
)
NULL_TARGETS = (
    ("TATE grid mean", "tate"),
    ("TCATE grid mean", "tcate"),
    ("Ref. TATE", "ref_tate"),
    ("Ref. TCATE", "ref_tcate"),
)


def _fmt(mean: float, se: float) -> str:
    return f"{mean:.4f}({se:.4f})"


def _fmt_pct(pct: float, se: float) -> str:
    return f"{pct:.1f}({se:.1f})"


def _panel(title: str, span: int) -> str:
    return rf"\multicolumn{{{span}}}{{l}}{{\textit{{{title}}}}} \\"


def _wrap_table(
    *,
    label: str,
    caption: str,
    columns: str,
    head_lines: Sequence[str],
    body_lines: Sequence[str],
    note: str | None = None,
) -> str:
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        *head_lines,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note is not None:
        lines.extend([r"\vspace{2pt}", rf"{{\footnotesize {note}}}"])
    lines.extend([r"\end{table}", ""])
    return "\n".join(lines)


def _summary(
    frame: pd.DataFrame,
    metric: str,
    target: str,
    *,
    method: str = "cwdb_dr",
    law: bool = False,
    **filters: object,
) -> tuple[float, float]:
    values = S.seed_values(frame, metric, target, method=method, **filters)
    estimate, se, _ = S.summarize(values, law=law)
    return estimate, se


def _end_to_end(
    frame: pd.DataFrame,
    functional: str,
    method: str,
    *,
    law: bool = False,
    **filters: object,
) -> tuple[float, float]:
    metric = S.end_to_end_metric(method, functional)
    target = S.FUNCTIONAL_TARGET[functional]
    return _summary(frame, metric, target, method=method, law=law, **filters)


def _build_km_table(frame: pd.DataFrame) -> str:
    head = [
        _panel(r"Panel A: $K$ sensitivity at $M=10$", 7),
        r"DGP & $K$ & Law & Mean-quantile & Ref. TCATE plug-in"
        r" & Ref. TCATE AIPW & Ref. TATE AIPW \\",
    ]
    body: list[str] = []
    for panel, coordinate, values in (
        ("Panel A: $K$ sensitivity at $M=10$", "n_grid", K_VALUES),
        ("Panel B: $M$ sensitivity at $K=25$", "n_particles", M_VALUES),
    ):
        if panel.startswith("Panel B"):
            body.extend([r"\midrule", _panel(panel, 7)])
            body.append(
                r"DGP & $M$ & Law & Mean-quantile & Ref. TCATE plug-in"
                r" & Ref. TCATE AIPW & Ref. TATE AIPW \\"
            )
            body.append(r"\midrule")
        for design, dgp in S.DESIGN_DGP.items():
            for value in values:
                coordinate_filters = {
                    "grid": S.GRID_PRIMARY, "dgp": dgp, "n_train": 1000,
                }
                if coordinate == "n_grid":
                    coordinate_filters.update(n_grid=value, n_particles=10)
                else:
                    coordinate_filters.update(n_grid=25, n_particles=value)
                law = _summary(
                    frame, *S.LAW_METRIC, law=True, **coordinate_filters
                )
                meanq = _summary(frame, *S.MEANQ_METRIC, **coordinate_filters)
                tcate_plugin = _summary(
                    frame, "plugin_reference_tcate_rmse", S.FUNCTIONAL_TARGET["ref_tcate"],
                    **coordinate_filters,
                )
                tcate_aipw = _summary(
                    frame, "aipw_reference_tcate_rmse", S.FUNCTIONAL_TARGET["ref_tcate"],
                    **coordinate_filters,
                )
                tate_aipw = _summary(
                    frame, "aipw_reference_tate_abs_error", S.FUNCTIONAL_TARGET["ref_tate"],
                    **coordinate_filters,
                )
                cells = [
                    _fmt(*law), _fmt(*meanq), _fmt(*tcate_plugin),
                    _fmt(*tcate_aipw), _fmt(*tate_aipw),
                ]
                body.append(f"{design} & {value} & " + " & ".join(cells) + r" \\")
    caption = (
        r"Sensitivity of WCF to the quantile grid size $K$ and the particle "
        r"budget $M$ at $n=1000$ for designs LS0 to LS3 under the confirmatory "
        r"protocol. Panel A varies $K$ at $M=10$; Panel B varies $M$ at $K=25$. "
        r"Law entries are Monte Carlo means and functional entries are Monte "
        r"Carlo RMSEs over fifty paired replications, with Monte Carlo standard "
        r"errors in parentheses; lower is better. The native-grid target changes "
        r"with $K$, so Panel A is a stability check rather than a ranking of "
        r"grid resolutions. WCF's functional columns use its cross-fitted AIPW "
        r"layer; the plug-in column is shown for comparison."
    )
    return _wrap_table(
        label="tab:sensitivity-km",
        caption=caption,
        columns="llrrrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_factorial_table(frame: pd.DataFrame) -> str:
    head = [r"$K$ & $M=5$ & $M=10$ & $M=25$ \\"]
    body: list[str] = []
    for panel, design, dgp in (
        ("Panel A", "LS1", "IC1"), ("Panel B", "LS3", "IC3"),
    ):
        body.append(_panel(rf"{panel}: {design}", 4))
        body.append(r"\midrule")
        for k in K_VALUES:
            cells = []
            for m in M_VALUES:
                estimate, se = _end_to_end(
                    frame, "ref_tcate", "cwdb_dr",
                    grid=S.GRID_PRIMARY, dgp=dgp, n_train=1000,
                    n_grid=k, n_particles=m,
                )
                cells.append(_fmt(estimate, se))
            body.append(f"{k} & " + " & ".join(cells) + r" \\")
    caption = (
        r"Interaction grid for the quantile grid size $K$ and the particle "
        r"budget $M$ on LS1 and LS3 at $n=1000$ over fifty paired replications. "
        r"Entries are end-to-end reference-TCATE Monte Carlo RMSEs with Monte "
        r"Carlo standard errors in parentheses; lower is better, rows are $K$, "
        r"and columns are $M$."
    )
    return _wrap_table(
        label="tab:sensitivity-factorial",
        caption=caption,
        columns="lrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_resolution_table(frame: pd.DataFrame) -> str:
    columns = (
        ("Law", *S.LAW_METRIC),
        ("Mean-quantile", *S.MEANQ_METRIC),
        ("Ref. TCATE plug-in", "plugin_reference_tcate_rmse", S.FUNCTIONAL_TARGET["ref_tcate"]),
        ("Ref. TCATE AIPW", "aipw_reference_tcate_rmse", S.FUNCTIONAL_TARGET["ref_tcate"]),
        ("Ref. TATE AIPW", "aipw_reference_tate_abs_error", S.FUNCTIONAL_TARGET["ref_tate"]),
    )
    head = [
        r"DGP & Law & Mean-quantile & Ref. TCATE plug-in"
        r" & Ref. TCATE AIPW & Ref. TATE AIPW \\",
    ]
    body: list[str] = []
    for design, dgp in S.DESIGN_DGP.items():
        cells = []
        for _, metric, target in columns:
            pct, se = S.paired_percentage(
                frame, metric=metric, target_id=target, method="cwdb_dr",
                low=(25, 10), high=(49, 10),
                grid=S.GRID_PRIMARY, dgp=dgp, n_train=1000,
            )
            cells.append(_fmt_pct(pct, se))
        body.append(f"{design} & " + " & ".join(cells) + r" \\")
    caption = (
        r"Seed-paired resolution differences at $n=1000$ for designs LS0 to "
        r"LS3, computed as $K=49$ minus $K=25$ at $M=10$ over fifty paired "
        r"replications. Entries are percentage changes, so a negative value "
        r"means the larger grid has the smaller error; the value in parentheses "
        r"is the Monte Carlo standard error of the paired percentage difference "
        r"in percentage points. Law is a mean, the other columns are errors. "
        r"The native-grid target changes with $K$, so these differences do not "
        r"separate quadrature resolution from tail coverage."
    )
    return _wrap_table(
        label="tab:sensitivity-resolution",
        caption=caption,
        columns="lrrrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_assignment_table(frame: pd.DataFrame) -> str:
    head = [r"DGP & WCF & Causal-DRF & DRF \\"]
    body: list[str] = []
    for panel, n_train in (("Panel A", 500), ("Panel B", 1000)):
        body.append(_panel(rf"{panel}: $n={n_train}$", 4))
        body.append(r"\midrule")
        for dgp in S.SYM_DGPS:
            cells = []
            for method in METHOD_HEAD:
                estimate, se = _end_to_end(
                    frame, "ref_tcate", method,
                    grid=S.GRID_SYM, dgp=dgp, n_train=n_train,
                    n_grid=25, n_particles=10,
                )
                cells.append(_fmt(estimate, se))
            body.append(f"{dgp} & " + " & ".join(cells) + r" \\")
    caption = (
        r"Symmetric and nonlinear assignment designs at $n=500$ and $n=1000$ "
        r"over fifty paired replications. Entries are end-to-end reference-TCATE "
        r"Monte Carlo RMSEs with Monte Carlo standard errors in parentheses; "
        r"lower is better. WCF uses its cross-fitted AIPW layer and the forest "
        r"baselines use plug-in contrasts. SYM-RANDOM is a constant-propensity "
        r"calibration control, SYM-LIN uses a linear assignment index, and "
        r"SYM-NL and SYM-MU use nonlinear assignment with the same symmetric "
        r"outcome laws."
    )
    return _wrap_table(
        label="tab:sensitivity-assignment",
        caption=caption,
        columns="lrrr",
        head_lines=head,
        body_lines=body,
    )


def _build_propensity_table(frame: pd.DataFrame) -> str:
    head = [r"DGP & Logistic & Random forest & Gradient boosting & Oracle \\"]
    body: list[str] = []
    for panel, n_train in (("Panel A", 500), ("Panel B", 1000)):
        body.append(_panel(rf"{panel}: $n={n_train}$", 5))
        body.append(r"\midrule")
        for dgp in S.PROPENSITY_DGPS:
            cells = []
            for _, grid, method in FACTORIES:
                estimate, se = _end_to_end(
                    frame, "ref_tcate", method,
                    grid=grid, dgp=dgp, n_train=n_train,
                    n_grid=25, n_particles=10,
                )
                cells.append(_fmt(estimate, se))
            body.append(f"{dgp} & " + " & ".join(cells) + r" \\")
    diagnostic = []
    for label, grid, method in FACTORIES:
        pooled = []
        for dgp in S.PROPENSITY_DGPS:
            for n_train in (500, 1000):
                pooled.append(
                    S.seed_values(
                        frame, "diagnostic_dr_if_se_reference",
                        "NONE_OPERATIONAL", method=method,
                        grid=grid, dgp=dgp, n_train=n_train,
                        n_grid=25, n_particles=10,
                    )
                )
        values = pd.concat(pooled)
        diagnostic.append(f"{label.lower()} {values.mean():.3f}")
    note = (
        r"\textit{Note:} the confirmatory protocol reports the common "
        r"cross-fitted AIPW view, whose propensity layer is fixed across "
        r"columns, so the factory choice has little effect on the tabulated "
        r"errors. The estimator's internal reference-estimate "
        r"influence-function SE, a calibration diagnostic, is "
        + ", ".join(diagnostic[:-1]) + r", and " + diagnostic[-1]
        + r"; the gradient-boosting factory is roughly three to four times "
        r"less stable than the alternatives, consistent with the archive's "
        r"miscalibration diagnosis."
    )
    caption = (
        r"Propensity-model sensitivity on the SYM-NL and SYM-MU designs at "
        r"$n=500$ and $n=1000$ over fifty paired replications. Entries are "
        r"end-to-end reference-TCATE Monte Carlo RMSEs with Monte Carlo standard "
        r"errors in parentheses; lower is better. The columns replace the "
        r"default logistic propensity with a random-forest factory, a "
        r"gradient-boosting factory, and the oracle propensity, which is a "
        r"diagnostic and not a feasible competitor. The outcome learner, grid, "
        r"and particle budget are held fixed."
    )
    return _wrap_table(
        label="tab:sensitivity-propensity",
        caption=caption,
        columns="lrrrr",
        head_lines=head,
        body_lines=body,
        note=note,
    )


def _build_placebo_table(frame: pd.DataFrame) -> str:
    head = [r"Target & WCF & Causal-DRF & DRF \\"]
    body: list[str] = []
    for panel, n_train in (("Panel A", 500), ("Panel B", 1000)):
        body.append(_panel(rf"{panel}: $n={n_train}$", 4))
        body.append(r"\midrule")
        for label, functional in NULL_TARGETS:
            cells = []
            for method in METHOD_HEAD:
                metric = S.end_to_end_metric(method, functional)
                target = S.FUNCTIONAL_TARGET[functional]
                values = []
                for grid, dgp in S.NULL_COORDINATES:
                    values.append(
                        S.seed_values(
                            frame, metric, target, method=method,
                            grid=grid, dgp=dgp, n_train=n_train,
                            n_grid=25, n_particles=10,
                        )
                    )
                pooled = pd.concat(values)
                estimate, se, _ = S.pooled_summary(pooled)
                cells.append(_fmt(estimate, se))
            body.append(f"{label} & " + " & ".join(cells) + r" \\")
    caption = (
        r"Null companion designs with exactly zero treatment effects, so every "
        r"entry is a false-effect error. Entries are end-to-end Monte Carlo "
        r"RMSEs pooled over the six null companion designs and fifty paired "
        r"replications, with Monte Carlo standard errors in parentheses; lower "
        r"is better. WCF uses its cross-fitted AIPW layer and the forest "
        r"baselines use plug-in contrasts. The first two rows use the grid-mean "
        r"functional, and the last two rows use the reference-distance marginal "
        r"and conditional targets."
    )
    return _wrap_table(
        label="tab:sensitivity-placebo",
        caption=caption,
        columns="lrrr",
        head_lines=head,
        body_lines=body,
    )


def build(output_dir: Path) -> Mapping[str, Path]:
    frame = S.load_per_seed()
    outputs = {
        "wcf_sensitivity_km": _build_km_table(frame),
        "wcf_sensitivity_factorial": _build_factorial_table(frame),
        "wcf_sensitivity_resolution": _build_resolution_table(frame),
        "wcf_sensitivity_assignment": _build_assignment_table(frame),
        "wcf_sensitivity_propensity": _build_propensity_table(frame),
        "wcf_sensitivity_placebo": _build_placebo_table(frame),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, text in outputs.items():
        path = output_dir / f"{name}.tex"
        path.write_text(text, encoding="utf-8")
        written[name] = path
    return written


if __name__ == "__main__":
    for name, path in build(ROOT / "tables").items():
        print(f"{name}: {path}")
