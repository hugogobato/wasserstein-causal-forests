# WP3-T3: training-only coordinate scaling

**Input files:** `research/finite_grid_estimands.md`, `research/assumption_ledger.md`, and `research/wp3_odcf.py`.

**Source to adapt:** Athey, Tibshirani, and Wager (2019), Sections 2–3, and Ćevid et al. (2022), Section 3. These sources motivate adaptive split criteria, but they do not select the project-specific scalar-coordinate normalization. The selection below is an implementation decision, not a theorem.

**Assumptions used:** A11 and A12.

**Status:** `PASS` for the pre-simulation implementation decision, not for C3.2. The rule is selected analytically; the deterministic calibration is illustrative only.

## 1. Candidate rules

For each functional coordinate (j), let (d_j) denote its training-only dispersion. The integrated curve reference is

\[
d_Q=\left\{\frac{\sum_k w_k\operatorname{Var}(R_{ik})}{\sum_kw_k}\right\}^{1/2}.
\]

The candidates considered were `robust_sd`, with
\[
d_j=1.4826\operatorname{MAD}(R_{ij}),
\]
`mad`, with \(d_j=\operatorname{MAD}(R_{ij})\), and `null_score_se`, with
\[
d_j=\frac{\operatorname{sd}(R_{ij})}
{\sqrt{\sum_i\widehat e_i(1-\widehat e_i)}}.
\]
The scalar coordinate is multiplied by \(s_j=d_Q/d_j\). A zero curve dispersion uses the declared unit reference rather than suppressing a pure-functional signal.

All locations and dispersions are computed from training scores. The scientific scores and returned effects remain unscaled; \(s_j\) is used only in split geometry.

## 2. Analytic selection

The frozen rule is `robust_sd`. The factor `1.4826` makes the MAD Gaussian-consistent while retaining robustness to a small number of extreme inverse-propensity scores. Under a fixed score law with a positive finite MAD, both \(d_Q\) and \(d_j\) converge to finite positive constants, which is the behavior required by A11.

The raw `mad` rule has the same qualitative stability but changes the functional block by the arbitrary normal-consistency factor. It is therefore retained only as a sensitivity rule.

The `null_score_se` rule is excluded from the primary estimator. If the propensity stays bounded away from zero and one, then
\[
\sum_i\widehat e_i(1-\widehat e_i)\asymp n.
\]
Consequently its denominator is \(O_p(n^{-1/2})\) and the corresponding split scale is \(O_p(\sqrt n)\), rather than converging to a positive constant. It cannot satisfy A11 as written and would make the split criterion depend directly on training sample size.

## 3. Illustrative calibration

The deterministic script reports three diagnostics: `balance_cv`, the coefficient of variation across the integrated curve contribution and the three scaled functional contributions; `half_sample_instability`, the relative change in functional scales under one half-sample; and `duplicate_grid_instability`, the relative change after every curve grid point is duplicated and its weight divided by two. The legacy summary score is

\[
\texttt{balance\_cv}+2\,\texttt{half\_sample\_instability}
 +\texttt{duplicate\_grid\_instability}.
\]

This score is retained as a regression diagnostic only. It is not a principled loss for selecting a scale rule, because a coefficient of variation can look smaller when three scalar contributions jointly dominate the single integrated curve contribution. A single seed and one half-sample also cannot establish stability.

The calibration in `research/wp3_scaling.py` uses 240 training rows, 49 curve coordinates, the frozen trapezoidal weights, three functional coordinates, and seed 20260727. It reports:

| Rule | Balance CV | Half-sample instability | Duplicate-grid instability | Selection score |
|---|---:|---:|---:|---:|
| `robust_sd` | 1.06863 | 0.0124903 | \(1.36\times10^{-16}\) | 1.09361 |
| `mad` | 1.16013 | 0.0124903 | \(4.95\times10^{-17}\) | 1.18511 |
| `null_score_se` | 0.565861 | 0.301257 | \(1.39\times10^{-16}\) | 1.16837 |

The table is consistent with the analytic decision in this one realization, but it does not prove it. In particular, the apparently smaller `null_score_se` balance coefficient accompanies scalar contributions that are collectively much larger than the curve contribution. No comparative downstream outcome was used to choose `robust_sd`.

## Checks run

Command: `python3 research/wp3_scaling.py`.

Observed implementation check: the primary rule is finite and positive, and exact grid duplication leaves it unchanged up to floating-point error.

Unresolved questions: C3.2 remains an `adapt` claim. Its proof must state conditions ensuring positive population MADs and convergence of cross-fitted score dispersions. The calibration does not establish a valid asymptotic split criterion.

`PASS`
