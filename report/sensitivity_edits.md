# Sensitivity edits for `report/wcf_main.tex`

Surgical replacements that integrate the WCF sensitivity study. Every `old` block is copied verbatim from `report/wcf_main.tex` and should occur exactly once. Edit 3 is intentionally a no edit. The references `sec:resolution-design` and `sec:assignment-design` are defined by `report/sensitivity_methods_fragment.tex`; `sec:results-assignment` is expected to be defined by the new sensitivity results fragment.

## Edit 1: Abstract, resolution retention and propensity-model sensitivity

Line 31 (final sentence of the abstract).

**Old:**
```latex
The results support WCF as a conditional-law learner with functional calibration under observed confounding, while delimiting its performance under finite particle budgets and limited overlap.
```

**New:**
```latex
A dedicated resolution study retained $K=25$ and $M=10$ after a pre-fixed stability rule failed; the larger $K=49$ grid improves full-range law error through extreme-tail coverage but ties on interior levels and costs more. The functional calibration is stable under a competent flexible (random-forest) propensity, while the default gradient-boosting factory is miscalibrated. The results support WCF as a conditional-law learner with functional calibration under observed confounding, while delimiting its performance under finite particle budgets and limited overlap.
```

**Why:** Reports the resolution retention outcome with the K=49 tail-coverage trade-off and the propensity-factory result in two sentences, keeping the existing concluding sentence and register.

## Edit 2: Introduction, scope of the simulation programme

Line 49 (first sentence of the evaluation paragraph).

**Old:**
```latex
We evaluate WCF in simulation studies designed to vary the treatment effect on location and distributional shape, the degree of overlap, the relationship between treatment assignment and prognostic covariates, and the presence of a point mass.
```

**New:**
```latex
We evaluate WCF in simulation studies designed to vary the treatment effect on location and distributional shape, the degree of overlap, the relationship between treatment assignment and prognostic covariates, and the presence of a point mass. The simulation programme now includes a resolution sensitivity study with common and interior 199-level targets and a family of assignment mechanisms separating linear, nonlinear, and prognosis-dependent assignment, with an alignment pair and a propensity-model check.
```

**Why:** Adds the one required sentence announcing the new sensitivity components at their first mention.

## Edit 3: Contribution paragraph

No edit. The contribution paragraph claims representation, calibration, and the reference-distance estimands, and makes no claim about grid or particle resolution or about the propensity-model family, so the new study creates no inconsistency there.

## Edit 4: Implementation, retained resolution and sensitivity settings

Line 321 (closing sentence of the paragraph).

**Old:**
```latex
The fixed $K$ and $M$ are computational choices; the present comparisons do not establish sensitivity to either resolution.
```

**New:**
```latex
The fixed $K$ and $M$ are computational choices; a dedicated sensitivity study retains $K=25$ and $M=10$. The sensitivity study uses three outcome folds and contrast candidates $\{0,50,500\}$ for every design and is reported in Sections~\ref{sec:resolution-design} and~\ref{sec:assignment-design}.
```

**Why:** Replaces the open sensitivity claim with the retained setting and points to the new study, and clarifies the folds and contrast candidates because the earlier sentences mention two folds for some designs.

## Edit 5: Propensity nuisance, forward pointer to the empirical check

Line 217 (third sentence of the paragraph).

**Old:**
```latex
Logistic regression is a working choice for the experiments.
```

**New:**
```latex
Logistic regression is a working choice for the experiments. An empirical sensitivity check of the calibration under flexible propensity factories is reported in Section~\ref{sec:results-assignment}.
```

**Why:** Adds the required forward pointer while leaving the overlap, convergence, clipping, and asymptotic-conditions claims unchanged.

## Edit 6: Discussion limitations, resolution study outcome

Line 365 (third sentence of the limitations paragraph).

**Old:**
```latex
The reported fits use one grid resolution and particle budget; stability to these choices remains to be established.
```

**New:**
```latex
A dedicated sensitivity study retained $K=25$ and $M=10$. The alternative $K=49$ improves full-range law error through extreme-tail coverage but ties on interior levels and loses on the IC3 marginal reference effect at higher cost, while $M=25$ yields no material interior gain.
```

**Why:** States the resolution outcome in place of the stale open question and leaves every other limitation sentence untouched.

## Edit 7: Conclusion, examination status of budgets and propensity models

Line 369 (final sentence).

**Old:**
```latex
These results motivate further evaluation of WCF with alternative representation budgets, flexible propensity models, and applied distribution-valued outcomes.
```

**New:**
```latex
These results motivate an applied evaluation of WCF with distribution-valued outcomes, now that the sensitivity study has examined alternative representation budgets and flexible propensity models.
```

**Why:** Removes the implication that budgets and propensity models are still unexamined while preserving the call for an applied evaluation.

## Edit 8: Abstract, assignment-mechanism results

Line 31 (sentence following the simulation-error range sentence).

**Old:**
```latex
Additional symmetric and nonlinear designs reveal variation in the ranking across estimands.
```

**New:**
```latex
Additional symmetric and nonlinear designs reveal variation in the ranking across estimands, with WCF at parity or better across the assignment mechanisms studied.
```

**Why:** Minimal clause so the abstract no longer understates the assignment-mechanism results, which show parity under linear and constant assignment and clearer gains under nonlinear assignment.
