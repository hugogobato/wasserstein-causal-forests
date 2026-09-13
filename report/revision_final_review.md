# Integrated manuscript review

## Scope and source

Root source: `report/wcf_main.tex`, with the generated `wcf_revision_*.tex` table inputs and `figures_generated/wcf_pipeline.tex`. Bibliography: `report/wcf_references.bib`. This is a review of the revised manuscript and its numerical presentation, not a new proof of consistency or efficiency of WCF. The earlier `revision_*_section.tex` files are drafting proposals; the root manuscript supersedes them.

The original source contained no theorem environments. The mathematical review therefore covers its definitions, identification statements, scoring objective, shared-leaf algebra, AIPW identity, and representation claims. Existing implementation and results were inspected without changing experiment runners or historical simulation outputs.

## Claim and evidence coverage

| Claim | Check and final scope |
|---|---|
| Quantile geometry | For common positive weights summing to one and sorted coordinates, the weighted Euclidean formula is exactly W2 between the corresponding discrete scalar laws. It approximates W2 between the original outcome distributions. Generic d, K, and M are not simulation constants. |
| Identification | Consistency, conditional exchangeability for each potential outcome, and positivity identify each conditional marginal law. A coupling of the two potential outcomes is not identified. |
| Functional estimands | Finite expected absolute functional values are assumed; nonempty moderator strata are required. TATE/TCATE transform each observed outcome before averaging. Reference effects average the distance to a fixed quantile vector. Negative values mean movement toward the reference. |
| Particle updates | Checked the preconditioner `-M/w`, observed-arm gradient, shared split criterion, weighted isotonic projection, initialization and backtracking against `cwdb/model.py` and `arm_shared_tree.py`. |
| Shared-leaf shrinkage | With share s=n1/(n0+n1), updates are pooled minus s*c*gap and pooled plus (1-s)*c*gap. Substitution proves exact preservation of the count-weighted mean and shrinkage of the gap by c. The old symmetric formula was rejected. |
| AIPW identity | Conditioning on X replaces A by e and observed residuals by the arm-mean residuals. Subtracting the true contrast gives `(e_dagger-e) * ((m1_dagger-mu1)/e_dagger + (m0_dagger-mu0)/(1-e_dagger))`. The two nuisance-correctness branches follow algebraically. This is not a theorem about nuisance convergence for the fixed-budget learner. |
| Clipping and tuning | Fixed clipping can break propensity correctness. Penalty selection is reused across nuisance folds rather than nested; the paper makes no efficiency or coverage claim for that procedure. |
| Two-part mixture | pi denotes probability of the nondegenerate vector. The explicit zero atom has weight 1-pi. Finite particle laws can themselves contain zeros; the rejected claim of structural impossibility is not in the final paper. Scalar two-part references motivate the construction by analogy, not as evidence of a completed distribution-valued application. |
| Empirical comparisons | Only `cwdb_dr` is displayed as calibrated WCF. Optional two-part estimates are explicitly uncalibrated. WCF losses in the D suite and per-functional IC tables remain visible. There is no universal assignment-driven superiority or “first doubly robust distributional method” claim. |
| Experimental coverage | Existing final-WCF results cover nonlinear prognosis and symmetric quantile contours. The new nonlinear-assignment helper is a construction probe only. K/M and flexible-propensity estimator studies remain planned. |

## Numerical and presentation verification

`build_wcf_revision_assets.py` reads four merged result artifacts, filters the exact setting and seeds, requires complete methods/designs and ten replication entries, averages arm/function rows within each replication, and then computes means and Monte Carlo SEs. It emits both CSV ledgers and the exact LaTeX inputs used by the root source.

`check_wcf_revision_assets.py` independently recomputed every ledger mean and SE by grouping the raw source records by seed. All 348 aggregate and 336 per-functional summaries matched to tolerance 5e-10. Results are in `revision_asset_checks.json`. This validates the aggregation, not the original model fits or every oracle quadrature error.

`check_revision_identities.py` verifies exact rational examples for both AIPW branches, clipping bias, leaf shrinkage, representable zero mass, noncommutation of expected distance and distance of the mean, and nonconstant assignment unrelated to outcomes. Results are in `revision_identity_checks.json`.

The existing targeted energy, shared-tree, and contrast-regularization tests completed: 37 passed in the final run. An earlier specialist started a broader test selection and interrupted its lengthy DGP portion; that partial run is not represented as a completed suite. No implementation change required retraining the historical experiments.

The root compiled with pdfLaTeX and BibTeX. The final 16-page PDF had no undefined citations, undefined references, overfull boxes, or LaTeX warnings. The title page, method diagram, simulation equations, principal results, appendix tables, and references were rendered and inspected. Diagram wrapping, an orphaned definition lead-in, and appendix pagination were improved during this inspection.

## Reconciliation of specialist proposals

The parent rejected a proposed reference effect that applied a distance to an entire conditional law instead of applying it to each outcome vector, and corrected its reversed sign. The parent also removed ordinary WCF rows from the zero-inflation table because those labels could conceal an uncalibrated path, and replaced a draft assertion that particles cannot carry an exact atom. These changes illustrate why the section drafts are not the authoritative manuscript.

The methods specialist's suggestion to replace the 2026 Causal-DRF author list with an older two-author preprint was rejected. The authoritative publication is Näf, Park, and Susmann, PMLR 300:1000–1008 (2026), verified at https://proceedings.mlr.press/v300/naf26a.html. The final bibliography uses that published version. The general citations were metadata and claim-support checked by the citation specialist; primary source proofs were not rederived. The primary TCDA metadata was also independently verified at https://arxiv.org/abs/2607.28161 and matched to the supplied local manuscript for its organizing-framework attribution.

The earlier results audit's statement that WCF is superior on D2/D7/D8 reference targets was narrowed: the displayed final data show larger reference-TCATE errors than the forest baselines in those designs. The final paper states those losses explicitly. It also distinguishes IC's constant population moderator effects from heterogeneous effects in the nonlinear family.

## Independent review ledger

All delegated roles were requested as `gpt-5.6-luna`, reasoning `xhigh`, without recursive delegation. The parent integrated and checked the outputs.

| Role | Artifact / status |
|---|---|
| Methods specialist | `revision_methods_audit.md`; completed; proposals reconciled above |
| Citation verifier and editorial writer | `revision_citation_audit.md`; completed; bibliography updated |
| Results auditor and experiment planner | `revision_results_audit.md`, sensitivity plan, DGP probe; completed; deterministic asset follow-up completed |
| Fresh blind methods reviewer | No review artifact returned; treated as unavailable rather than silently counted as verification |
| Independent complete-paper writing reviewer | The requested `gpt-5.6-luna` call failed because the service reported that its usage limit had been reached; no review artifact was produced |

The independent specialist layer is therefore incomplete. The final status is an internally verified manuscript revision with three completed specialist reviews, deterministic numerical reconstruction, targeted implementation tests, exact identity checks, a clean compilation, and a page-by-page parent inspection. It is not labeled as having passed an additional blind complete-paper audit.

## Remaining limits

The manuscript supplies no new consistency, efficiency, or coverage theorem for the fixed-particle estimator, and no robustness to unmeasured confounding. Historical fold-setting provenance relies on the stored configuration and documented study-specific rerun, rather than a new execution of all models. No full sensitivity experiment, flexible propensity refactor, applied study, or new estimator benchmark was performed. The complete experimental plan labels these tasks as future work.
