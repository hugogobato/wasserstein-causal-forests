# Citation ledger: supplement and supplied TCDA source

Inspection date: 2026-09-24. Scope is limited to the citations requested for the frozen WCF supplement and the supplied TCDA paper, plus a scan for named tools used in those files. This is not a review of every bibliography entry or of the mathematical validity of the full TCDA paper. The local source files inspected were `source/paper__supplement.tex`, `source/paper__references.bib`, `source/TCDA__main.tex`, and `source/TCDA__references.bib`.

## Findings at a glance

1. The WCF supplement's TCDA theorem locators are off by one section. Its outcome-level results are Theorem 4.1, Definition 4.2, and Proposition 4.3 in both the supplied TCDA TeX and arXiv v1. The supplement cites Theorem 5.1, Definition 5.2, and Proposition 5.3. In section 5 of TCDA, Definition 5.2 is a distribution-level treatment effect; there is no Theorem 5.1 or Proposition 5.3.
2. Gneiting and Raftery's 2007 source is real and Theorem 5 is at the cited section, but the condition printed there has an internal sign inconsistency with the paper's own energy-score example. Read literally, it does not cover the WCF smoothed distance function. WCF's local Fourier/Gaussian-kernel argument does independently establish the strictness claim, so the mathematical claim need not depend on the defective attribution.
3. Chernozhukov et al. Theorem 5.1 is the DML result for ATE/ATTE using the standard scalar ATE score. It is directly relevant after setting the scalar outcome to the fixed transform $H=h(q_K(Y))$. Its theorem assumes a stronger, uniform rate and moment regime than WCF Theorem A.4. WCF explicitly supplies a separate proof under its stated conditions, so the citation is contextual support rather than the source of those weaker conditions.
4. Bang and Robins (2005) supports the broad doubly robust AIPW attribution. The WCF conditional remainder identity is also derived locally by conditioning on (X).
5. Barlow and Brunk (1972) is a real, correctly identified paper whose accessible abstract states the weighted isotonic least-squares problem. The full text was not accessible, so I could not verify that this particular article gives the weighted PAV algorithm. The PAV tool is cited, but its direct algorithm source remains to be confirmed or added.

No other uncited named algorithm or theorem was found in the bounded scan of the supplement and the relevant TCDA outcome-level section. Standard inequalities and limit steps in the local proofs are not treated as missing attributions.

## C-01. Gneiting and Raftery (2007), Section 5, Theorem 5

**Manuscript location / node.** WCF supplement, line 95; Proposition A.1 proof, node `supp:A1`. The claim is strict propriety of the smoothed energy loss for outer laws on the transformed quantile grid.

**Bibliography key and identity.** `gneiting2007proper`, *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102(477), 359–378 (2007), DOI [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437). Identity verified against the publisher record and the author-hosted PDF. The bibliography metadata matches.

**Searches and access.** Queries included “Gneiting Raftery 2007 Strictly Proper Scoring Rules Section 5 Theorem 5 full text” and the theorem's distinctive condition. Full author-hosted PDF inspected, including printed p. 369, Section 5.1, Theorem 5. Publisher DOI page also inspected. Local evidence copy: [author PDF](/home/hugo_souto/Stuff/Research/Wasserstein_Causal_Forests/research/audits/supplement_20260924/citation_evidence/Gneiting_Raftery_2007_author_copy.pdf), SHA-256 `d31a0c5f0ae8fec1a0a6544db5d056645b2d7296d71b44a2e8efb293c7d87ba2`.

**Source result and local use.** Theorem 5 gives strict propriety for a Euclidean radial score built from a continuous function of squared distance, under a complete-monotonicity condition, and finite expected pairwise score. Gneiting and Raftery use a reward convention; the WCF supplement uses the negative score as a loss, so the direction of optimization is reversed. Under the intended positive-derivative condition, the local choice is
\[
\psi_\varepsilon(t)=\sqrt{t+\varepsilon^2}-\varepsilon,
\qquad
\psi_\varepsilon'(t)=\frac{1}{2\sqrt{t+\varepsilon^2}}.
\]
Its derivative is completely monotone for $t>0$ and is nonconstant, including at $\varepsilon=0$. The transformed quantile-grid domain is the finite-dimensional Euclidean space in the theorem, and WCF explicitly assumes finite first moments. These are the right domain and moment conditions for the intended theorem.

**Content/applicability defect.** The published PDF prints the condition with \(-\psi'\) completely monotone. The theorem's subsequent energy-score example instead takes \(\psi(t)=t^{\beta/2}\), whose *positive* derivative is completely monotone, while its negative derivative is not. This is an internal sign inconsistency in the source. Under the printed condition, \(\psi_\varepsilon\) does not qualify. I found no authoritative correction in the searches made. Therefore identity is verified, the locator is verified, but applicability of Theorem 5 as printed is not established. The nearby Theorem 4 gives the general kernel-score construction but only propriety, not strictness.

**Effect on WCF claim.** The local proof proceeds beyond the citation: it writes the smoothed distance as a positive mixture of Gaussian-kernel discrepancies and uses the characteristic-function representation to show that each discrepancy is nonnegative and vanishes only for equal laws. That argument is sufficient for the stated strictness result under the stated finite-moment assumption. The citation should be corrected or recast as background, and the local strictness argument retained. Do not use Theorem 5's printed hypothesis as the sole justification.

**Disposition.** Identity verified; exact theorem located; source internally inconsistent as printed; literal applicability fails for the local \(\psi_\varepsilon\). Local proof supplies the needed claim. Source proof itself was not audited.

## C-02a. TCDA outcome-level identification, Theorem 5.1 citation

**Manuscript location / node.** WCF supplement, line 187, node `supp:A2–A3`: identification and transformation of an outcome-level representation followed by regression on covariates.

**Bibliography key and identity.** `souto_diamantis_tcda_2026`, Souto and Diamantis, *A Mathematical Framework for Topological Causal Data Analysis*, arXiv:2607.28161. Identity verified on the official [arXiv record](https://arxiv.org/abs/2607.28161), submitted 2026-07-30, v1. The supplied local TeX matches the title and authors. Local evidence copy: [arXiv v1 PDF](/home/hugo_souto/Stuff/Research/Wasserstein_Causal_Forests/research/audits/supplement_20260924/citation_evidence/TCDA_arXiv_2607.28161v1.pdf), SHA-256 `6e486eecf5e36070e280b209e9356523abc3de9e3e658a7619b32e2dd18ac9c4`.

**Locator check.** The supplied source uses section-based theorem numbering (`\newtheorem{theorem}{Theorem}[section]` and shared counters for definitions and propositions). Its relevant outcome-level results are Theorem 4.1, “Identification of the outcome-level topological ATE” (local lines 565–637); Definition 4.2, “Conditional topological average treatment effect” (645 onward); and Proposition 4.3, “Augmented remainder and product-rate bound” (691 onward). The official arXiv v1 HTML gives the same locators and formulas at [Theorem 4.1](https://arxiv.org/html/2607.28161v1#S4.T1), [Definition 4.2](https://arxiv.org/html/2607.28161v1#S4.D2), and [Proposition 4.3](https://arxiv.org/html/2607.28161v1#S4.P3).

The supplement's “Theorem 5.1” is not a theorem in the supplied TCDA source. Section 5 is distribution-level TCDA: Definition 5.1 is the interventional representation, Definition 5.2 is the distribution-level treatment effect, and Theorem 5.3 identifies the interventional outcome law. Thus Definition 5.2 exists but is not the outcome-level conditional effect; the other two cited locators do not identify the cited outcome-level results.

**Formal mapping and applicability.** Put $Z^a=H^a=h(q_K(Y^a))$ and $Z=H=h(q_K(Y))$. Under consistency, conditional exchangeability, positivity, and integrability, TCDA Theorem 4.1 identifies $\mathbb E[H^a]$ through its conditional mean. Definition 4.2 defines the corresponding conditional outcome-level effect; choosing the covariate summary $V=X$ yields the pointwise conditional target used by WCF. The local conversion from a learned conditional law to the conditional mean is stated and proved in Proposition A.3.

**Stability locator.** The phrase “identification and stability-transfer logic” is not fully supported by Theorem 4.1 and Definition 4.2 alone. The related general result is TCDA Theorem 6.1, “Coupling stability of the outcome-level effect” (local line 1407; official [arXiv v1 HTML](https://arxiv.org/html/2607.28161v1#S6.T1)). It concerns treatment-specific couplings of potential-outcome laws, not the exact conditional-law inequality in WCF Proposition A.3. That conditional inequality is derived locally by taking the infimum over couplings at each (x). A precise citation would use Theorem 4.1 and Definition 4.2 for identification/estimand, and optionally Theorem 6.1 for the general stability-transfer idea while keeping the conditional specialization local.

**Disposition.** Identity verified; source content supports the intended concepts; cited locators are wrong by section and partly point to different objects. Replace with 4.1 and 4.2, and use 6.1 only for the broader stability-transfer attribution. No TCDA proof was independently audited here.

## C-02b. TCDA augmented remainder, Proposition 5.3 citation

**Manuscript location / node.** WCF supplement, line 205, node `supp:A4-bias`.

**Source result.** The local TCDA Proposition 4.3 states the augmented arm-mean remainder for Banach-valued (Z), and gives a product bound under a positive lower bound on the fitted arm propensity and square-integrable nuisance errors. The official arXiv v1 and supplied TeX agree on the locator and formula.

**Mapping.** Specialize the Banach outcome to scalar $H=h(q_K(Y))$, set $m_a(x)=\mu_{a,h}(x)$, and take fitted arm propensities $g(x)$ and $1-g(x)$. Subtracting the two arm identities yields WCF's conditional scalar bias expression. Bounded fitted propensity and square-integrability are the scalar versions of the proposition's side conditions. The WCF text also derives the identity directly from conditional means.

**Disposition.** Identity and content verified; exact conceptual result and scalar mapping pass, subject to the local integrability/overlap assumptions. The citation locator is wrong: cite Proposition 4.3, not Proposition 5.3. Source proof itself was not audited here.

## C-03. Chernozhukov et al. (2018), Theorem 5.1

**Manuscript location / node.** WCF supplement, line 229 and Theorem A.4 proof, node `supp:A4-CLT`.

**Bibliography key and identity.** `chernozhukov2018double`, *Double/debiased machine learning for treatment and structural parameters*, The Econometrics Journal 21(1), C1–C68 (2018), DOI [10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097). Identity and bibliographic data verified against the [Oxford Academic full text](https://academic.oup.com/ectj/article/21/1/C1/5056401); the DOI in the BibTeX is correct.

**Searches and access.** Searched the full title with “Theorem 5.1”; inspected official OUP full text, including Section 5.1 and Theorem 5.1 at lines 1042–1100 of the browser text.

**Source result.** Theorem 5.1 gives uniform DML1/DML2 asymptotic normality and studentized inference for the ATE or ATTE under Assumption 5.1. For ATE it uses the standard scalar score with outcome regression and propensity nuisances, overlap, a bounded conditional outcome-residual variance, $q>2$ moment bounds, foldwise nuisance restrictions, and a product-rate bound of order $o(N^{-1/2})$. It states the score is efficient.

**Mapping and limitation.** WCF's fixed scalar target is an ATE in the transformed scalar outcome $H=h(q_K(Y))$; its score is exactly the ATE/AIPW score with $Y$ replaced by $H$. This makes Theorem 5.1 relevant. However, Theorem A.4 states fixed-law conditions based on finite second moment of $H$, a uniform conditional residual-variance bound, foldwise $L^2$ consistency, and a little-o product rate. These do not assert the source theorem's uniform-over-$P$, $q>2$, and nuisance moment-envelope conditions. WCF then gives a separate foldwise Chebyshev/CLT/LLN proof under its own conditions. Consequently the source theorem is a methodological analogue and precedent, not a direct implication for all of Theorem A.4 under its weaker hypotheses.

**Disposition.** Identity and Theorem 5.1 content verified; the scalar ATE score mapping is exact. Applicability as background passes; applicability as the sole theorem supporting WCF's weaker conditions does not. Keep the local proof and consider changing “the ATE specialization of ... Theorem 5.1” to wording that identifies the score as the usual DML ATE score and states that the following proof establishes WCF's conditions.

## C-04. Bang and Robins (2005), doubly robust identity

**Manuscript location / node.** WCF supplement, line 205, node `supp:A4-bias`.

**Bibliography key and identity.** `bang2005doubly`, *Doubly Robust Estimation in Missing Data and Causal Inference Models*, Biometrics 61(4), 962–973 (2005), DOI [10.1111/j.1541-0420.2005.00377.x](https://doi.org/10.1111/j.1541-0420.2005.00377.x). Publisher metadata and title/authors agree with the BibTeX. Wiley's page and an accessible full-text copy were inspected. Local evidence copy: [full-text copy](/home/hugo_souto/Stuff/Research/Wasserstein_Causal_Forests/research/audits/supplement_20260924/citation_evidence/Bang_Robins_2005_source_copy.pdf), SHA-256 `3f1b87b4ae3275ba78b3c8672c122b7293d25986b384bd3316c450512a463056`.

**Source result and mapping.** The article develops doubly robust estimators for missing-data and binary-treatment causal models, including an augmented inverse-probability-weighted treatment-effect representation. This supports the WCF text's broad attribution to the usual doubly robust identity. It does not supply the exact WCF conditional remainder in its displayed notation; WCF obtains that identity directly by replacing treatment indicators times $H$ with their conditional means. Nor is the 2005 parametric-model result the source of WCF's cross-fitting/product-rate asymptotics.

**Disposition.** Identity verified; broad attribution passes; exact formula is a local algebraic specialization, not an imported equation. Source proof itself was not audited.

## C-05. Barlow and Brunk (1972), isotonic projection / weighted PAV

**Manuscript location / node.** WCF supplement, line 70, implementation description for \(\Pi_{\mathcal Q_K}^w\).

**Bibliography key and identity.** `barlow1972isotonic`, Barlow and Brunk, “The Isotonic Regression Problem and Its Dual,” JASA 67(337), 140–147 (1972), DOI [10.1080/01621459.1972.10481216](https://doi.org/10.1080/01621459.1972.10481216). The publisher page verifies title, authors, volume/pages, and states in its abstract that weighted least squares is minimized under order constraints.

**Searches and access.** Queried the article title with “PAVA” and “pool adjacent violators.” T&F abstract inspected. Direct PDF access returned HTTP 403; no full article was inspected. I also checked the official Stanford record for the authors' 1970 report with the same title, but the repository object itself was not accessible through the browser. An attempted Project Euclid download of Ayer et al. (1955) was blocked by its bot page.

**Mathematical mapping.** The WCF objective is exactly the weighted least-squares projection onto a monotone cone on a finite total order. The article's abstract supports that optimization problem. Uniqueness follows from strict convexity, and nonexpansiveness from metric projection onto a closed convex set in the weighted Hilbert norm; these facts are elementary consequences of the local projection setup. The remaining source claim is computational: that weighted pool-adjacent-violators computes this optimizer. The accessible abstract does not establish that algorithm or its weighted version.

**Disposition and follow-up.** Identity verified; content verified for the isotonic least-squares formulation; algorithm attribution content unverifiable due full-text access limit. The PAV algorithm is not wholly uncited, but this particular source-result pairing is not confirmed. Either inspect the Barlow–Brunk full article, or retain it for the weighted objective and add a direct algorithm reference (for example Ayer et al. (1955), *An Empirical Distribution Function for Sampling with Incomplete Information*, Annals of Mathematical Statistics 26(4), 641–647, DOI [10.1214/aoms/1177728423](https://doi.org/10.1214/aoms/1177728423), or the 1972 Barlow–Bartholomew–Bremner–Brunk monograph on isotonic regression). Do not present this access limitation as evidence that the existing source is wrong.

## Missing-source scan and evidence record

The supplement explicitly cites its named external tools: weighted PAV (`barlow1972isotonic`), DRF/Causal-DRF, the Gneiting–Raftery scoring-rule result, TCDA, Bang–Robins, and Chernozhukov et al. The Section 4 TCDA sources for Banach/Pettis/Bochner facts and its classical persistence-stability lemmas are cited in the supplied TCDA manuscript. I found no additional genuinely borrowed named algorithm/theorem in the audited passages that lacks any source; the actionable gap is narrower: the exact PAV algorithm source should be verified or supplemented. Local derivations (for example the grid-summary Lipschitz constants, AIPW conditional algebra, and the Gaussian-kernel Fourier strictness proof) should remain attributed to the local proofs rather than assigned a new external source.

External material saved under `citation_evidence/`: full author copy of Gneiting–Raftery (2007), official arXiv v1 PDF of TCDA, and the accessible Bang–Robins (2005) full-text copy. Their source URLs are recorded above. Chernozhukov et al. was inspected in the official OUP full HTML; its exact theorem text is summarized above. For Barlow–Brunk and Ayer et al., only publisher/institutional bibliographic metadata or abstracts were accessible; full-text access failed as described above.
