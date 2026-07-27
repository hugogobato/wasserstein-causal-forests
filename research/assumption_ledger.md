# Frozen assumption ledger

This is the single source of truth for the assumptions in the theory plan. Later work packages cite assumption IDs rather than silently restating or strengthening them. An assumption may be weakened only in WP8, with the replacement recorded in `decision_log.md`.

| ID | Assumption | Used by | Status and discharge |
|---|---|---|---|
| A1 | Outer units are iid draws from a region superpopulation | WP2–WP7 | Load-bearing core. Spatial dependence is a WP8 extension, not part of the main theorem. |
| A2 | Consistency, no hidden treatment versions, and no interference between outer units | WP2, WP7, WP10 | Load-bearing for the own-treatment potential-outcome notation. Spillovers require an exposure mapping or partial-interference extension. |
| A3 | \(\{Q^{(0)},Q^{(1)}\}\perp Z\mid X\) | WP2–WP7 | Load-bearing causal assumption. Assess by DAG, negative controls, and sensitivity analysis. |
| A4 | Uniform overlap \(e(X)\in[\eta,1-\eta]\) | WP2–WP7 | Load-bearing for orthogonal scores; empirical gate in WP9 and WP10. |
| A5 | Confirmatory \(V\) has dimension at most two, or \(G<\infty\) subgroups are frozen independently of confirmation outcomes | WP7 | Prevents false ordinary-EIF claims for high-dimensional pointwise CATE. |
| A6 | \(\mu_{\mathrm{raw}}^{(z)}\in\mathcal P_2([0,\infty))\), \(Q^{(z)}=\log(1+Q_{\mathrm{raw}}^{(z)})\in L^2(\mathcal P)\), \(E\|Q^{(z)}\|_{L^2}^{2+\delta}<\infty\), and \(E\|(Q^{(z)}(p_1),\ldots,Q^{(z)}(p_K))\|_2^{2+\delta}<\infty\) | WP2–WP7 | Makes the frozen log1p transform well-defined and separately controls the fixed point-evaluation coordinates, which an \(L^2\) moment alone does not control. |
| A7 | \(T_1,\ldots,T_J\) are fixed and measurable, \(E|T_j(Q_{\mathrm{raw}}^{(z)})|^{2+\delta}<\infty\), \(E\|U_K(Q^{(z)})\|_2^{2+\delta}<\infty\), and functional-specific means or denominators are bounded away from zero where required | WP2–WP7 | Gives the complete finite score vector a \(2+\delta\) moment. Each functional gets support, positivity, and tail checks. |
| A8 | Feasible latent-target claims use either 1. \(\min_i m_i\to\infty\) with explicit uniform conditional bias \(b_{m_i}\to0\) and \(L^2\) error \(v_{m_i}\to0\) at the rate required by the claim, or 2. an identified measurement, replicate, or validation model for the latent law | WP5–WP7 | Bounded \(m_i\) with an unrestricted random-distribution mixing law is not identified. Prove the iid empirical-quantile or survey-design conditions rather than assuming them. |
| A9 | Nuisance functions are cross-fitted and satisfy the product-rate conditions derived in WP4 | WP3–WP7 | Do not assume generic \(n^{-1/4}\) without deriving the norm used by the forest target. |
| A10 | Any later forest theorem assumes honesty, regularity, symmetry where required, subsamples \(s_n\), and shrinking leaves | WP4, WP6 | This is a post-G2 theory assumption to map exactly to GRF, Qiu, and DRF. The current finite prototype does not discharge it. |
| A11 | Coordinate scales \(s_j\) are estimated using training data only and converge to positive constants; the declared trapezoidal quantile weights are deterministic | WP3–WP7 | Prevents one scalar coordinate or grid size from dominating splits without treating fixed quadrature weights as estimated. |
| A12 | Core theory uses fixed \(K\) and fixed \(J\) | WP4, WP5, WP7 | Minimum viable theorem. Growing \(K_n\) belongs to WP6. |
| A13 | Conditional target maps are Lipschitz or Hölder in the metric required by the chosen forest theorem | WP4, WP6 | Characterize the fallback rate if only Hölder smoothness is credible. |
| A14 | Forest Monte Carlo error from a finite number of trees is negligible relative to sampling error | WP4, WP7 | Verify by tree-count stabilization; otherwise include it explicitly. |
| A15 | Projection regularity holds for any theorem claiming projection is first-order negligible | WP6, WP7 | Flat quantile regions are a separate nonregular regime. |
| A16 | The application has a credible assignment mechanism, sufficient independent outer units, and a documented household sampling design | WP10 | A publication gate, not a routine data-cleaning assumption. |
