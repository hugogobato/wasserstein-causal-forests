# Blind independent mathematical review

## 1. Verdict and scope

**Reviewer verdict: MATERIAL DEFECTS FOUND**, limited to a material ambiguity/inconsistency in the description of simulation design S3. The formal results A.1–A.6 and the other mathematical DGP assertions reviewed below have no material defect found; Theorem A.2 also needs an explicit finite-risk convention to be fully well-defined. This is a blind review artifact, not an adjudication of another audit.

The audited frozen source is `research/audits/supplement_20260924/source/paper__supplement.tex`, lines 1–319 and 321–394. I also read the frozen `paper__main.tex`, its bibliography, and the relevant portions of `TCDA__main.tex` and its bibliography for prerequisite definitions and local citation matching. SHA-256: supplement `0d7f9e61b07dda04a9c529a7883f8945ac27e9105295b82b05f48ffbf370115a`; main `d35f28cf7e5b4da495621cadf15429ec0ef786b0b2c8b3dfa0be14bc8ea3e8a5`; paper bibliography `8e236105f0276157cf1971cd3e57fee7eb04266fc52b785480c48a77277f218a`; TCDA source `e544298d5e270bfe66e63acdccfa5fcf24a1f71a22f570c85df5c44a48eaa11a`; TCDA bibliography `4e877f7245d1a9b1bce04cc82e35468bcebd8aa21350579e0055a8926ff751ac`.

No compilation or numerical run was performed. The main source supplies the definitions of the macros and states that h is integrable. I did not inspect current manuscript files, other audit reports, or worker outputs, and did not edit source files. I did not perform an adjudication pass.

## 2. Mathematical specification and foundation checks

For fixed K and positive weights summing to one, the unit-level observation Q is an ordered vector in the closed cone QK, while P_a(x) is the across-unit conditional law of Q^a given X=x. Consistency, conditional exchangeability, and positivity identify P_a(x) with the observed conditional law, up to the usual P_X-null-set choice of regular conditional version. The manuscript explicitly distinguishes this a.e. identification from values at a fixed covariate point. The finite-grid reference target is an outcome-level expectation of h(Q), not a transform of the arm-averaged law.

The target and AIPW layers use ordinary real-valued conditional means. Integrability is stated in the main source, and the stronger moment requirements for A.4 and A.5 are explicit. Particle labels are treated as arbitrary and no cross-arm particle coupling is used for individual effects. These objects are well-typed under the stated fixed K, positive-weight, and integrability conventions.

The causal DGPs are nonempty and have overlap: each assignment propensity is clipped to a closed interval bounded away from zero and one. The potential quantile vectors are ordered because z_k increases, the scale factor is positive, and the stated derivative of psi is strictly positive for 0 ≤ gamma < 1. The LS, S, and Z formulas keep gamma inside this range. Gaussian and finite Gaussian-mixture shocks have all finite moments, although their unbounded support means the compact-support transport conclusion of A.2 does not literally cover these simulation designs; the source states this limitation.

## 3. Coverage matrix

| Audit ID | Frozen supplement location | Main checks and stress tests | Status | Confidence |
|---|---:|---|---|---|
| B0 | 1–27 | Observation typing, causal identification, integrability imported from main, pointwise-version convention, target distinctions | No defect found | High |
| B1 | 30–72 | Empirical energy objective, gradient algebra, leaf shrinkage minimizer, weighted projection and descent claims | No defect found; weighted PAV implementation claim remains source-dependent | High |
| A1 | 76–109 | Score identity, smoothing integral, strictness for epsilon = 0 and epsilon > 0 | Verified under the stated finite-first-moment domain | High |
| A2 | 103–150 | Risk identity, oracle inequality, compact-support transport conversion, particle approximation, fixed-M obstruction | No defect found when displayed risks are finite and well-defined | High |
| A2a | 152–167 | Shared partition approximation, particle-cloud approximation, score-function net, sieve rate and balancing choice | No defect found | Medium-high |
| A3 | 171–187 | Wasserstein-to-functional transfer, marginal and stratum implications, pointwise limitation | Verified | High |
| B2 | 191–210 | Scalar AIPW score, exact bias identity, Cauchy–Schwarz product bound and one-correct-side consistency | Verified | High |
| A4 | 212–240 | Cross-fitted expansion, nuisance score L2 control, CLT, variance estimate, efficiency statement and tuning caveat | No defect found under stated conditions | Medium-high |
| A5 | 242–272 | Local score regression, denominator, undersmoothing, triangular-array CLT and local variance estimator | No defect found under stated conditions | Medium-high |
| B3 | 274–284 | Two-part law/mean decomposition, zero-atom convention, structural-state interpretation, participation caveat | No defect found | High |
| A6 | 286–301 | Mixture W1 coupling bound, scalar component-error bound, calibrated-score substitution | Verified | High |
| B4 | 303–319 | Grid representation bounds, measurement error, learned-reference derivative and dependence caveats | No defect found | High |
| D0 | 321–332 | Generic vector ordering, monotonicity of psi, treatment assignment | Verified | High |
| D1 | 334–355 | LS designs, positivity and target/moderator descriptions | No defect found | High |
| D2 | 357–378 | S designs, moments, equal-mean S3 construction and effect-heterogeneity description | Ambiguous/false as an unqualified constant-law-effect description; see F1 | High |
| D3 | 380–394 | Structural-zero designs, overlap, participation and component effects | No defect found | High |

## 4. Claim dossiers

### A.1: strict propriety with fixed smoothing

On the weighted Euclidean transform z_k = sqrt(w_k)q_k, the displayed integral representation is the standard identity for sqrt(r² + epsilon²) − epsilon. Substitution into the three terms defining D_epsilon expresses it as an integral of Gaussian-kernel MMD² values with strictly positive weight for every t > 0. For each t, Fourier inversion gives a positive Gaussian-weighted integral of the squared difference of characteristic functions. If D_epsilon is zero, this nonnegative integrand is zero for almost every t, hence for one such t, and the strictly positive Fourier weight implies equal characteristic functions and equal laws. The finite-first-moment condition makes each distance-score expectation finite and justifies the representation; the same reasoning includes epsilon = 0. Expansion also gives the score-regret identity with the stated factor 1/2. The strictness proof is locally self-contained, so the Gneiting citation is not a necessary proof leaf.

### A.2 and A.2a: excess risk and sieve comparison class

Conditioning on (X,A) yields R(F) − R(P) = one half the X expectation of the arm-weighted divergence. The approximate empirical minimization inequality plus two applications of the uniform deviation U_n yields R(Fhat) − R(P) ≤ 2U_n + b_n + o_n. Since e_a ≥ c, this gives the displayed factor 2/c. On compact C, probability laws are weakly compact; D_epsilon is continuous and vanishes only on the diagonal; W1 is continuous and separates laws. Thus the minimum divergence over the compact set W1 ≥ δ is positive when that set is nonempty. The diameter bound then gives the claimed integrated W1 control by first taking n to infinity and then δ down to zero.

For an i.i.d. M-cloud, direct expansion gives E D_epsilon(P_M,P) = M^-1 E d_epsilon(Z,Z'). This verifies the approximation calculation and the fixed-M non-atomic obstruction on compact C. The tail example at line 150 is valid: since the weights sum to one, ||j·1_K||_w=j and W1(F_j,G)=j^-1 j=1. With p=j^-1 and d=d_epsilon(0,j·1_K), E_{F_j,G}d=pd and E_{F_j,F_j}d=2p(1-p)d, so D_epsilon=2p²d=2j^-2(sqrt(j²+epsilon²)−epsilon)→0. This is a correct witness that vanishing energy divergence need not imply W1 convergence without tail control.

For A.2a, the Lipschitz conditional-law assumption makes the arm/cell mixture G_{a,j} within L_X sqrt(d) ell_n of P_a(x) in its cell. Changing the forecast law changes the score by at most 2W1. Replacing G_{a,j} by a random equal-weight cloud adds expected excess risk at most L_C/(2M_n); averaging over the finite cells implies a deterministic cloud tuple at least as good. Quantizing 2M_nJ_n particle locations in a compact subset of R^K produces a uniform score net with log-size O(KM_nJ_n log n) at mesh n^-1. Bounded-score concentration and the oracle inequality yield the stated order. Balancing ell_n, M_n^-1, and the complexity term gives ell_n^(d+3) of order log(n)/n. I found no missing dimension or quantifier factor that changes the rate.

There is one domain qualification for A.2: the source does not separately impose a finite first moment before defining R_epsilon and b_n, and differences of infinite risks need not exist. The theorem is sound on its intended domain where these risk quantities are real and the risk identity applies, but the moment/well-definedness condition should be stated explicitly. The later convergence premises do not repair an undefined expression.

### A.3, A.4, and supporting scalar-score claims

For any coupling of fitted and true laws, the Lipschitz inequality bounds the difference of h-expectations by L_h times expected transport cost; infimizing over couplings proves the first line. Subtraction and integration prove the treatment contrast and marginal bounds, and restriction to B followed by division by p_B proves the stratum form. The reverse triangle inequality gives the reference-distance constant one. Integrated consistency does not imply pointwise convergence; the source correctly states this limitation.

For the AIPW score, conditioning on X and simplifying gives

    E[phi(h;m,g) | X] − tau_h(X)
      = (g−e){(m_1−mu_1)/g + (m_0−mu_0)/(1−g)}.

The signs and denominators match the displayed score. Under g in [c,1−c], Cauchy–Schwarz gives the stated product bound. A.4’s nuisance product condition makes this conditional bias o_p(n^-1/2). Cross-fitting makes each fold’s evaluation observations independent of its fitted nuisances; bounded inverse weights, conditional residual variance, and L2 nuisance convergence give an o_p(1) conditional L2 score difference. The foldwise centered empirical differences are therefore o_p(n^-1/2), and the ordinary iid CLT applies to the true score. The same L2 control, LLN, and Cauchy–Schwarz support variance consistency. The assumptions are sufficient and the source correctly warns that the archived tuning/fold arrangement is not automatically covered.

### A.5 and A.6

For A.5, the kernel denominator divided by Nb^d converges to f_X(x). Local Holder bias is O(b^s) and disappears at scale sqrt(Nb^d) by undersmoothing. The nuisance conditional mean assumption is below the same scale, while its conditional variance contribution is o_p((Nb^d)^-1). For the true score, the local conditional moment bound gives a triangular-array Lindeberg condition, and the normalized variance after division by the squared denominator is v_h(x)∫L²/f_X(x). The local variance estimate is compatible with the stated local moment and score conditions. I found no counterexample within the specified independent-training and local-uniform setup.

The two-part decomposition is an algebraic mixture identity. The W1 bound follows by moving only the unmatched mass between 0 and the positive component, at cost no more than L_C|hat-pi−pi|, and coupling the shared component mass at cost pi W1(P-hat-plus,P-plus). The scalar bound follows by adding and subtracting the true component mean, using |h|≤B_h. The statement correctly flags that the component law is immaterial at pi=0, that fitted positive components may themselves put mass at 0, and that observed quantile-grid zero need not mean the underlying distribution is degenerate. A.6 follows.

### DGP checks and finding F1

The generic psi derivative differentiates to 1−gamma + gamma(z+1)^2/2, which is positive for gamma in [0,1), so the generated vectors remain ordered. Assignment clipping provides overlap. LS and Z moderator statements agree with the explicit treatment contrasts and propensities. S2 is a stochastic null because its arm-specific conditional shock laws coincide; S3 does have equal mean quantile curves because E exp(eta_0)=exp(0.45²/2), matching the deterministic treated scale multiplier. The Z designs keep pi and e in the stated clipped ranges.

**F1, moderate, confidence high: S3 is described inconsistently as having a covariate-constant distributional effect.** The frozen main text (line 139) describes S3 as a “distributional effect … that is constant across covariates,” while the supplement (lines 366–370) makes the baseline scale depend on X_4 and says the spread change is carried by X_4; it also designates X_4 as the active moderator. The equations clarify the distinction. Write v=0.45², z=z_k≠0 and c=exp(0.20x_4). In the control arm, Q_k=f(x)+xi_0+c exp(eta_0)z, with Var(xi_0)=0.40² and eta_0~N(0,v). In treatment, Q_k=f(x)+xi_1+c exp(v/2)z, with Var(xi_1)=0.15². The two means are equal, but

    Var(Q_k | A=0,X=x) − Var(Q_k | A=1,X=x)
      = 0.1375 + z² exp(0.40x_4+v)(exp(v)−1),

which varies with x_4. The log-scale parameter difference s_1−s_0=v/2 is constant, but the resulting conditional-law difference is not covariate constant. Thus the mean-curve claim is correct, while the unqualified distributional-effect claim is false if it refers to the law-level contrast or its spread feature, and otherwise ambiguous. A minimal clarification is to say that the log-scale parameter shift is constant while the conditional spread-law difference varies with X_4.

Counterexample certificate: construction is exactly the stated S3 DGP; the bounded covariates satisfy all DGP restrictions; the arm-specific shocks have the stated independent normal laws; gamma=0 is admissible; z_k is nonzero at any noncentral grid level. Equal means hold by the lognormal moment identity. The variance difference above is nonconstant in x_4. Certificate valid against the reading “conditional law effect is constant”; alternative narrow reading “s_1−s_0 is constant” is true and explains the ambiguity.

## 5. Citation record

| Source-result pair used locally | Local identity/content check | Applicability to the supplement | Access limit |
|---|---|---|---|
| Barlow and Brunk (1972), isotonic regression, cited for weighted PAV | Bibliographic key and DOI are present. The weighted projection onto QK is unique and nonexpansive by projection onto a closed convex set in the weighted Hilbert norm. | Mathematical projection claims checked; exact weighted algorithm/version was not independently authenticated. | External source not inspected. |
| Gneiting and Raftery (2007), Section 5/Theorem 5, cited in A.1 | Bib entry gives JASA 102(477), 359–378, DOI 10.1198/016214506000001437. Exact source theorem not inspected. | The supplement supplies its own Gaussian-kernel strictness proof, so theorem validity does not rely on the citation. | External source not inspected. |
| Souto and Diamantis (2026), Theorem 5.1 | Bib entry gives arXiv:2607.28161; corresponding theorem in the supplied frozen TCDA source identifies the outcome-level TATE under consistency, exchangeability and positivity. | Local A.3 identification/stability-transfer use is a scalar finite-grid specialization. | Frozen TCDA text inspected; external arXiv provenance not independently checked. |
| Souto and Diamantis (2026), Definition 5.2 | Shared theorem counter in the supplied TCDA source identifies the conditional-effect definition at the cited location. | Local conditional target conventions agree; no claim of pointwise identification beyond a.e. versions is made. | Same as above. |
| Souto and Diamantis (2026), Proposition 5.3 | Supplied TCDA source states and proves the augmented remainder and product-rate bound. | Scalar AIPW identity is recomputed directly and matches the cited result. | Same as above. |
| Bang and Robins (2005), doubly robust identity | Bib entry gives Biometrics 61(4), 962–973, DOI 10.1111/j.1541-0420.2005.00377.x. Exact source result not inspected. | Local scalar identity is derived and checked algebraically. | External source not inspected. |
| Chernozhukov et al. (2018), Theorem 5.1 | Bib entry gives Econometrics Journal 21(1), C1–C68, DOI 10.1111/ectj.12097. Exact source result not inspected. | A.4 gives its own sufficient-condition argument; no conclusion is accepted merely from this citation. | External source not inspected. |

The cited TCDA propositions were checked in the provided source text, which is useful content evidence but does not independently authenticate its arXiv record. No citation is called fabricated. The unverified Gneiting, Bang–Robins, Chernozhukov et al., and Barlow source details do not leave A.1, the AIPW bias identity, or A.4’s stated proof dependent on those sources. The weighted PAV implementation citation is the only specific algorithmic claim left citation-dependent.

## 6. Successful stress tests and limitations

The following attacks did not expose a defect: epsilon = 0 in A.1; coincident particles in the smoothed gradient; lambda = 0 and large lambda in the leaf solution; empty W1-separated sets in the compactness argument; non-attainment of the empirical infimum in A.2; fixed M with a non-atomic truth; stratum restriction with p_B>0; nonzero benchmark h(0); pi=0 in the two-part parameterization; and bounded overlap at each clipped DGP endpoint. Exact algebra also confirmed the treatment/control signs in the AIPW bias formula, the tail calculation in A.2, and the mixture coefficients in A.6.

The claim coverage is bounded to the listed frozen supplement ranges and prerequisite source passages. External sources were not web-authenticated; no source proofs were needed for the locally derived main arguments. The A.2 moment/well-definedness condition and S3 descriptor require clarification. The S3 discrepancy concerns design description and does not invalidate the S3 equations, the finite-grid estimands, or A.1–A.6.
