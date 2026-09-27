# Proof trace: cross-fitted inference, two-part laws, and benchmark scope

## Audit record

Audit ID: `supplement_20260924.inference-trace`

Task: proof-expansion support packet for source lines 189--319. This file records a reconstruction first and validation second. It does not authenticate external citations and does not assign a final manuscript-wide verdict.

Frozen source: `research/audits/supplement_20260924/source/paper__supplement.tex`, SHA-256 `0d7f9e61b07dda04a9c529a7883f8945ac27e9105295b82b05f48ffbf370115a`.

Actual role: single-agent proof expansion and local validation; no delegation; no manuscript edits. Source labels below refer to this frozen file. Source support is classified as `explicit`, `routine reconstruction`, or `cited result`. `Validation` is local to the displayed inference and does not certify citations.

Scope dependencies read in the same frozen source: causal identification and notation at lines 16--35; definitions of $H=h(Q)$, $\mu_{a,h}$, $\tau_h$, $\theta_h$, and $\theta_{h,B}$ at lines 27--35; the energy-law and functional-transfer results A.2--A.3 at lines 91--188. These are treated below as source premises where invoked, not re-audited here. The source's causal premises are consistency, $Y^a\perp A\mid X$, and $0<e(X)<1$; Theorem A.4 strengthens this to uniform overlap. External citations at lines 205, 229, and the TCDA cross-reference are not checked in this packet.

## Claim records and normalized quantifiers

### U1. Cross-fitted doubly robust consistency (unnumbered; lines 191--210)

For a fixed measurable scalar summary $h$, with fixed finite $K$ and $F\ge2$ folds independent of the observations and $|I_f|/n\to\rho_f\in(0,1)$, use nuisance functions fitted outside each evaluation fold. For each fold, fitted propensity $g=\widehat e^{(-f)}$ takes values in $[c,1-c]$ for a fixed $c>0$. Conditional on training, the displayed score identity applies pointwise in $x$. If the product of the foldwise $L^2(P_X)$ propensity error and summed outcome-regression error tends to zero and the foldwise score second moments are $O_p(1)$, the cross-fitted average is consistent for $\theta_h$. Outcome regression may have a nonzero limiting error if propensity error tends to zero and the outcome errors remain $L^2$ bounded; conversely, propensity error may persist if the outcome errors tend to zero and propensity errors remain $L^2$ bounded.

### A4. Scalar AIPW inference (Theorem A.4; lines 212--229)

For fixed $h$, fixed $K,w$, i.i.d. units, a fixed finite number of observation-independent folds with limiting positive fold shares, causal identification, true and estimated propensities in $[c,1-c]$, $E H^2<\infty$, and a uniformly bounded conditional residual variance in each arm, assume for every fold $f$ that $r_{e,f}=o_p(1)$, $r_{m,f}=o_p(1)$, and $r_{e,f}r_{m,f}=o_p(n^{-1/2})$. If $0<V_h=E\psi_h^2<\infty$, then the cross-fitted estimator has the stated asymptotic linear expansion and normal limit; the empirical score variance is consistent, and the resulting standard error is consistent. The influence function is asserted to be efficient for the fixed scalar target in the nonparametric i.i.d. model. Quantifiers are pointwise in the prespecified $h$ and target, not uniform over a data-selected family.

### U2. Known propensity with a possibly incorrect outcome limit (line 231)

Under the same fixed-fold evaluation structure, if $e$ is known exactly and $\widehat m_{a,h}^{(-f)}\to m_{a,h}^\dagger$ in foldwise $L^2(P_X)$ for fixed limits (not necessarily $\mu_{a,h}$), the text claims normal inference with influence function $\phi(h;m^\dagger,e)-\theta_h$ and score-based variance estimation, generally not efficient. The statement is pointwise for this fixed limit and relies on the same finite-moment and overlap conditions needed for the score to be square integrable.

### U3. Fixed stratum and finitely many summaries/strata (lines 233--240)

For a fixed, prespecified measurable $B$ with $p_B=P(X\in B)>0$, the ratio estimator is claimed to be root-$n$ normal under A.4's conditions, with influence function $1\{X\in B\}(\phi_0-\theta_{h,B})/p_B$ and the displayed plug-in variance. For any fixed finite collection of prespecified summaries and strata meeting their corresponding conditions, a joint finite-dimensional normal limit and empirical covariance estimate are asserted. The claim does not cover data-dependent selection or a uniform confidence band.

### A5. Local score estimator (Proposition A.5; lines 246--272)

For one fixed interior point $x$, an independent training sample and evaluation sample of size $N$, a bounded nonnegative compactly supported kernel $L$ with $\int L=1$ and $\int L^2>0$, $f_X$ continuous and positive at $x$, locally $s$-Hölder $\tau_h$ for fixed $0<s\le1$, continuous conditional true-score variance $v_h$ with $v_h(x)>0$, and locally bounded conditional $(2+\delta)$ moments for some $\delta>0$, assume clipped nuisance propensities bounded away from 0 and 1. Conditional on training, uniformly over the shrinking kernel neighborhood, require score error second moment $o_p(1)$ and absolute score error conditional mean $o_p((Nb^d)^{-1/2})$. For bandwidths with $b\to0$, $Nb^d\to\infty$, and $\sqrt{Nb^d}b^s\to0$, the local score ratio has the stated pointwise normal limit. A local variance estimator is also claimed consistent under the same local conditions. No uniform-in-$x$ claim is made.

### A6. Two-part mixture stability (Proposition A.6; lines 274--301)

Pointwise in $x$ and for either arm, if true and fitted positive-component laws are supported on a common set $C$ containing $0_K$ with finite diameter $L_C$, and $h$ is bounded by $B_h$ on $C$, then the displayed Wasserstein and scalar mean bounds hold. If the assembled scalar mean is cross-fitted, it can be substituted into the AIPW score, with Theorem A.4 applying whenever its scalar nuisance conditions are met. The component law at $\pi_a=0$ is arbitrary; products by $\pi_a$ are interpreted as zero. The structural-zero interpretation is explicitly only for the finite-grid observation, absent a separate equivalence assumption for the underlying distribution.

### U4. Grid approximation and learned reference (lines 303--319)

For each arm, if the full-distribution summary is $L$-Lipschitz in $W_2$ and the grid summary is its evaluation on the step-quantile reconstruction, the absolute target discrepancy is bounded by $L\sum_a E a_K(Y^a)$. For the distance contrast to an underlying benchmark $Y_\star$, the stated bound additionally includes twice the reconstruction error of its grid reference. The claims require the expectations/targets to be defined. If $K=K_n$ grows, the statistical conditions must hold along the sequence; ignoring representation bias in root-$n$ inference requires it to be $o(n^{-1/2})$ (in addition to the other CLT conditions).

For any fixed or random candidate reference $\widehat q_\star$ and fixed $q_\star$, the population reference contrast changes by at most $2\|\widehat q_\star-q_\star\|_w$. If both arm laws assign zero mass to $q_\star$, the derivative of the reference contrast in direction $v$ is the displayed difference of expected weighted unit vectors. An asymptotically linear estimate of the reference then contributes this derivative applied to its first-order error, under the additional joint/delta-method conditions needed for such a composition. An atom may destroy ordinary differentiability; absence of an atom is the stated local condition.

## Expansion pass: atomic proof DAGs

### U1. Score identity and consistency

**U1-N1** (lines 199--204). Active objects: fixed candidate measurable functions $(m_0,m_1,g)$ with $0<g(x)<1$, true $e(x)$ and conditional means $\mu_{a,h}(x)$. Premises: $E(H\mid X=x,A=a)=\mu_{a,h}(x)$ and $P(A=1\mid X=x)=e(x)$. Rule: conditional expectation and substitution into the score. For arm 1, $E[A(H-m_1)/g\mid X=x]=e(\mu_1-m_1)/g$; for arm 0, $E[(1-A)(H-m_0)/(1-g)\mid X=x]=(1-e)(\mu_0-m_0)/(1-g)$. Add the plug-in contrast $m_1-m_0$ and subtract $\tau_h=\mu_1-\mu_0$. Output: $E\{\phi(h;m,g)\mid X=x\}-\tau_h(x)=(g-e)\{(m_1-\mu_1)/g+(m_0-\mu_0)/(1-g)\}$. Support: `routine reconstruction` of the stated replacement calculation. Side condition: denominators nonzero; exact identity is conditional-a.e. Validation: valid.

**U1-N2** (lines 205--209). Premise: $g\in[c,1-c]$. Apply triangle inequality to the two terms in U1-N1, then Cauchy--Schwarz separately to $|E[(g-e)(m_a-\mu_a)/g_a]|$, with $g_1=g$, $g_0=1-g$ and both inverse factors at most $c^{-1}$. Output: $|E b_h(X)|\le c^{-1}\|g-e\|_2\sum_a\|m_a-\mu_a\|_2$. Support: `routine reconstruction`. Required integrability: the displayed $L^2$ norms finite. Validation: valid.

**U1-N3** (lines 191--210). Parent: U1-N2. Per fold, conditional on that fold's training data, expected validation score is $\theta_h+E b_{h,f}(X)$. The absolute bias is bounded by the product in U1-N2. The centered validation average has conditional variance at most its conditional second moment divided by fold size. With a fixed number of folds, positive limiting fold shares, and foldwise score second moments $O_p(1)$, Chebyshev gives an $o_p(1)$ centered aggregate. Product $o_p(1)$ plus this fluctuation yields $\widehat\theta_h\to_p\theta_h$. Support: `routine reconstruction`. Validation: valid under the stated interpretation that the score second-moment bound controls the validation conditional variance. Cross-fold dependence is immaterial for finite $F$ because the argument bounds each fold's term and then sums finitely many terms.

**U1-N4** (lines 210). If $r_m\to0$ and $r_e=O_p(1)$, their product tends to zero; if $r_e\to0$ and $r_m=O_p(1)$, likewise. Bounded clipping makes $r_e=O(1)$ when $e$ is also bounded, and the text separately assumes the remaining error bounded. Output: stated one-side consistency observation. Support: `routine reconstruction`. Validation: valid; it is a sufficient route, not a claim that arbitrary inconsistent nuisances work.

### A4. Scalar AIPW inference

**A4-N1** (lines 191--204, 213--219). For fold $f$, condition on training and put $m_a=\widehat m_{a,h}^{(-f)}$, $g=\widehat e^{(-f)}$. Apply U1-N1 pointwise and integrate over fresh $X$. Using overlap for both $g$ and $e$ gives $|E(\widehat\phi_f-\phi_0\mid\text{training})|\le c^{-1}r_{e,f}r_{m,f}=o_p(n^{-1/2})$. Summing the fixed $F$ fold means leaves an $o_p(n^{-1/2})$ bias contribution to $\widehat\theta_h-\theta_h$. Support: `routine reconstruction`; no invocation of DML citation is needed for this algebra. Validation: valid.

**A4-N2** (lines 213--217, 229). Expand the score difference into regression-error terms and propensity-error terms. Inverse weights and their differences are bounded/Lipschitz on $[c,1-c]$; conditional residual variance bounds $E[1\{A=a\}(H-\mu_a)^2\mid X]$ uniformly; hence its integrated squared norm is bounded by a constant times $r_{m,f}^2+r_{e,f}^2$ (the bounded propensity multiplier on outcome error is absorbed). Since both rates are $o_p(1)$, the fold conditional score difference has $L^2(P_O)$ norm $o_p(1)$. Support: `routine reconstruction`. Validation: valid. This is the local detail compressed into “bounded inverse weights, ... and L2 nuisance consistency.”

**A4-N3** (lines 228--229). For each fold let $D_{if}=\widehat\phi_i-\phi_{0,i}$. Conditional on training, evaluation observations in that fold are independent. Split the fold average into its conditional mean and centered average. The mean has size $o_p(n^{-1/2})$ by A4-N1. Its centered part has conditional variance $o_p(1)/|I_f|=o_p(n^{-1})$ by A4-N2 and $|I_f|/n\to\rho_f>0$; conditional Chebyshev gives $o_p(n^{-1/2})$. Sum over finite $F$. Output: $\widehat\theta_h-\theta_h=n^{-1}\sum_i\psi_h(O_i)+o_p(n^{-1/2})$. Support: `routine reconstruction`. Validation: valid.

**A4-N4** (lines 219, 229). $\psi_h$ is centered by definition and has finite, positive variance $V_h$. Apply the ordinary i.i.d. scalar CLT to $n^{-1/2}\sum_i\psi_h(O_i)$ and Slutsky to A4-N3. Output: the stated $N(0,V_h)$ limit. Support: `routine reconstruction`. Validation: valid.

**A4-N5** (lines 223, 229). A4-N2 implies $n^{-1}\sum_i(\widehat\phi_i-\phi_{0,i})^2=o_p(1)$ by conditional Markov foldwise and finite summation. The true-score empirical first and second moments converge by LLN since $E\phi_0^2<\infty$ (from $EH^2<\infty$ and bounded inverse propensity). Cauchy--Schwarz transfers these moments to fitted scores; A4-N3 gives $\widehat\theta_h\to_p\theta_h$. Output: $\widehat V_h\to_pV_h$ and $\sqrt{\widehat V_h/n}$ estimates the standard error. Support: `routine reconstruction`. Validation: valid.

**A4-N6** (line 226). The score at true regressions and propensity is the standard efficient influence function for a fixed ATE of scalar observed outcome $H$ in the nonparametric i.i.d. treatment model, assuming causal identification and positivity. Output: efficiency statement. Support: `cited result` / standard semiparametric result; citation verification outside scope. Validation: `citation-dependent` for the efficiency assertion; the local CLT proof above does not itself establish the tangent-space efficiency bound.

### U2. Exactly known propensity and incorrect outcome limit

**U2-N1** (line 231). Put $g=e$ in U1-N1. For every candidate outcome regression, the conditional score expectation equals $\tau_h(x)$ exactly. Thus no product-rate bias remains. Support: `routine reconstruction`. Validation: valid.

**U2-N2** (line 231). If $\widehat m_a\to m_a^\dagger$ in $L^2$ and $e$ is fixed known with bounded inverse weights, the fitted score converges in $L^2$ to $\phi(h;m^\dagger,e)$ using the same decomposition as A4-N2. Its expectation is $\theta_h$ by U2-N1; square integrability follows from $EH^2<\infty$ and $m_a^\dagger\in L^2$. Cross-fitting plus finite folds gives an asymptotic-linear expansion around this possibly inefficient score by the A4-N3 argument, now with zero conditional mean difference from the target for each fitted regression. CLT and empirical variance consistency follow as in A4-N4/N5, provided the limit variance is positive for a nondegenerate normal limit. Support: `routine reconstruction`. Validation: valid with those inherited conditions; if the limit variance is zero, the nondegenerate normal statement does not follow.

### U3. Stratum and finite-dimensional inference

**U3-N1** (lines 233--238). $\widehat p_B$ is the sample mean of Bernoulli indicators with mean $p_B>0$, so $\widehat p_B\to_p p_B$ and $\sqrt n(\widehat p_B-p_B)=O_p(1)$. Under A4's expansion, the stratum numerator has expansion around $p_B\theta_{h,B}$ with influence $1_B\phi_0-p_B\theta_{h,B}$. Ratio/delta method maps this to $1_B(\phi_0-\theta_{h,B})/p_B$. Output: the displayed influence function and root-$n$ limit if its variance is positive. Support: `routine reconstruction`. Validation: valid.

**U3-N2** (lines 238). Apply the same finite-second-moment empirical-moment argument as A4-N5 to the squared estimated stratum influence contributions; $\widehat p_B\to p_B$ and the numerator estimate is consistent. Output: displayed variance estimator is consistent. Support: `routine reconstruction`. Validation: valid.

**U3-N3** (line 238). For a fixed finite list, stack the influence functions, apply the multivariate i.i.d. CLT (equivalently Cramér--Wold to every fixed linear combination), and apply componentwise score consistency to the empirical covariance. Output: joint finite-dimensional normal limit. Support: `routine reconstruction`. Validation: valid when every listed target meets A4 and the list is fixed in advance. The nonselection/uniformity limitation follows from the pointwise finite-list quantifiers, not from a simultaneous result.

### A5. Local score estimator

**A5-N1** (lines 247--255). Condition on the independent training sample. Let $W_i=L((X_i-x)/b)$ and $D_N=\sum_iW_i$. By change of variables and continuity of $f_X$ at $x$, $E W_i/b^d\to f_X(x)\int L=f_X(x)$. Boundedness and compact support give $\operatorname{Var}(W_i)=O(b^d)$, so $D_N/(Nb^d)\to_p f_X(x)>0$ because $Nb^d\to\infty$. Therefore $P(D_N=0)\to0$. Support: `routine reconstruction`. Validation: valid.

**A5-N2** (lines 253--255, 265). For $E_i=\widehat\phi_i-\phi_{0,i}$, conditional mean contribution to $\sum_iW_iE_i/D_N$ is bounded in probability by the uniform conditional bias $o_p((Nb^d)^{-1/2})$ times $D_N/D_N$. Its centered conditional variance is at most $\sup_z E(E_i^2\mid X=z,training)\sum_iW_i^2/D_N^2= o_p(1)O_p((Nb^d)^{-1})=o_p((Nb^d)^{-1})$, since $\sum W_i^2=O_p(Nb^d)$. Output: fitted-score substitution is $o_p((Nb^d)^{-1/2})$. Support: `routine reconstruction`. Validation: valid.

**A5-N3** (lines 251, 256, 265). In the numerator centered at the target, split
\[
\sum_i W_i\{\phi_{0,i}-\tau_h(x)\}
=\sum_iW_i\{\phi_{0,i}-\tau_h(X_i)\}
+\sum_iW_i\{\tau_h(X_i)-\tau_h(x)\}.
\]
The second sum has expectation $O(Nb^{d+s})$ by local Holder continuity and bounded kernel support, and centered variance $O(Nb^{d+2s})$; after division by $D_N\asymp_p Nb^d$ and multiplication by $\sqrt{Nb^d}$, its centered fluctuation is $O_p(b^s)=o_p(1)$, while its mean is negligible by $\sqrt{Nb^d}b^s\to0$. Density continuity gives $D_N/(Nb^d)\to_p f_X(x)$ and does not change the bias order. Support: `routine reconstruction`. Validation: valid.

**A5-N4** (lines 251, 265). The CLT is for the *conditionally residual-centered* numerator $\sum_iW_i\{\phi_{0,i}-\tau_h(X_i)\}$, not by itself for the unconditionally centered raw-score numerator. Its variance is $Nb^d f_X(x)v_h(x)\int L^2+o(Nb^d)$, by change of variables, continuity of $f_X,v_h$, and the local conditional variance definition. Local bounded conditional $(2+\delta)$ moments and bounded $L$ imply Lyapunov/Lindeberg as $Nb^d\to\infty$. The omitted conditional-mean component is handled separately in A5-N3: its mean is the undersmoothed bias and its centered fluctuation is smaller by $b^s$. Thus unconditional centering of raw scores would add only a variance term of relative order $b^{2s}=o(1)$, but this reduction is a routine step that should be made explicit. Divide by $D_N\sim_p Nb^df_X(x)$ and multiply by $\sqrt{Nb^d}$ to obtain variance $v_h(x)\int L^2/f_X(x)$. A5-N2/N3 and Slutsky give the stated limit. Support: `routine reconstruction`. Validation: valid; terse source wording is not by itself a gap because the conditional-mean decomposition reconstructs from the stated Holder condition.

**A5-N5** (lines 267--272). The density estimate $\widehat f_b=D_N/(Nb^d)$ converges to $f_X(x)$. The weighted residual second moment in $\widehat v_b$ converges to $v_h(x)$: the local conditional variance is continuous, local conditional $(2+\delta)$ moments give uniform integrability/LLN, true conditional means vary by $o(1)$ on the support, and the fitted-score error contributes $o_p(1)$ by its uniform conditional $L^2$ bound and Cauchy--Schwarz. Thus $\widehat v_b\int L^2/(Nb^d\widehat f_b)$ estimates the variance of the unscaled local estimator. Support: `routine reconstruction`. Validation: valid under the same local uniform conditions; target is pointwise, not a confidence band.

### A6. Two-part mixture bounds and score substitution

**A6-N1** (lines 276--299). At fixed $x,a$, write $M(p,R)=(1-p)\delta_0+pR$. Insert $M(\pi,\widehat P^+)$ between $M(\widehat\pi,\widehat P^+)$ and $M(\pi,P^+)$. Couple the common mass $\min(\pi,\widehat\pi)$ through the common component; couple the residual mass $|\widehat\pi-\pi|$ between $0$ and a point in $C$, at cost at most $L_C$ per unit mass. Couple the remaining $\pi$ component mass optimally between $\widehat P^+$ and $P^+$. Triangle inequality yields $W_1\le L_C|\widehat\pi-\pi|+\pi W_1(\widehat P^+,P^+)$. Support: `routine reconstruction`. Validation: valid, including $\pi=0$ with any assigned component version because its term is multiplied by zero.

**A6-N2** (lines 278--296, 299). Expand $\widehat m^{2p}-\mu=(\widehat\pi-\pi)(\widehat\mu^+-h(0))+\pi(\widehat\mu^+-\mu^+)$. Since both $|\widehat\mu^+|$ and $|h(0)|$ are at most $B_h$, triangle inequality gives the coefficient $2B_h$. Output: scalar inequality. Support: `routine reconstruction`. Validation: valid.

**A6-N3** (lines 296, 299--301). The AIPW score uses only the scalar regression prediction $m_{a,h}(X)$, so substituting the assembled two-part scalar mean preserves the algebraic bias identity U1-N1. If its cross-fitted $L^2$ nuisance errors and propensity meet A4, the A4 proof applies. The component $L^2$ error bound in line 301 follows by taking $L^2(P_X)$ norms of the pointwise scalar inequality and using Minkowski. Support: `routine reconstruction`. Validation: valid; no structural-state assumption is required for calibrating the scalar mean $E[h(Q)|A,X]$.

### U4. Grid resolution and learned reference

**U4-N1** (lines 305--309). For each potential outcome $Y^a$, Lipschitzness gives $|h_\infty(\mathcal R_Kq_K(Y^a))-h_\infty(Y^a)|\le L a_K(Y^a)$. Take expectations, subtract arm-specific means, and apply triangle inequality over $a=0,1$. Output: first grid-target bound. For reference distance use the 1-Lipschitz property of $F\mapsto W_2(F,Y_\star)$, add the per-arm reconstruction discrepancy, and then the reference discrepancy $W_2(\mathcal R_Kq_\star^K,Y_\star)$ once per arm. Support: `routine reconstruction`. Validation: valid when the expectations and summaries are finite/well-defined. The source should be read with those typing/integrability conditions; a Lipschitz function on an unbounded metric space need not itself have finite expectation absent a first-moment premise.

**U4-N2** (lines 309--311). If $K=K_n$, the preceding fixed-$K$ theorem is not uniform in dimension unless its hypotheses are re-established along the sequence. A root-$n$ expansion centered at the full target includes the representation bias; to omit it from a centered root-$n$ limit its total contribution must be $o(n^{-1/2})$. Measurement error bound follows pointwise from Lipschitzness: $|h(\widetilde Q)-h(Q)|\le L_h\|\widetilde Q-Q\|_w$. Support: `routine reconstruction`. Validation: valid as scope conditions.

**U4-N3** (lines 313). For $T(r)=E\|Q^1-r\|_w-E\|Q^0-r\|_w$, reverse triangle inequality gives $|T(\widehat q_\star)-T(q_\star)|\le2\|\widehat q_\star-q_\star\|_w$, pointwise for every pair of references and hence for random references as well. Support: `routine reconstruction`. Validation: valid.

**U4-N4** (lines 313--319). For $q\ne q_\star$, directional derivative of $q_\star\mapsto\|q-q_\star\|_w$ along $v$ is $\langle q_\star-q,v\rangle_w/\|q_\star-q\|_w$. The derivative is bounded in absolute value by $\|v\|_w$; if both arm laws have no atom at $q_\star$, dominated convergence permits expectation of the directional derivative, and subtraction gives the displayed derivative of $T$. A delta-method influence contribution is obtained by applying this derivative to the reference estimator's first-order error, provided joint asymptotic linearity and the relevant differentiability remainder hold. Support: the displayed derivative is `routine reconstruction`; the influence-combination sentence is conditional, not a complete theorem. Validation: derivative valid; inferential composition needs the stated extra regularity. An atom can cause a cusp, but is not sufficient by itself to prove nondifferentiability of the contrast if arm-specific nonsmooth contributions cancel.

## Validation pass: attacks, counterexamples, and qualifications

### A4 and unnumbered doubly robust claims

No counterexample found to the scalar AIPW expansion under the theorem's explicit uniform overlap, fold independence, $L^2$ consistency, product rate, residual-variance bound, and finite fixed $F$. Important hypotheses are indeed present: the bound applies to both true and estimated propensities; the theorem is for a fixed scalar summary; nuisance fitting/tuning is outside the relevant evaluation fold; and the error conditions are required for every fold.

The proof's compressed conditional $L^2$ sentence is reconstructible: score differences split into outcome-regression differences and propensity differences multiplying true residuals. Uniform conditional residual variance and bounded inverse propensities control the latter. This is a routine omitted derivation, not a proof gap. The sentence “ordinary i.i.d. CLT applies” applies to the true score sequence, not to fitted cross-fold scores; A4-N3 supplies the needed fitted-score replacement.

The “one nuisance side can be inconsistent” claim is correct only with the stated boundedness of the other side's $L^2$ error and bounded score moments. No claim of consistency for two arbitrary inconsistent nuisances is justified. The fixed-known-propensity extension is algebraically valid: with $g=e$, the conditional score mean equals $\tau_h$ for every fitted regression. It still needs square-integrable limiting regressions and, for a nondegenerate normal limit, positive limiting score variance; the phrase “the same proof” inherits these qualifications.

Stratum attack: take $B$ with $p_B>0$ but arbitrarily small fixed probability. The variance can be large (scales with $p_B^{-1}$ in common cases) but remains finite for a fixed $B$ under A4 moments; the ratio proof is valid. If $p_B=0$, target/ratio are undefined, and the source explicitly excludes it. A data-selected $B$ or growing collection is outside the theorem, as the source says.

Finite-family attack: a fixed finite list permits multivariate CLT; a list growing with $n$ or chosen after looking at estimates would require simultaneous control. The source explicitly limits its claim, so no contradiction.

### A5 local limit and variance

Counterexample attack at the variance boundary: let $X$ have a positive continuous density at $x$, let $H=\tau(X)+\epsilon$ with conditional mean smooth and $\operatorname{Var}(\epsilon\mid X=z)=|z-x|^\alpha$ (for $\alpha>0$), and no nuisance error. This would make the local variance vanish at $x$, but it violates the explicit $v_h(x)>0$ condition and therefore is not a counterexample. A different attack with a fixed positive $v_h(x)$ and bounded conditional $(2+\delta)$ moments does not defeat the triangular-array CLT: bounded kernel weights and $Nb^d\to\infty$ give the needed Lindeberg control.

Serious failed attacks: (i) a nonsymmetric kernel can induce first-order spatial bias, but the proof only uses the general $O(b^s)$ Hölder bound and explicitly undersmooths so it is covered; (ii) a zero denominator can occur at finite $N$, but A5-N1 shows its probability tends to zero; (iii) replacing true scores by noisy fitted scores could dominate if their conditional mean were only $O((Nb^d)^{-1/2})$, but the stated condition is the strict little-o order and the centered error variance is also $o((Nb^d)^{-1})$.

No analytic counterexample found to Proposition A.5 or its local variance estimate under the literal uniform-over-neighborhood conditions. The estimator is pointwise; these hypotheses do not support uniform bands. The variance formula is for the unscaled estimator and equals $v_h(x)\int L^2/(Nb^d f_X(x))$ asymptotically, consistent with the displayed scaled CLT.

### A6 two-part stability

Mass-transfer attack: if $\widehat\pi>\pi$, only excess mass $\widehat\pi-\pi$ needs to move from zero into the fitted component; the common $\pi$ component mass is coupled across positive-component laws. If $\widehat\pi<\pi$, the unmatched true component mass moves to zero. Both cost at most $L_C|\widehat\pi-\pi|$. The factor on component-law distance is the common mass $\pi$, so the asymmetric-looking displayed inequality is valid. Scalar bound uses $|\widehat\mu^+-h(0)|\le2B_h$.

Boundary attack $\pi=0$: $P^+$ is not identified, but the product term is zero and any chosen version is immaterial for the mixture and scalar target. For positive but vanishing $\pi$, poor estimation of $P^+$ is attenuated by $\pi$. No counterexample found. The required support of both component laws in a common finite-diameter $C$ is essential for the displayed $L_C$ bound and is explicit.

### Grid and random-reference claims

The two-arm grid bound follows by armwise coupling and triangle inequality; the reference reconstruction term appears twice because there are two arm distances. The claim is not an arbitrary-grid convergence theorem: $E a_K(Y^a)$ must itself vanish at an adequate rate. For root-$n$ transfer the representation bias must be negligible at the CLT scale or explicitly included.

Random reference attack: in one dimension with both arms having an atom at $q_\star=0$, $E|Q^a-r|$ has a cusp contribution proportional to the atom mass. The derivative may fail. This satisfies the source's warning that an atom *can* invalidate ordinary differentiability. However, atom presence is not a universal failure certificate for a contrast: if arm laws are identical, the two cusp terms cancel. Thus “can invalidate” is accurately modal, while absence of atoms in both arms is a sufficient condition for the displayed derivative by dominated convergence.

The $2\|\widehat q_\star-q_\star\|_w$ perturbation bound is deterministic and survives dependence between benchmark and outcome samples. To use an estimated reference for inference about a fixed benchmark, a first-order expansion additionally needs a joint limit (or appropriate sample splitting) and a differentiability remainder; the source flags this rather than asserting the complete expansion unconditionally. A reference error $o_p(n^{-1/2})$ is sufficient to ignore the plug-in perturbation by the Lipschitz bound, regardless of differentiability.

## Local finding ledger

1. **No defect found (high confidence) in A4's scalar expansion and variance consistency**, conditional on its explicit fixed-fold and nuisance assumptions. External validity/efficiency citation has not been authenticated here.
2. **No defect found (medium-high confidence) in the local score CLT and variance formula A5** under its strong local uniform conditions. The proof compresses routine kernel-array details but they reconstruct from the assumptions.
3. **No defect found (high confidence) in the A6 mixture inequalities** under the common-support and bounded-summary premises.
4. **No defect found (medium confidence) in the grid/reference perturbation and derivative formulas**; finite-expectation well-definedness and joint delta-method conditions should be understood as prerequisites where the text is conditional.
5. **Citation-dependent leaf:** A4's efficiency attribution and external DML/TCDA comparisons are outside the assigned citation-authentication scope.

This packet did not audit earlier Theorems A.2/A.3, the cited external theorem identities, the archived implementation, or any manuscript-wide dependency propagation. It found no valid analytic counterexample in its assigned scope.
