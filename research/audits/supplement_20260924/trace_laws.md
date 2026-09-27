# Supporting proof traces: law and fitting claims

## Scope and status

This packet expands and locally checks the requested claims in the frozen file
research/audits/supplement_20260924/source/paper__supplement.tex, lines 1–187. Its SHA-256 is
0d7f9e61b07dda04a9c529a7883f8945ac27e9105295b82b05f48ffbf370115a; visible repository HEAD was
b9d441398f70a993cd68d8c92e1f20da5e8560e9. Bibliography files were identified but their entries and
cited works were not checked; citation authentication belongs to the primary audit. No manuscript
files were edited, no other audit-worker outputs were opened, and no worker was delegated. The
claim-inventory script was run as a coverage aid; it did not identify claims in this restricted
span. A first display command returned the whole frozen TeX file; analysis and this report use only
lines 1–187.

Expansion records reconstruct the supplied reasoning without silently adding assumptions. Validation
checks those reconstructed steps locally. “Routine reconstruction” means a finite omitted derivation
from recorded premises. This is a bounded support artifact, not a global verdict.

## Coverage and claim specifications

| ID | Source | Claim and hypotheses |
|---|---|---|
| A1 | Proposition A.1, lines 76–101 | For probability laws F,G on Q_K with finite first weighted moments and fixed epsilon >= 0, expected score difference is D_epsilon(F,G)/2 >= 0, with equality iff F=G. |
| R-ID | Lines 103–109 | Conditional risk excess is the covariate average of arm-weighted divergences. |
| A2 | Theorem A.2, lines 113–139 | Under iid sampling, causal assumptions, overlap c <= e <= 1-c, deterministic equal-weight particle classes and approximate empirical minimization, integrated divergence is bounded by 2(2U_n+b_n+o_n)/c. Common compact support plus vanishing errors gives mean W_1 convergence. |
| APP | Unnumbered approximation claims, lines 141–150 | Finite-class/net bounds, particle approximation identity, fixed-M obstruction, algorithmic caveats, and tail example. |
| A2a | Corollary A.2a, lines 152–167 | Shared-partition particle sieve has the displayed energy-risk bound under compact support and W_1-Lipschitz conditional laws. |
| A3 | Proposition A.3, lines 171–181 | W_1 law error transfers to Lipschitz summary means, treatment contrasts, marginal effects, and stratum effects. |
| GRAD | Equation (app-gradient), lines 36–52 | Smoothed negative weighted-preconditioned gradient formula. |
| SHR | Lines 54–65 | Penalized leaf minimizer has a shrunk contrast and preserves the count-weighted mean. |
| PROJ | Lines 67–70 | Weighted isotonic projection is unique and nonexpansive; PAVA computes it. |
| LIP | Lines 183–187 | Listed summary Lipschitz constants and downstream consistency/evaluation implications. |

Throughout, all w_k are positive and sum to one; Q_K is the closed ordered cone; the norm is
||v||_w=(sum_k w_k v_k^2)^(1/2); d_epsilon(r)=sqrt(r^2+epsilon^2)-epsilon. D_epsilon compares
outer laws. W_1 compares laws on vectors.

## Atomic expansion and validation

### A.1 and the conditional risk identity

| Node / span | Inputs, rule, and output | Side conditions and validation |
|---|---|---|
| A1-1, lines 78–89, 94–95 | Expand E_G S(F;V)=E d(Z,V)-1/2 E d(Z,Z') and E_G S(G;V)=1/2 E d(V,V'). Their difference is cross - half FF - half GG, exactly D_epsilon(F,G)/2. | Z,Z' iid F and V,V' iid G. Finite first moments make each expectation finite since d_epsilon(p,q) <= ||p-q||_w. Routine reconstruction. |
| A1-2, lines 95–100 | Map q to z=(sqrt(w_1)q_1,...,sqrt(w_K)q_K), preserving distances and equality of laws. Insert the displayed integral representation of d_epsilon. The constant 1 cancels in the F-G signed measure product, leaving an integral of Gaussian-kernel energies. | The weight exp(-epsilon^2 t)t^(-3/2)/(2sqrt(pi)) is positive for all t>0. The transform is invertible. |
| A1-3, lines 100–101 | Fourier inversion writes each Gaussian-kernel energy as a positive Gaussian-weighted integral of |phi_F-phi_G|^2. It is nonnegative and vanishes iff characteristic functions, hence laws, agree. Integration against the positive t-weight proves strictness. | The local Fourier argument supplies the needed characteristic property. Gneiting, Section 5, Theorem 5, is cited as a support route but was not authenticated here. Finite first moments ensure finite divergence, including epsilon=0. |
| R-1, lines 12–16, 103–106 | Condition on X=x,A=a, use Q|X,A=a ~ P_a(x), apply the A.1 identity, multiply by e_a(x), sum arms and integrate X. Obtain R(F)-R(P)=1/2 E_X sum_a e_a(X)D_epsilon(F_a(X),P_a(X)). | The conditional identity is valid a.e. where conditional first moments are finite. Integrating and subtracting marginal risks requires finite/integrable risks, or a separately defined finite excess. This condition is not explicit; see C1. |
| R-2, lines 103–109 | Since D_epsilon >= 0 and e_a>0, zero risk excess implies D_epsilon(F_a(X),P_a(X))=0 a.s. for each arm, hence F_a=P_a by A.1. | Pointwise positivity identifies at a zero integral; the uniform bound e_a>=c is needed for A.2's quantitative inequality. |

Validation: no defect found in A.1's algebra or strictness for its stated finite-moment domain. Strict
propriety concerns the unrestricted population law and does not imply exact recovery by a restricted
finite-particle class.

### Theorem A.2

Inputs: iid units; stated causal assumptions and c <= e <= 1-c; deterministic nonempty class F_n;
Fhat in F_n; U_n=sup_F |Rhat(F)-R(F)|, b_n=inf_F (R(F)-R(P)); and Rhat(Fhat)<=inf_F Rhat(F)+o_n,
with o_n>=0.

| Node / span | Inputs, rule, and output | Side conditions and validation |
|---|---|---|
| A2-1, lines 116–124, 133–134 | R(Fhat)<=Rhat(Fhat)+U_n<=inf Rhat+o_n+U_n<=inf R+o_n+2U_n=R(P)+b_n+o_n+2U_n. The penultimate inequality follows from Rhat(F)>=R(F)-U_n for every F. | Routine reconstruction. No infimum need be attained. The passage's approximate-minimizer explanation is sufficient though not necessary. Risks and differences must be well-defined and finite for this algebra; see C1. |
| A2-2, lines 103–106, 114, 133–134 | The risk identity and e_a>=c yield R(Fhat)-R(P)>=c/2 E_X sum_a D_epsilon(Fhat_a,P_a). Combine with A2-1 for the displayed oracle bound. | Correct on the finite-risk domain. |
| A2-3, lines 125, 133–134 | If U_n,b_n,o_n ->p 0, the nonnegative oracle bound implies E_X sum_a D_epsilon ->p 0. | Direct convergence-in-probability step. |
| A2-4, lines 125, 134–138 | On compact C, P(C) is weakly compact, D_epsilon is continuous on P(C)^2, and W_1 is continuous in the weak topology. The closed set of pairs with W_1>=delta is compact. Its minimum kappa_delta of D is attained and positive, since the zero set is the diagonal. | d_epsilon is bounded continuous on C^2; W_1 metrizes weak convergence on compact C. If the set is empty, all distances are <delta. |
| A2-5, lines 135–139 | For W_1<delta use W_1<=delta; otherwise W_1<=diam(C)<=diam(C)D/kappa_delta. Thus W_1<=delta+diam(C)D/kappa_delta. Integrate over fresh X, let n tend to infinity and then delta decrease to zero. | This proves the stated mean transport convergence in probability. It does not give uniform or pointwise-in-x convergence. |

### Unnumbered approximation claims, lines 141–150

| Node / span | Atomic reconstruction | Validation and edge cases |
|---|---|---|
| APP-1, line 141 | For N_n deterministic score functions with bounded scores, apply bounded-score concentration to each and union bound: U_n=O_p(sqrt(log(2N_n)/n)). For a uniform delta_n-net in sup norm, approximate each score and add at most 2delta_n; the stated O_p(delta_n+sqrt(log(2N_n)/n)) follows. | Correct order under iid sampling and the usual measurability/separability convention for suprema. |
| APP-2, lines 143–148 | Put mu=E d_epsilon(Z,Z'). Cross expectation for P_M versus P is mu; population pair expectation is mu; empirical pair expectation is (1-1/M)mu because M diagonal terms vanish. Thus E D_epsilon(P_M,P)=mu/M. On compact C, mu<=diam(C), so some cloud has divergence no greater than the mean. | Exact for every M>=1 and epsilon>=0. |
| APP-3, line 148 | Laws supported on at most M points are weakly closed on compact C: represent locations in C^M and weights in the compact simplex, then extract a convergent subsequence. A nonatomic law lies outside this closed set and cannot be approached in W_1 by fixed-M laws. | A positive-P_X-measure set of such conditional laws obstructs integrated recovery; a single null covariate point does not. |
| APP-4, lines 150–151 | For F_j=(1-1/j)delta_0+j^(-1)delta_(j 1_K), G=delta_0 and ||j 1_K||_w=j. Cross expectation is j^(-1)d(0,j1); FF pair expectation is 2j^(-1)(1-j^(-1))d(0,j1). Substitution yields D=2j^(-2)(sqrt(j^2+epsilon^2)-epsilon)->0. W_1=j^(-1)j=1; the grid mean changes by 1. | Exact tail counterexample to energy convergence implying W_1 convergence without tail control. |

The algorithmic caveat at line 150 follows from A.2's separate optimization and approximation terms:
accepted-step descent does not establish o_n->p0, and fixed depth/iteration budgets do not establish
b_n->0.

### Corollary A.2a

Inputs: common compact support C; X in [0,1]^d; W_1-Lipschitz conditional laws; deterministic cubes
of side ell_n; arbitrary M_n-particle clouds per arm and cell; approximate global empirical
minimization.

| Node / span | Atomic reconstruction | Validation |
|---|---|---|
| A2a-1, lines 153, 162–164 | G_(a,j)=Law(Q|A=a,X in cell j) is a mixture of P_a(x') over x' in that cell. Couple components to P_a(x); Lipschitzness and cell diameter sqrt(d)ell_n give W_1(G_(a,j),P_a(x))<=L_X sqrt(d)ell_n. | Treatment can tilt mixture weights, but all x' stay in the cell. Zero arm/cell-probability cells have zero risk weight and can be filled arbitrarily. |
| A2a-2, line 163 | The attraction term changes by at most W_1(F,G). Couple two independent forecast draws for the self-interaction term; changing both arguments costs at most 2W_1, and its half factor leaves W_1. Thus the score changes by at most 2W_1, yielding excess at most 2L_X sqrt(d)ell_n. | d_epsilon is 1-Lipschitz in each vector argument for all epsilon>=0. |
| A2a-3, lines 143–148, 163 | APP-2 gives E D(P_M,G)<=diam(C)/M. Score-risk excess is D/2, so expected excess from particle replacement is <=diam(C)/(2M_n). Some deterministic collection of clouds attains no more than its expected risk. Hence b_n<=2L_X sqrt(d)ell_n+diam(C)/(2M_n). | This is a comparison-class existence bound, not a fitted-WCF rate. |
| A2a-4, lines 165–166 | A delta-net for one location in C has size <=(C_0/delta)^K. Quantizing 2M_nJ_n locations gives log cardinality <=2KM_nJ_n log(C_0/delta); matched-particle coupling changes the forecast W_1 by <=delta, hence score functions by <=2delta. | Ambient K-dimensional covering bound is valid for compact C. |
| A2a-5, lines 153–160, 165–166 | Set delta=n^(-1). Bounded-score concentration gives U_n=O_p(sqrt(KM_nJ_n log(n)/n)), absorbing the n^(-1) net error. Substitute U_n and b_n into A.2 to obtain the displayed energy bound. With J_n=O(ell_n^(-d)) and ell_n=M_n^(-1)=t, balancing t and sqrt(t^(-(d+1))log(n)/n) gives t^(d+3)~log(n)/n. | Constants may depend on overlap, C,w,K,d,L_X. The balancing controls only the first three terms. |

Validation: the displayed rate correctly retains +o_n. Risk consistency at the balanced choice also
requires o_n->p0. The line 167 sentence that this construction retains risk consistency does not
restate that condition; nearby caveats at lines 150 and 167 signal the intended optimizer requirement.
Record candidate C2, not a false rate claim.

### Proposition A.3 and summary claims

| Node / span | Atomic reconstruction | Validation |
|---|---|---|
| A3-1, lines 172–181 | For any coupling (Z,V), Lipschitzness gives |E h(Z)-E h(V)|<=L_h E||Z-V||_w. Take the infimum over couplings to obtain the arm-mean bound. | First moments imply h-integrability because |h(q)|<=|h(0)|+L_h||q||_w. Infimum need not be attained. |
| A3-2, lines 174–178 | Subtract the two arm means and use the triangle inequality to obtain the contrast bound. Integrate its absolute value for the marginal bound; restrict to B and divide by p_B>0 for the stratum analogue. | Correct whenever the displayed W_1 expectations are finite. |
| A3-3, lines 178, 181 | Reverse triangle inequality gives | ||q-q_star||_w-||r-q_star||_w |<=||q-r||_w, so L_(h_star)=1 for every fixed reference. | Direct. |
| LIP-1, line 183 | Weighted Cauchy–Schwarz gives grid-mean constant sqrt(sum w_k)=1. Weighted centering is an orthogonal projection off constants, operator norm one, giving the same bound for SD. Coordinate j has dual norm w_j^(-1/2). A normalized mean over nonempty J has dual norm (sum_(j in J)w_j)^(-1/2). | Direct weighted-norm calculations. Skewness is not continuous at constant vectors under the usual standardized-moment definition; its variance denominator vanishes and directional limits can differ. |
| LIP-2, lines 185–187 | A.2's integrated transport convergence plus A.3 gives integrated, marginal, and fixed-stratum summary consistency. Pointwise or uniform summary convergence needs pointwise or uniform transport control. For independent evaluation, decompose error into population plug-in error plus centered sample average; bounded h on C bounds predictions and gives O_p(N^(-1/2)). | Correct. Integrated error may vanish while error stays at a fixed nonatomic point on a shrinking neighborhood. Root-n normality does not follow from law consistency alone. TCDA attribution at line 187 was not checked. |

### Gradient, shrinkage, and projection

| ID / span | Expansion and validation |
|---|---|
| GRAD, lines 36–50 | From S_(epsilon,M)=M^(-1)sum_m d(p_m,Q_i)-(2M^2)^(-1)sum_(m,l)d(p_m,p_l), differentiate: attraction derivative is w_k(p_mk-Q_ik)/(M r). The ordered pair sum has row and column contributions, totaling 2w_k sum_l(p_mk-p_lk)/r; after its prefactor this is -w_k/M^2 times the sum. Multiplication by -M/w_k gives exactly (app-gradient). Positive epsilon prevents zero denominators at coincident vectors. |
| SHR, lines 54–65 | For f=sum_a n_a||v_a-u_a||_2^2+lambda||v_1-v_0||_2^2, first-order conditions are n_0(v_0-u_0)+lambda(v_0-v_1)=0 and n_1(v_1-u_1)+lambda(v_1-v_0)=0. Adding yields the count-weighted mean bar(u); solving for v_1-v_0 yields [n_0n_1/(n_0+n_1)]/[n_0n_1/(n_0+n_1)+lambda] times Delta(u). Resolving mean and contrast gives the displayed formulas. | Requires n_0,n_1>0 and lambda>=0. Strict convexity gives uniqueness. Lambda=0 recovers raw means; increasing lambda decreases contrast and preserves weighted mean. The leaf minimum-arm-count constraint excludes zero counts. |
| PROJ, lines 67–70 | Q_K is nonempty, closed, convex; positive weights define a Hilbert norm. Strict convexity/coercivity gives a unique projection. Projection variational inequalities imply ||p-q||_w^2<=<p-q,x-y>_w<=||p-q||_w||x-y||_w; divide unless p=q. | Uniqueness and nonexpansiveness checked directly. Weighted PAVA computation/citation was not authenticated here. Backtracking proves nonincrease only for accepted steps, not global optimization. |

## Edge-case checks

1. At epsilon=0, A.1's integral weight stays positive on t>0 and the finite-moment divergence remains finite. The gradient claim is only for epsilon>0.
2. At M=1, the empirical pair term is diagonal zero; the particle identity becomes E D(delta_Z,P)=E d(Z,Z'), matching the formula.
3. At K=1, Q_K=R and projection is the identity. Coincident particles and ties remain covered because epsilon>0 smooths the gradient and the cone projection is unique.
4. At lambda=0, shrinkage returns u_0,u_1; as lambda grows without bound, contrast tends to zero. If an arm count were zero with lambda=0, the minimizer would be nonunique and c_lambda would be 0/0; the stated arm-specific leaf constraint rules this out.
5. The line 150 tail example is an exact failure of energy-to-W_1 transfer without tail control.
6. With X uniform and an error indicator on a shrinking interval around x_0, integrated error vanishes while error at x_0 remains one. This witnesses the pointwise caveat at line 185. At a positive-mass atom, this construction would fail.

## Candidate gaps and unresolved conditions

### C1: marginal risk integrability is unstated

Evidence: lines 76–82 impose finite first moments for each law in A.1. Lines 103–106 define R by
integrating conditional scores but do not require integrability over X. A.2 lines 113–124 likewise
give no unconditional moment/envelope condition. Conditional first moments can be finite almost
everywhere while their integral over X is infinite; then marginal risks can be infinite and
R(F)-R(P), or the supremum defining U_n, can involve undefined infinity-minus-infinity terms.

Strongest conclusion justified: the conditional identity is valid a.e. where conditional moments
are finite. Its integrated form and A.2's oracle algebra require finite/well-defined risks or a
specified finite excess. The convergence premise U_n,b_n,o_n->p0 appears to impose a finite-risk
reading for the consistency consequence, but the standalone displayed bound does not state it.
This is a well-definedness/assumption-completeness candidate, not a counterexample to a finite-risk
reading and not an established false theorem.

### C2: A.2a risk consistency needs vanishing optimization error

Evidence: the rate at lines 155–159 contains +o_n; line 160 balances only the first three terms.
A.2 requires o_n->p0 for consistency. The sentence at line 167 says the construction retains risk
consistency without restating this condition. The rate itself is correctly stated. This is a
candidate omitted condition; nearby warnings may intend approximate minimization with vanishing
error, so no false claim is established under that reading.

No other unsupported inference was found in the requested trace blocks. External citation identity
and support, PAVA implementation, the TCDA attribution, and supplement material after line 187 were
outside this packet.

## Execution limits

Actual role: bounded support-audit trace worker. Model/reasoning setting was not exposed. Tools:
local read-only shell, parser-assisted inventory, and exact hand calculations. Delegation: none.
Independent blind review/adjudication: not performed in this bounded packet. No numerical code was
needed; requested edge cases were checked algebraically. This artifact is evidence for the
coordinator's audit, not a standalone final report.
