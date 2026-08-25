# Frozen assumption ledger

**Ledger ID:** `G0-A1-A17-v1`  
**Applies to:** finite-grid oracle program  
**Status:** frozen after Phase 0  
**Companion contract:** `research/estimand_contract.md`

This ledger preserves A1 through A17 from
`research_phases/_phase_shared.md`. The clarification column records the exact
mathematical strength used by Phase 0. A later phase may strengthen an
assumption only by issuing a new ledger version and recording the scope change.

| ID | Frozen assumption | Phase 0 clarification | Role and failure consequence |
|---|---|---|---|
| A1 | \(A\in\{0,1\}\). | Treatment labels are exactly 0 and 1. | Defines the two arm marginal laws. Multi-arm use requires a new contract. |
| A2 | \(Y=Y^A\). | The observed random measure equals the potential random measure under the realized treatment, with no hidden version mismatch in the quantile extraction step. | Consistency. Failure invalidates the observable g-formula. |
| A3 | \((Y^0,Y^1)\perp A\mid X\). | Conditional independence is required only for observational identification; randomized designs may use their known assignment law. No cross-arm independence of \(Y^0\) and \(Y^1\) is assumed. | Identifies each arm marginal conditional law. |
| A4 | \(0<e(X)<1\) on the target population. | Identification requires positivity almost surely. Stable estimation may impose \(e(X)\in[\eta,1-\eta]\) on a trimmed target population, but \(\eta\) must be declared rather than inferred after outcomes are examined. | Makes both observed arm regressions available on the target support. |
| A5 | \(Y^a\in\mathcal P_2(\mathbb R)\) almost surely. | This guarantees finite within-measure second moments, but not the outer moment needed by a random-measure barycenter. Whenever `BARY-A` is reported, also require \(E\{W_2^2(Y^a,\nu_0)\mid X=x\}<\infty\) for one, hence every, fixed \(\nu_0\in\mathcal P_2(\mathbb R)\), for \(P_X\)-almost every \(x\). | Types the continuum Wasserstein objects. Without the added outer clause, the supporting barycenter target is not certified. |
| A6 | Each \(Y_i\) is directly observed in the oracle regime. | The stored quantile vector is exact or validated to a tolerance named in the data manifest. Inner empirical samples, histograms, and measurement error are outside Version 1. | Prevents unmodeled inner-sampling uncertainty. |
| A7 | \(q(Y)\in\mathcal Q_K\) with declared positive quadrature weights. | Use \(0<u_1<\cdots<u_K<1\), \(w_k>0\), and \(\sum_k w_k=1\). The generalized-inverse convention, grid, and weights are immutable within an evaluation manifest. \(W_{2,K}\) means the declared quadrature metric, not exact continuum \(W_2\). | Makes \(d_W\) a genuine metric and fixes the implemented geometry. |
| A8 | \(P_a^K(x)\) has the moments required by the selected proper score. | For \(S_\varepsilon\), require \(E\{\lVert q(Y^a)\rVert_W\mid X=x\}<\infty\) for \(P_X\)-almost every \(x\), plus integrability over the target \(X\)-law for marginal risks. A conditional second moment may be imposed for optimization diagnostics, but is not needed for score propriety. | Ensures finite population energy risk. |
| A9 | Each \(T_j\) is fixed, measurable, and integrable. | Banach-valued targets are strongly measurable and Bochner integrable: \(E\lVert T_j(Y^a)\rVert_{\mathcal B}<\infty\). This clause also applies to every reported reference-distance functional. A functional extracted from \(P_a^K\) must be \(h_j\{q(Y)\}\); otherwise it is an explicitly separate continuum oracle target. | Makes TATE and TCATE well-defined and prevents finite-grid overclaiming. |
| A10 | \(\nu_\star\) is external or frozen before outcome analysis. | \(\nu_\star\in\mathcal P_2(\mathbb R)\). Exact reference effects require \(E W_2(Y^a,\nu_\star)<\infty\); grid reference effects use the frozen \(q(\nu_\star)\). Data-adaptive construction occurs inside training folds only. | Defines reference effects without outcome leakage. |
| A11 | \(V=g(X)\) is pre-specified without outcome leakage. | The map \(g\), moderator support, smoothing rule, and reporting grid are frozen before confirmatory evaluation. Conditional claims hold \(P_V\)-almost everywhere unless additional smoothness is stated. | Protects confirmatory TCATE interpretation. |
| A12 | Only marginal laws \(P_0^K(x)\) and \(P_1^K(x)\) are claimed. | No particle index, common random seed, shared tree, or posterior iteration creates a causal coupling. Joint-potential-outcome targets and individual-effect distributions are excluded. | Avoids nonidentification. |
| A13 | The selected score is strictly proper over the unrestricted finite-grid law space. | The certified score is the weighted energy loss \(S_\varepsilon\) in `research/cwdb_validity_certificate.md`, on \(\mathcal P_1(\mathcal Q_K,d_W)\). The collision parameter \(\varepsilon\geq0\) is part of the score definition, and the same value must be used in objective and gradient. Both \(\varepsilon=0\) and the certified smooth form are strictly proper. | Identifies \(P_a^K(x)\) as the unrestricted population minimizer. A mismatched score/gradient fails G0. |
| A14 | Every reported output is invariant to particle permutation. | Invariance is required under independent within-arm permutations. Gradients are permutation equivariant; public summaries, scores, and causal contrasts are invariant. | Removes artificial labels and cross-arm pairing. |
| A15 | Tuning, propensity estimation, and reference construction respect sample splitting. | Preprocessing, grids learned from data, overlap trimming, nuisance fits, early stopping, and hyperparameter selection use only the allowed training fold. The evaluation manifest and test outcomes remain sealed. | Prevents leakage and optimistic comparisons. |
| A16 | Inner-sampling uncertainty is absent. | The Version 1 likelihood and loss condition on the oracle \(Y_i\) or validated \(q(Y_i)\). No uncertainty interval may be described as covering latent inner distributions. | Delimits the MVP claim. |
| A17 | Fixed-\(M\) output targets the score projection, with approximation error tracked across \(M\). | The class is \(\mathcal P_M^{\mathrm{emp}}(\mathcal Q_K)=\{M^{-1}\sum_{m=1}^M\delta_{p_m}:p_m\in\mathcal Q_K\}\), allowing repeated particles. The target is \(P_{a,M,\varepsilon}^{K,\star}(x)\in\arg\min_{P\in\mathcal P_M^{\mathrm{emp}}}E\{S_\varepsilon(P,q(Y))\mid A=a,X=x\}\). The score manifest records \(M\) and \(\varepsilon\). | Prevents exact-law claims at nonrepresentable fixed \(M\). |

## Derived identification conditions

A1 through A4 are interpreted on a standard Borel observation space so the
regular conditional laws used in the g-formula exist. Positivity and
conditional exchangeability identify targets only on the target covariate
support. Neither condition licenses extrapolation.

The conditional law of \(q(Y^a)\) is identified for each arm, as are
expectations of integrable functions of one potential outcome. The joint law
of \((Y^0,Y^1)\) is not identified under this ledger. A shared learner may
borrow statistical strength across arms, but it does not change that fact.

## Moment hierarchy

The following distinctions are mandatory:

| Object | Minimum moment used in Phase 0 |
|---|---|
| finite-grid energy score | \(E\lVert q(Y^a)\rVert_W<\infty\) |
| continuum reference effect | \(E W_2(Y^a,\nu_\star)<\infty\) |
| Banach-valued outcome functional | \(E\lVert T_j(Y^a)\rVert_{\mathcal B}<\infty\) |
| conditional continuum barycenter | \(E\{W_2^2(Y^a,\nu_0)\mid X\}<\infty\) almost surely |

Membership of each realized \(Y^a\) in \(\mathcal P_2(\mathbb R)\) does not by
itself imply any of the outer expectations in this table.

## Frozen gate rule

An analysis passes this ledger only when every reported target names its
applicable assumptions. In particular, C-WDB full-law claims cite A1 through
A8 and A12 through A17; continuum barycenter claims additionally invoke the
outer clause of A5; outcome-level functional claims invoke A9; reference
effects invoke A10; and confirmatory TCATE claims invoke A11.

Failure of A13, A14, or A17 stops C-WDB. Failure of the added outer moment
clause in A5 removes the barycenter target but does not invalidate finite-grid
law targets.
