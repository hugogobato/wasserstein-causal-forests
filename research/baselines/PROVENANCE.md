# Provenance and Licensing for the WP2-C Forest Baselines

**Phase:** `research_phases/Phase 4 - DRF Baselines Implementation.md`
**Date:** 2026-07-30
**Scope:** every external artefact the W-DRF-T, paper-DRF, and Causal-DRF
baselines depend on.

This file exists because WP1-B (software provenance) was deliberately deferred
in Phase 1 and the C-WDB implementation sidestepped it by being clean-room. The
DRF baselines cannot sidestep it: one of them links a GPL-3.0 package, and the
Causal-DRF rerun uses the authors' supplied implementation and its pinned
causal-clean dependency.

---

## 1. W-DRF-T (WP2-C1): the pinned `drf` package

| Field | Value |
|---|---|
| Package | `drf`, Distributional Random Forests |
| Pinned version | `1.3.1` |
| Source | CRAN, `https://cloud.r-project.org/src/contrib/drf_1.3.1.tar.gz` |
| Upstream repository | `https://github.com/lorismichel/drf` |
| Source tarball SHA256 | `6b3e9bf28d636fa9be012602883e236723813ec7be19b5e5d64a33251455279d` |
| CRAN packaging date | 2026-02-02 |
| License | **GPL-3** |
| Authors | Jeffrey Näf (cre), Loris Michel, Domagoj Ćevid |
| Method reference | Ćevid, Michel, Näf, Meinshausen, Bühlmann (2022), *JMLR* 23(333):1-79, `doi:10.48550/arXiv.2005.14458` |
| Build environment | R 4.3.3, x86_64-pc-linux-gnu, built 2026-07-30 |

Install command used:

```r
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages(c("transport", "drf"))
```

`transport` (0.15-4) is a hard `Imports:` dependency of `drf` and was installed
with it.

### Why CRAN and not the local snapshot

`code/drf-master` is a working copy of the upstream GitHub repository. Section
4.3 of the research plan records its fingerprint (SHA256 of
`core/src/splitting/FourierSplittingRule.cpp` =
`a5da1275153b2b9d1b2ffc52232c1cea3d4d9156224d12abdf51a4558d13c555`) and notes
that it carries no nested Git history, so its commit cannot be reconstructed.
It is additionally unbuildable as it stands: `r-package/drf/src/` uses symbolic
links into `bindings/` and `core/src/`, and in this checkout those links were
flattened into one-line text files holding their target paths. The pinned CRAN
release is a versioned, checksummable artefact and is used instead. The local
snapshot is retained only as the **E3 read-only evidence** behind EV4 (the
public DRF snapshot is ordinary predictive DRF, not Causal-DRF); no file from
it is compiled, copied, or distributed.

### GPL-3.0 isolation boundary

- `drf` is reached **only through its exported R API** (`drf()`, `predict()`),
  from `research/baselines/drf_tlearner.R`.
- No `drf` source, header, or object file is vendored into this repository.
- No project file is linked against `drf.so`.
- `research/baselines/causal_drf_r/` shares **no code** with `drf`; it is an
  independent implementation written from the Causal-DRF paper and is not a
  derivative work of the `drf` package. It does not load the `drf` namespace.
- Consequence: the GPL-3.0 copyleft obligation is confined to whatever a
  distribution of `drf` itself would require. Should the project ever ship a
  combined binary artefact, this boundary must be re-examined before release.

### Labelling constraint

`drf_tlearner.R` fits **two ordinary predictive forests**, one per arm, with no
shared partition and no treatment-aware split criterion. It must be labelled
`W-DRF-T` everywhere. Calling it "Causal-DRF" would misattribute a method that
this construction does not implement. The label is asserted in
`tests/test_drf_tlearner.R`.

The `W-` prefix records the **input geometry only**. Because
`||z - z'||^2 = W_{2,K}^2(q, q')` for `z = diag(sqrt(w)) q`, the Gaussian kernel
DRF already uses is exactly a discretized Wasserstein-RBF kernel. This is a
representation map (F0.5 of the research plan), not a contribution, and the
manuscript must not present it as one.

One deviation from the `drf` defaults is required and is not a tuning choice:
`response.scaling` is forced to `FALSE`. The default standardizes each response
column, which would replace the exact `W_2` geometry of `z` with an arbitrary
coordinatewise rescaling and make the geometry identity above false in the
fitted object.

---

## 2. Paper DRF benchmark: the supplied `drfinference` implementation

Section 11's `DRF` column is not the cheaper W-DRF-T adapter. It reproduces the
separate-forest benchmark used in the Causal-DRF paper's comparison: two
ordinary predictive DRF ensembles are fitted separately by treatment arm, each
with 2500 total trees represented as 50 half-sample groups of 50 trees. The
driver sources the supplied implementation at
`code/drfinference-main/drf-foo.R`, including its `drfCI()` and `predictdrf()`
half-sampling and weight-averaging code. The pinned CRAN `drf` package is used
through that public R API; no package source is copied.

The paper implementation keeps its default `response.scaling = TRUE`, so it is
reported as `DRF` rather than as W-DRF-T. It is a faithful paper benchmark and
not a new model created by this project. The Section 11 rerun is stored under
`results/main/shard_original_drf_*.parquet` and included in the extended main
coordinates through `results/manifests/original_drf_manifest.json`, preserving
the historical 4110-cell gate manifest.

---

## 3. Causal-DRF (WP2-C2): authors' implementation

| Field | Value |
|---|---|
| Paper | *Causal-DRF: Conditional Kernel Treatment Effect Estimation using Distributional Random Forest* |
| Authors | Jeffrey Näf (University of Geneva), Junhyung Park (ETH Zürich), Herbert Susmann (NYU Grossman School of Medicine) |
| Venue | AISTATS 2026, PMLR volume 300 |
| Preprint | `arXiv:2411.08778`, v2 (2 April 2026) |
| Author code | `code/causal_drf_paper-main/` |
| Causal package | `herbps10/drf`, branch `causal-clean` |
| Causal package commit | `0a1a508444176b5b1553f13e832be93a374b0af2` |
| PDF used | `arxiv.org/pdf/2411.08778v2`, SHA256 `4838d2ee173cb52559afff01639841f94c3f4619a90d96b2ca82655f41d41a1a` |
| LaTeX source used | `arxiv.org/e-print/2411.08778v2`, file `aistats_Revision.tex`, SHA256 `f9a26af72b4cf0304bca35586a526865557e90a4b9c163b250f7e92995fc7124` |

The local `code/causal_drf_paper-main/` directory is the authors' simulation
repository. Its setup script calls `drf()` with the treatment indicator and
uses the causal-clean prediction API. The G3 adapter preserves the exact fit
call and exports the package's arm-specific prediction weights, rather than the
paper script's witness-only output, because the common scorer needs the full arm
laws on the shared evaluation bank.

The arXiv PDF carries only the eleven-page main body. The appendices, which
contain the exact splitting criterion and the algorithm pseudocode, were read
from the LaTeX source tarball. Both artefacts are checksummed above; neither is
committed to this repository.

### What was implemented from which location

| Ingredient | Source location in the paper |
|---|---|
| CKTE definition `tau_k(x)` | eq. (1), Section 1 |
| Weighted-MMD split criterion | eq. (2), Section 3; exact form with the `\|I_L\|\|I_R\|/(\|I_L\|+\|I_R\|)^2` factor in Appendix C, eq. (C.1) |
| Random Fourier approximation of the criterion | Appendix C, eq. (C.2); `omega_b ~ N_d(0, sigma^-2 I_d)` |
| Signed prediction weights | eq. (3), Section 3 |
| Subforest grouping and half-sampling | Section 3 (paragraph after eq. 3); Algorithm 1, Appendix C |
| Honesty, `alpha`-regularity, both-arm leaf minimums | Forest assumptions (F1), (F4), (F5), Appendix D.1 |
| Gaussian kernel and median-heuristic bandwidth | Appendix C, final paragraph; Gretton et al. (2012) |
| Resampled test statistic and confidence band | eqs. (4), (5), (6), Section 3 |
| Reproduction target | Appendix B, simulation regimes 1-4 and Table 3 |

The project-specific adapter is
`research/baselines/g3_causal_drf_original_driver.R`. It records the source
path, branch, commit, tree count, and confidence-group size in every output
metadata record. The corresponding merged rerun is
`results/merged_original_causal_drf/main_results.parquet`.

### Licensing and isolation

The causal-clean `drf` package is used through its exported R API and is not
vendored into this repository. The original simulation repository remains
under its upstream licensing terms. The project adapter is separate from the
package source and carries this project's license.

---

## 4. WP2-C3 (Du et al. WRF adapter): still dormant

`research/baselines/wrf.R` is **not** provided. WP2-C3 is dormant pending the
G1 method audit's determination of applicability, and Phase 1 did not close
that determination. Nothing in this directory should be read as a WRF baseline.

---

## 5. Reproducing this environment

```bash
Rscript -e 'packageVersion("drf")'          # 1.3.1
Rscript tests/test_drf_tlearner.R           # WP2-C1 verification
Rscript tests/test_causal_drf.R             # WP2-C2 verification
python research/checks/g3_gate_flags.py --claimant cwdb_r3_cvridge \
  --original-causal-drf --with-repair
```
