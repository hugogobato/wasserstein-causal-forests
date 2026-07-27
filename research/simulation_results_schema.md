# Simulation-result schema

This is the WP0 schema definition for the long-format result table. It is documentation, not a simulation implementation.

## Required columns

Every result row must contain the following fields:

| Field | Meaning |
|---|---|
| `claim_id` | Stable claim or numerical-hook identifier, for example `C3.3` or `N4a`. |
| `dgp_id` | Data-generating-process identifier, for example `D0` through `D8`. |
| `observation_regime` | One of `oracle_latent`, `feasible_growing_inner`, `identified_measurement_model`, or `empirical_proxy`; determines which truth comparison is valid. |
| `evaluation_manifest_id` | Immutable identifier for the frozen evaluation points, weights, truth standardizers, and simultaneous-coverage family. |
| `n_regions` | Number of outer region units. |
| `inner_n` | Balanced inner sample size or a design label such as `heterogeneous`. |
| `seed` | Integer random seed for the replicate. |
| `method` | Frozen method or baseline name. |
| `metric` | Target-specific metric name. |
| `value` | Numeric measured value. |

For heterogeneous inner samples, optional columns `inner_n_min` and `inner_n_max` record the range. A result may include additional columns, but the required names above cannot be renamed or omitted.

## Structural WP0 check

Expected required-name set:

```text
claim_id, dgp_id, observation_regime, evaluation_manifest_id, n_regions, inner_n, seed, method, metric, value
```

Measured against this schema definition: all ten names are present exactly once.

Verdict: `PASS` for the WP0 structural check. Empirical validation of actual simulation rows remains pending until WP9 produces output.
