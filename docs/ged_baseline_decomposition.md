# GED baseline decomposition

This follow-up reuses an existing completed discrete-GED main run. It submits no
Gurobi solves and trains no encoder. Structural and geometric GED are analysed
separately.

The central structural comparison is:

1. `continuous_total_score`: the exact RelationFormer total when reconstructable,
   otherwise the existing `relationformer_loss_proxy` without renaming it as exact;
2. `binary_local_total_*`: node and edge disagreements after thresholding in the
   original 120 query slots under the saved Hungarian assignment;
3. `saved_matching_cost_*`: a valid structural edit cost constrained to that saved
   assignment;
4. `optimized_ged_*`: the previously certified Gurobi optimum.

Raw structural costs use unit node operations and the configured edge weight.
Normalized costs divide by `max(GT nodes + edge_weight * GT edges, 1)`. Under the
current GED accounting, edges incident to inserted or deleted nodes are charged as
edge operations, so compatible fixed-slot binary and saved-matching costs may be
identical. Any observed difference is exported rather than assumed away.

The prespecified combined structural-damage outcome is the mean of within-threshold
percentile ranks of topology L1, branchpoint error, saved-correspondence connectivity
disagreement, false cycle-closing edges, and fragmentation. Ties receive average
ranks. Missing components are omitted row-wise; individual outcomes remain available.

All ranking comparisons pair checkpoints from the same source patch and threshold.
Primary accuracy excludes outcome or score ties, while a separately labelled column
awards score ties half credit. Confidence intervals use deterministic source-patch
bootstrap resampling, keeping all checkpoints and thresholds from a patch clustered.

`duplicate_candidate_count` counts active predicted-node pairs separated by at most
one quarter of the median GT edge length. `subdivision_candidate_count` counts
unmatched active degree-two predicted nodes. These are fixed graph-only diagnostics,
not ground-truth ambiguity annotations.

Run on Jean Zay with:

```bash
MAIN_RUN=/path/to/completed/main/run \
  bash cluster/jean_zay/submit_ged_baseline_decomposition.sh
```

The analysis writes to `$MAIN_RUN/baseline_decomposition`. A completed directory is
never replaced unless `GED_DECOMPOSITION_OVERWRITE=YES` is explicitly set.
