# Conventional GED go/no-go study

## Hypothesis

The existing correspondence-conditioned soft edit cost resembles RelationFormer training losses. This staged experiment asks whether a conventional hard-graph GED, with a newly optimized node correspondence and a minimum edit path, contains stable structural information beyond those losses. It uses saved real validation predictions from epochs 200, 300, 400 and 500. Handmade corruptions are not primary evidence, and this milestone does not train an encoder.

## Distances and terminology

The **soft edit cost** in the earlier study uses saved correspondence and probabilities. **Conventional GED** here thresholds a prediction into an undirected simple graph, jointly optimizes a partial one-to-one node mapping, and charges node/edge insertions and deletions. Standard GED is not automatically topology-aware: Betti changes are evaluated separately.

Two variants are stored separately:

- `structural`: unit node edits, configurable edge edits (primary weight 1), zero substitution cost;
- `geometric`: the same edits plus `min(||x-y|| / r, 2)`, where `r` is the patch's median normalized GT edge length. An edgeless GT uses the normalized patch diagonal `sqrt(3)`. The cap equals delete-plus-insert cost.

Predicted nodes satisfy `p >= tau_node`; predicted edges require both endpoints active and raw symmetric relation probability `q >= tau_edge`. Effective probability `p_i p_j q_ij` is diagnostic only. Primary thresholds are `(0.5,0.5)`; pilot/main also evaluate `(0.3,0.3)` and `(0.7,0.7)`.

Node/edge counts, Betti numbers, endpoint/branchpoint counts, degree histograms, and largest-component fraction are graph-invariant. Matched connectivity disagreement, disconnected matched GT branches, missing GT bridges, and false cycle-closing edges explicitly depend on the saved RelationFormer correspondence and are labeled separately in every result.

The Gurobi backend solves a binary quadratic assignment objective. `optimal` is a certified GED. A finite `timeout` result is only a GED incumbent upper bound, accompanied by `lower_bound` and `relative_gap`. The saved RelationFormer matching gives a correspondence-constrained edit-cost upper bound; every certified optimized result is checked against it.

## RelationFormer loss reconstruction

Saved tensors permit exact reconstruction of node classification, matched-coordinate L1, and cardinality for the saved assignment. Exact total training loss is not available because predicted box sizes were not exported and edge training randomly selected relation direction while the export retains both directions. The deterministic symmetric edge value and explicitly weighted composite are labeled audit proxies. The resolved configuration is copied into each run, so exact terms can be added later without changing result identity.

## Stages

1. **Preflight:** validate Gurobi import/license/tiny solve and stream every raw shard into a cross-epoch prediction index.
2. **Smoke:** 2 paired source patches, 4 epochs, primary threshold, both variants, 10-second solve limit.
3. **Pilot:** 12 paired patches, 3 thresholds, both variants, 60-second limit. Review runtime, optimality and gaps.
4. **Main:** 50 paired patches, 3 thresholds, both variants, 300-second limit. `submit_main.sh` refuses to run unless `ALLOW_MAIN=YES`; do not submit before a pilot decision.

Each array task writes one atomic JSON record and skips a valid completed record on rerun. The analysis keeps source patches together for cross-validation and bootstrap resampling.

## Decision

Stop if existing losses predict GED almost completely, optimized and saved-matching costs are nearly identical, residual GED is structurally uninformative, rankings do not improve, threshold conclusions are unstable, or most labels remain high-gap incumbents. Consider GREED-style encoder training only when stable residual information, meaningful correspondence changes, improved structural ranking, threshold robustness and adequate solver label quality all support a go decision.

## Outputs

Completed stages write `report.md`, `summary.json`, `associations.csv`, `bootstrap_intervals.csv`, `ged_results.csv.gz`, `joined_analysis.csv.gz`, `ranking_results.csv`, `disagreement_cases.csv`, `solver_diagnostics.csv`, `sample_manifest.csv`, plots in PNG/PDF, and `COMPLETED.json`.
