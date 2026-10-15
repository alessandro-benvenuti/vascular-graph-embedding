# Study design and conventions

## Schemas

- `vascular-soft-graph/v1`: coordinates `[S,3]`, presence `[S]`, symmetric conditional adjacency `[S,S]`, stable node IDs.
- `vascular-correspondence/v1`: one `slot_to_gt` integer per candidate slot; `-1` means no GT coordinate target.
- `vascular-edit-event/v1`: family, requested/realized counts, seed, randomized parameters, anchors, ownership, rejection reason.
- `vascular-pair/v1`: immutable pair ID and pair-level provenance.
- `vascular-distance-result/v1`: raw components, explicit denominators, normalized components, weights, coordinate scale, and total.

Empty GT node/edge denominators are replaced by one and raw counts remain exported. The all-7,140-pairs denominator is diagnostic only. Inserted/background nodes have no matched-coordinate penalty. Soft deletion retains incident conditional probabilities by default, making endpoint gating explicit; `zero_conditional` is an alternate declared policy.

## Severity and interpretation

Confidence severity is direction-specific: lower confidence is more severe for retained/missing elements, while higher confidence is more severe for inserted elements. Family intensity levels must not be pooled as a universal scale. Count and intensity correlations are partially built into the objective; cross-family rankings, topology sensitivity, normalization stability, paired changes, and counterexamples are more informative.

Thresholded topology uses graph homology without triangle filling: `beta0` is component count and `beta1 = |E|-|V|+beta0`. Endpoints, bifurcations and isolated nodes are also reported. A generated loop event is accepted only when cycle rank increases. Observed empirical cycle membership must not be interpreted as unique causal loop closure.

## Empirical sampling

The sampler selects complete patch count vectors conditional on checkpoint and approximate GT size, then retrieves related source event rows. It preserves source profile/event IDs. It never claims that independently sampled scalar marginals preserve a spatial joint distribution. Missing displacement directions and full attachment lists require either focused raw-shard extraction or configurable geometric assumptions. Strict subdivision always has exactly two structural attachments. The empirical “split” label remains heuristic.

Capacity/eligibility failures are explicit outcomes. Bounded retries may be configured, but constraints must not be silently relaxed. Requested-versus-realized profiles and fallback use belong in sampling diagnostics.

## Planned controlled composites

Named composites are the only way operations may intentionally share affected elements: missing node + shortcut; missing node + elsewhere missing edge; extra node + attachments; duplicate + altered attachments; split + branch redistribution; subdivision + loop shortcut; displacement + confidence reduction; multiple mild edits versus one severe structural edit. They should be expanded as named generator methods before production use, so ownership remains auditable.

## Solver benchmark

The declared objective contains node assignment, insertions/deletions through dummies, and edge-induced quadratic terms. The included exhaustive backend exactly certifies tiny equal-cardinality cases and checks `d* <= C_known`. A production MIQP backend should report incumbent, lower bound, gap, runtime, status and certification. A timeout incumbent is not an exact GED label. Node-only Hungarian matching, if reported, is an incomplete baseline because it omits quadratic edge coupling.

