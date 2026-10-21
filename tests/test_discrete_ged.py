from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vascular_graph_embedding.ged.baseline import saved_matching_cost
from vascular_graph_embedding.ged.decomposition import (
    fixed_slot_disagreement, graph_size_normalizer, grouped_cv_extended,
    mapping_change_count, multithreshold_aggregate, paired_checkpoint_rows,
    patch_bootstrap, ranking_accuracy, sequential_gains, threshold_stability,
)
from vascular_graph_embedding.ged.extraction import extract_graphs
from vascular_graph_embedding.ged.loader import EXPECTED_PAIRS, build_prediction_index, parse_record, softmax_positive
from vascular_graph_embedding.ged.losses import classification_loss, reconstruct_losses
from vascular_graph_embedding.ged.metrics import graph_summary
from vascular_graph_embedding.ged.model import DistanceResult, HardGraph, PredictionRecord, canonical_edges
from vascular_graph_embedding.ged.selection import select_samples
from vascular_graph_embedding.ged.solver import ExhaustiveGraphEditDistance, GurobiGraphEditDistance, mapping_cost
from vascular_graph_embedding.ged.statistics import grouped_bootstrap_correlation, grouped_linear_cv


def graph(nodes, edges):
    return HardGraph(np.asarray(nodes, dtype=float).reshape(-1, 3), np.asarray(edges, dtype=int).reshape(-1, 2))


def record(node_logits=None, edge_logits=None):
    pairs = np.asarray([(i, j) for i in range(120) for j in range(i + 1, 120)], dtype=np.int64)
    node_logits = np.asarray(node_logits if node_logits is not None else [[1, 0]] * 120, dtype=float)
    if edge_logits is None: edge_logits = np.tile([1.0, 0.0], (EXPECTED_PAIRS, 1))
    raw = {
        "sample_id": "sample", "checkpoint": "epoch_0200", "epoch": 200,
        "nodes": np.zeros((120, 3)), "node_logits": node_logits,
        "gt_nodes": np.asarray([[0, 0, 0]], dtype=float), "gt_edges": np.empty((0, 2), dtype=int),
        "edge_logits_forward": edge_logits, "edge_logits_reverse": edge_logits,
        "matched_queries": np.asarray([0]), "matched_gt": np.asarray([0]), "model_input_shape": (64, 64, 64),
    }
    return parse_record(raw, pairs)


def test_forward_reverse_logits_are_averaged_before_softmax():
    pairs = np.asarray([(i, j) for i in range(120) for j in range(i + 1, 120)], dtype=np.int64)
    forward = np.tile([4.0, 0.0], (EXPECTED_PAIRS, 1)); reverse = np.tile([0.0, 2.0], (EXPECTED_PAIRS, 1))
    raw = {
        "sample_id": "x", "checkpoint": "c", "epoch": 1, "nodes": np.zeros((120, 3)),
        "node_logits": np.tile([0.0, 1.0], (120, 1)), "gt_nodes": np.empty((0, 3)),
        "gt_edges": np.empty((0, 2)), "edge_logits_forward": forward, "edge_logits_reverse": reverse,
        "matched_queries": np.empty(0), "matched_gt": np.empty(0), "model_input_shape": (64, 64, 64),
    }
    parsed = parse_record(raw, pairs)
    assert parsed.edge_probabilities[0] == pytest.approx(softmax_positive(np.asarray([[2.0, 1.0]]))[0])


def test_hard_extraction_includes_threshold_boundary_and_uses_raw_edge_probability():
    logits = np.tile([1.0, 0.0], (120, 1)); logits[:2] = [0.0, 0.0]
    edges = np.tile([1.0, 0.0], (EXPECTED_PAIRS, 1)); edges[0] = [0.0, 0.0]
    parsed = record(logits, edges); predicted, _ = extract_graphs(parsed, .5, .5)
    assert predicted.source_slots.tolist() == [0, 1]
    assert predicted.edges.tolist() == [[0, 1]]
    assert predicted.effective_edge_probabilities[0] == pytest.approx(.125)


def test_reconstructed_node_and_coordinate_terms_on_synthetic_tensors():
    parsed = record()
    config = {"loss": {
        "enabled": ["class", "nodes", "cardinality", "edges"],
        "weights": {"class": 4.0, "nodes": 2.0, "cardinality": .8, "edges": 6.0},
        "node": {"classification": {"name": "weighted_cross_entropy", "class_weights": [.25, .75]}},
        "edge": {
            "classification": {"name": "cross_entropy", "class_weights": [1.0, 1.0]},
            "balancing": {"mode": "none"},
        },
    }}
    result = reconstruct_losses(parsed, config)
    labels = np.zeros(120, dtype=int); labels[0] = 1
    expected = classification_loss(parsed.node_logits, labels, config["loss"]["node"]["classification"])
    assert result["node_loss"] == pytest.approx(expected)
    assert result["coordinate_loss"] == pytest.approx(0)
    assert result["relationformer_total_loss"] is None
    assert "random directed relation" in " ".join(result["missing_exact_terms"])


def test_edge_canonicalization_deduplicates_and_rejects_loops():
    assert canonical_edges([[2, 1], [1, 2]], 3).tolist() == [[1, 2]]
    with pytest.raises(ValueError): canonical_edges([[1, 1]], 2)


def test_beta0_beta1_path_and_triangle():
    path = graph([[0,0,0],[1,0,0],[2,0,0]], [[0,1],[1,2]])
    triangle = graph([[0,0,0],[1,0,0],[0,1,0]], [[0,1],[1,2],[0,2]])
    assert graph_summary(path)["beta0"] == 1 and graph_summary(path)["beta1"] == 0
    assert graph_summary(triangle)["beta1"] == 1


@pytest.mark.parametrize("variant", ["structural", "geometric"])
def test_identical_graph_has_zero_cost_and_is_permutation_invariant(variant):
    original = graph([[0,0,0],[1,0,0],[2,0,0]], [[0,1],[1,2]])
    permuted = graph([[2,0,0],[0,0,0],[1,0,0]], [[1,2],[2,0]])
    solver = ExhaustiveGraphEditDistance(variant=variant)
    assert solver.compute(original, original).distance == pytest.approx(0)
    assert solver.compute(permuted, original).distance == pytest.approx(0)


def test_single_node_insertion_and_deletion():
    empty = graph(np.empty((0,3)), np.empty((0,2)))
    one = graph([[0,0,0]], np.empty((0,2)))
    solver = ExhaustiveGraphEditDistance()
    assert solver.compute(empty, one).distance == pytest.approx(1)
    assert solver.compute(one, empty).distance == pytest.approx(1)


def test_single_edge_insertion_deletion_and_path_triangle():
    nodes = [[0,0,0],[1,0,0],[0,1,0]]
    path = graph(nodes, [[0,1],[1,2]]); triangle = graph(nodes, [[0,1],[1,2],[0,2]])
    result = ExhaustiveGraphEditDistance().compute(path, triangle)
    assert result.distance == pytest.approx(1)
    assert result.edge_insertions + result.edge_deletions == 1


def test_pure_edge_subdivision_cost():
    truth = graph([[0,0,0],[2,0,0]], [[0,1]])
    subdivision = graph([[0,0,0],[1,0,0],[2,0,0]], [[0,1],[1,2]])
    assert ExhaustiveGraphEditDistance().compute(subdivision, truth).distance == pytest.approx(2)


def test_coordinate_substitution_is_scaled_and_capped():
    truth = graph([[0,0,0]], np.empty((0,2))); far = graph([[100,0,0]], np.empty((0,2)))
    cost, _, metadata = mapping_cost(far, truth, [(0,0)], variant="geometric", edge_weight=1)
    assert cost == pytest.approx(2); assert metadata["coordinate_substitution_cost"] == pytest.approx(2)


def test_saved_matching_is_upper_bound_for_exhaustive_optimum():
    predicted = HardGraph([[0,0,0],[1,0,0]], [[0,1]], source_slots=[0,1])
    truth = graph([[0,0,0],[1,0,0]], [[0,1]])
    parsed = record(); parsed = PredictionRecord(**{**parsed.__dict__, "matched_queries": np.asarray([0]), "matched_gt": np.asarray([0])})
    baseline = saved_matching_cost(predicted, truth, parsed, variant="structural", edge_weight=1)
    optimum = ExhaustiveGraphEditDistance().compute(predicted, truth)
    assert baseline["saved_matching_edit_cost"] >= optimum.distance


@pytest.mark.parametrize("variant", ["structural", "geometric"])
def test_gurobi_matches_independent_exhaustive_enumeration_on_tiny_graphs(variant):
    pytest.importorskip("gurobipy")
    predicted = graph([[0,0,0],[1,0,0],[0,1,0]], [[0,1],[1,2]])
    truth = graph([[0,0,0],[0,1,0],[1,0,0]], [[0,1],[1,2],[0,2]])
    exhaustive = ExhaustiveGraphEditDistance(variant=variant).compute(predicted, truth)
    optimized = GurobiGraphEditDistance(variant=variant, timeout=30).compute(predicted, truth)
    assert optimized.status == "optimal"
    assert optimized.distance == pytest.approx(exhaustive.distance)


def test_timeout_status_serializes_without_claiming_optimality():
    result = DistanceResult(None, "timeout", 1.0, 2.0, 3.0, 1/3, 1, 0, 1, 0, 1, [], {"optimality_certified": False})
    restored = json.loads(json.dumps(result.as_dict()))
    assert restored["status"] == "timeout" and restored["distance"] is None
    assert restored["upper_bound"] == 3.0 and not restored["solver_metadata"]["optimality_certified"]


def test_deterministic_stratified_selection(tmp_path: Path):
    root = tmp_path
    for epoch in (200, 500):
        folder = root / f"epoch_{epoch:04d}"; folder.mkdir()
        pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(20)], "gt_nodes": np.arange(20)+2,
            "gt_edges": np.arange(20)+1, "node_cost": np.arange(20)+(500-epoch)/100,
            "coordinate_cost": np.arange(20)/10, "edge_effective_cost": np.arange(20)/5,
            "gt_components": 1, "gt_cycle_rank": 0, "t0.5_components": [1,2]*10,
            "t0.5_cycle_rank": [0,1]*10, "t0.5_false_loop_edges": [0,1]*10,
        }).to_csv(folder / "patches.csv", index=False)
    first = select_samples(root, [200,500], 8, 7)
    second = select_samples(root, [200,500], 8, 7)
    assert first.sample_id.tolist() == second.sample_id.tolist(); assert len(first) == 8


def test_streaming_shard_index_and_cross_epoch_membership(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pairs = torch.combinations(torch.arange(120), r=2)
    torch.save(pairs, tmp_path / "edge_pairs.pt")
    raw = {
        "sample_id": "streamed", "checkpoint": "checkpoint", "epoch": 200,
        "nodes": torch.zeros((120, 3)), "node_logits": torch.tensor([[1.0, 0.0]] * 120),
        "gt_nodes": torch.tensor([[0.0, 0.0, 0.0]]), "gt_edges": torch.empty((0, 2), dtype=torch.long),
        "edge_logits_forward": torch.tensor([[1.0, 0.0]] * EXPECTED_PAIRS),
        "edge_logits_reverse": torch.tensor([[1.0, 0.0]] * EXPECTED_PAIRS),
        "matched_queries": torch.tensor([0]), "matched_gt": torch.tensor([0]),
        "model_input_shape": (64, 64, 64),
    }
    for epoch in (200, 300):
        folder = tmp_path / f"epoch_{epoch:04d}" / "raw"; folder.mkdir(parents=True)
        value = dict(raw); value["epoch"] = epoch; value["checkpoint"] = f"epoch_{epoch:04d}"
        torch.save([value], folder / "shard_00000.pt")
    output = tmp_path / "index.csv"
    rows = build_prediction_index(tmp_path, [200, 300], output)
    assert len(rows) == 2 and output.is_file()


def test_grouped_bootstrap_and_cross_validation_keep_groups():
    frame = pd.DataFrame({
        "sample_id": np.repeat([f"s{i}" for i in range(10)], 3),
        "x": np.tile([0.,1.,2.], 10), "z": np.tile([2.,1.,0.], 10),
    }); frame["y"] = 2 * frame.x + .5 * frame.z
    interval = grouped_bootstrap_correlation(frame, "x", "y", "sample_id", draws=20, seed=3)
    metrics, predictions = grouped_linear_cv(frame, "y", ["x", "z"], "sample_id", folds=5, seed=3)
    assert interval["groups"] == 10 and interval["draws"] == 20
    assert metrics["r2"] == pytest.approx(1) and len(predictions) == len(frame)


def two_node_slot_record(*, second_active=True, edge_active=True):
    parsed = record()
    node_probabilities = np.zeros(120); node_probabilities[0] = .9
    node_probabilities[1] = .9 if second_active else .1
    edge_probabilities = np.zeros(EXPECTED_PAIRS); edge_probabilities[0] = .9 if edge_active else .1
    return replace(
        parsed,
        node_probabilities=node_probabilities,
        edge_probabilities=edge_probabilities,
        gt_nodes=np.asarray([[0., 0., 0.], [1., 0., 0.]]),
        gt_edges=np.asarray([[0, 1]], dtype=int),
        matched_queries=np.asarray([0, 1]), matched_gt=np.asarray([0, 1]),
    )


def test_fixed_slot_binary_node_edge_disagreement_and_normalization():
    compatible = fixed_slot_disagreement(two_node_slot_record(), .5, .5)
    assert compatible["binary_node_disagreement"] == 0
    assert compatible["binary_edge_disagreement"] == 0
    inactive = fixed_slot_disagreement(two_node_slot_record(second_active=False), .5, .5)
    assert inactive["binary_node_disagreement"] == 1
    assert inactive["binary_edge_disagreement"] == 1
    assert inactive["binary_local_total_raw"] == 2
    assert inactive["binary_local_total_normalized"] == pytest.approx(2 / 3)
    assert graph_size_normalizer(2, 1) == 3


def test_binary_and_valid_edit_cost_agree_when_incident_edges_are_charged():
    parsed = two_node_slot_record(second_active=False)
    binary = fixed_slot_disagreement(parsed, .5, .5)
    predicted = HardGraph([[0, 0, 0]], np.empty((0, 2)), source_slots=[0])
    truth = graph([[0, 0, 0], [1, 0, 0]], [[0, 1]])
    saved = saved_matching_cost(predicted, truth, parsed, variant="structural", edge_weight=1)
    assert binary["binary_local_total_raw"] == saved["saved_matching_edit_cost"] == 2


def test_intentionally_bad_saved_mapping_is_improved_by_graph_structure():
    predicted = HardGraph([[0,0,0],[1,0,0],[2,0,0]], [[0,1]], source_slots=[0,1,2])
    truth = graph([[0,0,0],[1,0,0],[2,0,0]], [[0,2]])
    parsed = replace(record(), gt_nodes=truth.coordinates, gt_edges=truth.edges,
                     matched_queries=np.asarray([0,1,2]), matched_gt=np.asarray([0,1,2]))
    saved = saved_matching_cost(predicted, truth, parsed, variant="structural", edge_weight=1)
    optimized = ExhaustiveGraphEditDistance().compute(predicted, truth)
    improvement = saved["saved_matching_edit_cost"] - optimized.distance
    assert improvement >= 0
    assert improvement == pytest.approx(2)
    assert mapping_change_count(saved["saved_matching_node_mapping"], optimized.node_mapping) > 0


def ranking_fixture():
    rows = []
    for sample in ("a", "b", "c"):
        for epoch, value in zip((200, 300, 400, 500), (4., 3., 2., 1.)):
            rows.append({"sample_id": sample, "epoch": epoch, "damage": value,
                         "continuous": value if sample != "c" else 1., "binary": value,
                         "saved": value, "optimized": value})
    return pd.DataFrame(rows)


def test_paired_rankings_ties_sequential_gains_and_patch_bootstrap_are_deterministic():
    frame = ranking_fixture(); scores = ["continuous", "binary", "saved", "optimized"]
    pairs = paired_checkpoint_rows(frame, scores, ["damage"], threshold_label=.5)
    assert len(pairs) == 3 * 6 * 4
    accuracy = ranking_accuracy(pairs); continuous = accuracy[accuracy.score == "continuous"].iloc[0]
    assert continuous.score_ties == 6 and continuous.determinate_pairs == 12
    gains = sequential_gains(accuracy, scores)
    assert gains.iloc[0].thresholding_gain == pytest.approx(0)
    first = patch_bootstrap(pairs, scores, draws=20, seed=11)
    second = patch_bootstrap(pairs, scores, draws=20, seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_grouped_cv_cross_threshold_stability_and_multithreshold_aggregation():
    rows = []
    for sample_id in [f"s{i}" for i in range(8)]:
        offset = int(sample_id[1:])
        for epoch in (200, 300, 400, 500):
            for threshold in (.3, .5, .7):
                x = offset + epoch / 1000 + threshold
                rows.append({"sample_id": sample_id, "epoch": epoch, "node_threshold": threshold,
                             "x": x, "y": 2*x, "score": x, "damage": 3*x})
    frame = pd.DataFrame(rows)
    metrics, predictions = grouped_cv_extended(frame, "y", ["x"], "sample_id", folds=4, seed=2)
    assert metrics["r2"] == pytest.approx(1) and metrics["spearman"] == pytest.approx(1)
    assert len(predictions) == len(frame)
    stability = threshold_stability(frame, ["score"])
    np.testing.assert_allclose(
        stability.loc[stability.record_type == "rank_correlation", "spearman"].to_numpy(float), 1
    )
    assert stability.loc[stability.record_type == "checkpoint_flip", "flip_rate"].iloc[0] == 0
    multi = multithreshold_aggregate(frame, ["score"], ["damage"])
    assert len(multi) == 8 * 4 and (multi.threshold == "multithreshold").all()
