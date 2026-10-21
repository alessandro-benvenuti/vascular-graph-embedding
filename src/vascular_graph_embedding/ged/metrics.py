from __future__ import annotations

from collections import Counter

import numpy as np

from .model import HardGraph, PredictionRecord


def adjacency(graph: HardGraph) -> np.ndarray:
    value = np.zeros((graph.node_count, graph.node_count), dtype=bool)
    if len(graph.edges):
        value[graph.edges[:, 0], graph.edges[:, 1]] = True
        value[graph.edges[:, 1], graph.edges[:, 0]] = True
    return value


def component_labels(graph: HardGraph) -> np.ndarray:
    matrix = adjacency(graph)
    labels = np.full(graph.node_count, -1, dtype=np.int64)
    component = 0
    for start in range(graph.node_count):
        if labels[start] >= 0:
            continue
        labels[start] = component
        stack = [start]
        while stack:
            for neighbour in np.flatnonzero(matrix[stack.pop()]):
                neighbour = int(neighbour)
                if labels[neighbour] < 0:
                    labels[neighbour] = component; stack.append(neighbour)
        component += 1
    return labels


def graph_summary(graph: HardGraph) -> dict[str, float | int]:
    matrix = adjacency(graph)
    degrees = matrix.sum(axis=1) if graph.node_count else np.empty(0, dtype=int)
    beta0 = len(np.unique(component_labels(graph))) if graph.node_count else 0
    beta1 = graph.edge_count - graph.node_count + beta0
    largest = 0
    if graph.node_count:
        largest = max(Counter(component_labels(graph).tolist()).values())
    result = {
        "nodes": graph.node_count, "edges": graph.edge_count,
        "beta0": beta0, "beta1": beta1,
        "endpoints": int(np.count_nonzero(degrees == 1)),
        "branchpoints": int(np.count_nonzero(degrees >= 3)),
        "largest_component_fraction": float(largest / max(1, graph.node_count)),
    }
    return result


def bridge_edges(graph: HardGraph) -> set[tuple[int, int]]:
    matrix = adjacency(graph)
    discovery = np.full(graph.node_count, -1, dtype=int)
    low = np.full(graph.node_count, -1, dtype=int)
    parent = np.full(graph.node_count, -1, dtype=int)
    result: set[tuple[int, int]] = set(); clock = 0
    def visit(node: int) -> None:
        nonlocal clock
        discovery[node] = low[node] = clock; clock += 1
        for raw in np.flatnonzero(matrix[node]):
            neighbour = int(raw)
            if discovery[neighbour] < 0:
                parent[neighbour] = node; visit(neighbour)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    result.add(tuple(sorted((node, neighbour))))
            elif neighbour != parent[node]:
                low[node] = min(low[node], discovery[neighbour])
    for node in range(graph.node_count):
        if discovery[node] < 0: visit(node)
    return result


def structural_metrics(predicted: HardGraph, truth: HardGraph, record: PredictionRecord) -> dict[str, float | int]:
    pred = graph_summary(predicted); gt = graph_summary(truth)
    pred_adj = adjacency(predicted); gt_adj = adjacency(truth)
    pred_degree = pred_adj.sum(1).astype(int); gt_degree = gt_adj.sum(1).astype(int)
    max_degree = max([0, *pred_degree.tolist(), *gt_degree.tolist()])
    pred_hist = np.bincount(pred_degree, minlength=max_degree + 1)
    gt_hist = np.bincount(gt_degree, minlength=max_degree + 1)

    query_to_gt = dict(zip(record.matched_queries.tolist(), record.matched_gt.tolist()))
    local_to_gt = {local: query_to_gt[int(slot)] for local, slot in enumerate(predicted.source_slots) if int(slot) in query_to_gt}
    mapped_gt_to_local = {gt_id: local for local, gt_id in local_to_gt.items()}
    disagreements = 0; missing_matched = 0
    for left in range(truth.node_count):
        for right in range(left + 1, truth.node_count):
            if left not in mapped_gt_to_local or right not in mapped_gt_to_local:
                continue
            value = bool(pred_adj[mapped_gt_to_local[left], mapped_gt_to_local[right]])
            target = bool(gt_adj[left, right])
            disagreements += int(value != target)
            missing_matched += int(target and not value)
    gt_bridges = bridge_edges(truth)
    missing_bridges = 0
    for left, right in gt_bridges:
        if left not in mapped_gt_to_local or right not in mapped_gt_to_local:
            missing_bridges += 1
        elif not pred_adj[mapped_gt_to_local[left], mapped_gt_to_local[right]]:
            missing_bridges += 1
    pred_bridges = bridge_edges(predicted)
    false_cycle_edges = 0
    for left, right in map(tuple, predicted.edges.tolist()):
        target_edge = False
        if left in local_to_gt and right in local_to_gt:
            target_edge = bool(gt_adj[local_to_gt[left], local_to_gt[right]])
        if not target_edge and tuple(sorted((left, right))) not in pred_bridges:
            false_cycle_edges += 1
    result = {
        "predicted_node_count": pred["nodes"], "predicted_edge_count": pred["edges"],
        "gt_node_count": gt["nodes"], "gt_edge_count": gt["edges"],
        "absolute_node_count_error": abs(int(pred["nodes"]) - int(gt["nodes"])),
        "absolute_edge_count_error": abs(int(pred["edges"]) - int(gt["edges"])),
        "predicted_beta0": pred["beta0"], "predicted_beta1": pred["beta1"],
        "gt_beta0": gt["beta0"], "gt_beta1": gt["beta1"],
        "abs_delta_beta0": abs(int(pred["beta0"]) - int(gt["beta0"])),
        "abs_delta_beta1": abs(int(pred["beta1"]) - int(gt["beta1"])),
        "topology_l1": abs(int(pred["beta0"]) - int(gt["beta0"])) + abs(int(pred["beta1"]) - int(gt["beta1"])),
        "endpoint_count_error": abs(int(pred["endpoints"]) - int(gt["endpoints"])),
        "branchpoint_count_error": abs(int(pred["branchpoints"]) - int(gt["branchpoints"])),
        "degree_histogram_l1_error": int(np.abs(pred_hist - gt_hist).sum()),
        "largest_component_fraction": pred["largest_component_fraction"],
        "matched_gt_connectivity_disagreement": disagreements,
        "matched_gt_branches_disconnected": missing_matched,
        "missing_bridge_edges": missing_bridges,
        "false_cycle_closing_edges": false_cycle_edges,
        "fragmentation": max(0, int(pred["beta0"]) - int(gt["beta0"])),
    }
    result["combined_structural_score"] = (
        result["topology_l1"] + result["branchpoint_count_error"]
        + result["matched_gt_connectivity_disagreement"]
        + result["false_cycle_closing_edges"]
    )
    return result
