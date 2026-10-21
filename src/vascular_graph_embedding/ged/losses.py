from __future__ import annotations

import math
from typing import Any

import numpy as np

from .model import PredictionRecord


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = values.max(axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def classification_loss(logits: np.ndarray, labels: np.ndarray, configuration: dict[str, Any]) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    log_prob = _log_softmax(logits)
    target_log = log_prob[np.arange(len(labels)), labels]
    weights = np.asarray(configuration["class_weights"], dtype=np.float64)
    selected_weights = weights[labels]
    if configuration["name"] == "focal":
        gamma = float(configuration.get("focal_gamma", 2.0))
        probability = np.exp(target_log)
        losses = -np.maximum(1.0 - probability, np.finfo(np.float64).eps) ** gamma * target_log * selected_weights
    else:
        losses = -target_log * selected_weights
    return float(losses.sum() / max(selected_weights.sum(), np.finfo(np.float64).eps))


def ratio_upsample(logits: np.ndarray, labels: np.ndarray, ratio: float, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    positives = logits[labels == 1]; negatives = logits[labels == 0]
    if not len(positives) or not len(negatives): return logits, labels
    actual = len(positives) / float(len(negatives))
    if actual < ratio - tolerance:
        count = int(len(negatives) * ratio - len(positives)); source = positives; label = 1
    elif ratio + tolerance < actual:
        count = int(len(positives) / ratio - len(negatives)); source = negatives; label = 0
    else: return logits, labels
    if count <= 0: return logits, labels
    indices = np.arange(count) % len(source)
    return np.concatenate((logits, source[indices])), np.concatenate((labels, np.full(count, label, dtype=np.int64)))


def reconstruct_losses(record: PredictionRecord, resolved_config: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct exact saved-tensor terms and explicitly named proxies.

    Node classification, cardinality, and matched-coordinate L1 are exact for
    the saved assignment.  Box logits were not exported.  Training edge loss
    used randomly oriented relation logits; the export permits a deterministic
    symmetric-logit proxy but cannot recreate that RNG choice exactly.
    """
    loss_config = resolved_config.get("loss", {})
    labels = np.zeros(len(record.node_logits), dtype=np.int64)
    labels[record.matched_queries] = 1
    node_configuration = loss_config.get("node", {}).get("classification", {
        "name": "weighted_cross_entropy", "class_weights": [0.25, 0.75]
    })
    node_loss = classification_loss(record.node_logits, labels, node_configuration)
    cardinality = abs(int(np.argmax(record.node_logits, axis=-1).sum()) - len(record.gt_nodes)) / len(record.node_logits)
    target_for_query = np.full(len(record.node_logits), -1, dtype=np.int64)
    target_for_query[record.matched_queries] = record.matched_gt
    coordinate_loss = float(np.abs(
        record.coordinates[record.matched_queries] - record.gt_nodes[record.matched_gt]
    ).sum() / max(1, len(record.matched_gt)))

    matched = set(record.matched_queries.tolist())
    query_to_gt = dict(zip(record.matched_queries.tolist(), record.matched_gt.tolist()))
    gt_edges = set(map(tuple, record.gt_edges.tolist()))
    pair_mask = np.asarray([
        int(left) in matched and int(right) in matched for left, right in record.edge_pairs
    ], dtype=bool)
    pairs = record.edge_pairs[pair_mask]
    averaged_logits = 0.5 * (
        record.edge_logits_forward[pair_mask] + record.edge_logits_reverse[pair_mask]
    )
    edge_labels = np.asarray([
        int(tuple(sorted((query_to_gt[int(left)], query_to_gt[int(right)]))) in gt_edges)
        for left, right in pairs
    ], dtype=np.int64)
    edge_configuration = loss_config.get("edge", {}).get("classification", {
        "name": "cross_entropy", "class_weights": [1.0, 1.0]
    })
    balancing = loss_config.get("edge", {}).get("balancing", {})
    if balancing.get("mode") == "ratio_upsample":
        averaged_logits, edge_labels = ratio_upsample(
            averaged_logits, edge_labels,
            float(balancing.get("positive_to_negative_ratio", 0.15)),
            float(balancing.get("tolerance", 0.02)),
        )
    edge_proxy = classification_loss(averaged_logits, edge_labels, edge_configuration) if len(edge_labels) else 0.0

    node_cost = float(np.abs(record.node_probabilities - labels).sum())
    coordinate_cost = float((
        record.node_probabilities[record.matched_queries]
        * np.linalg.norm(
            record.coordinates[record.matched_queries] - record.gt_nodes[record.matched_gt], axis=1
        )
    ).sum())
    target_adjacency = np.zeros((len(record.node_logits), len(record.node_logits)), dtype=float)
    reverse = {int(gt): int(query) for query, gt in query_to_gt.items()}
    for left, right in record.gt_edges:
        q_left, q_right = reverse[int(left)], reverse[int(right)]
        target_adjacency[q_left, q_right] = target_adjacency[q_right, q_left] = 1.0
    target_pairs = target_adjacency[record.edge_pairs[:, 0], record.edge_pairs[:, 1]]
    edge_raw_cost = float(np.abs(record.edge_probabilities - target_pairs).sum())
    edge_effective_cost = float(np.abs(record.effective_edge_probabilities - target_pairs).sum())

    weights = loss_config.get("weights", {})
    enabled = loss_config.get("enabled", [])
    reconstructable = {"class": node_loss, "nodes": coordinate_loss, "cardinality": cardinality, "edges": edge_proxy}
    proxy_total = sum(float(weights.get(name, 0.0)) * value for name, value in reconstructable.items() if name in enabled)
    topology = resolved_config.get("topology", {})
    enabled_betti = [
        name for name in ("betti_h0", "betti_h1")
        if topology.get(name, {}).get("enabled", False)
    ]
    exact_total_available = not any(name in enabled for name in ("boxes", "edges")) and not enabled_betti
    missing_terms = []
    if "boxes" in enabled: missing_terms.append("box predictions were not saved")
    if "edges" in enabled: missing_terms.append("training edge loss used a random directed relation realization; only both exported directions remain")
    if enabled_betti: missing_terms.append("exact enabled Betti criterion values were not exported with the raw prediction record")
    return {
        "node_loss": node_loss, "coordinate_loss": coordinate_loss,
        "cardinality_loss": float(cardinality), "edge_loss_proxy": edge_proxy,
        "betti_h0_loss": None, "betti_h1_loss": None,
        "relationformer_total_loss": proxy_total if exact_total_available else None,
        "audit_composite_proxy": proxy_total,
        "node_cost": node_cost, "coordinate_cost": coordinate_cost,
        "edge_raw_cost": edge_raw_cost, "edge_effective_cost": edge_effective_cost,
        "loss_reconstruction": "exact" if exact_total_available else "audit_proxy_missing_boxes_and_random_directed_edge_realization",
        "missing_exact_terms": missing_terms,
        "audit_proxy_weights": {name: float(weights.get(name, 0.0)) for name in reconstructable},
    }
