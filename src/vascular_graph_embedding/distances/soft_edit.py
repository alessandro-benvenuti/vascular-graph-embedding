from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vascular_graph_embedding.graphs import Correspondence, SoftGraph
from .base import DistancePlugin, register_distance

DISTANCE_SCHEMA = "vascular-distance-result/v1"


@register_distance
@dataclass
class SoftGeometricEditCost(DistancePlugin):
    """Correspondence-conditioned surrogate; not a metric or exact GED."""

    name = "soft_geometric_edit_v1"
    lambda_node: float = 1.0
    lambda_coordinate: float = 1.0
    lambda_edge: float = 1.0
    normalization: str = "gt"  # raw or gt
    coordinate_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def compute(self, gt: SoftGraph, candidate: SoftGraph, correspondence: Correspondence, **kwargs: Any) -> dict[str, Any]:
        if candidate.slots != len(correspondence.slot_to_gt):
            raise ValueError("candidate and correspondence slot counts differ")
        gt_active = np.flatnonzero(gt.presence > 0.5)
        gt_index = {int(identity): k for k, identity in enumerate(gt_active)}
        y_node = np.zeros(candidate.slots, dtype=float)
        matched = []
        for slot, identity in enumerate(correspondence.slot_to_gt):
            if int(identity) in gt_index:
                y_node[slot] = 1.0
                matched.append((slot, gt_index[int(identity)]))
        node_raw = float(np.abs(candidate.presence - y_node).sum())
        scale = np.asarray(self.coordinate_scale, dtype=float)
        coordinate_raw = float(sum(
            candidate.presence[s] * np.linalg.norm((candidate.coordinates[s] - gt.coordinates[g]) * scale)
            for s, g in matched
        ))
        gt_edge = gt.effective_adjacency
        mapping = correspondence.slot_to_gt
        valid = mapping >= 0
        safe = np.where(valid, mapping, 0)
        target_edges = gt_edge[np.ix_(safe, safe)] * (valid[:, None] & valid[None, :])
        edge_raw = float(np.abs(candidate.effective_adjacency - target_edges)[np.triu_indices(candidate.slots, 1)].sum())
        gt_nodes = int(len(gt_active))
        gt_edges = int(np.count_nonzero(np.triu(gt_edge > 0.5, 1)))
        candidate_pairs = candidate.slots * (candidate.slots - 1) // 2
        node_den = max(gt_nodes, 1) if self.normalization == "gt" else 1
        coordinate_den = max(gt_nodes, 1) if self.normalization == "gt" else 1
        edge_den = max(gt_edges, 1) if self.normalization == "gt" else 1
        node = node_raw / node_den
        coordinate = coordinate_raw / coordinate_den
        edge = edge_raw / edge_den
        total = self.lambda_node * node + self.lambda_coordinate * coordinate + self.lambda_edge * edge
        return {
            "schema": DISTANCE_SCHEMA, "plugin": self.name,
            "normalization": self.normalization,
            "empty_graph_convention": "zero denominators are replaced by one",
            "coordinate_scale": list(self.coordinate_scale),
            "node_raw": node_raw, "coordinate_raw": coordinate_raw, "edge_raw": edge_raw,
            "gt_node_denominator": gt_nodes, "gt_edge_denominator": gt_edges,
            "candidate_pair_denominator_diagnostic": candidate_pairs,
            "node_normalized": node, "coordinate_normalized": coordinate, "edge_normalized": edge,
            "lambda_node": self.lambda_node, "lambda_coordinate": self.lambda_coordinate,
            "lambda_edge": self.lambda_edge, "total_cost": total,
        }

