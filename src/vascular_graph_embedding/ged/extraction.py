from __future__ import annotations

import numpy as np

from .model import HardGraph, PredictionRecord


def extract_graphs(
    record: PredictionRecord, node_threshold: float, edge_threshold: float,
) -> tuple[HardGraph, HardGraph]:
    if not 0 <= node_threshold <= 1 or not 0 <= edge_threshold <= 1:
        raise ValueError("graph extraction thresholds must lie in [0,1]")
    active_slots = np.flatnonzero(record.node_probabilities >= float(node_threshold))
    slot_to_local = np.full(len(record.node_probabilities), -1, dtype=np.int64)
    slot_to_local[active_slots] = np.arange(len(active_slots))
    live = (
        (record.edge_probabilities >= float(edge_threshold))
        & (slot_to_local[record.edge_pairs[:, 0]] >= 0)
        & (slot_to_local[record.edge_pairs[:, 1]] >= 0)
    )
    selected_pairs = record.edge_pairs[live]
    predicted_edges = np.column_stack((
        slot_to_local[selected_pairs[:, 0]], slot_to_local[selected_pairs[:, 1]],
    )) if len(selected_pairs) else np.empty((0, 2), dtype=np.int64)
    predicted = HardGraph(
        record.coordinates[active_slots], predicted_edges, active_slots,
        record.edge_probabilities[live], record.effective_edge_probabilities[live],
    )
    truth = HardGraph(
        record.gt_nodes, record.gt_edges, np.arange(len(record.gt_nodes), dtype=np.int64)
    )
    return predicted, truth
