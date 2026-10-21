from __future__ import annotations

from .model import HardGraph, PredictionRecord
from .solver import mapping_cost


def saved_matching_cost(
    predicted: HardGraph, truth: HardGraph, record: PredictionRecord,
    *, variant: str, edge_weight: float,
) -> dict:
    query_to_gt = dict(zip(record.matched_queries.tolist(), record.matched_gt.tolist()))
    mapping = [
        (local, int(query_to_gt[int(slot)]))
        for local, slot in enumerate(predicted.source_slots)
        if int(slot) in query_to_gt
    ]
    cost, operations, metadata = mapping_cost(
        predicted, truth, mapping, variant=variant, edge_weight=edge_weight
    )
    return {"saved_matching_edit_cost": cost, "saved_matching_node_mapping": [list(pair) for pair in mapping], **operations, **metadata}
