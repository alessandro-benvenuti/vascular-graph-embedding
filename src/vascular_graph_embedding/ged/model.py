from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def canonical_edges(edges: np.ndarray, node_count: int) -> np.ndarray:
    values = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(values):
        return np.empty((0, 2), dtype=np.int64)
    if np.any(values < 0) or np.any(values >= int(node_count)):
        raise ValueError("edge endpoint is outside the graph")
    values = np.sort(values, axis=1)
    if np.any(values[:, 0] == values[:, 1]):
        raise ValueError("self-loops are not valid in a simple graph")
    return np.unique(values, axis=0)


@dataclass(frozen=True)
class HardGraph:
    coordinates: np.ndarray
    edges: np.ndarray
    source_slots: np.ndarray | None = None
    edge_probabilities: np.ndarray | None = None
    effective_edge_probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape [nodes, 3]")
        if not np.isfinite(coordinates).all():
            raise ValueError("graph coordinates must be finite")
        edges = canonical_edges(self.edges, len(coordinates))
        slots = (
            np.arange(len(coordinates), dtype=np.int64)
            if self.source_slots is None
            else np.asarray(self.source_slots, dtype=np.int64)
        )
        if slots.shape != (len(coordinates),) or len(np.unique(slots)) != len(slots):
            raise ValueError("source slots must be unique and match node count")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "source_slots", slots)
        for name in ("edge_probabilities", "effective_edge_probabilities"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=np.float64)
                if value.shape != (len(edges),) or not np.isfinite(value).all():
                    raise ValueError(f"{name} must contain one finite value per edge")
                object.__setattr__(self, name, value)

    @property
    def node_count(self) -> int:
        return len(self.coordinates)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    checkpoint: str
    epoch: int
    coordinates: np.ndarray
    node_logits: np.ndarray
    node_probabilities: np.ndarray
    gt_nodes: np.ndarray
    gt_edges: np.ndarray
    edge_pairs: np.ndarray
    edge_logits_forward: np.ndarray
    edge_logits_reverse: np.ndarray
    edge_probabilities: np.ndarray
    effective_edge_probabilities: np.ndarray
    matched_queries: np.ndarray
    matched_gt: np.ndarray
    model_input_shape: tuple[int, int, int]


@dataclass
class DistanceResult:
    distance: float | None
    status: str
    runtime_seconds: float
    lower_bound: float | None
    upper_bound: float | None
    relative_gap: float | None
    node_insertions: int | None
    node_deletions: int | None
    node_substitutions: int | None
    edge_insertions: int | None
    edge_deletions: int | None
    node_mapping: list[list[int]] = field(default_factory=list)
    solver_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
