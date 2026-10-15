from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

GRAPH_SCHEMA = "vascular-soft-graph/v1"
CORRESPONDENCE_SCHEMA = "vascular-correspondence/v1"


@dataclass
class Correspondence:
    """Map each slot to a GT identity; -1 denotes an inserted/background slot."""

    slot_to_gt: np.ndarray
    schema: str = CORRESPONDENCE_SCHEMA

    def __post_init__(self) -> None:
        self.slot_to_gt = np.asarray(self.slot_to_gt, dtype=np.int64)
        if self.slot_to_gt.ndim != 1:
            raise ValueError("slot_to_gt must be one-dimensional")
        real = self.slot_to_gt[self.slot_to_gt >= 0]
        if len(np.unique(real)) != len(real):
            raise ValueError("a GT identity may be matched to at most one slot")

    def permuted(self, order: np.ndarray) -> "Correspondence":
        return Correspondence(self.slot_to_gt[np.asarray(order, dtype=int)])


@dataclass
class SoftGraph:
    """Fixed-slot soft graph with explicit stable identities and provenance."""

    coordinates: np.ndarray
    presence: np.ndarray
    conditional_edges: np.ndarray
    node_ids: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = GRAPH_SCHEMA

    def __post_init__(self) -> None:
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        self.presence = np.asarray(self.presence, dtype=float)
        self.conditional_edges = np.asarray(self.conditional_edges, dtype=float)
        n = len(self.presence)
        if self.coordinates.shape != (n, 3):
            raise ValueError("coordinates must have shape [slots, 3]")
        if self.conditional_edges.shape != (n, n):
            raise ValueError("conditional_edges must have shape [slots, slots]")
        if len(self.node_ids) != n or len(set(self.node_ids)) != n:
            raise ValueError("node_ids must be unique and match slot count")
        if not np.allclose(self.conditional_edges, self.conditional_edges.T):
            raise ValueError("conditional_edges must be symmetric")
        if not np.allclose(np.diag(self.conditional_edges), 0):
            raise ValueError("self-loops are forbidden")
        if np.any((self.presence < 0) | (self.presence > 1)):
            raise ValueError("presence probabilities must lie in [0, 1]")
        if np.any((self.conditional_edges < 0) | (self.conditional_edges > 1)):
            raise ValueError("conditional edge probabilities must lie in [0, 1]")

    @property
    def slots(self) -> int:
        return len(self.presence)

    @property
    def effective_adjacency(self) -> np.ndarray:
        result = self.presence[:, None] * self.presence[None, :] * self.conditional_edges
        np.fill_diagonal(result, 0.0)
        return result

    def copy(self) -> "SoftGraph":
        return SoftGraph(
            self.coordinates.copy(), self.presence.copy(), self.conditional_edges.copy(),
            list(self.node_ids), dict(self.provenance), self.schema,
        )

    def permuted(self, order: np.ndarray) -> "SoftGraph":
        order = np.asarray(order, dtype=int)
        return SoftGraph(
            self.coordinates[order], self.presence[order],
            self.conditional_edges[np.ix_(order, order)],
            [self.node_ids[i] for i in order], dict(self.provenance), self.schema,
        )

    def compact(self, threshold: float = 0.0) -> tuple["SoftGraph", np.ndarray]:
        keep = np.flatnonzero(self.presence > threshold)
        return self.permuted(keep), keep

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "coordinates": self.coordinates,
            "presence": self.presence,
            "conditional_edges": self.conditional_edges,
            "node_ids": np.asarray(self.node_ids),
            "schema": np.asarray(self.schema),
        }

    @classmethod
    def from_compact(
        cls, compact: "SoftGraph", original_slots: np.ndarray, total_slots: int,
        *, background_presence: float = 0.0,
    ) -> "SoftGraph":
        original_slots = np.asarray(original_slots, dtype=int)
        if len(original_slots) != compact.slots or np.any(original_slots >= total_slots):
            raise ValueError("invalid compact-to-fixed slot mapping")
        x = np.zeros((total_slots, 3), dtype=float)
        p = np.full(total_slots, background_presence, dtype=float)
        q = np.zeros((total_slots, total_slots), dtype=float)
        ids = [f"background:{i}" for i in range(total_slots)]
        x[original_slots] = compact.coordinates
        p[original_slots] = compact.presence
        q[np.ix_(original_slots, original_slots)] = compact.conditional_edges
        for new, old in enumerate(original_slots):
            ids[int(old)] = compact.node_ids[new]
        return cls(x, p, q, ids, dict(compact.provenance))


def clean_fixed_graph(coordinates: np.ndarray, edges: np.ndarray, slots: int = 120) -> tuple[SoftGraph, Correspondence]:
    coordinates = np.asarray(coordinates, dtype=float)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    n = len(coordinates)
    if n > slots:
        raise ValueError(f"GT graph has {n} nodes, exceeding {slots}-slot capacity")
    x = np.zeros((slots, 3), dtype=float)
    p = np.zeros(slots, dtype=float)
    q = np.zeros((slots, slots), dtype=float)
    x[:n], p[:n] = coordinates, 1.0
    for u, v in edges:
        if u == v or min(u, v) < 0 or max(u, v) >= n:
            raise ValueError(f"invalid edge {(u, v)}")
        q[u, v] = q[v, u] = 1.0
    ids = [f"gt:{i}" if i < n else f"background:{i}" for i in range(slots)]
    corr = np.full(slots, -1, dtype=int)
    corr[:n] = np.arange(n)
    return SoftGraph(x, p, q, ids), Correspondence(corr)

