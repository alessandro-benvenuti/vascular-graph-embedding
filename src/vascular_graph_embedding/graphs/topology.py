from __future__ import annotations

import numpy as np


def topology_summary(presence: np.ndarray, effective: np.ndarray, *, threshold: float = 0.5) -> dict[str, int]:
    active = np.flatnonzero(np.asarray(presence) >= threshold)
    active_set = set(active.tolist())
    edges = [(i, j) for i in active for j in active if i < j and effective[i, j] >= threshold]
    adjacency = {int(i): set() for i in active}
    for i, j in edges:
        adjacency[int(i)].add(int(j)); adjacency[int(j)].add(int(i))
    components = 0
    unseen = set(active_set)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour); stack.append(neighbour)
    degrees = [len(adjacency[int(i)]) for i in active]
    return {
        "nodes": len(active), "edges": len(edges), "beta0": components,
        "beta1": len(edges) - len(active) + components,
        "endpoints": sum(d == 1 for d in degrees),
        "bifurcations": sum(d >= 3 for d in degrees),
        "isolated_nodes": sum(d == 0 for d in degrees),
    }

