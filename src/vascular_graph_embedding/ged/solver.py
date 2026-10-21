from __future__ import annotations

from abc import ABC, abstractmethod
import itertools
import math
import time
from typing import Iterable

import numpy as np

from .model import DistanceResult, HardGraph


def geometric_scale(truth: HardGraph, epsilon: float = 1e-8) -> tuple[float, str]:
    if truth.edge_count:
        lengths = np.linalg.norm(
            truth.coordinates[truth.edges[:, 0]] - truth.coordinates[truth.edges[:, 1]], axis=1
        )
        positive = lengths[lengths > epsilon]
        if len(positive):
            return max(float(np.median(positive)), epsilon), "median_gt_edge_length_normalized"
    return math.sqrt(3.0), "normalized_patch_diagonal_fallback"


def substitution_matrix(predicted: HardGraph, truth: HardGraph, variant: str) -> tuple[np.ndarray, dict]:
    if variant == "structural":
        return np.zeros((predicted.node_count, truth.node_count)), {"coordinate_scale": None}
    if variant != "geometric":
        raise ValueError("GED variant must be structural or geometric")
    scale, convention = geometric_scale(truth)
    if not predicted.node_count or not truth.node_count:
        values = np.empty((predicted.node_count, truth.node_count))
    else:
        values = np.linalg.norm(
            predicted.coordinates[:, None, :] - truth.coordinates[None, :, :], axis=-1
        ) / scale
        values = np.minimum(values, 2.0)
    return values, {"coordinate_scale": scale, "coordinate_scale_convention": convention, "substitution_cap": 2.0}


def mapping_cost(
    predicted: HardGraph, truth: HardGraph, mapping: Iterable[tuple[int, int]],
    *, variant: str, edge_weight: float,
) -> tuple[float, dict[str, int], dict]:
    pairs = [(int(left), int(right)) for left, right in mapping]
    pred_ids = [left for left, _ in pairs]; gt_ids = [right for _, right in pairs]
    if len(set(pred_ids)) != len(pred_ids) or len(set(gt_ids)) != len(gt_ids):
        raise ValueError("node mapping must be one-to-one")
    if any(left < 0 or left >= predicted.node_count for left in pred_ids) or any(right < 0 or right >= truth.node_count for right in gt_ids):
        raise ValueError("node mapping contains an out-of-range index")
    substitutions, metadata = substitution_matrix(predicted, truth, variant)
    lookup = dict(pairs); gt_edges = set(map(tuple, truth.edges.tolist()))
    matched_edges = 0
    for left, right in predicted.edges:
        if int(left) in lookup and int(right) in lookup:
            target = tuple(sorted((lookup[int(left)], lookup[int(right)])))
            matched_edges += int(target in gt_edges)
    operations = {
        "node_insertions": truth.node_count - len(pairs),
        "node_deletions": predicted.node_count - len(pairs),
        "node_substitutions": len(pairs),
        "edge_insertions": truth.edge_count - matched_edges,
        "edge_deletions": predicted.edge_count - matched_edges,
    }
    coordinate = sum(float(substitutions[left, right]) for left, right in pairs)
    cost = (
        operations["node_insertions"] + operations["node_deletions"] + coordinate
        + float(edge_weight) * (operations["edge_insertions"] + operations["edge_deletions"])
    )
    return float(cost), operations, {**metadata, "coordinate_substitution_cost": coordinate, "matched_edges": matched_edges}


class GraphDistance(ABC):
    @abstractmethod
    def compute(self, predicted_graph: HardGraph, gt_graph: HardGraph) -> DistanceResult: ...


class ExhaustiveGraphEditDistance(GraphDistance):
    def __init__(self, variant: str = "structural", edge_weight: float = 1.0, timeout: float = 30.0, max_nodes: int = 8):
        self.variant = variant; self.edge_weight = float(edge_weight)
        self.timeout = float(timeout); self.max_nodes = int(max_nodes)

    def compute(self, predicted_graph: HardGraph, gt_graph: HardGraph) -> DistanceResult:
        if max(predicted_graph.node_count, gt_graph.node_count) > self.max_nodes:
            raise ValueError("exhaustive GED graph exceeds max_nodes")
        start = time.monotonic(); best = math.inf; best_mapping: list[tuple[int, int]] = []
        best_ops = None; best_meta = {}; evaluated = 0; complete = True
        n, m = predicted_graph.node_count, gt_graph.node_count
        for count in range(min(n, m) + 1):
            for pred_subset in itertools.combinations(range(n), count):
                for gt_subset in itertools.combinations(range(m), count):
                    for permutation in itertools.permutations(gt_subset):
                        if time.monotonic() - start > self.timeout:
                            complete = False; break
                        mapping = list(zip(pred_subset, permutation))
                        cost, operations, metadata = mapping_cost(
                            predicted_graph, gt_graph, mapping,
                            variant=self.variant, edge_weight=self.edge_weight,
                        )
                        evaluated += 1
                        if cost < best - 1e-12:
                            best, best_mapping, best_ops, best_meta = cost, mapping, operations, metadata
                    if not complete: break
                if not complete: break
            if not complete: break
        runtime = time.monotonic() - start
        return DistanceResult(
            distance=best if complete and math.isfinite(best) else None,
            status="optimal" if complete else "timeout",
            runtime_seconds=runtime,
            lower_bound=best if complete else 0.0,
            upper_bound=best if math.isfinite(best) else None,
            relative_gap=0.0 if complete else None,
            node_mapping=[list(pair) for pair in best_mapping],
            solver_metadata={"backend": "exhaustive", "evaluated_mappings": evaluated, "optimality_certified": complete, **best_meta},
            **(best_ops or {key: None for key in ("node_insertions", "node_deletions", "node_substitutions", "edge_insertions", "edge_deletions")}),
        )


class GurobiGraphEditDistance(GraphDistance):
    def __init__(self, variant: str = "structural", edge_weight: float = 1.0, timeout: float = 300.0, seed: int = 0, initial_mapping: Iterable[tuple[int, int]] | None = None):
        self.variant = variant; self.edge_weight = float(edge_weight)
        self.timeout = float(timeout); self.seed = int(seed)
        self.initial_mapping = [] if initial_mapping is None else [(int(i), int(j)) for i, j in initial_mapping]

    def compute(self, predicted_graph: HardGraph, gt_graph: HardGraph) -> DistanceResult:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as error:
            raise RuntimeError("Gurobi backend requested but gurobipy cannot be imported") from error
        start = time.monotonic(); n, m = predicted_graph.node_count, gt_graph.node_count
        substitutions, scale_metadata = substitution_matrix(predicted_graph, gt_graph, self.variant)
        model = gp.Model("vascular_discrete_ged")
        model.Params.OutputFlag = 0; model.Params.TimeLimit = self.timeout
        model.Params.Seed = self.seed; model.Params.Threads = 1; model.Params.NonConvex = 2
        x = model.addVars(n, m, vtype=GRB.BINARY, name="match")
        initial = set(self.initial_mapping)
        for i in range(n):
            for j in range(m): x[i, j].Start = 1.0 if (i, j) in initial else 0.0
        for left in range(n): model.addConstr(gp.quicksum(x[left, right] for right in range(m)) <= 1)
        for right in range(m): model.addConstr(gp.quicksum(x[left, right] for left in range(n)) <= 1)
        objective = float(n + m + self.edge_weight * (predicted_graph.edge_count + gt_graph.edge_count))
        objective += gp.quicksum((float(substitutions[i, j]) - 2.0) * x[i, j] for i in range(n) for j in range(m))
        if predicted_graph.edge_count and gt_graph.edge_count:
            matched_edge_terms = []
            for i, k in predicted_graph.edges:
                for j, l in gt_graph.edges:
                    matched_edge_terms.extend((x[int(i), int(j)] * x[int(k), int(l)], x[int(i), int(l)] * x[int(k), int(j)]))
            objective += -2.0 * self.edge_weight * gp.quicksum(matched_edge_terms)
        model.setObjective(objective, GRB.MINIMIZE); model.optimize()
        status_names = {
            GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "timeout", GRB.INFEASIBLE: "infeasible",
            GRB.INTERRUPTED: "interrupted", GRB.SUBOPTIMAL: "suboptimal",
        }
        status = status_names.get(model.Status, f"gurobi_status_{model.Status}")
        has_incumbent = model.SolCount > 0
        mapping = [[i, j] for i in range(n) for j in range(m) if has_incumbent and x[i, j].X > 0.5]
        operations = {key: None for key in ("node_insertions", "node_deletions", "node_substitutions", "edge_insertions", "edge_deletions")}
        operation_metadata = {}
        if has_incumbent:
            checked, operations, operation_metadata = mapping_cost(
                predicted_graph, gt_graph, mapping, variant=self.variant, edge_weight=self.edge_weight
            )
            if abs(checked - float(model.ObjVal)) > 1e-5 * max(1.0, abs(checked)):
                raise RuntimeError(f"decoded GED {checked} disagrees with Gurobi objective {model.ObjVal}")
        upper = float(model.ObjVal) if has_incumbent else None
        lower = float(model.ObjBound) if model.Status not in {GRB.INFEASIBLE} else None
        gap = None
        if upper is not None and lower is not None:
            gap = max(0.0, upper - lower) / max(abs(upper), 1e-12)
        return DistanceResult(
            distance=upper if status == "optimal" else None, status=status, runtime_seconds=time.monotonic() - start,
            lower_bound=lower, upper_bound=upper, relative_gap=gap,
            node_mapping=mapping, solver_metadata={
                "backend": "gurobi_miqp", "gurobi_version": ".".join(map(str, gp.gurobi.version())),
                "seed": self.seed, "time_limit": self.timeout, "threads": 1,
                "warm_start_pairs": len(self.initial_mapping),
                "optimality_certified": status == "optimal", "node_pair_variables": n * m,
                **scale_metadata, **operation_metadata,
            }, **operations,
        )
