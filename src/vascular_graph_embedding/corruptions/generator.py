from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from vascular_graph_embedding.graphs import Correspondence, SoftGraph
from vascular_graph_embedding.graphs.topology import topology_summary

EDIT_SCHEMA = "vascular-edit-event/v1"
PAIR_SCHEMA = "vascular-pair/v1"


@dataclass
class EditEvent:
    family: str
    requested: int
    realized: int
    seed: int
    parameters: dict[str, Any]
    anchors: list[str] = field(default_factory=list)
    owned_nodes: list[str] = field(default_factory=list)
    owned_edges: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    schema: str = EDIT_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairResult:
    graph: SoftGraph
    correspondence: Correspondence
    events: list[EditEvent]
    metadata: dict[str, Any]


class CorruptionGenerator:
    """Deterministic, ownership-aware corruption engine for fixed-slot graphs."""

    def __init__(self, seed: int, *, threshold: float = 0.5):
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.threshold = threshold
        self.owned_nodes: set[int] = set()
        self.owned_edges: set[tuple[int, int]] = set()

    @staticmethod
    def _edge(i: int, j: int) -> tuple[int, int]:
        return (min(i, j), max(i, j))

    def _choose(self, eligible: list[Any], count: int) -> list[Any]:
        if count <= 0 or not eligible:
            return []
        indices = self.rng.choice(len(eligible), size=min(count, len(eligible)), replace=False)
        return [eligible[int(i)] for i in np.atleast_1d(indices)]

    def apply(self, graph: SoftGraph, corr: Correspondence, family: str, count: int = 1, **parameters: Any) -> EditEvent:
        method: Callable[..., EditEvent] | None = getattr(self, f"_{family}", None)
        if method is None:
            raise KeyError(f"unknown corruption family {family!r}")
        return method(graph, corr, int(count), parameters)

    def generate(self, base: SoftGraph, correspondence: Correspondence, operations: list[dict[str, Any]], *, pair_id: str = "pair") -> PairResult:
        graph, corr = base.copy(), Correspondence(correspondence.slot_to_gt.copy())
        events = []
        for operation in operations:
            spec = dict(operation)
            family = str(spec.pop("family")); count = int(spec.pop("count", 1))
            events.append(self.apply(graph, corr, family, count, **spec))
        return PairResult(graph, corr, events, {
            "schema": PAIR_SCHEMA, "pair_id": pair_id, "seed": self.seed,
            "requested_events": sum(e.requested for e in events),
            "realized_events": sum(e.realized for e in events),
        })

    def _event(self, family: str, count: int, params: dict[str, Any], anchors: list[int], *, nodes: list[int] = [], edges: list[tuple[int, int]] = [], reason: str | None = None) -> EditEvent:
        self.owned_nodes.update(nodes); self.owned_edges.update(self._edge(*e) for e in edges)
        return EditEvent(family, count, len(anchors), self.seed, params,
                         [str(i) for i in anchors], [str(i) for i in nodes],
                         [f"{min(e)}-{max(e)}" for e in edges], reason)

    def _retained_node_confidence(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        confidence = float(p.get("confidence", 0.8))
        eligible = [i for i, gt in enumerate(c.slot_to_gt) if gt >= 0 and i not in self.owned_nodes]
        chosen = self._choose(eligible, count)
        g.presence[chosen] = confidence
        return self._event("retained_node_confidence", count, {**p, "confidence": confidence}, chosen, nodes=chosen, reason=None if len(chosen)==count else "insufficient eligible nodes")

    def _coordinate_displacement(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        magnitude = float(p.get("magnitude", 0.02)); mode = str(p.get("direction", "isotropic"))
        eligible = [i for i, gt in enumerate(c.slot_to_gt) if gt >= 0 and i not in self.owned_nodes]
        chosen = self._choose(eligible, count)
        for i in chosen:
            neighbours = np.flatnonzero(g.conditional_edges[i] > 0.5)
            vector = self.rng.normal(size=3)
            if mode in {"along_vessel", "transverse"} and len(neighbours):
                tangent = g.coordinates[neighbours].mean(axis=0) - g.coordinates[i]
                if np.linalg.norm(tangent) > 1e-12:
                    tangent /= np.linalg.norm(tangent)
                    if mode == "along_vessel": vector = tangent * (1 if self.rng.random() > .5 else -1)
                    else: vector -= vector.dot(tangent) * tangent
            if np.linalg.norm(vector) < 1e-12: vector = np.array([1., 0., 0.])
            g.coordinates[i] += magnitude * vector / np.linalg.norm(vector)
        return self._event("coordinate_displacement", count, {**p, "magnitude": magnitude, "direction": mode}, chosen, nodes=chosen, reason=None if len(chosen)==count else "insufficient eligible nodes")

    def _missing_node(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        residual = float(p.get("residual_confidence", 0.05)); incident = str(p.get("incident_policy", "retain_conditional"))
        eligible = [i for i, gt in enumerate(c.slot_to_gt) if gt >= 0 and i not in self.owned_nodes]
        chosen = self._choose(eligible, count)
        for i in chosen:
            g.presence[i] = residual
            if incident == "zero_conditional": g.conditional_edges[i, :] = 0; g.conditional_edges[:, i] = 0
            elif incident != "retain_conditional": raise ValueError("incident_policy must be retain_conditional or zero_conditional")
        return self._event("missing_node", count, {**p, "residual_confidence": residual, "incident_policy": incident}, chosen, nodes=chosen, reason=None if len(chosen)==count else "insufficient eligible nodes")

    def _missing_edge(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        residual = float(p.get("residual_probability", 0.05))
        eligible = [(i,j) for i in range(g.slots) for j in range(i+1,g.slots) if g.conditional_edges[i,j] > .5 and self._edge(i,j) not in self.owned_edges and i not in self.owned_nodes and j not in self.owned_nodes]
        chosen = self._choose(eligible, count)
        for i,j in chosen: g.conditional_edges[i,j] = g.conditional_edges[j,i] = residual
        return self._event("missing_edge", count, {**p, "residual_probability": residual}, [x for e in chosen for x in e], edges=chosen, reason=None if len(chosen)==count else "insufficient eligible edges")

    def _free_slots(self, g: SoftGraph, c: Correspondence) -> list[int]:
        return [i for i in range(g.slots) if c.slot_to_gt[i] < 0 and g.presence[i] <= self.threshold and i not in self.owned_nodes]

    def _generic_extra_node(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        confidence = float(p.get("confidence", .8)); degree = int(p.get("attachments", 1)); offset = float(p.get("offset", .03)); relation = float(p.get("relation_probability", .8))
        slots = self._choose(self._free_slots(g,c), count); anchors=[]; edges=[]
        active = [i for i,x in enumerate(c.slot_to_gt) if x>=0 and i not in self.owned_nodes]
        for slot in slots:
            if not active: break
            attach = self._choose(active, min(degree,len(active))); source=attach[0]
            direction=self.rng.normal(size=3); direction/=max(np.linalg.norm(direction),1e-12)
            g.coordinates[slot]=g.coordinates[source]+offset*direction; g.presence[slot]=confidence
            g.node_ids[slot]=f"extra:{self.seed}:{slot}"; anchors.append(source)
            for node in attach: g.conditional_edges[slot,node]=g.conditional_edges[node,slot]=relation; edges.append((slot,node))
        return self._event("generic_extra_node", count, {**p,"confidence":confidence,"attachments":degree,"offset":offset,"relation_probability":relation}, anchors, nodes=slots, edges=edges, reason=None if len(slots)==count else "graph capacity exhausted")

    def _duplicate(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        p={"attachments":p.get("attachments",1),"offset":p.get("offset",.01),"confidence":p.get("confidence",.8),"relation_probability":p.get("relation_probability",.8),**p}
        event=self._generic_extra_node(g,c,count,p); event.family="duplicate"
        for slot in map(int,event.owned_nodes): g.node_ids[slot]=g.node_ids[slot].replace("extra:","duplicate:")
        return event

    def _false_edges(self, g: SoftGraph) -> list[tuple[int,int]]:
        active=np.flatnonzero(g.presence>self.threshold)
        return [(int(i),int(j)) for pos,i in enumerate(active) for j in active[pos+1:] if g.conditional_edges[i,j] <= self.threshold and self._edge(int(i),int(j)) not in self.owned_edges and i not in self.owned_nodes and j not in self.owned_nodes]

    def _generic_false_edge(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        confidence=float(p.get("confidence",.8)); chosen=self._choose(self._false_edges(g),count)
        for i,j in chosen:g.conditional_edges[i,j]=g.conditional_edges[j,i]=confidence
        return self._event("generic_false_edge",count,{**p,"confidence":confidence},[x for e in chosen for x in e],edges=chosen,reason=None if len(chosen)==count else "insufficient non-edges")

    def _path_exists(self,g:SoftGraph,u:int,v:int)->bool:
        seen={u}; stack=[u]
        while stack:
            i=stack.pop()
            if i==v:return True
            for j in np.flatnonzero(g.effective_adjacency[i]>=self.threshold):
                if int(j) not in seen:seen.add(int(j));stack.append(int(j))
        return False

    def _loop_forming_edge(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        confidence=float(p.get("confidence",.9)); before=topology_summary(g.presence,g.effective_adjacency,threshold=self.threshold)["beta1"]
        eligible=[e for e in self._false_edges(g) if self._path_exists(g,*e)]; chosen=self._choose(eligible,count)
        applied=[]
        for i,j in chosen:
            # Guarantee that the *effective* inserted edge crosses the topology threshold.
            required=min(1.0,(self.threshold+1e-9)/max(g.presence[i]*g.presence[j],1e-12))
            actual=max(confidence,required)
            if actual<=1.0:g.conditional_edges[i,j]=g.conditional_edges[j,i]=actual;applied.append((i,j))
        chosen=applied
        after=topology_summary(g.presence,g.effective_adjacency,threshold=self.threshold)["beta1"]
        if after-before != len(chosen): raise RuntimeError("loop-forming insertion failed to increase cycle rank")
        return self._event("loop_forming_edge",count,{**p,"confidence":confidence,"cycle_rank_increase":after-before},[x for e in chosen for x in e],edges=chosen,reason=None if len(chosen)==count else "insufficient connected non-edges")

    def _subdivision(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        fraction=float(p.get("fraction",.5)); transverse=float(p.get("transverse_offset",0)); confidence=float(p.get("confidence",.9)); relation=float(p.get("relation_probability",.9)); retain=bool(p.get("retain_direct_edge",False))
        eligible=[e for e in [(i,j) for i in range(g.slots) for j in range(i+1,g.slots) if g.conditional_edges[i,j]>.5] if self._edge(*e) not in self.owned_edges]
        edges_chosen=self._choose(eligible,min(count,len(self._free_slots(g,c)))); slots=self._choose(self._free_slots(g,c),len(edges_chosen)); owned=[]
        for slot,(u,v) in zip(slots,edges_chosen):
            point=(1-fraction)*g.coordinates[u]+fraction*g.coordinates[v]
            tangent=g.coordinates[v]-g.coordinates[u]; direction=self.rng.normal(size=3); direction-=direction.dot(tangent)*tangent/max(tangent.dot(tangent),1e-12); direction/=max(np.linalg.norm(direction),1e-12)
            g.coordinates[slot]=point+transverse*direction;g.presence[slot]=confidence;g.node_ids[slot]=f"subdivision:{self.seed}:{slot}"
            if not retain:g.conditional_edges[u,v]=g.conditional_edges[v,u]=0
            for node in (u,v):g.conditional_edges[slot,node]=g.conditional_edges[node,slot]=relation;owned.append((slot,node))
        return self._event("subdivision",count,{**p,"fraction":fraction,"transverse_offset":transverse,"confidence":confidence,"relation_probability":relation,"retain_direct_edge":retain,"structural_attachments":2},[x for e in edges_chosen for x in e],nodes=slots,edges=owned+edges_chosen,reason=None if len(slots)==count else "capacity or eligible-edge constraint")

    def _split_node(self, g: SoftGraph, c: Correspondence, count: int, p: dict[str, Any]) -> EditEvent:
        separation=float(p.get("separation",.02));confidence=float(p.get("confidence",.9));relation=float(p.get("relation_probability",.9))
        eligible=[i for i,gt in enumerate(c.slot_to_gt) if gt>=0 and i not in self.owned_nodes and np.count_nonzero(g.conditional_edges[i]>.5)>=2]
        sources=self._choose(eligible,min(count,len(self._free_slots(g,c))));slots=self._choose(self._free_slots(g,c),len(sources));owned=[]
        for slot,source in zip(slots,sources):
            neigh=list(map(int,np.flatnonzero(g.conditional_edges[source]>.5)));self.rng.shuffle(neigh);moved=neigh[:max(1,len(neigh)//2)]
            direction=self.rng.normal(size=3);direction/=max(np.linalg.norm(direction),1e-12)
            g.coordinates[slot]=g.coordinates[source]+separation*direction;g.presence[slot]=confidence;g.node_ids[slot]=f"split:{self.seed}:{slot}"
            for node in moved:g.conditional_edges[source,node]=g.conditional_edges[node,source]=0;g.conditional_edges[slot,node]=g.conditional_edges[node,slot]=relation;owned.extend([(source,node),(slot,node)])
            g.conditional_edges[source,slot]=g.conditional_edges[slot,source]=relation;owned.append((source,slot))
        return self._event("split_node",count,{**p,"separation":separation,"confidence":confidence,"relation_probability":relation,"branch_partition":"random_half"},sources,nodes=sources+slots,edges=owned,reason=None if len(slots)==count else "capacity or degree constraint")

    def _combine(self, family: str, count: int, parameters: dict[str, Any], parts: list[EditEvent]) -> EditEvent:
        return EditEvent(family, count, min((part.realized for part in parts), default=0), self.seed,
                         {**parameters, "components": [part.family for part in parts]},
                         [x for part in parts for x in part.anchors],
                         [x for part in parts for x in part.owned_nodes],
                         [x for part in parts for x in part.owned_edges],
                         "; ".join(x for x in (part.rejection_reason for part in parts) if x) or None)

    def _missing_node_false_shortcut(self, g, c, count, p):
        return self._combine("missing_node_false_shortcut",count,p,[self._missing_node(g,c,count,{"residual_confidence":p.get("residual_confidence",.05)}),self._generic_false_edge(g,c,count,{"confidence":p.get("edge_confidence",.8)})])

    def _missing_node_missing_edge_elsewhere(self, g, c, count, p):
        return self._combine("missing_node_missing_edge_elsewhere",count,p,[self._missing_node(g,c,count,{"residual_confidence":p.get("residual_confidence",.05)}),self._missing_edge(g,c,count,{"residual_probability":p.get("residual_probability",.05)})])

    def _duplicate_altered_attachments(self, g, c, count, p):
        return self._duplicate(g,c,count,{**p,"attachments":int(p.get("attachments",2))})

    def _subdivision_loop_shortcut(self, g, c, count, p):
        return self._combine("subdivision_loop_shortcut",count,p,[self._subdivision(g,c,count,{"retain_direct_edge":False,**p}),self._loop_forming_edge(g,c,count,{"confidence":p.get("edge_confidence",.9)})])

    def _displacement_confidence_reduction(self, g, c, count, p):
        displacement=self._coordinate_displacement(g,c,count,{"magnitude":p.get("magnitude",.02),"direction":p.get("direction","isotropic")})
        confidence=float(p.get("confidence",.7))
        for slot in map(int,displacement.owned_nodes):g.presence[slot]=confidence
        return EditEvent("displacement_confidence_reduction",count,displacement.realized,self.seed,{**p,"confidence":confidence},displacement.anchors,displacement.owned_nodes,displacement.owned_edges,displacement.rejection_reason)

