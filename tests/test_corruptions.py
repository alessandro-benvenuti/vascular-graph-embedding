import numpy as np

from vascular_graph_embedding.corruptions import CorruptionGenerator
from vascular_graph_embedding.graphs.model import clean_fixed_graph
from vascular_graph_embedding.graphs.topology import topology_summary


def base(slots=8):return clean_fixed_graph(np.array([[0,0,0],[.1,0,0],[.2,0,0],[.3,0,0]]),np.array([[0,1],[1,2],[2,3]]),slots)


def test_determinism_and_capacity():
    graph,corr=base(5);ops=[{"family":"generic_extra_node","count":2,"confidence":.8}]
    one=CorruptionGenerator(4).generate(graph,corr,ops);two=CorruptionGenerator(4).generate(graph,corr,ops)
    assert np.array_equal(one.graph.coordinates,two.graph.coordinates)
    assert one.events[0].realized==1 and one.events[0].rejection_reason


def test_ownership_prevents_unrelated_reuse():
    graph,corr=base();gen=CorruptionGenerator(1);first=gen.apply(graph,corr,"missing_node",1);second=gen.apply(graph,corr,"coordinate_displacement",3)
    assert not set(first.owned_nodes)&set(second.owned_nodes)


def test_split_redistributes_and_subdivision_has_two_attachments():
    graph,corr=base();split=CorruptionGenerator(2).generate(graph,corr,[{"family":"split_node","count":1}]);assert split.events[0].realized==1
    inserted=int([x for x in split.events[0].owned_nodes if int(x)>=4][0]);assert np.count_nonzero(split.graph.conditional_edges[inserted]>.5)>=2
    sub=CorruptionGenerator(3).generate(*base(),[{"family":"subdivision","count":1,"retain_direct_edge":False}]);event=sub.events[0];inserted=int(event.owned_nodes[0]);assert np.count_nonzero(sub.graph.conditional_edges[inserted]>.5)==2


def test_loop_increases_cycle_rank():
    graph,corr=base();before=topology_summary(graph.presence,graph.effective_adjacency)["beta1"];result=CorruptionGenerator(2).generate(graph,corr,[{"family":"loop_forming_edge","count":1}]);after=topology_summary(result.graph.presence,result.graph.effective_adjacency)["beta1"];assert after==before+1


def test_missing_node_incident_policy():
    graph,corr=base();result=CorruptionGenerator(5).generate(graph,corr,[{"family":"missing_node","count":1,"incident_policy":"zero_conditional"}]);slot=int(result.events[0].owned_nodes[0]);assert not result.graph.conditional_edges[slot].any()


def test_named_composite_records_components():
    graph,corr=base();result=CorruptionGenerator(7).generate(graph,corr,[{"family":"displacement_confidence_reduction","count":1,"magnitude":.1,"confidence":.6}]);event=result.events[0];slot=int(event.owned_nodes[0]);assert event.realized==1 and result.graph.presence[slot]==.6 and not np.allclose(result.graph.coordinates[slot],graph.coordinates[slot])

