import numpy as np
import pytest

from vascular_graph_embedding.distances.soft_edit import SoftGeometricEditCost
from vascular_graph_embedding.graphs.model import Correspondence, SoftGraph, clean_fixed_graph


def base(slots=6):
    return clean_fixed_graph(np.array([[.1,.2,.3],[.2,.2,.3],[.3,.2,.3]]),np.array([[0,1],[1,2]]),slots)


def test_clean_zero_and_components():
    graph,corr=base();result=SoftGeometricEditCost().compute(graph,graph,corr)
    assert result["total_cost"]==0
    changed=graph.copy();changed.presence[0]=.5;changed.coordinates[1,0]+=.2;changed.conditional_edges[1,2]=changed.conditional_edges[2,1]=.5
    result=SoftGeometricEditCost(normalization="raw").compute(graph,changed,corr)
    assert result["node_raw"]==pytest.approx(.5)
    assert result["coordinate_raw"]==pytest.approx(.2)
    assert result["edge_raw"]>0


def test_inserted_node_has_no_coordinate_penalty():
    graph,corr=base();changed=graph.copy();changed.presence[4]=.8;changed.coordinates[4]=[100,100,100]
    result=SoftGeometricEditCost(normalization="raw").compute(graph,changed,corr)
    assert result["coordinate_raw"]==0 and result["node_raw"]==pytest.approx(.8)


def test_query_permutation_invariance():
    graph,corr=base();changed=graph.copy();changed.presence[0]=.7;changed.coordinates[2,1]+=.1
    before=SoftGeometricEditCost().compute(graph,changed,corr)["total_cost"]
    order=np.array([4,2,0,5,1,3]);after=SoftGeometricEditCost().compute(graph,changed.permuted(order),corr.permuted(order))["total_cost"]
    assert after==pytest.approx(before)


def test_symmetry_effective_adjacency_and_round_trip():
    graph,_=base();assert np.allclose(graph.effective_adjacency,graph.effective_adjacency.T);assert np.allclose(np.diag(graph.effective_adjacency),0)
    compact,slots=graph.compact();fixed=SoftGraph.from_compact(compact,slots,graph.slots)
    assert np.allclose(fixed.effective_adjacency,graph.effective_adjacency)


def test_graph_size_normalization():
    small,corr_s=clean_fixed_graph(np.array([[0,0,0],[1,0,0]]),np.array([[0,1]]),4);large,corr_l=clean_fixed_graph(np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]]),np.array([[0,1],[1,2],[2,3]]),6)
    a=small.copy();a.presence[:2]=.5;b=large.copy();b.presence[:4]=.5
    cost=SoftGeometricEditCost(normalization="gt")
    assert cost.compute(small,a,corr_s)["node_normalized"]==pytest.approx(cost.compute(large,b,corr_l)["node_normalized"])

