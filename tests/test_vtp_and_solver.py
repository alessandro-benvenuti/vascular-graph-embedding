from pathlib import Path
import numpy as np
import pytest

from vascular_graph_embedding.data.vtp import read_vtp_graph
from vascular_graph_embedding.distances.optimized import exhaustive_correspondence
from vascular_graph_embedding.distances.soft_edit import SoftGeometricEditCost
from vascular_graph_embedding.graphs.model import clean_fixed_graph


def test_ascii_vtp_normalization_and_edges(tmp_path:Path):
    path=tmp_path/"g.vtp";path.write_text('''<VTKFile><PolyData><Piece><Points><DataArray format="ascii">0 0 0 9 19 29</DataArray></Points><Lines><DataArray Name="connectivity">0 1</DataArray><DataArray Name="offsets">2</DataArray></Lines></Piece></PolyData></VTKFile>''')
    points,edges=read_vtp_graph(path,shape=(10,20,30),coordinate_space="voxel");assert np.allclose(points[1],[.9,.95,29/30]);assert edges.tolist()==[[0,1]]


def test_solver_objective_and_known_upper_bound():
    graph,corr=clean_fixed_graph(np.array([[0,0,0],[1,0,0],[2,0,0]]),np.array([[0,1],[1,2]]),3);candidate=graph.permuted(np.array([2,0,1]));known=corr.permuted(np.array([2,0,1]));cost=SoftGeometricEditCost(normalization="raw");incumbent=cost.compute(graph,candidate,known)["total_cost"];result=exhaustive_correspondence(graph,candidate,cost)
    assert result["optimality_certified"] and result["objective"]==pytest.approx(incumbent) and result["objective"]<=incumbent

