import numpy as np
from vascular_graph_embedding.data.storage import PairArchiveWriter,load_pair,save_pair
from vascular_graph_embedding.distances.soft_edit import SoftGeometricEditCost
from vascular_graph_embedding.graphs.model import clean_fixed_graph


def test_saved_pairs_support_new_distance_without_regeneration(tmp_path):
    graph,corr=clean_fixed_graph(np.array([[0,0,0],[1,0,0]]),np.array([[0,1]]),4);path=tmp_path/"pair.npz";save_pair(path,graph,graph,corr,{"pair_id":"x"});mtime=path.stat().st_mtime_ns;gt,candidate,corr2,_=load_pair(path);SoftGeometricEditCost(lambda_edge=3).compute(gt,candidate,corr2);assert path.stat().st_mtime_ns==mtime


def test_archive_pair_round_trip(tmp_path):
    graph,corr=clean_fixed_graph(np.array([[0,0,0],[1,0,0]]),np.array([[0,1]]),4);archive=tmp_path/"pairs.zip"
    with PairArchiveWriter(archive) as writer:writer.write("x.npz",graph,graph,corr,{"pair_id":"x"})
    gt,candidate,loaded,meta=load_pair(f"{archive}::x.npz");assert meta["pair_id"]=="x";assert np.array_equal(candidate.effective_adjacency,gt.effective_adjacency);assert np.array_equal(loaded.slot_to_gt,corr.slot_to_gt)


def test_archive_finalization_is_idempotent(tmp_path):
    graph,corr=clean_fixed_graph(np.array([[0,0,0],[1,0,0]]),np.array([[0,1]]),4);archive=tmp_path/"pairs.zip";writer=PairArchiveWriter(archive);writer.__enter__();writer.write("x.npz",graph,graph,corr,{"pair_id":"x"});writer.__exit__(None,None,None);writer.__exit__(None,None,None);assert archive.is_file()

