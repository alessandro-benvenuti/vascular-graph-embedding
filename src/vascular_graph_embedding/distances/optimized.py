from __future__ import annotations

import itertools, time
import numpy as np

from vascular_graph_embedding.graphs import Correspondence, SoftGraph
from .soft_edit import SoftGeometricEditCost


def exhaustive_correspondence(gt: SoftGraph, candidate: SoftGraph, cost: SoftGeometricEditCost, *, max_active: int = 8, timeout: float = 30.0) -> dict:
    """Exact permutation search for tiny equal-cardinality cases under the declared objective."""
    start=time.monotonic();cand=np.flatnonzero(candidate.presence>.5);targets=np.flatnonzero(gt.presence>.5)
    if len(cand)!=len(targets) or len(cand)>max_active:raise ValueError("exhaustive solver requires equal active cardinality at most max_active")
    best=float("inf");best_corr=None;evaluated=0;complete=True
    for permutation in itertools.permutations(targets.tolist()):
        if time.monotonic()-start>timeout:complete=False;break
        mapping=np.full(candidate.slots,-1,dtype=int);mapping[cand]=permutation
        value=cost.compute(gt,candidate,Correspondence(mapping))["total_cost"];evaluated+=1
        if value<best:best=value;best_corr=mapping.copy()
    return {"objective":best,"lower_bound":best if complete else None,"gap":0.0 if complete else None,"runtime_seconds":time.monotonic()-start,"status":"optimal" if complete else "timeout","optimality_certified":complete,"evaluated":evaluated,"slot_to_gt":best_corr.tolist() if best_corr is not None else None}

