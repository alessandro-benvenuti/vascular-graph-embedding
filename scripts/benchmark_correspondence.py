#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scipy.optimize import linear_sum_assignment
from vascular_graph_embedding.data.storage import load_pair
from vascular_graph_embedding.distances.soft_edit import SoftGeometricEditCost
from vascular_graph_embedding.distances.optimized import exhaustive_correspondence


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--limit",type=int,default=25);p.add_argument("--timeout",type=float,default=30);a=p.parse_args();run=Path(a.run);manifest=pd.read_csv(run/"pair_manifest.csv.gz");rows=[];cost=SoftGeometricEditCost()
    for _,record in manifest.head(a.limit).iterrows():
        stored=str(record.path);pair_path=f"{run}/{stored}" if "::" not in stored else f"{run}/{stored.split('::',1)[0]}::{stored.split('::',1)[1]}";gt,candidate,corr,_=load_pair(pair_path);incumbent=cost.compute(gt,candidate,corr)["total_cost"]
        active_c=[i for i,presence in enumerate(candidate.presence) if presence>.5];active_g=[i for i,presence in enumerate(gt.presence) if presence>.5]
        base={"pair_id":record.pair_id,"incumbent_objective":incumbent,"backend":"exhaustive","status":"ineligible"}
        if len(active_c)==len(active_g) and len(active_c)<=8:
            result=exhaustive_correspondence(gt,candidate,cost,timeout=a.timeout);base.update(result);base["known_is_upper_bound"]=result["objective"]<=incumbent+1e-10
        rows.append(base)
    pd.DataFrame(rows).to_csv(run/"solver_results.csv.gz",index=False);print(f"wrote {len(rows)} solver benchmark rows")
if __name__=="__main__":main()

