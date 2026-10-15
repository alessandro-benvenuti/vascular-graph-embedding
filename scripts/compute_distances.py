#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from vascular_graph_embedding.data.storage import load_pair
from vascular_graph_embedding.distances import get_distance
from vascular_graph_embedding.experiments.common import load_config


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--run",required=True);a=p.parse_args();cfg=load_config(a.config);run=Path(a.run);manifest=pd.read_csv(run/"pair_manifest.csv.gz");rows=[]
    variants=cfg.get("distances",[{"plugin":"soft_geometric_edit_v1","normalization":"gt"}])
    for _,record in manifest.iterrows():
        gt,candidate,corr,meta=load_pair(run/record["path"])
        for index,spec in enumerate(variants):
            options=dict(spec);name=options.pop("plugin");options["coordinate_scale"]=tuple(options.get("coordinate_scale",(1,1,1)));plugin=get_distance(name,**options);result=plugin.compute(gt,candidate,corr)
            baseline=gt.copy();graph_cfg=cfg.get("graph",{});background=float(graph_cfg.get("background_presence",0));baseline.presence[corr.slot_to_gt>=0]=float(graph_cfg.get("retained_presence",1));baseline.presence[corr.slot_to_gt<0]=background;baseline.conditional_edges[baseline.conditional_edges>0]=float(graph_cfg.get("retained_edge_probability",1));base_result=plugin.compute(gt,baseline,corr)
            rows.append({**record.to_dict(),"variant":index,**result,"background_baseline_cost":base_result["total_cost"],"change_from_background":result["total_cost"]-base_result["total_cost"]})
    pd.DataFrame(rows).to_csv(run/"distance_results.csv.gz",index=False);print(f"wrote {len(rows)} distance rows")
if __name__=="__main__":main()

