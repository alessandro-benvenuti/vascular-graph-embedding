#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

from vascular_graph_embedding.corruptions import CorruptionGenerator
from vascular_graph_embedding.corruptions.empirical import EmpiricalSampler
from vascular_graph_embedding.data.manifest import load_patch_index, stratified_select
from vascular_graph_embedding.data.storage import PairArchiveWriter, save_pair
from vascular_graph_embedding.data.vtp import read_vtp_graph
from vascular_graph_embedding.experiments.common import load_config, prepare_run
from vascular_graph_embedding.graphs.model import clean_fixed_graph


def fixture(index:int,slots:int):
    n=5+index%4;x=np.stack([np.linspace(.1,.8,n),np.full(n,.5),np.full(n,.5)],axis=1);edges=np.stack([np.arange(n-1),np.arange(1,n)],axis=1);return clean_fixed_graph(x,edges,slots)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("\0".join(map(str,parts)).encode()).digest()[:8],"big")%(2**31)


def load_bases(config, count, slots, synthetic):
    data=config.get("data",{});vtp_root=Path(str(data.get("vtp_root","")));index_path=Path(str(data.get("patch_index","")))
    if synthetic:
        return [(f"synthetic-{i}",f"fixture-{i//2}",*fixture(i,slots),{"source":"local synthetic fixture"}) for i in range(count)]
    if not vtp_root.is_dir() or not index_path.is_file():
        raise FileNotFoundError("real data paths unavailable; pass --synthetic-fixtures only for local implementation tests")
    index=load_patch_index(index_path);files=sorted(vtp_root.glob("*_graph.vtp"));rows=[]
    patient_col=next((c for c in index.columns if c.lower()=="patient_id"),None);id_cols=[c for c in index.columns if c.lower() in {"sample_id","patch_id","id","name"}]
    lookup={str(row[c]):row for _,row in index.iterrows() for c in id_cols if pd.notna(row[c])}
    shape=tuple(map(int,config.get("graph",{}).get("volume_shape",[64,64,64])));storage=str(config.get("graph",{}).get("coordinate_storage","voxel"))
    candidate_files=files if len(files)<=max(2000,count*10) else [files[i] for i in np.linspace(0,len(files)-1,max(2000,count*10),dtype=int)]
    for path in candidate_files:
        sample=path.name.removesuffix("_graph.vtp");meta=lookup.get(sample);patient=str(meta[patient_col]) if meta is not None and patient_col else sample.split("_")[0]
        coordinates,edges=read_vtp_graph(path,shape=shape,coordinate_space=storage);graph,corr=clean_fixed_graph(coordinates,edges,slots)
        degree=np.bincount(edges.reshape(-1),minlength=len(coordinates)) if len(edges) else np.zeros(len(coordinates),int)
        lengths=np.linalg.norm(coordinates[edges[:,0]]-coordinates[edges[:,1]],axis=1) if len(edges) else np.array([])
        from vascular_graph_embedding.graphs.topology import topology_summary
        topo=topology_summary(graph.presence,graph.effective_adjacency)
        rows.append({"sample_id":sample,"patient_id":patient,"path":str(path),"gt_nodes":len(coordinates),"gt_edges":len(edges),"components":topo["beta0"],"mean_edge_length":float(lengths.mean()) if len(lengths) else 0,"bifurcations":int(np.count_nonzero(degree>=3)),"has_cycle":topo["beta1"]>0,"graph":graph,"corr":corr})
    tabular=pd.DataFrame([{k:v for k,v in row.items() if k not in {"graph","corr"}} for row in rows]);selected=stratified_select(tabular,count,int(config.get("seed",0)))
    by_id={row["sample_id"]:row for row in rows};return [(sample,by_id[sample]["patient_id"],by_id[sample]["graph"],by_id[sample]["corr"],{k:v for k,v in by_id[sample].items() if k not in {"graph","corr"}}) for sample in selected.sample_id]


def controlled_specs(config):
    families=config["study"]["families"];counts=config["study"].get("counts",[0,1,2,4]);levels=config["study"].get("intensity_levels",[.2,.4,.6,.8,1.]);reps=int(config["study"].get("repetitions",5))
    parameter={"retained_node_confidence":lambda z:{"confidence":1-.8*z},"coordinate_displacement":lambda z:{"magnitude":.05*z},"missing_node":lambda z:{"residual_confidence":1-z},"missing_edge":lambda z:{"residual_probability":1-z},"generic_extra_node":lambda z:{"confidence":z,"offset":.05*z,"attachments":1},"duplicate":lambda z:{"confidence":z,"offset":.02*z},"split_node":lambda z:{"separation":.04*z},"subdivision":lambda z:{"transverse_offset":.02*z},"generic_false_edge":lambda z:{"confidence":z},"loop_forming_edge":lambda z:{"confidence":max(.51,z)},"missing_node_false_shortcut":lambda z:{"residual_confidence":1-z,"edge_confidence":z},"missing_node_missing_edge_elsewhere":lambda z:{"residual_confidence":1-z,"residual_probability":1-z},"duplicate_altered_attachments":lambda z:{"confidence":z,"offset":.02*z,"attachments":2},"subdivision_loop_shortcut":lambda z:{"transverse_offset":.02*z,"edge_confidence":max(.51,z)},"displacement_confidence_reduction":lambda z:{"magnitude":.05*z,"confidence":1-.8*z}}
    for family in families:
        for count in counts:
            for level_index,intensity in enumerate(levels):
                for rep in range(reps):yield family,count,level_index,float(intensity),rep,parameter.get(family,lambda z:{})(float(intensity))


def empirical_operation(row):
    raw=str(row.get("event_type"));mapping={"duplicate_node":"duplicate","subdivision_node":"subdivision","loop_closing_false_edge":"loop_forming_edge"};family=mapping.get(raw,raw)
    def number(name,default):
        value=row.get(name,default)
        return default if pd.isna(value) else float(value)
    params={}
    if family=="missing_node":params={"residual_confidence":number("confidence",.05)}
    elif family=="missing_edge":params={"residual_probability":number("raw_edge_probability",.05)}
    elif family in {"generic_extra_node","duplicate"}:params={"confidence":number("confidence",.8),"offset":number("nearest_node_distance_normalized",.03),"attachments":max(1,int(number("incident_edge_count",1))),"relation_probability":number("incident_probability_mean",.8)}
    elif family=="split_node":params={"confidence":number("confidence",.8),"separation":number("nearest_node_distance_normalized",.02),"relation_probability":number("incident_probability_mean",.8)}
    elif family=="subdivision":params={"confidence":number("confidence",.8),"fraction":number("subdivision_fraction",.5),"transverse_offset":number("transverse_offset_normalized",0),"relation_probability":number("incident_probability_mean",.8),"retain_direct_edge":False}
    elif family in {"generic_false_edge","loop_forming_edge"}:params={"confidence":number("raw_edge_probability",.8)}
    return {"family":family,"count":1,**params,"source_event_id":row.get("source_event_id")}


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--dry-run",action="store_true");p.add_argument("--synthetic-fixtures",action="store_true",help="local tests only; never represents a cluster data run");p.add_argument("--patch-offset",type=int,default=0);p.add_argument("--patch-limit",type=int);a=p.parse_args();cfg=load_config(a.config)
    patch_count=int(cfg["study"].get("patches",10));mode=cfg["study"].get("mode","controlled");specs=list(controlled_specs(cfg)) if mode=="controlled" else list(range(int(cfg["study"].get("pairs_per_patch",5))));pairs=patch_count*len(specs)
    if a.dry_run:print(json.dumps({"patches":patch_count,"conditions_per_patch":len(specs),"pairs":pairs,"estimated_gib_at_11_kib_per_pair":round(pairs*11*1024/(1024**3),3),"pair_members":pairs,"recommended_storage":"zip archives per shard"},indent=2));return
    cfg={**cfg,"shard":{"patch_offset":a.patch_offset,"patch_limit":a.patch_limit}}
    out=prepare_run(cfg,a.output);(out/"pairs").mkdir(exist_ok=True);manifest=[];events=[];slots=int(cfg.get("graph",{}).get("slots",120));base_seed=int(cfg.get("seed",0));background=float(cfg.get("graph",{}).get("background_presence",0));retained=float(cfg.get("graph",{}).get("retained_presence",1));retained_edge=float(cfg.get("graph",{}).get("retained_edge_probability",1));all_bases=load_bases(cfg,patch_count,slots,a.synthetic_fixtures);end=None if a.patch_limit is None else a.patch_offset+a.patch_limit;bases=all_bases[a.patch_offset:end];pair_store=str(cfg.get("graph",{}).get("pair_store","files"));archive_context=PairArchiveWriter(out/"pairs"/"pairs.zip") if pair_store=="zip" else None
    empirical=EmpiricalSampler(cfg["data"]["empirical_root"],base_seed) if mode=="empirical" else None
    selection=[]
    writer=archive_context.__enter__() if archive_context is not None else None
    archive_success=False
    try:
      for local_patch,(patch_id,patient_id,gt,corr,source_meta) in enumerate(bases):
        patch=a.patch_offset+local_patch
        selection.append({"patch_id":patch_id,"patient_id":patient_id,**source_meta})
        iteration_specs=specs
        for item in iteration_specs:
            if mode=="controlled":family,count,level,intensity,rep,params=item;source_profile=None;source_events=[]
            else:
                rep=int(item);profile,source_events=empirical.draw(approximate_gt_nodes=int(np.count_nonzero(gt.presence>.5)));operations=[empirical_operation(row) for row in source_events]
                represented={}
                for operation in operations:represented[operation["family"]]=represented.get(operation["family"],0)+1
                mapping={"duplicate_node":"duplicate","subdivision_node":"subdivision","loop_closing_false_edge":"loop_forming_edge"}
                for key,value in profile.items():
                    if str(key).startswith("count_") and pd.notna(value):
                        family_name=mapping.get(str(key)[6:],str(key)[6:]);missing=max(0,int(float(value))-represented.get(family_name,0));operations.extend({"family":family_name,"count":1,"attribute_source":"geometric_default_missing_raw_event"} for _ in range(missing))
                family="empirical_mixture";count=sum(x["count"] for x in operations);level=-1;intensity=float(profile.get("difficulty_score",np.nan));params={};source_profile=profile
            seed=stable_seed(base_seed,patch_id,family,count,level,rep);base=gt.copy();base.presence[corr.slot_to_gt>=0]=retained;base.presence[corr.slot_to_gt<0]=background;base.conditional_edges[base.conditional_edges>0]=retained_edge
            ops=[{"family":family,"count":count,**params}] if mode=="controlled" else operations
            for operation in ops:operation.pop("source_event_id",None)
            generator=CorruptionGenerator(seed);result=generator.generate(base,corr,ops,pair_id=f"p{patch:04d}-{family}-c{count}-i{level}-r{rep}")
            pair_id=result.metadata["pair_id"];pair_meta={**result.metadata,"patch_id":patch_id,"patient_id":patient_id,"family":family,"count":count,"intensity_level":level,"intensity":intensity,"repetition":rep,"source_profile_id":None if source_profile is None else source_profile.get("source_profile_id")}
            if writer is not None:
                member=pair_id+".npz";writer.write(member,gt,result.graph,result.correspondence,pair_meta);relative=f"pairs/pairs.zip::{member}"
            else:
                relative=str(Path("pairs")/(pair_id+".npz"));save_pair(out/relative,gt,result.graph,result.correspondence,pair_meta)
            realized=sum(event.realized for event in result.events);reasons=";".join(filter(None,(event.rejection_reason for event in result.events))) or None;manifest.append({"pair_id":pair_id,"path":str(relative),"patch_id":patch_id,"patient_id":patient_id,"family":family,"requested_count":count,"realized_count":realized,"intensity_level":level,"intensity":intensity,"repetition":rep,"seed":seed,"status":"ok" if realized==count else "skipped_or_partial","rejection_reason":reasons,"source_profile_id":None if source_profile is None else source_profile.get("source_profile_id")})
            source_event_ids=[row.get("source_event_id") for row in source_events]
            for event in result.events:events.append({"pair_id":pair_id,"source_profile_id":None if source_profile is None else source_profile.get("source_profile_id"),"source_event_ids":json.dumps(source_event_ids),**event.as_dict(),"parameters":json.dumps(event.parameters),"anchors":json.dumps(event.anchors),"owned_nodes":json.dumps(event.owned_nodes),"owned_edges":json.dumps(event.owned_edges)})
      archive_success=True
    finally:
      if archive_context is not None:archive_context.__exit__(None if archive_success else RuntimeError,None,None)
    pd.DataFrame(manifest).to_csv(out/"pair_manifest.csv.gz",index=False);pd.DataFrame(events).to_csv(out/"edit_events.csv.gz",index=False);pd.DataFrame(selection).to_csv(out/"selection_manifest.csv",index=False)
    diagnostics=out/"sampling_diagnostics";diagnostics.mkdir(exist_ok=True);pd.DataFrame(manifest)[["pair_id","requested_count","realized_count","status","rejection_reason","source_profile_id"]].to_csv(diagnostics/"requested_vs_realized.csv",index=False)
    print(f"wrote {len(manifest)} reusable pairs to {out}")
if __name__=="__main__":main()

