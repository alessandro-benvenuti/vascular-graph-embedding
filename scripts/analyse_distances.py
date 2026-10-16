#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from vascular_graph_embedding.analysis.statistics import associations,cluster_bootstrap_mean
from vascular_graph_embedding.data.storage import atomic_json,load_pair
from vascular_graph_embedding.graphs.topology import topology_summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);a=p.parse_args();run=Path(a.run);df=pd.read_csv(run/"distance_results.csv.gz");plots=run/"plots";plots.mkdir(exist_ok=True)
    summaries=[]
    for (variant,family),group in df.groupby(["variant","family"]):
        assoc=associations(group,"realized_count","total_cost");boot=cluster_bootstrap_mean(group,"total_cost","patient_id",draws=300);summaries.append({"variant":int(variant),"family":family,**assoc,**{f"bootstrap_{k}":v for k,v in boot.items()}})
    fig,ax=plt.subplots(figsize=(8,5))
    for family,group in df[df.variant==df.variant.min()].groupby("family"):
        means=group.groupby("realized_count")["total_cost"].mean();ax.plot(means.index,means.values,marker="o",label=family)
    ax.set(xlabel="realized count",ylabel="mean surrogate cost");ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(plots/"cost_vs_count.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    fixed=df[(df.variant==df.variant.min()) & (df.realized_count==1)]
    for family,group in fixed.groupby("family"):
        means=group.groupby("intensity_level")["total_cost"].mean();ax.plot(means.index,means.values,marker="o",label=family)
    ax.set(xlabel="within-family intensity level",ylabel="mean surrogate cost");ax.legend(fontsize=5,ncol=3);fig.tight_layout();fig.savefig(plots/"cost_vs_intensity.png",dpi=160);plt.close(fig)
    component=df.groupby("family")[["node_normalized","coordinate_normalized","edge_normalized"]].mean();component.plot.bar(stacked=True,figsize=(9,5)).figure.tight_layout();plt.savefig(plots/"component_contributions.png",dpi=160);plt.close()
    topology=[]
    for _,row in df.drop_duplicates("pair_id").iterrows():
        stored=str(row.path);pair_path=f"{run}/{stored}" if "::" not in stored else f"{run}/{stored.split('::',1)[0]}::{stored.split('::',1)[1]}";gt,candidate,_,_=load_pair(pair_path);before=topology_summary(gt.presence,gt.effective_adjacency);after=topology_summary(candidate.presence,candidate.effective_adjacency);topology.append({"pair_id":row.pair_id,**{f"gt_{k}":v for k,v in before.items()},**{f"corrupted_{k}":v for k,v in after.items()}})
    pd.DataFrame(topology).to_csv(run/"topology_results.csv.gz",index=False);pd.DataFrame(summaries).to_csv(run/"associations.csv",index=False)
    merged=df.merge(pd.DataFrame(topology),on="pair_id",how="left");merged["delta_beta1"]=merged.corrupted_beta1-merged.gt_beta1
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(merged.realized_count,merged.total_cost,c=merged.delta_beta1,alpha=.35,s=12);ax.set(xlabel="realized events",ylabel="surrogate cost");fig.tight_layout();fig.savefig(plots/"empirical_mixture_scatter.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));df.boxplot(column="total_cost",by="family",ax=ax,rot=90);fig.suptitle("");fig.tight_layout();fig.savefig(plots/"cost_distributions.png",dpi=160);plt.close(fig)
    weights=df.groupby("variant")[["node_normalized","coordinate_normalized","edge_normalized","total_cost"]].mean();weights.plot.bar(figsize=(7,4)).figure.tight_layout();plt.savefig(plots/"weight_sensitivity.png",dpi=160);plt.close()
    trajectories=df[(df.variant==df.variant.min()) & (df.realized_count==1)].copy();fig,ax=plt.subplots(figsize=(7,4))
    for _,group in trajectories.groupby(["patch_id","family","repetition"]):ax.plot(group.intensity_level,group.total_cost,color="tab:blue",alpha=.05)
    ax.set(xlabel="within-family intensity level",ylabel="paired surrogate cost");fig.tight_layout();fig.savefig(plots/"paired_trajectories.png",dpi=160);plt.close(fig)
    ordered=df[df.variant==df.variant.min()].sort_values("total_cost");similar=[]
    values=ordered.total_cost.to_numpy()
    for i,row in ordered.iterrows():
        candidates=ordered[(ordered.family!=row.family)&((ordered.total_cost-row.total_cost).abs()<=max(.01,.02*abs(row.total_cost)))]
        if not candidates.empty:
            other=candidates.iloc[0];similar.append({"pair_a":row.pair_id,"family_a":row.family,"cost_a":row.total_cost,"pair_b":other.pair_id,"family_b":other.family,"cost_b":other.total_cost});
        if len(similar)>=20:break
    pd.DataFrame(similar).to_csv(run/"similar_cost_different_damage.csv",index=False)
    monotonic=[]
    for keys,group in trajectories.groupby(["patch_id","family","repetition"]):
        ordered_group=group.sort_values("intensity_level");changes=np.diff(ordered_group.total_cost.to_numpy());monotonic.append({"patch_id":keys[0],"family":keys[1],"repetition":keys[2],"comparisons":len(changes),"decreases":int(np.count_nonzero(changes < -1e-12)),"ties":int(np.count_nonzero(np.abs(changes)<=1e-12))})
    pd.DataFrame(monotonic).to_csv(run/"monotonicity_diagnostics.csv",index=False)
    examples=df.drop_duplicates("pair_id").head(4);fig=plt.figure(figsize=(10,8))
    for pos,(_,row) in enumerate(examples.iterrows(),1):
        stored=str(row.path);pair_path=f"{run}/{stored}" if "::" not in stored else f"{run}/{stored.split('::',1)[0]}::{stored.split('::',1)[1]}";gt,candidate,_,_=load_pair(pair_path);ax=fig.add_subplot(2,2,pos,projection="3d");active=np.flatnonzero(candidate.presence>.05);ax.scatter(candidate.coordinates[active,2],candidate.coordinates[active,1],candidate.coordinates[active,0],c=candidate.presence[active],vmin=0,vmax=1,cmap="viridis",s=12);ax.scatter(gt.coordinates[gt.presence>.5,2],gt.coordinates[gt.presence>.5,1],gt.coordinates[gt.presence>.5,0],facecolors="none",edgecolors="black",s=18);ax.set_title(row.family,fontsize=8)
    fig.tight_layout();fig.savefig(plots/"graph_overlays_3d.png",dpi=160);plt.close(fig)
    summary={"pairs":int(df.pair_id.nunique()),"distance_rows":len(df),"families":sorted(df.family.unique()),"notes":["This is a correspondence-conditioned surrogate, not a proven metric or exact GED.","Count/intensity associations are partly built into the cost definition.","Different corruption families do not share a universal intensity scale.","Validation-derived empirical profiles are development input, not untouched final evaluation.","No detected thresholded error does not imply zero soft cost."]};atomic_json(run/"summary.json",summary)
    if os.environ.get("SLURM_JOB_ID"):
        execution_note=f"This report was completed on Jean Zay by Slurm job `{os.environ['SLURM_JOB_ID']}`. This validates the current stage only; controlled, empirical, and solver stages require their own verification."
    else:
        execution_note="This report was generated locally. Jean Zay verification remains pending until cluster outputs are supplied."
    report="# Corruption-distance study\n\n"+"\n".join(f"- {note}" for note in summary["notes"])+f"\n\n## Execution status\n\n{execution_note}\n"
    (run/"report.md").write_text(report);atomic_json(run/"COMPLETED.json",{"status":"complete","summary":"summary.json"});print(f"report complete: {run}")
if __name__=="__main__":main()

