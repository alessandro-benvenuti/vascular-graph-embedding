#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vascular_graph_embedding.data.storage import load_pair
from vascular_graph_embedding.graphs.topology import topology_summary


def pair_path(run: Path, stored: str) -> str:
    if "::" not in stored:
        return str(run / stored)
    archive, member = stored.split("::", 1)
    return f"{run / archive}::{member}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    manifest = pd.read_csv(run / "pair_manifest.csv.gz")
    rows=[]
    for record in manifest.itertuples(index=False):
        gt,candidate,_,_=load_pair(pair_path(run,str(record.path)))
        before=topology_summary(gt.presence,gt.effective_adjacency)
        after=topology_summary(candidate.presence,candidate.effective_adjacency)
        rows.append({"pair_id":record.pair_id,**{f"gt_{key}":value for key,value in before.items()},**{f"corrupted_{key}":value for key,value in after.items()}})
    result=pd.DataFrame(rows)
    if len(result)!=len(manifest) or result.pair_id.duplicated().any():
        raise RuntimeError("topology output does not cover every unique pair")
    target=run/"topology_results.csv.gz";temporary=target.with_name(target.name+".tmp")
    result.to_csv(temporary,index=False,compression="gzip");temporary.replace(target)
    print(f"wrote {len(result)} topology rows to {target}")


if __name__ == "__main__":
    main()

