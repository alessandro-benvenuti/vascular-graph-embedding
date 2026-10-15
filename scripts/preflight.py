#!/usr/bin/env python3
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np

from vascular_graph_embedding.data.storage import load_pair, save_pair
from vascular_graph_embedding.graphs.model import clean_fixed_graph


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--output");args=parser.parse_args()
    import yaml
    config=yaml.safe_load(Path(args.config).read_text()); checks=[]
    for key,path in config.get("data",{}).items():
        if path in (None,""):continue
        checks.append({"check":f"path:{key}","ok":Path(str(path)).exists(),"value":str(path)})
    checks.extend({"check":f"dependency:{name}","ok":importlib.util.find_spec(name) is not None} for name in ("numpy","pandas","yaml","scipy","matplotlib"))
    checks.append({"check":"solver:gurobipy","ok":importlib.util.find_spec("gurobipy") is not None,"optional":True})
    graph,corr=clean_fixed_graph(np.array([[0,0,0],[.1,.2,.3]]),np.array([[0,1]]),slots=4)
    target=Path(args.output or "/tmp/vge_preflight_pair.npz");save_pair(target,graph,graph,corr,{"round_trip":True});loaded=load_pair(target)
    checks.append({"check":"graph_round_trip","ok":np.array_equal(loaded[1].effective_adjacency,graph.effective_adjacency)})
    data=config.get("data",{}); checks.append({"check":"split_is_train","ok":str(data.get("split","train")).lower()=="train"})
    failed=[x for x in checks if not x["ok"] and not x.get("optional")]
    print(json.dumps({"checks":checks,"ok":not failed},indent=2));return bool(failed)

if __name__=="__main__":raise SystemExit(main())

