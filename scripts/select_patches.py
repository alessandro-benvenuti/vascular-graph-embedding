#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from vascular_graph_embedding.data.manifest import load_patch_index, stratified_select


def main():
    p=argparse.ArgumentParser();p.add_argument("--patch-index",required=True);p.add_argument("--output",required=True);p.add_argument("--count",type=int,required=True);p.add_argument("--seed",type=int,default=0);a=p.parse_args()
    selected=stratified_select(load_patch_index(a.patch_index),a.count,a.seed);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);selected.to_csv(out,index=False);print(f"selected {len(selected)} training patches -> {out}")
if __name__=="__main__":main()

