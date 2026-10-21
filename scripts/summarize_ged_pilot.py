#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

parser=argparse.ArgumentParser();parser.add_argument("--run",required=True);args=parser.parse_args();run=Path(args.run)
frame=pd.read_csv(run/"solver_diagnostics.csv")
summary={"solves":len(frame),"optimal_rate":float((frame.status=="optimal").mean()),"timeouts":int((frame.status=="timeout").sum()),"median_runtime_seconds":float(frame.runtime_seconds.median()),"p95_runtime_seconds":float(frame.runtime_seconds.quantile(.95)),"median_gap":float(frame.relative_gap.dropna().median()) if frame.relative_gap.notna().any() else None,"max_predicted_nodes":int(frame.predicted_node_count.max())}
print(json.dumps(summary,indent=2))
