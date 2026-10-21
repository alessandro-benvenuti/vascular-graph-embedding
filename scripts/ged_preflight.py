#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import yaml

from vascular_graph_embedding.data.storage import atomic_json
from vascular_graph_embedding.ged.loader import build_prediction_index
from vascular_graph_embedding.ged.model import HardGraph
from vascular_graph_embedding.ged.solver import GurobiGraphEditDistance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["data"]["prediction_root"])
    output_root = Path(config["output_root"]); output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "prediction_index.csv"

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise RuntimeError("gurobipy is unavailable; load gurobi/13.0.1 and activate the GED environment") from error
    model = gp.Model("license_preflight"); model.Params.OutputFlag = 0
    value = model.addVar(lb=0, ub=1, vtype=GRB.BINARY); model.setObjective(value, GRB.MINIMIZE); model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi license preflight failed with status {model.Status}")

    tiny_a = HardGraph([[0, 0, 0], [1, 0, 0]], [[0, 1]])
    tiny_b = HardGraph([[0, 0, 0], [1, 0, 0]], [[0, 1]])
    result = GurobiGraphEditDistance(timeout=10).compute(tiny_a, tiny_b)
    if result.status != "optimal" or abs(float(result.distance or 0)) > 1e-8:
        raise RuntimeError(f"tiny GED preflight failed: {result.as_dict()}")

    rows = build_prediction_index(root, list(map(int, config["epochs"])), index_path)
    atomic_json(output_root / "PREFLIGHT_COMPLETED.json", {
        "status": "complete", "gurobi_version": ".".join(map(str, gp.gurobi.version())),
        "prediction_index": str(index_path), "records": len(rows),
        "epochs": list(map(int, config["epochs"])), "unique_samples": len({row["sample_id"] for row in rows}),
        "edge_pairs_sha256": hashlib.sha256((root / "edge_pairs.pt").read_bytes()).hexdigest(),
    })
    print(json.dumps(json.loads((output_root / "PREFLIGHT_COMPLETED.json").read_text()), indent=2))


if __name__ == "__main__": main()
