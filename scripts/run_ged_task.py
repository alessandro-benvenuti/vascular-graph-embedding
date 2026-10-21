#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from vascular_graph_embedding.data.storage import atomic_json
from vascular_graph_embedding.ged.baseline import saved_matching_cost
from vascular_graph_embedding.ged.extraction import extract_graphs
from vascular_graph_embedding.ged.loader import load_indexed_record
from vascular_graph_embedding.ged.losses import reconstruct_losses
from vascular_graph_embedding.ged.metrics import structural_metrics
from vascular_graph_embedding.ged.solver import GurobiGraphEditDistance


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True)
    parser.add_argument("--task-id", required=True, type=int); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); run = Path(args.run); output = run / "results" / f"task_{args.task_id:06d}.json"
    if output.is_file() and not args.overwrite:
        existing = json.loads(output.read_text())
        if existing.get("record_status") == "complete": print(f"valid task already complete: {output}"); return
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    tasks = pd.read_csv(run / "task_manifest.csv"); selected = tasks[tasks.task_id == args.task_id]
    if len(selected) != 1: raise KeyError(f"task {args.task_id} is absent or duplicated")
    task = selected.iloc[0]; index = pd.read_csv(Path(config["output_root"]) / "prediction_index.csv", dtype={"sample_id": str})
    found = index[(index.sample_id == str(task.sample_id)) & (index.epoch == int(task.epoch))]
    if len(found) != 1: raise KeyError(f"prediction index lookup failed for {task.sample_id}, epoch {task.epoch}")
    record = load_indexed_record(found.iloc[0].to_dict(), Path(config["data"]["prediction_root"]))
    predicted, truth = extract_graphs(record, float(task.node_threshold), float(task.edge_threshold))
    graph_path = run / "hard_graphs" / f"task_{args.task_id:06d}.npz"
    graph_path.parent.mkdir(exist_ok=True)
    graph_temporary = graph_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        graph_temporary,
        predicted_coordinates=predicted.coordinates,
        predicted_source_slots=predicted.source_slots,
        predicted_edges=predicted.edges,
        predicted_edge_raw_probabilities=predicted.edge_probabilities,
        predicted_edge_effective_probabilities=predicted.effective_edge_probabilities,
        gt_coordinates=truth.coordinates,
        gt_edges=truth.edges,
    )
    graph_temporary.replace(graph_path)
    metrics = structural_metrics(predicted, truth, record); losses = reconstruct_losses(record, config["relationformer_config"])
    variant = str(task.variant); edge_weight = float(config["costs"]["edge_weight"])
    baseline = saved_matching_cost(predicted, truth, record, variant=variant, edge_weight=edge_weight)
    solver = GurobiGraphEditDistance(
        variant=variant, edge_weight=edge_weight, timeout=float(task.timeout_seconds),
        seed=int(config["solver"]["seed"]), initial_mapping=baseline["saved_matching_node_mapping"],
    )
    result = solver.compute(predicted, truth)
    if result.status == "optimal" and result.distance is not None and result.distance > baseline["saved_matching_edit_cost"] + 1e-6:
        raise RuntimeError(
            f"optimized GED {result.distance} exceeds saved-matching upper bound {baseline['saved_matching_edit_cost']}"
        )
    baseline_row = {
        "saved_matching_edit_cost": baseline["saved_matching_edit_cost"],
        "saved_matching_node_mapping": baseline["saved_matching_node_mapping"],
        **{f"saved_matching_{key}": value for key, value in baseline.items() if key not in {"saved_matching_edit_cost", "saved_matching_node_mapping"}},
    }
    row = {
        "record_status": "complete", "task_id": int(task.task_id), "sample_id": record.sample_id,
        "patient_id": "unknown", "checkpoint": record.checkpoint, "epoch": record.epoch,
        "node_threshold": float(task.node_threshold), "edge_threshold": float(task.edge_threshold),
        "ged_variant": variant, "edge_weight": edge_weight,
        "hard_graph_path": str(graph_path.relative_to(run)),
        "predicted_edge_raw_probability_mean": float(predicted.edge_probabilities.mean()) if predicted.edge_count else None,
        "predicted_edge_effective_probability_mean": float(predicted.effective_edge_probabilities.mean()) if predicted.edge_count else None,
        **result.as_dict(), **baseline_row,
        **losses, **metrics, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    manifest = pd.read_csv(run / "sample_manifest.csv", dtype={"sample_id": str})
    identity = manifest[manifest.sample_id == record.sample_id]
    if len(identity) == 1: row["patient_id"] = str(identity.iloc[0].get("patient_id", "unknown"))
    atomic_json(output, row); print(json.dumps({key: row[key] for key in ("task_id", "sample_id", "epoch", "ged_variant", "status", "distance", "relative_gap")}, indent=2))


if __name__ == "__main__": main()
