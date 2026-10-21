#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from vascular_graph_embedding.data.storage import atomic_json
from vascular_graph_embedding.experiments.common import config_hash, git_sha
from vascular_graph_embedding.ged.selection import select_samples


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp"); frame.to_csv(temporary, index=False); temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--run", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--threshold", action="append", help="NODE,EDGE; may be repeated")
    parser.add_argument("--variant", action="append", choices=("structural", "geometric"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); run = Path(args.run)
    if args.sample_count is not None: config["stage"]["sample_count"] = args.sample_count
    if args.timeout_seconds is not None: config["solver"]["timeout_seconds"] = args.timeout_seconds
    if args.threshold:
        config["thresholds"] = [dict(zip(("node", "edge"), map(float, value.split(",")))) for value in args.threshold]
    if args.variant: config["variants"] = args.variant
    relationformer_path = Path(config["data"]["relationformer_config_path"])
    if not relationformer_path.is_file():
        raise FileNotFoundError(f"saved RelationFormer resolved config is missing: {relationformer_path}")
    config["relationformer_config"] = yaml.safe_load(relationformer_path.read_text())
    if args.overwrite and run.exists(): shutil.rmtree(run)
    if (run / "COMPLETED.json").is_file() and not args.overwrite: raise FileExistsError(f"completed run exists: {run}")
    if run.exists() and any(run.iterdir()) and not args.overwrite:
        if (run / "task_manifest.csv").is_file(): print(f"prepared run already exists: {run}"); return
        raise FileExistsError(f"nonempty incomplete run exists: {run}")
    run.mkdir(parents=True, exist_ok=True); (run / "results").mkdir(exist_ok=True)
    output_root = Path(config["output_root"]); index_path = output_root / "prediction_index.csv"
    if not index_path.is_file(): raise FileNotFoundError(f"run preflight first; missing {index_path}")
    index = pd.read_csv(index_path, dtype={"sample_id": str})
    selected = select_samples(
        Path(config["data"]["prediction_root"]), list(map(int, config["epochs"])),
        int(config["stage"]["sample_count"]), int(config["selection_seed"]),
    )
    available = set(zip(index.sample_id.astype(str), index.epoch.astype(int)))
    missing = [(str(sample), int(epoch)) for sample in selected.sample_id for epoch in config["epochs"] if (str(sample), int(epoch)) not in available]
    if missing: raise ValueError(f"selected predictions missing from raw index: {missing[:5]}")
    atomic_csv(selected, run / "sample_manifest.csv")
    tasks = []; task_id = 0
    for sample_id in selected.sample_id.astype(str):
        for epoch in map(int, config["epochs"]):
            for threshold in config["thresholds"]:
                for variant in config["variants"]:
                    tasks.append({
                        "task_id": task_id, "sample_id": sample_id, "epoch": epoch,
                        "node_threshold": float(threshold["node"]), "edge_threshold": float(threshold["edge"]),
                        "variant": str(variant), "timeout_seconds": float(config["solver"]["timeout_seconds"]),
                    }); task_id += 1
    atomic_csv(pd.DataFrame(tasks), run / "task_manifest.csv")
    resolved = run / "resolved_config.yaml"; temporary = resolved.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False)); temporary.replace(resolved)
    repository = Path(os.environ.get("VGE_REPO_DIR", Path.cwd())).resolve()
    atomic_json(run / "run_metadata.json", {
        "schema": "discrete-ged-run/v1", "git_sha": git_sha(repository),
        "config_hash": config_hash(config), "python": platform.python_version(),
        "input_root": config["data"]["prediction_root"], "prediction_index": str(index_path),
        "tasks": len(tasks), "samples": len(selected), "stage": config["stage"]["name"],
        "prediction_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "relationformer_config_sha256": hashlib.sha256(relationformer_path.read_bytes()).hexdigest(),
        "host": platform.node(), "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    print(json.dumps({"run": str(run), "samples": len(selected), "tasks": len(tasks)}, indent=2))


if __name__ == "__main__": main()
