#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from vascular_graph_embedding.data.storage import atomic_json


TOPOLOGY_METRICS = ("abs_delta_beta0", "abs_delta_beta1", "topology_l1")


def correlations(frame: pd.DataFrame, metric: str) -> dict[str, float | int]:
    clean = frame[[metric, "total_cost"]].dropna()
    if len(clean) < 2 or clean[metric].nunique() < 2:
        return {"n": len(clean), "pearson": np.nan, "spearman": np.nan}
    return {
        "n": len(clean),
        "pearson": float(pearsonr(clean[metric], clean["total_cost"]).statistic),
        "spearman": float(spearmanr(clean[metric], clean["total_cost"]).statistic),
    }


def analyse(distances: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    required_distance = {"pair_id", "variant", "family", "total_cost"}
    required_topology = {
        "pair_id",
        "gt_beta0",
        "gt_beta1",
        "corrupted_beta0",
        "corrupted_beta1",
    }
    missing_distance = required_distance.difference(distances.columns)
    missing_topology = required_topology.difference(topology.columns)
    if missing_distance:
        raise ValueError(f"distance table is missing columns: {sorted(missing_distance)}")
    if missing_topology:
        raise ValueError(f"topology table is missing columns: {sorted(missing_topology)}")
    if topology["pair_id"].duplicated().any():
        raise ValueError("topology table contains duplicate pair IDs")

    frame = distances.merge(topology, on="pair_id", how="inner", validate="many_to_one")
    expected = distances["pair_id"].nunique()
    observed = frame["pair_id"].nunique()
    if observed != expected:
        raise ValueError(f"topology covers {observed} of {expected} distance pairs")

    frame["abs_delta_beta0"] = (frame["corrupted_beta0"] - frame["gt_beta0"]).abs()
    frame["abs_delta_beta1"] = (frame["corrupted_beta1"] - frame["gt_beta1"]).abs()
    frame["topology_l1"] = frame["abs_delta_beta0"] + frame["abs_delta_beta1"]

    rows: list[dict[str, object]] = []
    for variant, variant_frame in frame.groupby("variant", sort=True):
        for metric in TOPOLOGY_METRICS:
            rows.append({
                "scope": "all_families",
                "variant": int(variant),
                "family": "all",
                "topology_metric": metric,
                **correlations(variant_frame, metric),
            })
        for family, family_frame in variant_frame.groupby("family", sort=True):
            for metric in TOPOLOGY_METRICS:
                rows.append({
                    "scope": "within_family",
                    "variant": int(variant),
                    "family": family,
                    "topology_metric": metric,
                    **correlations(family_frame, metric),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    distance_path = run / "distance_results.csv.gz"
    topology_path = run / "topology_results.csv.gz"
    output_path = run / "topology_ged_associations.csv"

    distances = pd.read_csv(distance_path, low_memory=False)
    topology = pd.read_csv(topology_path, low_memory=False)
    result = analyse(distances, topology)

    temporary = output_path.with_name(output_path.name + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(output_path)

    primary_variant = int(result["variant"].min())
    overall = result[
        (result["scope"] == "all_families")
        & (result["variant"] == primary_variant)
    ]
    print(f"overall topology-GED associations for primary variant {primary_variant}:")
    print(overall.to_string(index=False))

    marker = run / "TOPOLOGY_ASSOCIATIONS_COMPLETED.json"
    atomic_json(marker, {
        "status": "complete",
        "output": output_path.name,
        "rows": int(len(result)),
        "pairs": int(distances["pair_id"].nunique()),
        "variants": int(distances["variant"].nunique()),
    })
    print(f"saved {len(result)} association rows to {output_path}")


if __name__ == "__main__":
    main()
