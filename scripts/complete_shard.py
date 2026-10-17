#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import pandas as pd

from vascular_graph_embedding.data.storage import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    args = parser.parse_args()
    shard = Path(args.shard)
    pairs = pd.read_csv(shard / "pair_manifest.csv.gz")
    distances = pd.read_csv(shard / "distance_results.csv.gz")
    if pairs.pair_id.duplicated().any():
        raise RuntimeError("duplicate pair IDs within shard")
    variants = int(distances.variant.nunique())
    if len(distances) != len(pairs) * variants:
        raise RuntimeError("distance results do not cover every pair and variant")
    archives = sorted({str(value).split("::", 1)[0] for value in pairs.path if "::" in str(value)})
    archive_members = 0
    for relative in archives:
        archive = shard / relative
        with zipfile.ZipFile(archive) as handle:
            corrupt = handle.testzip()
            if corrupt is not None:
                raise RuntimeError(f"corrupt archive member: {archive}::{corrupt}")
            archive_members += len(handle.namelist())
    if archives and archive_members != len(pairs):
        raise RuntimeError(f"archive contains {archive_members} members for {len(pairs)} manifest rows")
    topology_path=shard/"topology_results.csv.gz"
    topology_rows=None
    if topology_path.is_file():
        topology=pd.read_csv(topology_path)
        if len(topology)!=len(pairs) or topology.pair_id.duplicated().any():
            raise RuntimeError("topology results do not cover every unique pair")
        topology_rows=int(len(topology))
    atomic_json(shard / "SHARD_COMPLETED.json", {
        "status": "complete", "pairs": int(len(pairs)),
        "distance_rows": int(len(distances)), "distance_variants": variants,
        "pair_archives": len(archives), "archive_members": archive_members,
        "archive_integrity": "verified" if archives else "not_applicable",
        "topology_rows": topology_rows,
    })


if __name__ == "__main__":
    main()

