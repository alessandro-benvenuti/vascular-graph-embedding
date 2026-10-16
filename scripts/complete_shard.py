#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vascular_graph_embedding.data.storage import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    args = parser.parse_args()
    shard = Path(args.shard)
    pairs = pd.read_csv(shard / "pair_manifest.csv.gz")
    distances = pd.read_csv(shard / "distance_results.csv.gz")
    atomic_json(shard / "SHARD_COMPLETED.json", {
        "status": "complete", "pairs": int(len(pairs)),
        "distance_rows": int(len(distances)),
    })


if __name__ == "__main__":
    main()

