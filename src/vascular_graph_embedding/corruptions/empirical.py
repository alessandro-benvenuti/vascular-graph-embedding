from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


class EmpiricalSampler:
    """Sample complete profiles and related event rows; never independent event counts."""

    def __init__(self, root: str | Path, seed: int):
        self.root = Path(root); self.rng = np.random.default_rng(seed)
        with gzip.open(self.root / "patch_profiles.csv.gz", "rt") as handle:
            self.profiles = pd.read_csv(handle)
        with gzip.open(self.root / "corruption_events.csv.gz", "rt") as handle:
            self.events = pd.read_csv(handle)

    def draw(self, *, epoch: int | None = None, approximate_gt_nodes: int | None = None) -> tuple[dict, list[dict]]:
        pool = self.profiles
        if epoch is not None: pool = pool[pool["epoch"].astype(int) == int(epoch)]
        if approximate_gt_nodes is not None and "gt_nodes" in pool:
            delta = (pool["gt_nodes"].astype(int) - approximate_gt_nodes).abs()
            pool = pool[delta <= max(5, int(delta.quantile(.25)))]
        if pool.empty: raise ValueError("no empirical profiles satisfy conditioning")
        profile = pool.iloc[int(self.rng.integers(len(pool)))].to_dict()
        keys = [k for k in ("sample_id","epoch") if k in self.events and k in profile]
        related = self.events
        for key in keys: related = related[related[key].astype(str) == str(profile[key])]
        records = related.to_dict("records")
        for index,row in enumerate(records): row["source_event_id"] = row.get("event_index", index)
        profile["source_profile_id"] = profile.get("sample_id", profile.get("profile_id"))
        profile["sampling_note"] = "complete count vector; attributes retained from source event rows"
        return profile, records

