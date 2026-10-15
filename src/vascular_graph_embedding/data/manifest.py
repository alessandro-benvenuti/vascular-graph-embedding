from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def stratified_select(frame: pd.DataFrame, count: int, seed: int, *, patient_column: str = "patient_id") -> pd.DataFrame:
    """Deterministic size-stratified, patient-balanced selection."""
    if count <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    data = frame.copy()
    if patient_column not in data: data[patient_column] = "unknown"
    for column in ("gt_nodes", "gt_edges", "components", "mean_edge_length", "bifurcations", "has_cycle"):
        if column not in data: data[column] = 0
    for column in ("gt_nodes", "gt_edges", "mean_edge_length", "bifurcations"):
        try: data[column + "_stratum"] = pd.qcut(data[column], min(4, data[column].nunique()), duplicates="drop").astype(str)
        except ValueError: data[column + "_stratum"] = "all"
    strata = ["gt_nodes_stratum", "gt_edges_stratum", "components", "bifurcations_stratum", "has_cycle"]
    payload = data.astype(str).agg("\0".join, axis=1) + f"\0{seed}"
    data["_rank"] = payload.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    data = data.sort_values([patient_column, *strata, "_rank"])
    selected, patient_counts = [], {}
    groups = [group for _, group in data.groupby(strata, dropna=False, sort=True)]
    while len(selected) < min(count, len(data)):
        progress = False
        for group in groups:
            available = group[~group.index.isin(selected)]
            if available.empty: continue
            best = min(available.index, key=lambda idx: (patient_counts.get(str(data.at[idx,patient_column]),0), data.at[idx,"_rank"]))
            selected.append(best); patient=str(data.at[best,patient_column]);patient_counts[patient]=patient_counts.get(patient,0)+1;progress=True
            if len(selected) >= min(count,len(data)): break
        if not progress: break
    result = data.loc[selected].copy()
    result["selection_order"] = np.arange(len(result)); result["selection_seed"] = seed
    result["selection_logic"] = "proportional strata round-robin with patient-load tie break"
    return result.drop(columns=["_rank"])


def load_patch_index(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    lowered = {str(c).lower(): c for c in frame.columns}
    for candidate in ("split", "dataset_split", "partition"):
        if candidate in lowered:
            frame = frame[frame[lowered[candidate]].astype(str).str.lower() == "train"]
            break
    return frame.reset_index(drop=True)

