from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{int(seed)}|{value}".encode()).hexdigest()


def _read_epoch_tables(root: Path, epochs: list[int]) -> pd.DataFrame:
    frames = []
    for epoch in epochs:
        path = Path(root) / f"epoch_{epoch:04d}" / "patches.csv"
        if not path.is_file(): raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False); frame["epoch"] = int(epoch)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    memberships = data.groupby("epoch")["sample_id"].apply(lambda values: set(map(str, values)))
    reference = memberships.iloc[0]
    if any(values != reference for values in memberships.iloc[1:]):
        raise ValueError("patch tables do not contain identical sample IDs across epochs")
    data["sample_id"] = data["sample_id"].astype(str)
    return data


def select_samples(root: Path, epochs: list[int], count: int, seed: int) -> pd.DataFrame:
    data = _read_epoch_tables(root, epochs)
    first, last = min(epochs), max(epochs)
    early = data[data.epoch == first].set_index("sample_id")
    late = data[data.epoch == last].set_index("sample_id")
    common = sorted(set(early.index) & set(late.index))
    table = early.loc[common].copy()
    audit_columns = [name for name in ("node_cost", "coordinate_cost", "edge_effective_cost") if name in table]
    table["audit_score"] = sum(pd.to_numeric(table[name], errors="coerce").fillna(0) for name in audit_columns)
    late_score = pd.Series(0.0, index=common)
    for name in audit_columns:
        late_score = late_score.add(
            pd.to_numeric(late.loc[common, name], errors="coerce").fillna(0),
            fill_value=0,
        )
    table["audit_change_early_to_late"] = late_score.to_numpy() - table["audit_score"].to_numpy()
    def numeric(*names: str) -> pd.Series:
        for name in names:
            if name in table:
                return pd.to_numeric(table[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=table.index)
    size = numeric("gt_nodes")
    beta0 = numeric("t0.5_components", "t0.5_beta0")
    beta1 = numeric("t0.5_cycle_rank", "t0.5_beta1")
    false_loops = numeric("t0.5_false_loop_edges")
    gt_components = numeric("gt_components", "gt_beta0")
    table["loss_quantile"] = pd.qcut(table.audit_score.rank(method="first"), 4, labels=False, duplicates="drop")
    table["size_quantile"] = pd.qcut(size.rank(method="first"), 3, labels=False, duplicates="drop")
    table["topology_stratum"] = np.select(
        [false_loops > 0, beta0 > gt_components, beta1 > numeric("gt_cycle_rank", "gt_beta1")],
        ["false_loop", "fragmented", "cycle_excess"], default="no_detected_topology_error",
    )
    table["change_stratum"] = np.where(table.audit_change_early_to_late <= 0, "improved", "worsened")
    table["selection_stratum"] = (
        "loss" + table.loss_quantile.astype(str) + "_size" + table.size_quantile.astype(str)
        + "_" + table.topology_stratum.astype(str) + "_" + table.change_stratum.astype(str)
    )
    table["selection_rank"] = [stable_rank(seed, sample_id) for sample_id in table.index]
    ordered = table.reset_index().sort_values(["selection_stratum", "selection_rank"])
    chosen = []
    groups = [group for _, group in ordered.groupby("selection_stratum", sort=True)]
    position = 0
    while len(chosen) < min(count, len(ordered)):
        progress = False
        for group in groups:
            if position < len(group):
                chosen.append(group.iloc[position]); progress = True
                if len(chosen) == min(count, len(ordered)): break
        if not progress: break
        position += 1
    result = pd.DataFrame(chosen)
    provenance = {}
    manifest_path = Path(root) / "validation_manifest.json"
    if manifest_path.is_file():
        import json
        for row in json.loads(manifest_path.read_text()): provenance[str(row["sample_id"])] = row.get("provenance") or {}
    result["patient_id"] = [
        provenance.get(str(sample_id), {}).get("patient_id", provenance.get(str(sample_id), {}).get("source_id", "unknown"))
        for sample_id in result.sample_id
    ]
    result["selection_seed"] = int(seed); result["available_epochs"] = ",".join(map(str, epochs))
    keep = [
        "sample_id", "patient_id", "selection_stratum", "selection_seed", "available_epochs",
        "gt_nodes", "gt_edges", "audit_score", "audit_change_early_to_late", *audit_columns,
    ]
    return result[[name for name in keep if name in result]].reset_index(drop=True)
