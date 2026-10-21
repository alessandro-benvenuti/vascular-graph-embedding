from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .model import PredictionRecord, canonical_edges


EXPECTED_SLOTS = 120
EXPECTED_PAIRS = EXPECTED_SLOTS * (EXPECTED_SLOTS - 1) // 2


def softmax_positive(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim < 2 or logits.shape[-1] != 2 or not np.isfinite(logits).all():
        raise ValueError("binary logits must be finite with final dimension 2")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values[..., 1] / values.sum(axis=-1)


def torch_load(path: Path):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to read RelationFormer raw shards; load the Jean Zay "
            "pytorch-gpu module or install torch in the environment"
        ) from error
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_edge_pairs(root: Path) -> np.ndarray:
    pairs = np.asarray(torch_load(Path(root) / "edge_pairs.pt"), dtype=np.int64)
    pairs = canonical_edges(pairs, EXPECTED_SLOTS)
    expected = np.asarray(
        [(i, j) for i in range(EXPECTED_SLOTS) for j in range(i + 1, EXPECTED_SLOTS)],
        dtype=np.int64,
    )
    if pairs.shape != (EXPECTED_PAIRS, 2) or not np.array_equal(pairs, expected):
        raise ValueError("edge_pairs.pt must contain all 7,140 canonical query pairs")
    return pairs


def _array(record, key, dtype=None):
    if key not in record:
        raise KeyError(f"raw prediction is missing {key!r}")
    value = record[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def parse_record(raw: dict, edge_pairs: np.ndarray) -> PredictionRecord:
    coordinates = _array(raw, "nodes", np.float64).reshape(-1, 3)
    node_logits = _array(raw, "node_logits", np.float64)
    gt_nodes = _array(raw, "gt_nodes", np.float64).reshape(-1, 3)
    gt_edges = canonical_edges(_array(raw, "gt_edges", np.int64), len(gt_nodes))
    forward = _array(raw, "edge_logits_forward", np.float64)
    reverse = _array(raw, "edge_logits_reverse", np.float64)
    matched_queries = _array(raw, "matched_queries", np.int64).reshape(-1)
    matched_gt = _array(raw, "matched_gt", np.int64).reshape(-1)
    shape = tuple(int(value) for value in raw["model_input_shape"])

    if coordinates.shape != (EXPECTED_SLOTS, 3):
        raise ValueError("each prediction must contain exactly 120 query coordinates")
    if node_logits.shape != (EXPECTED_SLOTS, 2):
        raise ValueError("node_logits must have shape [120,2]")
    if forward.shape != (EXPECTED_PAIRS, 2) or reverse.shape != (EXPECTED_PAIRS, 2):
        raise ValueError("each edge-logit direction must have shape [7140,2]")
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("model_input_shape must contain three positive axes")
    for name, value in (
        ("coordinates", coordinates), ("GT coordinates", gt_nodes),
        ("node logits", node_logits), ("forward logits", forward),
        ("reverse logits", reverse),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contain non-finite values")
    if matched_queries.shape != matched_gt.shape:
        raise ValueError("saved matching arrays have different lengths")
    if len(np.unique(matched_queries)) != len(matched_queries) or len(np.unique(matched_gt)) != len(matched_gt):
        raise ValueError("saved matching must be one-to-one")
    if np.any(matched_queries < 0) or np.any(matched_queries >= EXPECTED_SLOTS):
        raise ValueError("matched query index is out of range")
    if np.any(matched_gt < 0) or np.any(matched_gt >= len(gt_nodes)):
        raise ValueError("matched GT index is out of range")
    if len(matched_gt) != len(gt_nodes) or set(matched_gt.tolist()) != set(range(len(gt_nodes))):
        raise ValueError("every GT node must be represented exactly once in saved matching")

    node_probabilities = softmax_positive(node_logits)
    edge_logits = 0.5 * (forward + reverse)
    edge_probabilities = softmax_positive(edge_logits)
    effective = (
        node_probabilities[edge_pairs[:, 0]]
        * node_probabilities[edge_pairs[:, 1]]
        * edge_probabilities
    )
    for name, value in (("node", node_probabilities), ("edge", edge_probabilities), ("effective edge", effective)):
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError(f"invalid {name} probabilities")

    return PredictionRecord(
        sample_id=str(raw["sample_id"]), checkpoint=str(raw["checkpoint"]),
        epoch=int(raw["epoch"]), coordinates=coordinates, node_logits=node_logits,
        node_probabilities=node_probabilities, gt_nodes=gt_nodes, gt_edges=gt_edges,
        edge_pairs=edge_pairs, edge_logits_forward=forward, edge_logits_reverse=reverse,
        edge_probabilities=edge_probabilities, effective_edge_probabilities=effective,
        matched_queries=matched_queries, matched_gt=matched_gt, model_input_shape=shape,
    )


def iter_epoch_records(root: Path, epoch: int) -> Iterator[tuple[Path, int, PredictionRecord]]:
    root = Path(root)
    pairs = load_edge_pairs(root)
    raw_dir = root / f"epoch_{int(epoch):04d}" / "raw"
    shards = sorted(raw_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no raw shards found in {raw_dir}")
    for shard in shards:
        records = torch_load(shard)
        if not isinstance(records, list):
            raise TypeError(f"{shard} does not contain a list of prediction records")
        for offset, raw in enumerate(records):
            yield shard, offset, parse_record(raw, pairs)


def build_prediction_index(root: Path, epochs: list[int], output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    memberships: dict[int, set[str]] = {}
    for epoch in epochs:
        seen: set[str] = set()
        for shard, offset, record in iter_epoch_records(root, epoch):
            if record.sample_id in seen:
                raise ValueError(f"duplicate sample {record.sample_id!r} at epoch {epoch}")
            seen.add(record.sample_id)
            rows.append({
                "sample_id": record.sample_id, "epoch": int(epoch),
                "checkpoint": record.checkpoint, "shard": str(shard), "offset": offset,
                "shard_size_bytes": shard.stat().st_size,
                "shard_mtime_ns": shard.stat().st_mtime_ns,
                "gt_nodes": len(record.gt_nodes), "gt_edges": len(record.gt_edges),
            })
        memberships[int(epoch)] = seen
    reference_epoch = int(epochs[0])
    reference = memberships[reference_epoch]
    for epoch, values in memberships.items():
        if values != reference:
            missing = sorted(reference - values)[:5]
            extra = sorted(values - reference)[:5]
            raise ValueError(f"sample IDs differ at epoch {epoch}; missing={missing}, extra={extra}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(output)
    return rows


def load_indexed_record(index_row: dict[str, str], root: Path) -> PredictionRecord:
    pairs = load_edge_pairs(root)
    shard = Path(index_row["shard"]); stat = shard.stat()
    if "shard_size_bytes" in index_row and int(index_row["shard_size_bytes"]) != stat.st_size:
        raise RuntimeError(f"indexed shard size changed after preflight: {shard}")
    if "shard_mtime_ns" in index_row and int(index_row["shard_mtime_ns"]) != stat.st_mtime_ns:
        raise RuntimeError(f"indexed shard modification time changed after preflight: {shard}")
    records = torch_load(shard)
    return parse_record(records[int(index_row["offset"])], pairs)
