from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vascular_graph_embedding.graphs import Correspondence, SoftGraph


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_pair(path: Path, gt: SoftGraph, candidate: SoftGraph, correspondence: Correspondence, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    arrays = {f"gt_{k}": v for k,v in gt.to_npz_dict().items()}
    arrays.update({f"candidate_{k}": v for k,v in candidate.to_npz_dict().items()})
    arrays["slot_to_gt"] = correspondence.slot_to_gt
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_pair(path: Path) -> tuple[SoftGraph, SoftGraph, Correspondence, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        def graph(prefix: str) -> SoftGraph:
            return SoftGraph(z[f"{prefix}_coordinates"], z[f"{prefix}_presence"], z[f"{prefix}_conditional_edges"], z[f"{prefix}_node_ids"].astype(str).tolist())
        return graph("gt"), graph("candidate"), Correspondence(z["slot_to_gt"]), json.loads(str(z["metadata_json"]))

