from __future__ import annotations

import json
import io
import atexit
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from vascular_graph_embedding.graphs import Correspondence, SoftGraph

_ARCHIVE_CACHE: dict[str, zipfile.ZipFile] = {}


@atexit.register
def _close_archives() -> None:
    for handle in _ARCHIVE_CACHE.values():
        handle.close()
    _ARCHIVE_CACHE.clear()


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


def _pair_arrays(gt: SoftGraph, candidate: SoftGraph, correspondence: Correspondence, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = {f"gt_{k}": v for k,v in gt.to_npz_dict().items()}
    arrays.update({f"candidate_{k}": v for k,v in candidate.to_npz_dict().items()})
    arrays["slot_to_gt"] = correspondence.slot_to_gt
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return arrays


class PairArchiveWriter:
    """Atomically create one ZIP container for many compressed pair NPZ records."""

    def __init__(self, path: Path):
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.handle: zipfile.ZipFile | None = None

    def __enter__(self) -> "PairArchiveWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = zipfile.ZipFile(self.temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
        return self

    def write(self, member: str, gt: SoftGraph, candidate: SoftGraph, correspondence: Correspondence, metadata: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("pair archive is not open")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **_pair_arrays(gt, candidate, correspondence, metadata))
        self.handle.writestr(member, buffer.getvalue())

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            self.handle.close()
        if exc_type is None:
            if self.temporary.exists():
                self.temporary.replace(self.path)
            elif self.path.exists():
                # Treat repeated finalization as success only when the already
                # published archive is structurally intact.
                with zipfile.ZipFile(self.path) as published:
                    corrupt = published.testzip()
                if corrupt is not None:
                    raise RuntimeError(f"published pair archive is corrupt at {corrupt}")
            else:
                raise FileNotFoundError(f"neither temporary nor published pair archive exists for {self.path}")


def load_pair(path: Path | str) -> tuple[SoftGraph, SoftGraph, Correspondence, dict[str, Any]]:
    value = str(path)
    if "::" in value:
        archive, member = value.split("::", 1)
        handle = _ARCHIVE_CACHE.setdefault(archive, zipfile.ZipFile(archive))
        source: Path | io.BytesIO = io.BytesIO(handle.read(member))
    else:
        source = Path(value)
    with np.load(source, allow_pickle=False) as z:
        def graph(prefix: str) -> SoftGraph:
            return SoftGraph(z[f"{prefix}_coordinates"], z[f"{prefix}_presence"], z[f"{prefix}_conditional_edges"], z[f"{prefix}_node_ids"].astype(str).tolist())
        return graph("gt"), graph("candidate"), Correspondence(z["slot_to_gt"]), json.loads(str(z["metadata_json"]))

