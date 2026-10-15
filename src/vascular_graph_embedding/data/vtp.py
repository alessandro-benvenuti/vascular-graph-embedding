from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def normalize_coordinates(points: np.ndarray, shape: tuple[int, int, int], coordinate_space: str) -> np.ndarray:
    """Return canonical normalized D/H/W coordinates using index / axis_size."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, >=3]")
    if coordinate_space == "normalized":
        result = points[:, :3].copy()
    elif coordinate_space == "voxel":
        shape_array = np.asarray(shape, dtype=float)
        if shape_array.shape != (3,) or np.any(shape_array <= 0):
            raise ValueError("shape must contain three positive axis sizes")
        result = points[:, :3] / shape_array
    else:
        raise ValueError("coordinate_space must be 'normalized' or 'voxel'")
    return result


def _ascii_vtp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    point_array = root.find("./PolyData/Piece/Points/DataArray")
    connectivity = root.find("./PolyData/Piece/Lines/DataArray[@Name='connectivity']")
    offsets = root.find("./PolyData/Piece/Lines/DataArray[@Name='offsets']")
    if point_array is None or connectivity is None or offsets is None:
        raise ValueError("unsupported VTP: expected ASCII Points and Lines arrays")
    if point_array.get("format", "ascii") != "ascii":
        raise ValueError("non-ASCII VTP requires the optional pyvista dependency")
    points = np.fromstring(point_array.text or "", sep=" ", dtype=float).reshape(-1, 3)
    conn = np.fromstring(connectivity.text or "", sep=" ", dtype=int)
    ends = np.fromstring(offsets.text or "", sep=" ", dtype=int)
    edges, start = [], 0
    for end in ends:
        cell = conn[start:end]; start = int(end)
        if len(cell) != 2:
            raise ValueError("every VTP line cell must contain exactly two points")
        edges.append(cell)
    return points, np.asarray(edges, dtype=int).reshape(-1, 2)


def read_vtp_graph(path: str | Path, *, shape: tuple[int, int, int], coordinate_space: str = "voxel") -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    try:
        import pyvista  # type: ignore
    except ImportError:
        points, edges = _ascii_vtp(path)
    else:
        graph = pyvista.read(path)
        points = np.asarray(graph.points, dtype=float)
        flat = np.asarray(graph.lines, dtype=int)
        if flat.size % 3 or (flat.size and not np.all(flat.reshape(-1, 3)[:, 0] == 2)):
            raise ValueError("every VTP line cell must contain exactly two points")
        edges = flat.reshape(-1, 3)[:, 1:]
    return normalize_coordinates(points, shape, coordinate_space), edges

