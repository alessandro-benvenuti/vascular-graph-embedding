from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vascular_graph_embedding.graphs import Correspondence, SoftGraph


class DistancePlugin(ABC):
    name: str

    @abstractmethod
    def compute(self, gt: SoftGraph, candidate: SoftGraph, correspondence: Correspondence, **kwargs: Any) -> dict[str, Any]: ...


_REGISTRY: dict[str, type[DistancePlugin]] = {}


def register_distance(plugin: type[DistancePlugin]) -> type[DistancePlugin]:
    if not plugin.name:
        raise ValueError("distance plugin requires a non-empty name")
    _REGISTRY[plugin.name] = plugin
    return plugin


def get_distance(name: str, **kwargs: Any) -> DistancePlugin:
    if name not in _REGISTRY:
        raise KeyError(f"unknown distance plugin {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)

