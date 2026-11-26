"""Private helper for copying Rustworkx graphs together with their payloads."""

from __future__ import annotations

import copy
from typing import Any, TypeVar

import rustworkx as rx

GraphT = TypeVar("GraphT", rx.PyGraph, rx.PyDiGraph)

# Payload values of these types are immutable and may be shared between copies.
_IMMUTABLE_TYPES = frozenset({str, int, float, bool, bytes, type(None)})


def _copy_payload(payload: Any) -> Any:
    """Return an independent copy of one node or edge payload.

    Payloads are normally flat dicts of scalars and strings, so each value is
    kept as-is when its type is immutable. Anything else (for example a list
    from a Polars list column, or a tuple that might hold one) is deep-copied,
    which keeps the common case cheap and the rare nested case correct.
    """
    if isinstance(payload, dict):
        return {
            key: value if type(value) in _IMMUTABLE_TYPES else copy.deepcopy(value)
            for key, value in payload.items()
        }
    if type(payload) in _IMMUTABLE_TYPES:
        return payload
    return copy.deepcopy(payload)


def copy_graph_with_payloads(graph: GraphT) -> GraphT:
    """Copy a Rustworkx graph and give every node and edge its own payload copy.

    ``graph.copy()`` is shallow: the copy shares payload objects with the
    original, so mutating a payload dict on one is visible on the other. This
    keeps node and edge indices identical, including holes left by removed
    nodes, then replaces each node and edge payload. Graph-level ``.attrs`` is
    not copied and stays shared with the original.
    """
    duplicate = graph.copy()
    for node_index in duplicate.node_indices():
        duplicate[node_index] = _copy_payload(duplicate[node_index])
    for edge_index in duplicate.edge_indices():
        duplicate.update_edge_by_index(
            edge_index, _copy_payload(duplicate.get_edge_data_by_index(edge_index))
        )
    return duplicate
