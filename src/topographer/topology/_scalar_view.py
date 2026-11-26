"""Read a scalar network through its public surface.

Topology code uses this instead of ``ScalarNetwork`` private attributes. The returned
Rustworkx graph is the defensive copy from ``ScalarNetwork.graph``, so node and edge
indices match the network snapshot.
"""

from __future__ import annotations

from typing import Any

import rustworkx as rx

from topographer.graph import ScalarNetwork


def network_graph(network: ScalarNetwork) -> rx.PyGraph:
    """Return a payload-copied Rustworkx graph for ``network``."""
    return network.graph


def vertex_scalar_values(network: ScalarNetwork) -> dict[Any, float]:
    """Return each vertex id mapped to its finite scalar, as stored on the graph."""
    graph = network.graph
    id_attr = network.vertex_id_attr
    scalar_attr = network.scalar_attr
    values: dict[Any, float] = {}
    for index in graph.node_indices():
        payload = graph.get_node_data(index)
        values[payload[id_attr]] = float(payload[scalar_attr])
    return values
