"""Hamming landscape example subpackage."""

from __future__ import annotations

from topographer.examples.hamming.network import (
    HAMMING_EXAMPLE_SCALARS,
    HAMMING_SIMPLIFICATION_THRESHOLD,
    HAMMING_TREE_NODE_ATTRS,
    build_hamming_example_graph,
    build_hamming_scalar_network,
    build_hamming_topology,
    hamming_coordinates,
    hamming_distance,
    hypercube_coordinates,
    hypercube_edge_table,
    hypercube_node_table,
)

__all__ = [
    "HAMMING_EXAMPLE_SCALARS",
    "HAMMING_SIMPLIFICATION_THRESHOLD",
    "HAMMING_TREE_NODE_ATTRS",
    "build_hamming_example_graph",
    "build_hamming_scalar_network",
    "build_hamming_topology",
    "hamming_coordinates",
    "hamming_distance",
    "hypercube_coordinates",
    "hypercube_edge_table",
    "hypercube_node_table",
]
