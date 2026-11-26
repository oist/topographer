"""Canonical example datasets and fixture builders for topographer."""

from __future__ import annotations

from topographer.examples.cube import (
    CUBE_SIMPLIFICATION_THRESHOLD,
    CUBE_TREE_NODE_ATTRS,
    build_cube_scalar_network,
    build_cube_topology,
    build_two_peak_terrain_scalar_network,
    cube_coordinates,
    cube_edge_table,
    cube_node_table,
    two_peak_terrain_edge_table,
    two_peak_terrain_node_table,
)
from topographer.examples.hamming import (
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
    "CUBE_SIMPLIFICATION_THRESHOLD",
    "CUBE_TREE_NODE_ATTRS",
    "HAMMING_EXAMPLE_SCALARS",
    "HAMMING_SIMPLIFICATION_THRESHOLD",
    "HAMMING_TREE_NODE_ATTRS",
    "build_cube_scalar_network",
    "build_cube_topology",
    "build_hamming_example_graph",
    "build_hamming_scalar_network",
    "build_hamming_topology",
    "build_two_peak_terrain_scalar_network",
    "cube_coordinates",
    "cube_edge_table",
    "cube_node_table",
    "hamming_coordinates",
    "hamming_distance",
    "hypercube_coordinates",
    "hypercube_edge_table",
    "hypercube_node_table",
    "two_peak_terrain_edge_table",
    "two_peak_terrain_node_table",
]
