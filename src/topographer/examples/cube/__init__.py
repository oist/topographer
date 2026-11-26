"""Cube 3D landscape example subpackage."""

from __future__ import annotations

from topographer.examples.cube.network import (
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

__all__ = [
    "CUBE_SIMPLIFICATION_THRESHOLD",
    "CUBE_TREE_NODE_ATTRS",
    "build_cube_scalar_network",
    "build_cube_topology",
    "build_two_peak_terrain_scalar_network",
    "cube_coordinates",
    "cube_edge_table",
    "cube_node_table",
    "two_peak_terrain_edge_table",
    "two_peak_terrain_node_table",
]
