"""Network loading, generation, and the 3-cube workspace."""

from __future__ import annotations

import polars as pl

from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology


def cube_coordinates() -> dict[str, tuple[float, float]]:
    """Return fixed (x, y) coordinates for the 8 cube vertices."""
    return {
        f"{value:03b}": (
            float(int(f"{value:03b}"[0]) + 0.48 * int(f"{value:03b}"[2])),
            float(int(f"{value:03b}"[1]) + 0.34 * int(f"{value:03b}"[2])),
        )
        for value in range(8)
    }


def cube_node_table() -> pl.DataFrame:
    """Return the eight cube vertices with a normalized terrain scalar function and attached (x, y) coordinates.

    Vertex IDs are three-bit strings. ``height`` has two local minima joined at
    vertex ``"000"`` and two maxima split at vertex ``"111"``, making the cube
    a compact deterministic merge-tree regression fixture rather than merely
    a graph-construction example.
    """
    heights = {
        "000": 0.5,
        "001": 0.0,
        "010": 0.6,
        "011": 1.0,
        "100": 0.1,
        "101": 0.7,
        "110": 0.9,
        "111": 0.8,
    }
    coords = cube_coordinates()
    return pl.DataFrame(
        {
            "node_id": list(heights),
            "height": list(heights.values()),
            "label": list(heights),
            "x": [coords[label][0] for label in heights],
            "y": [coords[label][1] for label in heights],
        }
    )


def cube_edge_table() -> pl.DataFrame:
    """Return the twelve cube edges with generic distance and capacity scalars."""
    rows = []
    for value in range(8):
        source = f"{value:03b}"
        for bit_index in range(3):
            target_value = value ^ (1 << bit_index)
            if value >= target_value:
                continue
            rows.append(
                {
                    "source": source,
                    "target": f"{target_value:03b}",
                    "distance": 1.0,
                    "capacity": float(bit_index + 1),
                }
            )
    return pl.DataFrame(rows)


def build_cube_scalar_network() -> ScalarNetwork:
    """Build the generic cube example with vertex (including x, y) and edge scalar attributes."""
    g = Graph.from_tables(
        cube_node_table(),
        cube_edge_table(),
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height", "label", "x", "y"),
        edge_attribute_cols=("distance", "capacity"),
    )
    return ScalarNetwork(g, scalar_attr="height")


def two_peak_terrain_node_table() -> pl.DataFrame:
    """Return a planar two-peak terrain for split-tree teaching examples."""
    rows = [
        ("left peak", -3.2, 1.00),
        ("left upper", -2.7, 0.86),
        ("left outer ridge", -1.9, 0.78),
        ("left inner ridge", -2.7, 0.72),
        ("left lower ridge", -1.6, 0.63),
        ("left saddle approach", -0.7, 0.56),
        ("right peak", 3.2, 0.90),
        ("right upper", 2.7, 0.80),
        ("right outer ridge", 1.9, 0.70),
        ("right inner ridge", 2.7, 0.64),
        ("right lower ridge", 1.6, 0.57),
        ("right saddle approach", 0.7, 0.53),
        ("saddle", 0.0, 0.48),
        ("left lower shoulder", -1.1, 0.43),
        ("right lower shoulder", 1.1, 0.40),
        ("lower left", -0.6, 0.32),
        ("lower right", 0.5, 0.30),
        ("lower outer ridge", -0.4, 0.22),
        ("lower inner ridge", 0.3, 0.14),
        ("minimum", 0.0, 0.05),
    ]
    return pl.DataFrame(
        {
            "node_id": list(range(len(rows))),
            "label": [row[0] for row in rows],
            "x": [row[1] for row in rows],
            "y": [row[2] for row in rows],
            "height": [row[2] for row in rows],
        }
    )


def two_peak_terrain_edge_table() -> pl.DataFrame:
    """Return the deterministic ridge edges for :func:`two_peak_terrain_node_table`."""
    return pl.DataFrame(
        {
            "source": [
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                5,
                6,
                6,
                7,
                7,
                8,
                8,
                9,
                9,
                10,
                11,
                5,
                12,
                13,
                13,
                11,
                14,
                14,
                15,
                15,
                16,
                16,
                17,
                17,
                18,
                3,
                9,
            ],
            "target": [
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                12,
                7,
                8,
                8,
                9,
                9,
                10,
                10,
                11,
                11,
                12,
                13,
                15,
                16,
                14,
                14,
                16,
                15,
                16,
                17,
                17,
                18,
                18,
                19,
                19,
                13,
                14,
            ],
        }
    )


def build_two_peak_terrain_scalar_network() -> ScalarNetwork:
    """Build the fixed planar two-peak scalar network used by the sweep tutorial."""
    g = Graph.from_tables(
        two_peak_terrain_node_table(),
        two_peak_terrain_edge_table(),
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height", "label", "x", "y"),
    )
    return ScalarNetwork(g, scalar_attr="height")


CUBE_SIMPLIFICATION_THRESHOLD = 0.10
CUBE_TREE_NODE_ATTRS = ("label", "height", "x", "y")


def build_cube_topology() -> ScalarNetworkTopology:
    """Build the cube workspace with its join and split trees already computed."""
    topology = ScalarNetworkTopology(build_cube_scalar_network())
    topology.join_tree()
    topology.split_tree()
    return topology
