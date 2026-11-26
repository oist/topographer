"""Network loading, generation, and Hamming hypercube workspaces."""

from __future__ import annotations

from itertools import combinations

import polars as pl
import rustworkx as rx

from topographer.biology import Alphabet
from topographer.graph import HammingGraph, ScalarNetwork
from topographer.topology import (
    ScalarNetworkTopology,
)

HAMMING_EXAMPLE_SCALARS: dict[int, dict[str, float]] = {
    2: {"00": 0.20, "01": 0.60, "10": 0.80, "11": 0.40},
    3: {
        "000": 0.08,
        "001": 0.74,
        "010": 0.29,
        "011": 0.91,
        "100": 0.44,
        "101": 0.67,
        "110": 0.53,
        "111": 0.22,
    },
    4: {
        "0000": 0.058823529411764705,
        "0001": 0.11764705882352941,
        "0010": 0.17647058823529413,
        "0011": 0.23529411764705882,
        "0100": 0.29411764705882354,
        "0101": 0.35294117647058826,
        "0110": 0.4117647058823529,
        "0111": 0.75,
        "1000": 0.5882352941176471,
        "1001": 0.6470588235294118,
        "1010": 0.7058823529411765,
        "1011": 0.7647058823529411,
        "1100": 0.8235294117647058,
        "1101": 0.8823529411764706,
        "1110": 0.9411764705882353,
        "1111": 0.47058823529411764,
    },
}


def hamming_distance(left: str, right: str) -> int:
    """Return the Hamming distance between two equal-length bitstrings."""
    return sum(char_left != char_right for char_left, char_right in zip(left, right, strict=True))


def hamming_coordinates(label: str) -> tuple[float, float]:
    """Return a stable square, cube, or hypercube projection for a bitstring."""
    bits = [int(bit) for bit in label]
    if len(bits) == 2:
        return float(bits[0]), float(bits[1])
    if len(bits) == 3:
        return float(bits[0] + 0.48 * bits[2]), float(bits[1] + 0.34 * bits[2])
    if len(bits) == 4:
        return (
            float(bits[0] + 0.48 * bits[2] + 0.21 * bits[3]),
            float(bits[1] + 0.34 * bits[2] + 0.67 * bits[3]),
        )
    raise ValueError("The committed example suite supports dimensions 2, 3, and 4.")


def hypercube_coordinates() -> dict[str, tuple[float, float]]:
    """Return fixed 2D layout coordinates for the 16 4-cube vertices."""
    return {f"{val:04b}": hamming_coordinates(f"{val:04b}") for val in range(16)}


def hypercube_node_table() -> pl.DataFrame:
    """Return the 16 4-cube vertices with canonical coordinates and scalar values."""
    scalars = HAMMING_EXAMPLE_SCALARS[4]
    coords = hypercube_coordinates()
    return pl.DataFrame(
        [
            {
                "node_id": label,
                "sequence": label,
                "scalar": float(scalars[label]),
                "weight": int(label.count("1")),
                "x": coords[label][0],
                "y": coords[label][1],
            }
            for label in sorted(scalars)
        ]
    )


def hypercube_edge_table() -> pl.DataFrame:
    """Return the 32 4-cube edges with distance 1."""
    scalars = HAMMING_EXAMPLE_SCALARS[4]
    rows = []
    sorted_labels = sorted(scalars)
    for left, right in combinations(sorted_labels, 2):
        if hamming_distance(left, right) == 1:
            rows.append({"source": left, "target": right, "distance": 1.0})
    return pl.DataFrame(rows)


def build_hamming_example_graph(dimension: int) -> rx.PyGraph:
    """Build a deterministic scalar Hamming graph with generic node payloads."""
    scalar_by_label = HAMMING_EXAMPLE_SCALARS.get(int(dimension))
    if scalar_by_label is None:
        raise ValueError(f"Unsupported Hamming example dimension {dimension!r}.")
    graph = rx.PyGraph()
    node_index_by_label: dict[str, int] = {}
    for label in sorted(scalar_by_label):
        x, y = hamming_coordinates(label)
        node_index_by_label[label] = graph.add_node(
            {
                "label": label,
                "scalar": float(scalar_by_label[label]),
                "weight": int(label.count("1")),
                "x": x,
                "y": y,
            }
        )
    for left, right in combinations(sorted(scalar_by_label), 2):
        if hamming_distance(left, right) == 1:
            graph.add_edge(node_index_by_label[left], node_index_by_label[right], {"distance": 1})
    return graph


HAMMING_SIMPLIFICATION_THRESHOLD = 0.09
HAMMING_TREE_NODE_ATTRS = ("label", "sequence", "scalar", "weight", "x", "y")


def build_hamming_scalar_network(dimension: int = 4) -> ScalarNetwork:
    """Build the binary Hamming hypercube scalar network of the given dimension (2, 3 or 4)."""
    if dimension == 4:
        nodes = hypercube_node_table()
        alphabet = Alphabet("01")
        hg = HammingGraph(
            nodes,
            sequence_col="sequence",
            alphabet=alphabet,
            hamming_radius=1,
            vertex_attribute_cols=("scalar", "weight", "x", "y"),
        )
        return ScalarNetwork(hg, scalar_attr="scalar")

    scalars = HAMMING_EXAMPLE_SCALARS.get(dimension)
    if scalars is None:
        raise ValueError(f"Unsupported Hamming dimension: {dimension}")
    nodes_data = [
        {
            "node_id": label,
            "sequence": label,
            "scalar": float(scalars[label]),
            "weight": int(label.count("1")),
            "x": hamming_coordinates(label)[0],
            "y": hamming_coordinates(label)[1],
        }
        for label in sorted(scalars)
    ]
    nodes_df = pl.DataFrame(nodes_data)
    alphabet = Alphabet("01")
    hg = HammingGraph(
        nodes_df,
        sequence_col="sequence",
        alphabet=alphabet,
        hamming_radius=1,
        vertex_attribute_cols=("scalar", "weight", "x", "y"),
    )
    return ScalarNetwork(hg, scalar_attr="scalar")


def build_hamming_topology(dimension: int = 4) -> ScalarNetworkTopology:
    """Build a Hamming hypercube workspace with its join and split trees already computed."""
    topology = ScalarNetworkTopology(build_hamming_scalar_network(dimension))
    topology.join_tree()
    topology.split_tree()
    return topology
