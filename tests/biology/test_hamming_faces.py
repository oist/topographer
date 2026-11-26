"""Tests for the private categorical Hamming face builder."""

from __future__ import annotations

import pytest
import rustworkx as rx

from topographer.biology._faces import Face
from topographer.biology._sequences import build_categorical_hamming_faces
from topographer.graph import Graph, ScalarNetwork


def _build_hypercube_network(dim: int = 2) -> ScalarNetwork:
    # Build H(dim, 2) boolean hypercube
    graph = rx.PyGraph()
    seq_to_node = {}
    num_nodes = 1 << dim

    for i in range(num_nodes):
        seq = format(i, f"0{dim}b")
        # Injective fitness scalar: e.g. i / num_nodes
        scalar = float(i) / float(num_nodes)
        node_idx = graph.add_node({"sequence": seq, "fitness": scalar})
        seq_to_node[seq] = node_idx

    for i in range(num_nodes):
        seq_a = format(i, f"0{dim}b")
        for bit in range(dim):
            j = i ^ (1 << bit)
            if i < j:
                seq_b = format(j, f"0{dim}b")
                graph.add_edge(seq_to_node[seq_a], seq_to_node[seq_b], {})

    g = Graph.from_rustworkx(graph, vertex_id_attr="sequence", vertex_attribute_cols=("fitness",))
    return ScalarNetwork(g, scalar_attr="fitness")


def test_hamming_2d_hypercube_faces() -> None:
    # H(2,2): 4 vertices, 4 edges, 1 square face
    faces = build_categorical_hamming_faces(_build_hypercube_network(2))
    assert len(faces) == 1
    assert faces[0].payload["face_kind"] == "square"
    assert len(faces[0].vertices) == 4


def test_hamming_3d_hypercube_faces() -> None:
    # H(3,2): 8 vertices, 12 edges, 6 square faces (cube boundary)
    faces = build_categorical_hamming_faces(_build_hypercube_network(3))
    assert len(faces) == 6
    assert all(f.payload["face_kind"] == "square" for f in faces)
    assert len({frozenset(f.vertices) for f in faces}) == 6


def test_face_validation() -> None:
    assert Face(vertices=(0, 1, 2), face_id="f1").vertices == (0, 1, 2)
    with pytest.raises(ValueError, match="at least 3"):
        Face(vertices=(0, 1))
    with pytest.raises(ValueError, match="distinct"):
        Face(vertices=(0, 1, 1))
