"""Tests for topographer.graph.SequenceGraph."""

from __future__ import annotations

import polars as pl

from topographer.biology import DNA_ALPHABET
from topographer.graph import SequenceGraph


def test_sequence_graph_basic() -> None:
    vertices = pl.DataFrame(
        {
            "var_id": [1, 2, 3],
            "seq": ["AA", "AC", "CC"],
            "fitness": [0.1, 0.5, 0.9],
        }
    )
    edges = pl.DataFrame(
        {
            "src": [1, 2],
            "dst": [2, 3],
            "weight": [1.0, 1.0],
        }
    )
    graph = SequenceGraph(
        vertices,
        edges,
        vertex_id_col="var_id",
        sequence_col="seq",
        source_id_col="src",
        target_id_col="dst",
        alphabet=DNA_ALPHABET,
        vertex_attribute_cols=("fitness",),
        edge_attribute_cols=("weight",),
    )

    assert graph.vertex_count == 3
    assert graph.edge_count == 2
    assert graph.vertex_ids == [1, 2, 3]
    assert graph.neighbors(2) == [1, 3]
    assert graph.vertex_id_for_sequence("AC") == 2


def test_sequence_graph_incident_subgraph_and_largest_component() -> None:
    # Diagnostic graph: F_A (1) -- N (2) -- F_B (3)
    vertices = pl.DataFrame(
        {
            "var_id": [1, 2, 3],
            "seq": ["AA", "AC", "CC"],
        }
    )
    edges = pl.DataFrame(
        {
            "src": [1, 2],
            "dst": [2, 3],
        }
    )
    g = SequenceGraph(
        vertices,
        edges,
        vertex_id_col="var_id",
        sequence_col="seq",
        source_id_col="src",
        target_id_col="dst",
        alphabet=DNA_ALPHABET,
    )
    # Core is {1, 3}
    core = g.induced_subgraph([1, 3])
    # Core has 2 isolated vertices (no direct edge 1 -- 3)
    assert core.edge_count == 0

    # incident_subgraph from {1, 3} retains edges incident to 1 or 3
    # which includes (1, 2) and (2, 3)
    expanded = g.incident_subgraph([1, 3])
    assert expanded.vertex_count == 3
    assert expanded.edge_count == 2

    lcc = expanded.largest_connected_component()
    assert lcc.vertex_count == 3


def test_sequence_graph_added_attributes_survive_transformations() -> None:
    vertices = pl.DataFrame(
        {
            "var_id": [1, 2, 3],
            "seq": ["AA", "AC", "CC"],
        }
    )
    edges = pl.DataFrame(
        {
            "src": [1, 2],
            "dst": [2, 3],
        }
    )
    g = SequenceGraph(
        vertices,
        edges,
        vertex_id_col="var_id",
        sequence_col="seq",
        source_id_col="src",
        target_id_col="dst",
        alphabet=DNA_ALPHABET,
    )
    # Add vertex attribute after construction
    new_v_attrs = pl.DataFrame(
        {
            "var_id": [1, 2, 3],
            "extra_score": [0.1, 0.5, 0.9],
        }
    )
    g.add_vertex_attributes(new_v_attrs, vertex_id_col="var_id", attribute_cols=["extra_score"])

    # Induced subgraph
    sub = g.induced_subgraph([1, 2])
    raw_sub = sub.graph
    node_map = {d["var_id"]: d for d in raw_sub.nodes()}
    assert node_map[1].get("extra_score") == 0.1
    assert node_map[2].get("extra_score") == 0.5


def test_sequence_graph_rejects_mixed_vertex_identity_types() -> None:
    import pytest

    vertices = pl.DataFrame(
        {
            "var_id": pl.Series("var_id", [1, "x"], dtype=pl.Object),
            "seq": ["AA", "AC"],
        }
    )
    edges = pl.DataFrame(schema={"src": pl.Int64, "dst": pl.Int64})
    with pytest.raises(TypeError, match="Mixed vertex identity types are not permitted"):
        SequenceGraph(
            vertices,
            edges,
            vertex_id_col="var_id",
            sequence_col="seq",
            source_id_col="src",
            target_id_col="dst",
            alphabet=DNA_ALPHABET,
        )
