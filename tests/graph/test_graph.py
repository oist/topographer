"""Tests for topographer.graph.Graph interface and behaviors."""

from __future__ import annotations

import polars as pl
import pytest
import rustworkx as rx

from topographer.graph import Graph


def test_graph_from_tables_basic() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v1", "v2", "v3"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v1", "v2"],
            "dst": ["v2", "v3"],
            "dist": [0.5, 0.8],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("weight",),
        edge_attribute_cols=("dist",),
    )

    assert not graph.directed
    assert graph.vertex_count == 3
    assert graph.edge_count == 2
    assert graph.vertex_ids == ["v1", "v2", "v3"]
    assert graph.neighbors("v2") == ["v1", "v3"]
    assert graph.connected_components() == [["v1", "v2", "v3"]]


def test_graph_canonical_vertex_order() -> None:
    # Shuffle input vertices, check that vertex_ids is sorted deterministically
    vertices = pl.DataFrame(
        {
            "id": ["c", "a", "b"],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["c", "b"],
            "dst": ["a", "a"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
    )
    assert graph.vertex_ids == ["a", "b", "c"]
    assert graph.neighbors("a") == ["b", "c"]


def test_graph_defensive_copy() -> None:
    vertices = pl.DataFrame({"id": ["v1", "v2"]})
    edges = pl.DataFrame({"src": ["v1"], "dst": ["v2"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
    )
    raw_rx = graph.graph
    assert isinstance(raw_rx, rx.PyGraph)
    # Mutating raw_rx nodes or payloads should not affect graph
    raw_rx.add_node({"id": "v3"})
    assert graph.vertex_count == 2
    raw_rx.get_node_data(0)["extra"] = "leak"
    assert "extra" not in graph.graph.get_node_data(0)


def _graph_with_nested_payloads() -> Graph:
    vertices = pl.DataFrame(
        {
            "id": ["v1", "v2"],
            "fitness": [0.2, 0.8],
            "tags": [["a"], ["b"]],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v1"],
            "dst": ["v2"],
            "weight": [0.5],
            "notes": [["x"]],
        }
    )
    return Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("fitness", "tags"),
        edge_attribute_cols=("weight", "notes"),
    )


def test_graph_graph_isolates_nested_node_payload_values() -> None:
    graph = _graph_with_nested_payloads()
    leaked = graph.graph
    leaked.get_node_data(0)["tags"].append("leak")

    assert graph.graph.get_node_data(0)["tags"] == ["a"]


def test_graph_graph_isolates_nested_edge_payload_values() -> None:
    graph = _graph_with_nested_payloads()
    leaked = graph.graph
    leaked.get_edge_data_by_index(leaked.edge_indices()[0])["notes"].append("leak")

    fresh = graph.graph
    assert fresh.get_edge_data_by_index(fresh.edge_indices()[0])["notes"] == ["x"]


def test_graph_unknown_neighbor_raises() -> None:
    vertices = pl.DataFrame({"id": ["v1", "v2"]})
    edges = pl.DataFrame({"src": ["v1"], "dst": ["v2"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
    )
    with pytest.raises(KeyError, match="v99"):
        graph.neighbors("v99")


def test_graph_connected_components_ordering() -> None:
    vertices = pl.DataFrame({"id": ["x1", "y1", "y2", "y3", "z1", "z2"]})
    edges = pl.DataFrame(
        {
            "src": ["y1", "y2", "z1"],
            "dst": ["y2", "y3", "z2"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
    )
    # Ordered by decreasing size, then canonical identity order
    # component y: size 3 -> ["y1", "y2", "y3"]
    # component z: size 2 -> ["z1", "z2"]
    # component x: size 1 -> ["x1"]
    assert graph.connected_components() == [
        ["y1", "y2", "y3"],
        ["z1", "z2"],
        ["x1"],
    ]


def test_graph_add_vertex_attributes() -> None:
    vertices = pl.DataFrame({"id": ["v1", "v2"]})
    edges = pl.DataFrame({"src": ["v1"], "dst": ["v2"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
    )
    new_attrs = pl.DataFrame(
        {
            "id": ["v2", "v1"],
            "color": ["red", "blue"],
        }
    )
    graph.add_vertex_attributes(new_attrs, vertex_id_col="id", attribute_cols=["color"])
    raw = graph.graph
    nodes = {node["id"]: node for node in raw.nodes()}
    assert nodes["v1"]["color"] == "blue"
    assert nodes["v2"]["color"] == "red"


def test_graph_add_vertex_attributes_validation_failure_is_atomic() -> None:
    vertices = pl.DataFrame({"id": ["v1", "v2"]})
    edges = pl.DataFrame({"src": ["v1"], "dst": ["v2"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
    )
    # Missing v2, has unknown v3
    bad_attrs = pl.DataFrame(
        {
            "id": ["v1", "v3"],
            "color": ["blue", "green"],
        }
    )
    with pytest.raises(ValueError):
        graph.add_vertex_attributes(bad_attrs, vertex_id_col="id", attribute_cols=["color"])

    # Ensure no partial mutation
    raw = graph.graph
    assert "color" not in raw.nodes()[0]


def test_graph_add_edge_attributes_modifies_exported_payloads() -> None:
    vertices = pl.DataFrame({"id": ["v1", "v2"]})
    edges = pl.DataFrame({"src": ["v1"], "dst": ["v2"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
    )
    edge_attrs = pl.DataFrame(
        {
            "src": ["v1"],
            "dst": ["v2"],
            "weight": [0.75],
        }
    )
    graph.add_edge_attributes(
        edge_attrs,
        source_id_col="src",
        target_id_col="dst",
        attribute_cols=["weight"],
    )
    raw = graph.graph
    edge_data = [data for _, _, data in raw.weighted_edge_list()]
    assert len(edge_data) == 1
    assert edge_data[0].get("weight") == 0.75


def test_graph_vertex_identity_is_not_an_ordinary_attribute() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v1", "v2"],
            "scalar": [0.1, 0.9],
        }
    )
    edges = pl.DataFrame({"src": ["v1"], "dst": ["v2"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=["scalar"],
    )
    assert graph.vertex_attr_names == {"scalar"}
    assert "id" not in graph.vertex_attr_names


def test_graph_edge_identities_undirected_and_directed() -> None:
    # Undirected generated identity: canonical endpoint pair
    v = pl.DataFrame({"id": ["b", "a"]})
    e = pl.DataFrame({"src": ["b"], "dst": ["a"]})
    g_undir = Graph.from_tables(v, e, vertex_id_col="id", source_id_col="src", target_id_col="dst")
    assert g_undir.edge_ids == [("a", "b")]
    raw_e = [data for _, _, data in g_undir.graph.weighted_edge_list()]
    assert raw_e[0]["edge_id"] == ("a", "b")

    # Directed generated identity: preserves (src, dst)
    g_dir = Graph.from_tables(
        v, e, vertex_id_col="id", source_id_col="src", target_id_col="dst", directed=True
    )
    assert g_dir.edge_ids == [("b", "a")]
    raw_e_dir = [data for _, _, data in g_dir.graph.weighted_edge_list()]
    assert raw_e_dir[0]["edge_id"] == ("b", "a")


def test_graph_edge_identity_explicit_and_multigraph() -> None:
    v = pl.DataFrame({"id": ["v1", "v2"]})
    e = pl.DataFrame(
        {
            "eid": ["e1", "e2"],
            "src": ["v1", "v1"],
            "dst": ["v2", "v2"],
            "weight": [1.0, 2.0],
        }
    )
    # Multigraph with explicit edge identities
    g_multi = Graph.from_tables(
        v,
        e,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        edge_id_col="eid",
        edge_attribute_cols=["weight"],
        multigraph=True,
    )
    assert g_multi.edge_ids == ["e1", "e2"]
    assert g_multi.edge_id_attr == "eid"
    assert g_multi.edge_attr_names == {"weight"}
    assert "eid" not in g_multi.edge_attr_names

    raw_multi = [data for _, _, data in g_multi.graph.weighted_edge_list()]
    assert [d["eid"] for d in raw_multi] == ["e1", "e2"]

    # Multigraph without explicit edge identities must fail
    with pytest.raises(ValueError, match="Multigraph requires explicit"):
        Graph.from_tables(
            v,
            e,
            vertex_id_col="id",
            source_id_col="src",
            target_id_col="dst",
            multigraph=True,
        )


def test_graph_add_edge_attributes_rollback_is_atomic() -> None:
    v = pl.DataFrame({"id": ["v1", "v2"]})
    e = pl.DataFrame({"src": ["v1"], "dst": ["v2"]})
    g = Graph.from_tables(v, e, vertex_id_col="id", source_id_col="src", target_id_col="dst")

    # Table with one valid and one invalid edge
    bad_edge_attrs = pl.DataFrame(
        {
            "src": ["v1", "v1"],
            "dst": ["v2", "v99"],
            "flow": [10.0, 20.0],
        }
    )
    with pytest.raises(ValueError):
        g.add_edge_attributes(
            bad_edge_attrs,
            source_id_col="src",
            target_id_col="dst",
            attribute_cols=["flow"],
        )

    # Graph remains byte-for-byte unmodified
    assert "flow" not in g.edge_attr_names
    raw_edges = [data for _, _, data in g.graph.weighted_edge_list()]
    assert "flow" not in raw_edges[0]
