"""Tests for topographer.graph.ScalarNetwork."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.biology import DNA_ALPHABET
from topographer.graph import Graph, HammingGraph, ScalarNetwork, SequenceGraph


def test_scalar_network_basic_with_graph() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v1", "v2"],
            "fitness": [0.2, 0.8],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v1"],
            "dst": ["v2"],
            "weight": [1.0],
        }
    )
    g = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("fitness",),
        edge_attribute_cols=("weight",),
    )
    sn = ScalarNetwork(g, scalar_attr="fitness", edge_scalar_attr="weight")
    assert sn.scalar_attr == "fitness"
    assert sn.edge_scalar_attr == "weight"
    assert sn.vertex_count == 2
    assert sn.edge_count == 1
    assert sn.vertex_ids == ["v1", "v2"]


def test_scalar_network_with_hamming_and_sequence_graph() -> None:
    df = pl.DataFrame(
        {
            "seq": ["AA", "AC"],
            "fitness": [1.0, 2.0],
        }
    )
    hg = HammingGraph(
        df, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fitness",)
    )
    sn_h = ScalarNetwork(hg, scalar_attr="fitness")
    assert sn_h.vertex_ids == ["AA", "AC"]

    edges = pl.DataFrame({"src": ["AA"], "dst": ["AC"]})
    sg = SequenceGraph(
        df,
        edges,
        vertex_id_col="seq",
        sequence_col="seq",
        source_id_col="src",
        target_id_col="dst",
        alphabet=DNA_ALPHABET,
        vertex_attribute_cols=("fitness",),
    )
    sn_s = ScalarNetwork(sg, scalar_attr="fitness")
    assert sn_s.vertex_ids == ["AA", "AC"]


def test_scalar_network_rejects_disconnected_or_directed_or_empty() -> None:
    # Empty
    df_empty = pl.DataFrame({"id": [], "val": []}, schema={"id": pl.Utf8, "val": pl.Float64})
    e_empty = pl.DataFrame({"src": [], "dst": []}, schema={"src": pl.Utf8, "dst": pl.Utf8})
    g_empty = Graph.from_tables(
        df_empty, e_empty, vertex_id_col="id", source_id_col="src", target_id_col="dst"
    )
    with pytest.raises(ValueError, match="empty"):
        ScalarNetwork(g_empty, scalar_attr="val")

    # Disconnected
    df_disc = pl.DataFrame({"id": ["a", "b"], "val": [1.0, 2.0]})
    e_disc = pl.DataFrame({"src": [], "dst": []}, schema={"src": pl.Utf8, "dst": pl.Utf8})
    g_disc = Graph.from_tables(
        df_disc, e_disc, vertex_id_col="id", source_id_col="src", target_id_col="dst"
    )
    with pytest.raises(ValueError, match="disconnected"):
        ScalarNetwork(g_disc, scalar_attr="val")

    # Directed
    df_dir = pl.DataFrame({"id": ["a", "b"], "val": [1.0, 2.0]})
    e_dir = pl.DataFrame({"src": ["a"], "dst": ["b"]})
    g_dir = Graph.from_tables(
        df_dir, e_dir, vertex_id_col="id", source_id_col="src", target_id_col="dst", directed=True
    )
    with pytest.raises(ValueError, match="undirected"):
        ScalarNetwork(g_dir, scalar_attr="val")


def test_scalar_network_rejects_nonfinite_or_nonnumeric_scalars() -> None:
    df_nan = pl.DataFrame({"id": ["a", "b"], "val": [1.0, float("nan")]})
    e = pl.DataFrame({"src": ["a"], "dst": ["b"]})
    g = Graph.from_tables(
        df_nan,
        e,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    with pytest.raises(ValueError, match="finite"):
        ScalarNetwork(g, scalar_attr="val")

    df_str = pl.DataFrame({"id": ["a", "b"], "val": ["x", "y"]})
    g_str = Graph.from_tables(
        df_str,
        e,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    with pytest.raises(TypeError, match="numeric"):
        ScalarNetwork(g_str, scalar_attr="val")


def test_scalar_network_rejects_raw_backend_objects() -> None:
    import rustworkx as rx

    rg = rx.PyGraph()
    with pytest.raises(TypeError, match="Graph"):
        ScalarNetwork(rg, scalar_attr="val")  # type: ignore


def test_scalar_network_preserves_exact_identities_without_payload_confusion() -> None:
    # Regression fixture where ordinary attribute value equals another vertex's identity
    # and appears earlier in the payload dictionary than the identity key
    vertices = pl.DataFrame(
        {
            "other_name": ["v2", "v1"],
            "id": ["v1", "v2"],
            "fitness": [0.2, 0.8],
        }
    )
    edges = pl.DataFrame({"src": ["v1"], "dst": ["v2"]})
    g = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("other_name", "fitness"),
    )
    sn = ScalarNetwork(g, scalar_attr="fitness")
    assert sn.vertex_ids == ["v1", "v2"]
    # Check that vertex_scalars has correct mapping for v1 and v2
    assert sn._vertex_scalars["v1"] == 0.2
    assert sn._vertex_scalars["v2"] == 0.8


def test_scalar_network_graph_accessor_isolates_payloads(
    single_saddle_diamond: ScalarNetwork,
) -> None:
    network = single_saddle_diamond
    pristine = network.graph
    before_nodes = [dict(pristine[i]) for i in pristine.node_indices()]
    before_edges = [dict(d) for d in pristine.edges()]

    leaked = network.graph
    for idx in leaked.node_indices():
        leaked[idx]["scalar"] = -99.0
    for idx in leaked.edge_indices():
        leaked.get_edge_data_by_index(idx)["injected"] = True

    after = network.graph
    assert [dict(after[i]) for i in after.node_indices()] == before_nodes
    assert [dict(d) for d in after.edges()] == before_edges


def test_scalar_network_snapshot_isolated_from_graph_nested_payloads() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v1", "v2"],
            "fitness": [0.2, 0.8],
            "tags": [["a"], ["b"]],
        }
    )
    edges = pl.DataFrame({"src": ["v1"], "dst": ["v2"], "weight": [0.5]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("fitness", "tags"),
        edge_attribute_cols=("weight",),
    )
    network = ScalarNetwork(graph, scalar_attr="fitness", edge_scalar_attr="weight")

    graph.graph.get_node_data(0)["tags"].append("leak")

    assert network.graph.get_node_data(0)["tags"] == ["a"]
