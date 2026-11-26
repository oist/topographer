"""Tests for the private graph payload copy helper."""

from __future__ import annotations

import rustworkx as rx

from topographer.graph._payloads import copy_graph_with_payloads


def test_copy_preserves_indices_with_removed_nodes() -> None:
    graph = rx.PyDiGraph()
    a = graph.add_node({"name": "a"})
    b = graph.add_node({"name": "b"})
    c = graph.add_node({"name": "c"})
    graph.add_edge(a, c, {"w": 0.5})
    graph.remove_node(b)

    duplicate = copy_graph_with_payloads(graph)

    assert list(duplicate.node_indices()) == [a, c]
    assert duplicate[c] == {"name": "c"}
    assert duplicate.get_edge_data(a, c) == {"w": 0.5}


def test_copy_isolates_nested_mutable_values() -> None:
    graph = rx.PyGraph()
    a = graph.add_node({"tags": ["x"]})
    b = graph.add_node({"tags": ["y"]})
    edge = graph.add_edge(a, b, {"path": [1, 2]})

    duplicate = copy_graph_with_payloads(graph)
    duplicate[a]["tags"].append("z")
    duplicate.get_edge_data_by_index(edge)["path"].append(3)

    assert graph[a]["tags"] == ["x"]
    assert graph.get_edge_data_by_index(edge)["path"] == [1, 2]


def test_copy_preserves_edge_index_holes() -> None:
    graph = rx.PyDiGraph()
    a = graph.add_node({})
    b = graph.add_node({})
    c = graph.add_node({})
    e0 = graph.add_edge(a, b, {"w": 0})
    e1 = graph.add_edge(b, c, {"w": 1})
    e2 = graph.add_edge(a, c, {"w": 2})
    graph.remove_edge_from_index(e1)

    duplicate = copy_graph_with_payloads(graph)

    assert sorted(duplicate.edge_indices()) == [e0, e2]
    assert duplicate.get_edge_data_by_index(e0) == {"w": 0}
    assert duplicate.get_edge_data_by_index(e2) == {"w": 2}
    assert duplicate.get_edge_endpoints_by_index(e2) == (a, c)


def test_copy_keeps_parallel_edges_independent_in_multigraph() -> None:
    graph = rx.PyGraph(multigraph=True)
    a = graph.add_node({})
    b = graph.add_node({})
    first = graph.add_edge(a, b, {"w": 1})
    second = graph.add_edge(a, b, {"w": 2})

    duplicate = copy_graph_with_payloads(graph)
    duplicate.get_edge_data_by_index(first)["w"] = 99

    assert duplicate.num_edges() == 2
    assert duplicate.get_edge_data_by_index(second) == {"w": 2}
    assert graph.get_edge_data_by_index(first) == {"w": 1}


def test_copy_preserves_multigraph_flag() -> None:
    assert copy_graph_with_payloads(rx.PyGraph(multigraph=True)).multigraph is True
    assert copy_graph_with_payloads(rx.PyDiGraph(multigraph=False)).multigraph is False
