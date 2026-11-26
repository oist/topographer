"""Tests for topographer.graph.Graph adapters."""

from __future__ import annotations

import pytest
import rustworkx as rx

from topographer.graph import Graph


def test_graph_from_rustworkx() -> None:
    rx_graph = rx.PyGraph()
    rx_graph.add_node({"name": "node_b", "val": 20})
    rx_graph.add_node({"name": "node_a", "val": 10})
    rx_graph.add_edge(0, 1, {"dist": 1.5})

    graph = Graph.from_rustworkx(
        rx_graph,
        vertex_id_attr="name",
        vertex_attribute_cols=("val",),
        edge_attribute_cols=("dist",),
    )

    assert graph.vertex_ids == ["node_a", "node_b"]
    assert graph.vertex_count == 2
    assert graph.edge_count == 1
    assert graph.neighbors("node_a") == ["node_b"]


def test_graph_from_networkx() -> None:
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    G.add_node("beta", weight=2.0)
    G.add_node("alpha", weight=1.0)
    G.add_edge("alpha", "beta", cost=3.5)

    graph = Graph.from_networkx(
        G,
        vertex_attribute_cols=("weight",),
        edge_attribute_cols=("cost",),
    )

    assert graph.vertex_ids == ["alpha", "beta"]
    assert graph.vertex_count == 2
    assert graph.edge_count == 1
    assert graph.neighbors("alpha") == ["beta"]


def test_graph_from_igraph() -> None:
    ig = pytest.importorskip("igraph")
    g = ig.Graph(n=2, edges=[(0, 1)], directed=False)
    g.vs["my_id"] = ["v2", "v1"]
    g.vs["score"] = [200, 100]
    g.es["weight"] = [1.2]

    graph = Graph.from_igraph(
        g,
        vertex_id_attr="my_id",
        vertex_attribute_cols=("score",),
        edge_attribute_cols=("weight",),
    )

    assert graph.vertex_ids == ["v1", "v2"]
    assert graph.neighbors("v1") == ["v2"]


def test_adapters_equivalence() -> None:
    # Build identical graph via from_tables, from_rustworkx, and from_networkx
    import polars as pl
    import rustworkx as rx

    nx = pytest.importorskip("networkx")

    # 1. from_tables
    vt = pl.DataFrame({"id": ["b", "a"], "scalar": [0.8, 0.2]})
    et = pl.DataFrame({"src": ["b"], "dst": ["a"], "cost": [1.5]})
    g_tab = Graph.from_tables(
        vt,
        et,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=["scalar"],
        edge_attribute_cols=["cost"],
    )

    # 2. from_rustworkx
    rx_g = rx.PyGraph()
    n0 = rx_g.add_node({"id": "b", "scalar": 0.8})
    n1 = rx_g.add_node({"id": "a", "scalar": 0.2})
    rx_g.add_edge(n0, n1, {"cost": 1.5})
    g_rx = Graph.from_rustworkx(
        rx_g, vertex_id_attr="id", vertex_attribute_cols=["scalar"], edge_attribute_cols=["cost"]
    )

    # 3. from_networkx
    nx_g = nx.Graph()
    nx_g.add_node("b", scalar=0.8)
    nx_g.add_node("a", scalar=0.2)
    nx_g.add_edge("b", "a", cost=1.5)
    g_nx = Graph.from_networkx(
        nx_g, vertex_id_attr="id", vertex_attribute_cols=["scalar"], edge_attribute_cols=["cost"]
    )

    for other in (g_rx, g_nx):
        assert g_tab.vertex_ids == other.vertex_ids == ["a", "b"]
        assert g_tab.edge_ids == other.edge_ids == [("a", "b")]
        assert g_tab.vertex_attr_names == other.vertex_attr_names == {"scalar"}
        assert g_tab.edge_attr_names == other.edge_attr_names == {"cost"}
        assert g_tab.directed == other.directed is False
        assert g_tab.neighbors("a") == other.neighbors("a") == ["b"]
        assert g_tab.connected_components() == other.connected_components() == [["a", "b"]]

        # Defensive export check
        raw_tab = g_tab.graph
        raw_oth = other.graph
        assert raw_tab.num_nodes() == raw_oth.num_nodes() == 2
        assert raw_tab.num_edges() == raw_oth.num_edges() == 1


def test_mixed_vertex_identity_types_raise_clear_type_error() -> None:
    import polars as pl

    rx_graph = rx.PyGraph()
    rx_graph.add_node({"id": 1})
    rx_graph.add_node({"id": "x"})
    with pytest.raises(TypeError, match="Mixed vertex identity types are not permitted"):
        Graph.from_rustworkx(rx_graph, vertex_id_attr="id")

    with pytest.raises(TypeError, match="Mixed vertex identity types are not permitted"):
        Graph.from_tables(
            pl.DataFrame({"id": pl.Series("id", [1, "x"], dtype=pl.Object)}),
            pl.DataFrame(schema={"s": pl.Int64, "t": pl.Int64}),
            vertex_id_col="id",
            source_id_col="s",
            target_id_col="t",
        )


def test_mixed_vertex_identity_types_raise_clear_type_error_networkx() -> None:
    nx = pytest.importorskip("networkx")
    nx_graph = nx.Graph()
    nx_graph.add_node(1)
    nx_graph.add_node("x")
    with pytest.raises(TypeError, match="Mixed vertex identity types are not permitted"):
        Graph.from_networkx(nx_graph)


def test_mixed_vertex_identity_types_raise_clear_type_error_igraph() -> None:
    ig = pytest.importorskip("igraph")
    ig_graph = ig.Graph(n=2)
    ig_graph.vs["id"] = [1, "x"]
    with pytest.raises(TypeError, match="Mixed vertex identity types are not permitted"):
        Graph.from_igraph(ig_graph, vertex_id_attr="id")


def test_unorderable_vertex_identity_type_is_rejected() -> None:
    rx_graph = rx.PyGraph()
    rx_graph.add_node({"id": 1j})
    rx_graph.add_node({"id": 2j})
    with pytest.raises(TypeError, match="no native ordering"):
        Graph.from_rustworkx(rx_graph, vertex_id_attr="id")


def test_tuple_vertex_identities_with_incomparable_contents_are_rejected() -> None:
    # tuple has a native order, but (1,) and ("x",) cannot be compared.
    rx_graph = rx.PyGraph()
    rx_graph.add_node({"id": (1,)})
    rx_graph.add_node({"id": ("x",)})
    with pytest.raises(TypeError, match="'tuple' have no native ordering among these values"):
        Graph.from_rustworkx(rx_graph, vertex_id_attr="id")


def test_unorderable_vertex_identity_type_is_rejected_networkx() -> None:
    nx = pytest.importorskip("networkx")
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from([1j, 2j])
    with pytest.raises(TypeError, match="no native ordering"):
        Graph.from_networkx(nx_graph)


def test_unorderable_vertex_identity_type_is_rejected_igraph() -> None:
    ig = pytest.importorskip("igraph")
    ig_graph = ig.Graph(n=2)
    ig_graph.vs["id"] = [1j, 2j]
    with pytest.raises(TypeError, match="no native ordering"):
        Graph.from_igraph(ig_graph, vertex_id_attr="id")


def _assert_nested_payloads_isolated(graph: Graph) -> None:
    """Graph payloads must still hold the original nested values."""
    raw = graph.graph
    by_vertex = {graph.vertex_ids[i]: raw.get_node_data(i) for i in raw.node_indices()}
    assert by_vertex["v1"]["tags"] == ["a", {"k": [1]}]
    assert by_vertex["v1"]["fitness"] == 0.2
    edge_payload = raw.get_edge_data_by_index(raw.edge_indices()[0])
    assert edge_payload["notes"] == ["x", {"k": [1]}]
    assert edge_payload["weight"] == 0.5


def test_graph_from_rustworkx_isolates_nested_payloads() -> None:
    rx_graph = rx.PyGraph()
    tags = ["a", {"k": [1]}]
    notes = ["x", {"k": [1]}]
    n0 = rx_graph.add_node({"id": "v1", "fitness": 0.2, "tags": tags})
    n1 = rx_graph.add_node({"id": "v2", "fitness": 0.8, "tags": ["b"]})
    rx_graph.add_edge(n0, n1, {"weight": 0.5, "notes": notes})

    graph = Graph.from_rustworkx(
        rx_graph,
        vertex_id_attr="id",
        vertex_attribute_cols=("fitness", "tags"),
        edge_attribute_cols=("weight", "notes"),
    )
    tags.append("leak")
    tags[1]["k"].append(2)
    notes.append("leak")
    notes[1]["k"].append(2)

    _assert_nested_payloads_isolated(graph)


def test_graph_from_networkx_isolates_nested_payloads() -> None:
    nx = pytest.importorskip("networkx")
    tags = ["a", {"k": [1]}]
    notes = ["x", {"k": [1]}]
    nx_graph = nx.Graph()
    nx_graph.add_node("v1", fitness=0.2, tags=tags)
    nx_graph.add_node("v2", fitness=0.8, tags=["b"])
    nx_graph.add_edge("v1", "v2", weight=0.5, notes=notes)

    graph = Graph.from_networkx(
        nx_graph,
        vertex_id_attr="id",
        vertex_attribute_cols=("fitness", "tags"),
        edge_attribute_cols=("weight", "notes"),
    )
    tags.append("leak")
    tags[1]["k"].append(2)
    notes.append("leak")
    notes[1]["k"].append(2)

    _assert_nested_payloads_isolated(graph)


def test_graph_from_igraph_isolates_nested_payloads() -> None:
    ig = pytest.importorskip("igraph")
    tags = ["a", {"k": [1]}]
    notes = ["x", {"k": [1]}]
    ig_graph = ig.Graph(n=2, edges=[(0, 1)], directed=False)
    ig_graph.vs["id"] = ["v1", "v2"]
    ig_graph.vs["fitness"] = [0.2, 0.8]
    ig_graph.vs["tags"] = [tags, ["b"]]
    ig_graph.es["weight"] = [0.5]
    ig_graph.es["notes"] = [notes]

    graph = Graph.from_igraph(
        ig_graph,
        vertex_id_attr="id",
        vertex_attribute_cols=("fitness", "tags"),
        edge_attribute_cols=("weight", "notes"),
    )
    tags.append("leak")
    tags[1]["k"].append(2)
    notes.append("leak")
    notes[1]["k"].append(2)

    _assert_nested_payloads_isolated(graph)
