"""Tests for join and split merge tree construction, reduction, and simplification."""

from __future__ import annotations

import json

import polars as pl
import pytest
import rustworkx as rx

from topographer.examples import build_cube_scalar_network
from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology, build_tree_metrics
from topographer.topology._merge_tree import _directed_sink
from topographer.topology._tree_construction import build_join_tree
from topographer.topology._tree_reduction import reduce_merge_tree


def _build_branching_network(values: list[float]) -> ScalarNetwork:
    vertices = pl.DataFrame(
        {
            "id": [f"v{i}" for i in range(len(values))],
            "m": values,
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1", "v3", "v4"],
            "dst": ["v1", "v2", "v4", "v2"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("m",),
    )
    return ScalarNetwork(graph, scalar_attr="m")


def test_build_join_tree_detects_saddle_event() -> None:
    network = _build_branching_network([0.0, 0.2, 0.9, 0.1, 0.3])
    join = ScalarNetworkTopology(network).join_tree()

    assert join.scalar_attr == "m"
    assert join.kind == "join"
    assert join.augmented_graph.num_nodes() == 5
    assert join.augmented_graph.num_edges() == 4
    assert list(join.events["vertex_index"]) == [0, 3, 1, 4, 2]
    assert list(join.events["event"]) == [
        "minimum",
        "minimum",
        "regular",
        "regular",
        "join_saddle",
    ]

    edge_payloads = [
        dict(join.augmented_graph.get_edge_data_by_index(edge_index))
        for edge_index in join.augmented_graph.edge_indices()
    ]
    assert any(
        payload["from_repr"] == 1 and payload["to_saddle"] == 2 for payload in edge_payloads
    )


def test_merge_tree_requires_one_weak_component_and_no_self_loops() -> None:
    disconnected_v = pl.DataFrame({"id": ["v0", "v1"], "m": [0.0, 1.0]})
    disconnected_e = pl.DataFrame(
        {"src": [], "dst": []}, schema={"src": pl.String, "dst": pl.String}
    )
    disconnected_g = Graph.from_tables(
        disconnected_v,
        disconnected_e,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("m",),
    )
    with pytest.raises(ValueError, match="connected graph"):
        ScalarNetwork(disconnected_g, scalar_attr="m")

    self_loop_v = pl.DataFrame({"id": ["v0"], "m": [0.0]})
    self_loop_e = pl.DataFrame({"src": ["v0"], "dst": ["v0"]})
    self_loop_g = Graph.from_tables(
        self_loop_v,
        self_loop_e,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("m",),
    )
    self_loop_sn = ScalarNetwork(self_loop_g, scalar_attr="m")
    with pytest.raises(ValueError, match="self-loops"):
        ScalarNetworkTopology(self_loop_sn).join_tree()


def test_build_split_tree_detects_saddle_event_on_mirrored_scalars() -> None:
    network = _build_branching_network([1.0, 0.8, 0.1, 0.9, 0.7])
    split = ScalarNetworkTopology(network).split_tree()

    assert split.scalar_attr == "m"
    assert split.kind == "split"
    assert split.augmented_graph.num_nodes() == 5
    assert split.augmented_graph.num_edges() == 4
    assert list(split.events["vertex_index"]) == [0, 3, 1, 4, 2]
    assert list(split.events["event"]) == [
        "maximum",
        "maximum",
        "regular",
        "regular",
        "split_saddle",
    ]


def test_build_merge_tree_uses_tie_breaking_by_node_index() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2"],
            "m": [0.0, 0.5, 0.5],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1"],
            "dst": ["v1", "v2"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("m",),
    )
    network = ScalarNetwork(graph, scalar_attr="m")
    join = ScalarNetworkTopology(network).join_tree()

    assert list(join.events["vertex_index"]) == [0, 1, 2]
    assert list(join.events["event"]) == ["minimum", "regular", "regular"]


def test_cube_scalar_network_has_a_deterministic_join_tree() -> None:
    join = ScalarNetworkTopology(build_cube_scalar_network()).join_tree()

    assert join.scalar_attr == "height"
    assert join.augmented_graph.num_nodes() == 8
    assert join.augmented_graph.num_edges() == 7
    assert join.events.to_dicts() == [
        {"vertex_index": 1, "event": "minimum", "active_neighbor_components": 0},
        {"vertex_index": 4, "event": "minimum", "active_neighbor_components": 0},
        {"vertex_index": 0, "event": "join_saddle", "active_neighbor_components": 2},
        {"vertex_index": 2, "event": "regular", "active_neighbor_components": 1},
        {"vertex_index": 5, "event": "regular", "active_neighbor_components": 1},
        {"vertex_index": 7, "event": "regular", "active_neighbor_components": 1},
        {"vertex_index": 6, "event": "regular", "active_neighbor_components": 1},
        {"vertex_index": 3, "event": "regular", "active_neighbor_components": 1},
    ]


def test_scalar_network_requires_undirected_graph() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2"],
            "scalar": [0.0, 0.4, 1.0],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1"],
            "dst": ["v1", "v2"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
        directed=True,
    )
    with pytest.raises(ValueError, match="undirected"):
        ScalarNetwork(graph, scalar_attr="scalar")


def test_scalar_network_requires_an_explicit_declared_scalar() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1"],
            "m": [0.0, 0.8],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0"],
            "dst": ["v1"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("m",),
    )
    with pytest.raises(ValueError, match="fitness_wt"):
        ScalarNetwork(graph, scalar_attr="fitness_wt")
    with pytest.raises(TypeError, match="ScalarNetwork"):
        ScalarNetworkTopology(graph)  # type: ignore[arg-type]


def test_scalar_network_requires_an_unambiguous_declared_network_scalar() -> None:
    network = _build_branching_network([0.0, 0.2, 0.9, 0.1, 0.3])
    topology = ScalarNetworkTopology(network)
    join = topology.join_tree()
    assert join.scalar_attr == "m"


def test_build_merge_tree_rejects_empty_scalar_networks() -> None:
    with pytest.raises(ValueError, match="Graph cannot be empty."):
        g = Graph.from_tables(
            pl.DataFrame({"id": [], "height": []}, schema={"id": pl.String, "height": pl.Float64}),
            pl.DataFrame({"src": [], "dst": []}, schema={"src": pl.String, "dst": pl.String}),
            vertex_id_col="id",
            source_id_col="src",
            target_id_col="dst",
            vertex_attribute_cols=("height",),
        )
        ScalarNetwork(g, scalar_attr="height")


def test_build_split_tree_accepts_a_validated_scalar_network() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1"],
            "m": [1.0, 0.2],
            "site": ["summit", "valley"],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0"],
            "dst": ["v1"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("m", "site"),
    )
    network = ScalarNetwork(graph, scalar_attr="m")
    split = ScalarNetworkTopology(network).split_tree()

    assert split.scalar_attr == "m"
    edge_data = dict(split.augmented_graph.get_edge_data_by_index(0))
    assert set(edge_data) == {
        "source_scalar",
        "target_scalar",
        "scalar_difference",
        "weight",
        "merge_scalar",
        "from_repr",
        "to_saddle",
    }


def test_reduce_merge_tree_keeps_every_critical_vertex() -> None:
    """Reduction contracts only interior regular vertices and keeps all critical ones.

    A join tree keeps every minimum and join saddle, so its reduced size is set by the
    graph (for example a sparse empirical graph), not by the reduction.
    """
    graph = rx.PyGraph()
    for scalar in (0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 1.0):
        graph.add_node({"scalar": scalar})
    # Leaves 0-3 join hub 4, which continues through regular 5 to the top vertex 6.
    for source, target in [(0, 4), (1, 4), (2, 4), (3, 4), (4, 5), (5, 6)]:
        graph.add_edge(source, target, None)

    augmented = build_join_tree(graph, scalar_attr="scalar")
    events = augmented.event_table["event"].value_counts().to_dicts()
    event_counts = {row["event"]: row["count"] for row in events}
    assert event_counts == {"minimum": 4, "join_saddle": 1, "regular": 2}

    reduced = reduce_merge_tree(augmented.tree)

    reduced_vertices = {
        reduced.get_node_data(index)["vertex_index"] for index in reduced.node_indices()
    }
    # Regular vertex 5 is contracted; the root vertex 6 is kept as a boundary regular.
    assert reduced_vertices == {0, 1, 2, 3, 4, 6}
    non_regular_count = event_counts["minimum"] + event_counts["join_saddle"]
    assert reduced.num_nodes() == non_regular_count + 1


def test_reduce_merge_tree_preserves_contracted_vertex_chain_on_edges() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2", "v3", "v4"],
            "scalar": [0.0, 0.2, 0.4, 0.8, 0.1],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1", "v2", "v4"],
            "dst": ["v1", "v2", "v3", "v3"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    network = ScalarNetwork(graph, scalar_attr="scalar")
    join = ScalarNetworkTopology(network).join_tree()
    reduced = join.graph

    assert reduced.num_nodes() == 3
    assert reduced.num_edges() == 2

    edge_payloads = [dict(reduced.get_edge_data_by_index(i)) for i in reduced.edge_indices()]
    contracted_edge = next(e for e in edge_payloads if e["regular_count"] == 2)
    vertex_chain = json.loads(contracted_edge["vertex_chain_json"])

    assert contracted_edge["vertex_chain_count"] == 4
    assert [row["vertex_index"] for row in vertex_chain] == [0, 1, 2, 3]
    assert [row["event"] for row in vertex_chain] == [
        "minimum",
        "regular",
        "regular",
        "join_saddle",
    ]
    assert contracted_edge["from_repr"] == 0
    assert contracted_edge["to_saddle"] == 3


def test_reduce_merge_tree_preserves_contracted_vertex_chain_on_split_tree_edges() -> None:
    # Mirror of the join-tree case: scalars are 1 - (0.0, 0.2, 0.4, 0.8, 0.1), so the
    # superlevel sweep sees a maximum at v0, two regular vertices v1 and v2, and the split
    # saddle v3 where the v4 branch (maximum 0.9) meets the contracted chain.
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2", "v3", "v4"],
            "scalar": [1.0, 0.8, 0.6, 0.2, 0.9],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1", "v2", "v4"],
            "dst": ["v1", "v2", "v3", "v3"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    network = ScalarNetwork(graph, scalar_attr="scalar")
    split = ScalarNetworkTopology(network).split_tree()
    reduced = split.graph

    assert reduced.num_nodes() == 3
    assert reduced.num_edges() == 2

    edge_payloads = [dict(reduced.get_edge_data_by_index(i)) for i in reduced.edge_indices()]
    contracted_edge = next(e for e in edge_payloads if e["regular_count"] == 2)
    vertex_chain = json.loads(contracted_edge["vertex_chain_json"])

    assert contracted_edge["vertex_chain_count"] == 4
    assert [row["vertex_index"] for row in vertex_chain] == [0, 1, 2, 3]
    assert [row["event"] for row in vertex_chain] == [
        "maximum",
        "regular",
        "regular",
        "split_saddle",
    ]
    assert contracted_edge["from_repr"] == 0
    assert contracted_edge["to_saddle"] == 3


def _build_tied_extrema_network(values: list[float]) -> ScalarNetwork:
    """Build v0-v2, v1-v2, v2-v3 so that v2 and v3 tie as one global extremum."""
    vertices = pl.DataFrame({"id": [f"v{i}" for i in range(4)], "m": values})
    edges = pl.DataFrame({"src": ["v0", "v1", "v2"], "dst": ["v2", "v2", "v3"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("m",),
    )
    return ScalarNetwork(graph, scalar_attr="m")


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("join", [0.0, 0.2, 1.0, 1.0]),
        ("split", [1.0, 0.8, 0.0, 0.0]),
    ],
)
def test_simplify_keeps_sweep_root_for_tied_global_extrema(kind: str, values: list[float]) -> None:
    """The simplified tree is rooted at the swept-last tied extremum, vertex 3."""
    topology = ScalarNetworkTopology(_build_tied_extrema_network(values))
    tree = topology.join_tree() if kind == "join" else topology.split_tree()

    simplified = tree.simplify(0.9)

    graph = simplified.graph
    sources = [index for index in graph.node_indices() if graph.in_degree(index) == 0]
    assert len(sources) == 1
    assert graph.get_node_data(sources[0])["vertex_index"] == 3

    root = sources[0]

    persistence = simplified.persistence()
    extended = persistence.filter(pl.col("pair_kind") == "extended")
    assert extended["death_vertex_index"].to_list() == [3]
    simplified.lineages()  # must not raise on a tree with tied global extrema

    metrics = build_tree_metrics(simplified)
    root_row = metrics.critical_node_table.filter(pl.col("tree_node_index") == root)
    assert root_row["lineage_vertex_count"].to_list() == [graph.num_nodes()]
    assert root not in metrics.branch_table["source_tree_node_index"].to_list()
    assert root in metrics.branch_table["target_tree_node_index"].to_list()
    metrics_extended = metrics.persistence_table.filter(pl.col("pair_kind") == "extended")
    assert metrics_extended["death_tree_node_index"].to_list() == [root]


def test_directed_sink_returns_the_unique_root() -> None:
    tree = rx.PyDiGraph()
    leaf_a = tree.add_node({"vertex_index": 0})
    leaf_b = tree.add_node({"vertex_index": 1})
    root = tree.add_node({"vertex_index": 2})
    tree.add_edge(leaf_a, root, {})
    tree.add_edge(leaf_b, root, {})

    assert _directed_sink(tree) == root


def test_directed_sink_rejects_a_tree_with_two_sinks() -> None:
    tree = rx.PyDiGraph()
    source = tree.add_node({"vertex_index": 0})
    first_sink = tree.add_node({"vertex_index": 1})
    second_sink = tree.add_node({"vertex_index": 2})
    tree.add_edge(source, first_sink, {})
    tree.add_edge(source, second_sink, {})

    with pytest.raises(ValueError, match="exactly one directed root"):
        _directed_sink(tree)


def test_ordinary_persistence_pruning_keeps_the_elder_root_lineage() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2", "v3"],
            "scalar": [0.1, 0.8, 0.3, 0.9],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1", "v2"],
            "dst": ["v1", "v2", "v3"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    network = ScalarNetwork(graph, scalar_attr="scalar")
    tree = ScalarNetworkTopology(network).join_tree()

    simplified = tree.simplify(0.5)

    retained_vertex_indices = {
        simplified.graph.get_node_data(index)["vertex_index"]
        for index in simplified.graph.node_indices()
    }
    assert retained_vertex_indices == {0, 1, 3}


def _build_star_network(leaf_count: int) -> ScalarNetwork:
    """Build a star with ``leaf_count`` tied 0.0 leaves and one shallow 0.9 leaf."""
    leaf_ids = [f"leaf{i}" for i in range(leaf_count)] + ["shallow"]
    vertices = pl.DataFrame(
        {
            "id": ["centre", *leaf_ids],
            "scalar": [1.0, *[0.0] * leaf_count, 0.9],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["centre"] * len(leaf_ids),
            "dst": leaf_ids,
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    return ScalarNetwork(graph, scalar_attr="scalar")


def test_merge_tree_edge_order_is_deterministic_across_builds() -> None:
    """Repeated builds of one network give identical edge insertion order."""
    network = _build_star_network(12)

    def build_edge_lists() -> tuple[list, list, list]:
        tree = ScalarNetworkTopology(network).join_tree()
        return (
            list(tree.augmented_graph.weighted_edge_list()),
            list(tree.graph.weighted_edge_list()),
            list(tree.simplify(0.5).graph.weighted_edge_list()),
        )

    first = build_edge_lists()
    for _ in range(20):
        assert build_edge_lists() == first

    augmented_edges = first[0]
    assert len(augmented_edges) == 13
    # The shallow leaf (persistence 0.1) is pruned by simplify(0.5).
    tree = ScalarNetworkTopology(network).join_tree()
    assert len(tree.simplify(0.5).graph.edge_list()) < len(tree.graph.edge_list())
    # Pins the documented ascending node-index child order of `_orient_root_to_leaves`.
    root_children = [target for _, target, _ in augmented_edges]
    assert root_children == sorted(root_children)
