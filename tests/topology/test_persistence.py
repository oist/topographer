"""Tests for branch spans and persistence pair derivation on merge trees."""

from __future__ import annotations

import math

import polars as pl
import pytest
import rustworkx as rx

from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology
from topographer.topology._persistence import (
    build_tree_branch_span_table,
    build_tree_critical_branch_span_table,
    build_tree_critical_persistence_table,
    build_tree_extended_persistence_pair_table,
    build_tree_persistence_pair_table,
    build_tree_persistence_table,
)


def _build_scalar_network(values: list[float], edges: list[tuple[int, int]]) -> ScalarNetwork:
    vertices = pl.DataFrame(
        {
            "id": [f"v{i}" for i in range(len(values))],
            "scalar": values,
        }
    )
    edges_df = pl.DataFrame(
        {
            "src": [f"v{u}" for u, _ in edges],
            "dst": [f"v{v}" for _, v in edges],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges_df,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    return ScalarNetwork(graph, scalar_attr="scalar")


def test_branch_span_builder_uses_explicit_span_terminology() -> None:
    network = _build_scalar_network([0.1, 0.8, 0.3], [(0, 1), (1, 2)])
    join = ScalarNetworkTopology(network).join_tree()
    table = join.spans()

    assert {"extremum_scalar", "saddle_scalar", "branch_span"}.issubset(table.columns)
    assert {"birth_scalar", "death_scalar", "persistence"}.isdisjoint(table.columns)
    assert table["branch_span"].to_list() == pytest.approx([0.7, 0.5])


def test_deprecated_branch_span_names_preserve_their_schema() -> None:
    network = _build_scalar_network([0.1, 0.8, 0.3], [(0, 1), (1, 2)])
    join = ScalarNetworkTopology(network).join_tree()

    with pytest.warns(DeprecationWarning, match="branch spans"):
        table = build_tree_persistence_table(
            join._leaves_to_root_graph(),
            extremum_event="minimum",
            saddle_event="join_saddle",
        )

    assert {"birth_scalar", "death_scalar", "persistence"}.issubset(table.columns)
    assert {"extremum_scalar", "saddle_scalar", "branch_span"}.isdisjoint(table.columns)


def test_critical_branch_span_builder_and_deprecated_name_agree() -> None:
    network = _build_scalar_network([0.1, 0.8, 0.3], [(0, 1), (1, 2)])
    join = ScalarNetworkTopology(network).join_tree()
    tree = join._leaves_to_root_graph()
    branch_spans = build_tree_critical_branch_span_table(
        tree,
        source_events=("minimum",),
        target_events=("join_saddle",),
    )
    with pytest.warns(DeprecationWarning):
        legacy = build_tree_critical_persistence_table(
            tree,
            source_events=("minimum",),
            target_events=("join_saddle",),
        )

    assert branch_spans["branch_span"].to_list() == legacy["persistence"].to_list()
    assert branch_spans["source_scalar"].to_list() == legacy["birth_scalar"].to_list()


def test_elder_rule_join_and_split_pairs_are_not_all_branch_spans() -> None:
    network = _build_scalar_network(
        [0.1, 0.8, 0.3, 0.9],
        [(0, 1), (1, 2), (2, 3)],
    )
    topology = ScalarNetworkTopology(network)
    join_pairs = topology.join_tree().persistence().filter(pl.col("pair_kind") == "ordinary")
    split_pairs = topology.split_tree().persistence().filter(pl.col("pair_kind") == "ordinary")

    assert join_pairs.select("birth_vertex_index", "death_vertex_index").rows() == [(2, 1)]
    assert join_pairs["persistence"].to_list() == pytest.approx([0.5])
    assert split_pairs.select("birth_vertex_index", "death_vertex_index").rows() == [(1, 2)]
    assert split_pairs["persistence"].to_list() == pytest.approx([0.5])


def test_ordinary_persistence_reports_only_finite_elder_rule_saddle_deaths() -> None:
    network = _build_scalar_network(
        [0.1, 0.8, 0.3, 0.9],
        [(0, 1), (1, 2), (2, 3)],
    )
    join = ScalarNetworkTopology(network).join_tree()
    table = join.persistence().filter(pl.col("pair_kind") == "ordinary")

    assert table.select("birth_vertex_index", "death_vertex_index").rows() == [(2, 1)]
    assert table["persistence"].to_list() == pytest.approx([0.5])
    assert table["pair_kind"].to_list() == ["ordinary"]


def test_extended_persistence_adds_global_endpoint_pair_to_ordinary_rows() -> None:
    network = _build_scalar_network(
        [0.1, 0.8, 0.3, 0.9],
        [(0, 1), (1, 2), (2, 3)],
    )
    split = ScalarNetworkTopology(network).split_tree()
    table = split.persistence()

    assert table.select("birth_vertex_index", "death_vertex_index", "pair_kind").rows() == [
        (3, 0, "extended"),
        (1, 2, "ordinary"),
    ]
    assert table["persistence"].to_list() == pytest.approx([0.8, 0.5])


def test_persistence_pairs_optionally_include_null_essential_rows() -> None:
    network = _build_scalar_network(
        [0.1, 0.8, 0.3, 0.9],
        [(0, 1), (1, 2), (2, 3)],
    )
    join = ScalarNetworkTopology(network).join_tree()
    table = build_tree_persistence_pair_table(
        join._leaves_to_root_graph(),
        tree_kind="join",
        include_essential=True,
    )

    assert table["is_essential"].to_list() == [False, True]
    essential = table.filter("is_essential").row(0, named=True)
    assert essential["birth_vertex_index"] == 0
    assert essential["death_tree_node_index"] is None
    assert essential["death_scalar"] is None
    assert essential["persistence"] is None


def test_persistence_pairing_handles_multiway_saddles_and_symbolic_ties() -> None:
    network = _build_scalar_network(
        [0.0, 0.2, 0.4, 1.0],
        [(0, 3), (1, 3), (2, 3)],
    )
    table = ScalarNetworkTopology(network).join_tree().persistence()
    ord_table = table.filter(pl.col("pair_kind") == "ordinary")
    assert ord_table["birth_vertex_index"].to_list() == [1, 2]
    assert ord_table["persistence"].to_list() == pytest.approx([0.8, 0.6])

    tied_network = _build_scalar_network(
        [0.0, 0.0, 0.0, 0.0],
        [(0, 3), (1, 3), (2, 3)],
    )
    tied = ScalarNetworkTopology(tied_network).join_tree().persistence()
    tied_ord = tied.filter(pl.col("pair_kind") == "ordinary")
    tied_ext = tied.filter(pl.col("pair_kind") == "extended")
    assert tied_ord["birth_vertex_index"].to_list() == [1, 2]
    assert tied_ext["birth_vertex_index"].to_list() == [0]
    assert tied_ord["persistence"].to_list() == [0.0, 0.0]


def _leaves_to_root(tree: rx.PyDiGraph) -> rx.PyDiGraph:
    rev = rx.PyDiGraph()
    for idx in tree.node_indices():
        rev.add_node(tree.get_node_data(idx))
    for u, v, data in tree.weighted_edge_list():
        rev.add_edge(v, u, data)
    return rev


def test_augmented_and_reduced_trees_have_the_same_persistence_pairs() -> None:
    network = _build_scalar_network(
        [0.1, 0.8, 0.3, 0.9],
        [(0, 1), (1, 2), (2, 3)],
    )
    join = ScalarNetworkTopology(network).join_tree()

    augmented_pairs = build_tree_persistence_pair_table(
        _leaves_to_root(join.augmented_graph), tree_kind="join"
    )
    reduced_pairs = build_tree_persistence_pair_table(
        join._leaves_to_root_graph(), tree_kind="join"
    )

    columns = ["birth_vertex_index", "death_vertex_index", "persistence"]
    assert augmented_pairs.select(columns).equals(reduced_pairs.select(columns))


def test_augmented_and_reduced_trees_have_the_same_extended_pairs() -> None:
    network = _build_scalar_network(
        [0.1, 0.8, 0.3, 0.9],
        [(0, 1), (1, 2), (2, 3)],
    )
    split = ScalarNetworkTopology(network).split_tree()

    augmented_pairs = build_tree_extended_persistence_pair_table(
        _leaves_to_root(split.augmented_graph), tree_kind="split"
    )
    reduced_pairs = build_tree_extended_persistence_pair_table(
        split._leaves_to_root_graph(), tree_kind="split"
    )

    columns = ["birth_vertex_index", "death_vertex_index", "persistence", "pair_kind"]
    assert augmented_pairs.select(columns).equals(reduced_pairs.select(columns))


def test_elder_rule_pairs_obey_join_split_scalar_negation_duality() -> None:
    edges = [(0, 2), (1, 2), (2, 3)]
    values = [0.0, 0.25, 0.75, 1.0]
    join_net = _build_scalar_network(values, edges)
    split_net = _build_scalar_network([-value for value in values], edges)

    join_pairs = (
        ScalarNetworkTopology(join_net)
        .join_tree()
        .persistence()
        .filter(pl.col("pair_kind") == "ordinary")
    )
    split_pairs = (
        ScalarNetworkTopology(split_net)
        .split_tree()
        .persistence()
        .filter(pl.col("pair_kind") == "ordinary")
    )

    columns = ["birth_vertex_index", "death_vertex_index", "persistence"]
    assert join_pairs.select(columns).equals(split_pairs.select(columns))


def test_persistence_pairing_validates_kind_direction_and_cycles() -> None:
    network = _build_scalar_network([0.0, 1.0], [(0, 1)])
    tree = ScalarNetworkTopology(network).join_tree()._leaves_to_root_graph()
    with pytest.raises(ValueError, match="tree_kind"):
        build_tree_persistence_pair_table(tree, tree_kind="contour")
    with pytest.raises(ValueError, match="not monotone"):
        build_tree_persistence_pair_table(tree, tree_kind="split")

    cycle = rx.PyDiGraph()
    left = cycle.add_node({"vertex_index": 0, "scalar": 0.0, "event": "minimum"})
    right = cycle.add_node({"vertex_index": 1, "scalar": 0.0, "event": "regular"})
    cycle.add_edge(left, right, {})
    cycle.add_edge(right, left, {})
    with pytest.raises(ValueError, match="acyclic"):
        build_tree_persistence_pair_table(cycle, tree_kind="join")

    malformed = rx.PyDiGraph()
    malformed.add_node({"vertex_index": 0, "event": "minimum"})
    with pytest.raises(KeyError, match="missing scalar"):
        build_tree_persistence_pair_table(malformed, tree_kind="join")

    invalid_payload = rx.PyDiGraph()
    invalid_payload.add_node(0.0)
    with pytest.raises(TypeError, match="payload must be a dictionary"):
        build_tree_persistence_pair_table(invalid_payload, tree_kind="join")


def test_build_tree_branch_span_table_returns_join_pairs() -> None:
    tree = rx.PyDiGraph()
    minimum_a = tree.add_node(
        {"vertex_index": 0, "label": "00", "scalar": 0.10, "event": "minimum"}
    )
    minimum_b = tree.add_node(
        {"vertex_index": 1, "label": "01", "scalar": 0.30, "event": "minimum"}
    )
    saddle = tree.add_node(
        {"vertex_index": 2, "label": "11", "scalar": 0.80, "event": "join_saddle"}
    )
    tree.add_edge(minimum_a, saddle, {})
    tree.add_edge(minimum_b, saddle, {})

    table = build_tree_branch_span_table(
        tree,
        extremum_event="minimum",
        saddle_event="join_saddle",
    )

    assert table.height == 2
    assert table.columns == [
        "extremum_tree_node_index",
        "saddle_tree_node_index",
        "extremum_vertex_index",
        "saddle_vertex_index",
        "extremum_label",
        "saddle_label",
        "extremum_scalar",
        "saddle_scalar",
        "branch_span",
        "extremum_event",
        "saddle_event",
    ]
    assert table["extremum_vertex_index"].to_list() == [0, 1]
    assert table["branch_span"].to_list() == pytest.approx([0.7, 0.5])


def test_build_tree_branch_span_table_returns_split_pairs() -> None:
    tree = rx.PyDiGraph()
    maximum = tree.add_node({"vertex_index": 0, "label": "11", "scalar": 0.90, "event": "maximum"})
    saddle = tree.add_node(
        {"vertex_index": 1, "label": "10", "scalar": 0.40, "event": "split_saddle"}
    )
    regular = tree.add_node({"vertex_index": 2, "label": "00", "scalar": 0.10, "event": "regular"})
    tree.add_edge(maximum, saddle, {})
    tree.add_edge(saddle, regular, {})

    table = build_tree_branch_span_table(
        tree,
        extremum_event="maximum",
        saddle_event="split_saddle",
    )

    assert table.height == 1
    assert table.row(0, named=True)["extremum_vertex_index"] == 0
    assert table.row(0, named=True)["saddle_vertex_index"] == 1
    assert table.row(0, named=True)["branch_span"] == 0.5
    assert table["branch_span"].min() >= 0.0


def test_build_tree_branch_span_table_returns_empty_schema_when_no_pairs_match() -> None:
    tree = rx.PyDiGraph()
    source = tree.add_node({"vertex_index": 0, "scalar": 0.20, "event": "regular"})
    target = tree.add_node({"vertex_index": 1, "scalar": 0.60, "event": "join_saddle"})
    tree.add_edge(source, target, {})

    table = build_tree_branch_span_table(
        tree,
        extremum_event="minimum",
        saddle_event="join_saddle",
    )

    assert table.height == 0
    assert table.schema == {
        "extremum_tree_node_index": pl.Int64,
        "saddle_tree_node_index": pl.Int64,
        "extremum_vertex_index": pl.Int64,
        "saddle_vertex_index": pl.Int64,
        "extremum_label": pl.String,
        "saddle_label": pl.String,
        "extremum_scalar": pl.Float64,
        "saddle_scalar": pl.Float64,
        "branch_span": pl.Float64,
        "extremum_event": pl.String,
        "saddle_event": pl.String,
    }


def test_build_tree_branch_span_table_uses_explicit_generic_label_attribute() -> None:
    tree = rx.PyDiGraph()
    minimum = tree.add_node(
        {"vertex_index": 4, "site": "south", "scalar": 0.1, "event": "minimum"}
    )
    saddle = tree.add_node(
        {"vertex_index": 7, "site": "pass", "scalar": 0.8, "event": "join_saddle"}
    )
    tree.add_edge(minimum, saddle, {})

    table = build_tree_branch_span_table(
        tree,
        extremum_event="minimum",
        saddle_event="join_saddle",
        label_attr="site",
    )

    assert table.select("extremum_label", "saddle_label").row(0) == ("south", "pass")


def test_build_tree_critical_branch_span_table_includes_saddle_to_promoted_root() -> None:
    tree = rx.PyDiGraph()
    minimum = tree.add_node({"vertex_index": 0, "label": "low", "scalar": 0.0, "event": "minimum"})
    saddle = tree.add_node(
        {"vertex_index": 1, "label": "pass", "scalar": 0.5, "event": "join_saddle"}
    )
    root = tree.add_node({"vertex_index": 2, "label": "high", "scalar": 1.0, "event": "regular"})
    tree.add_edge(minimum, saddle, {})
    tree.add_edge(saddle, root, {})

    table = build_tree_critical_branch_span_table(
        tree,
        source_events=("minimum", "join_saddle"),
        target_events=("join_saddle", "maximum"),
        root_node_index=root,
        root_event="maximum",
    )

    assert table.select("source_label", "target_label", "source_event", "target_event").rows() == [
        ("pass", "high", "join_saddle", "maximum"),
        ("low", "pass", "minimum", "join_saddle"),
    ]
    assert table["branch_span"].to_list() == pytest.approx([0.5, 0.5])


def test_constructed_toy_trees_have_hand_verified_branch_spans() -> None:
    network = _build_scalar_network(
        [0.1, 0.8, 0.3, 0.9],
        [(0, 1), (1, 2), (2, 3)],
    )
    topology = ScalarNetworkTopology(network)
    join_table = topology.join_tree().spans()
    split_table = topology.split_tree().spans()

    assert join_table.select("extremum_vertex_index", "saddle_vertex_index").rows() == [
        (0, 1),
        (2, 1),
    ]
    assert join_table["branch_span"].to_list() == pytest.approx([0.7, 0.5])
    assert split_table.select("extremum_vertex_index", "saddle_vertex_index").rows() == [
        (3, 2),
        (1, 2),
    ]
    assert split_table["branch_span"].to_list() == pytest.approx([0.6, 0.5])


def test_join_and_negated_split_trees_have_matching_branch_spans() -> None:
    edges = [(0, 2), (1, 2), (2, 3)]
    values = [0.0, 0.25, 0.75, 1.0]
    join_net = _build_scalar_network(values, edges)
    split_net = _build_scalar_network([-value for value in values], edges)

    join_table = ScalarNetworkTopology(join_net).join_tree().spans()
    split_table = ScalarNetworkTopology(split_net).split_tree().spans()

    assert (
        join_table["extremum_vertex_index"].to_list()
        == split_table["extremum_vertex_index"].to_list()
    )
    assert (
        join_table["saddle_vertex_index"].to_list() == split_table["saddle_vertex_index"].to_list()
    )
    assert join_table["branch_span"].to_list() == pytest.approx(
        split_table["branch_span"].to_list()
    )


def test_constructed_tree_branch_spans_are_finite_nonnegative_and_bounded() -> None:
    network = _build_scalar_network(
        [0.1, 0.8, 0.3, 0.9],
        [(0, 1), (1, 2), (2, 3)],
    )
    topology = ScalarNetworkTopology(network)

    for tree in (topology.join_tree(), topology.split_tree()):
        table = tree.spans()
        assert table.height > 0
        assert all(math.isfinite(value) for value in table["branch_span"])
        assert table["branch_span"].min() >= 0.0
        assert table["branch_span"].max() <= 0.8


def test_symbolic_tie_perturbation_preserves_zero_branch_spans() -> None:
    network = _build_scalar_network(
        [0.0, 0.0, 0.0, 0.0],
        [(0, 3), (1, 3), (2, 3)],
    )
    table = ScalarNetworkTopology(network).join_tree().spans()

    assert table.height == 3
    assert table["branch_span"].to_list() == [0.0, 0.0, 0.0]
    assert table["extremum_scalar"].to_list() == [0.0, 0.0, 0.0]
    assert table["saddle_scalar"].to_list() == [0.0, 0.0, 0.0]


def test_branch_span_construction_rejects_disconnected_merge_input() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2", "v3", "v4", "v5"],
            "scalar": [0.0, 0.2, 0.8, 0.1, 0.3, 0.9],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1", "v3", "v4"],
            "dst": ["v2", "v2", "v5", "v5"],
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
    with pytest.raises(ValueError, match="connected graph"):
        ScalarNetwork(graph, scalar_attr="scalar")
