"""Tests for sequence-free tree vertex counts and Horton-Strahler orders."""

from __future__ import annotations

import polars as pl
import pytest

from topographer import Graph, ScalarNetwork, ScalarNetworkTopology
from topographer.topology import TreeMetrics, build_tree_metrics
from topographer.topology._tree_metrics import _rooted_merge_tree


def _network(
    fitness: list[float], edges: list[tuple[int, int]], ids: list[int] | None = None
) -> ScalarNetwork:
    ids = ids or [10 * (i + 1) for i in range(len(fitness))]
    vertices = pl.DataFrame({"variant": ids, "fitness": fitness})
    edge_table = pl.DataFrame({"source": [a for a, _ in edges], "target": [b for _, b in edges]})
    graph = Graph.from_tables(
        vertices,
        edge_table,
        vertex_id_col="variant",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness",),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def _fixture() -> ScalarNetwork:
    return _network(
        [0.9, 0.5, 0.8, 0.3, 0.1], [(10, 20), (20, 35), (30, 20), (35, 40)], [10, 20, 30, 35, 40]
    )


def _reference_horton_strahler(children: dict[int, list[int]], root: int) -> dict[int, int]:
    """Independent recursive Horton-Strahler reference."""

    orders: dict[int, int] = {}

    def visit(node: int) -> int:
        child_orders = [visit(child) for child in children[node]]
        if not child_orders:
            orders[node] = 1
        else:
            largest = max(child_orders)
            orders[node] = largest + 1 if child_orders.count(largest) >= 2 else largest
        return orders[node]

    visit(root)
    return orders


def _public_id(network: ScalarNetwork, tree, node: int) -> int:
    """Public vertex ID of a tree node."""

    public = dict(zip(network.graph.node_indices(), network.vertex_ids, strict=True))
    return public[int(tree.graph.get_node_data(node)["vertex_index"])]


def test_split_tree_fixture_values() -> None:
    network = _fixture()
    tree = ScalarNetworkTopology(network).split_tree()
    metrics = build_tree_metrics(tree)
    assert isinstance(metrics, TreeMetrics)

    def pid(node: object) -> int:
        return _public_id(network, tree, int(node))  # type: ignore[call-overload]

    branches = {
        (pid(r["source_tree_node_index"]), pid(r["target_tree_node_index"])): (
            r["branch_vertex_count"],
            r["lineage_vertex_count"],
            r["horton_strahler_order"],
        )
        for r in metrics.branch_table.to_dicts()
    }
    assert branches == {
        (10, 20): (1, 1, 1),
        (30, 20): (1, 1, 1),
        (20, 40): (2, 4, 2),
    }
    critical = {
        pid(r["tree_node_index"]): r["lineage_vertex_count"]
        for r in metrics.critical_node_table.to_dicts()
    }
    assert critical == {10: 1, 30: 1, 20: 3, 40: 5}

    persistence = {
        (pid(r["birth_tree_node_index"]), r["pair_kind"]): r["lineage_vertex_count"]
        for r in metrics.persistence_table.to_dicts()
    }
    assert persistence[30, "ordinary"] == 1
    assert [v for (_, kind), v in persistence.items() if kind == "extended"] == [5]
    assert set(tree.persistence().columns) < set(metrics.persistence_table.columns)
    assert {"lineage_vertex_count", "horton_strahler_order"} <= set(
        metrics.persistence_table.columns
    )
    assert metrics.persistence_table.height == tree.persistence().height


def test_join_tree_is_a_single_path() -> None:
    tree = ScalarNetworkTopology(_fixture()).join_tree()
    metrics = build_tree_metrics(tree)
    assert metrics.branch_table.height == 1
    row = metrics.branch_table.row(0, named=True)
    assert (row["branch_vertex_count"], row["lineage_vertex_count"]) == (4, 4)
    root = _rooted_merge_tree(tree).root
    critical = {r["tree_node_index"]: r for r in metrics.critical_node_table.to_dicts()}
    assert critical[root]["lineage_vertex_count"] == 5
    assert row["target_tree_node_index"] == root


@pytest.mark.parametrize("kind", ["split", "join"])
def test_unsimplified_tree_properties(kind: str) -> None:
    network = _network(
        [0.95, 0.4, 0.8, 0.2, 0.7, 0.55, 0.1, 0.6, 0.3],
        [(10, 20), (20, 30), (30, 40), (40, 50), (30, 60), (60, 70), (70, 80), (50, 90), (80, 90)],
    )
    topology = ScalarNetworkTopology(network)
    tree = topology.split_tree() if kind == "split" else topology.join_tree()
    metrics = build_tree_metrics(tree)
    rooted = _rooted_merge_tree(tree)
    branch = {int(r["source_tree_node_index"]): r for r in metrics.branch_table.to_dicts()}
    critical = {int(r["tree_node_index"]): r for r in metrics.critical_node_table.to_dicts()}

    assert rooted.root not in branch
    assert all(
        int(r["target_tree_node_index"]) != int(r["source_tree_node_index"])
        for r in branch.values()
    )
    total = sum(r["branch_vertex_count"] for r in branch.values()) + 1
    assert total == critical[rooted.root]["lineage_vertex_count"]
    assert total == network.graph.num_nodes()
    for node, row in branch.items():
        below = sum(branch[c]["lineage_vertex_count"] for c in rooted.children[node])
        assert row["lineage_vertex_count"] == row["branch_vertex_count"] + below
        assert row["branch_vertex_count"] == rooted.regular_count[node] + 1
    expected = _reference_horton_strahler(rooted.children, rooted.root)
    assert {n: int(r["horton_strahler_order"]) for n, r in critical.items()} == expected
    assert {n: int(r["horton_strahler_order"]) for n, r in branch.items()} == {
        n: expected[n] for n in branch
    }


def test_simplified_tree_still_folds() -> None:
    network = _fixture()
    tree = ScalarNetworkTopology(network).split_tree()
    simplified = tree.simplify(0.35)
    metrics = build_tree_metrics(simplified)
    assert metrics.branch_table.height < build_tree_metrics(tree).branch_table.height
    rooted = _rooted_merge_tree(simplified)
    total = metrics.critical_node_table.filter(pl.col("tree_node_index") == rooted.root)[
        "lineage_vertex_count"
    ][0]
    assert total == 4

    def pid(node: object) -> int:
        return _public_id(network, simplified, int(node))  # type: ignore[call-overload]

    branches = {
        (pid(r["source_tree_node_index"]), pid(r["target_tree_node_index"])): (
            r["branch_vertex_count"],
            r["lineage_vertex_count"],
        )
        for r in metrics.branch_table.to_dicts()
    }
    assert branches == {(10, 20): (1, 1), (20, 40): (2, 3)}


def test_single_vertex_network() -> None:
    network = _network([0.5], [], [10])
    tree = ScalarNetworkTopology(network).split_tree()
    metrics = build_tree_metrics(tree)
    assert metrics.branch_table.height == 0
    assert metrics.critical_node_table["lineage_vertex_count"].to_list() == [1]
    pairs = metrics.persistence_table
    assert pairs.height == 1
    row = pairs.row(0, named=True)
    assert row["pair_kind"] == "extended"
    assert row["birth_scalar"] == row["death_scalar"]
    assert row["lineage_vertex_count"] == 1


def test_non_merge_tree_raises_type_error() -> None:
    with pytest.raises(TypeError):
        build_tree_metrics(object())  # type: ignore[arg-type]
