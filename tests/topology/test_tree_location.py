from __future__ import annotations

import rustworkx as rx

from topographer.topology._tree_construction import (
    build_join_tree,
    build_split_tree,
)
from topographer.topology._tree_location import build_tree_location_tables
from topographer.topology._tree_reduction import reduce_merge_tree


def _build_test_graph() -> rx.PyGraph:
    # 5-node graph:
    # Node 0 (m=0.0, label="A") -> Node 1 (m=0.2, label="B") -> Node 2 (m=0.8, label="C")
    # Node 3 (m=0.1, label="D") -> Node 4 (m=0.3, label="E") -> Node 2 (m=0.8, label="C")
    graph = rx.PyGraph()
    labels = ["A", "B", "C", "D", "E"]
    values = [0.0, 0.2, 0.8, 0.1, 0.3]
    for label, val in zip(labels, values, strict=True):
        graph.add_node({"m": val, "label": label})
    graph.add_edge(0, 1, None)
    graph.add_edge(1, 2, None)
    graph.add_edge(3, 4, None)
    graph.add_edge(4, 2, None)
    return graph


def test_build_tree_location_tables_partitions_nodes() -> None:
    graph = _build_test_graph()
    # Join tree: minima are 0 (m=0.0) and 3 (m=0.1).
    # Saddle is 2 (m=0.8).
    # Nodes 1 and 4 are regular points.
    join_result = build_join_tree(graph, scalar_attr="m", node_attrs=("label",))
    reduced_tree = reduce_merge_tree(join_result.tree)

    tables = build_tree_location_tables(reduced_tree)

    # All 5 nodes must be partitioned across regular members and critical nodes.
    assert tables.branch_member_table.height + tables.critical_node_table.height == 5

    # Check regular vertices (1 and 4)
    member_vertices = sorted(tables.branch_member_table.get_column("vertex_index").to_list())
    assert member_vertices == [1, 4]
    assert "label" in tables.branch_member_table.columns
    assert "source_tree_node_index" in tables.branch_member_table.columns
    assert "target_tree_node_index" in tables.branch_member_table.columns
    assert "scalar" in tables.branch_member_table.columns

    # Check critical vertices (0, 2, 3)
    critical_vertices = sorted(tables.critical_node_table.get_column("vertex_index").to_list())
    assert critical_vertices == [0, 2, 3]
    assert "label" in tables.critical_node_table.columns
    assert "event" in tables.critical_node_table.columns
    assert "tree_node_index" in tables.critical_node_table.columns


def test_build_tree_location_tables_split_tree() -> None:
    graph = _build_test_graph()
    split_result = build_split_tree(graph, scalar_attr="m", node_attrs=("label",))
    reduced_tree = reduce_merge_tree(split_result.tree)

    tables = build_tree_location_tables(reduced_tree)

    total_vertices = tables.branch_member_table.height + tables.critical_node_table.height
    assert total_vertices == 5
    all_vertices = sorted(
        tables.branch_member_table.get_column("vertex_index").to_list()
        + tables.critical_node_table.get_column("vertex_index").to_list()
    )
    assert all_vertices == [0, 1, 2, 3, 4]
