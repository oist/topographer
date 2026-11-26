from __future__ import annotations

import polars as pl
import rustworkx as rx

from topographer.plotting import build_tree_plot_data
from topographer.plotting._swarm import SwarmConfig, compute_branch_local_swarm


def _build_simple_tree_plot_data():
    # Tree with root 2, children 0 and 1:
    # Edge (0, 2) and Edge (1, 2)
    tree = rx.PyDiGraph()
    n0 = tree.add_node({"scalar": 0.0, "event": "minimum", "vertex_index": 0})
    n1 = tree.add_node({"scalar": 0.1, "event": "minimum", "vertex_index": 1})
    n2 = tree.add_node({"scalar": 1.0, "event": "maximum", "vertex_index": 2})
    tree.add_edge(n0, n2, {"source_scalar": 0.0, "target_scalar": 1.0})
    tree.add_edge(n1, n2, {"source_scalar": 0.1, "target_scalar": 1.0})
    return build_tree_plot_data(tree, root_kind="maximum")


def test_compute_branch_local_swarm_sparse_centered():
    plot_data = _build_simple_tree_plot_data()
    # Member table with one stop codon on branch (0, 2)
    member_table = pl.DataFrame(
        [
            {
                "source_tree_node_index": 0,
                "target_tree_node_index": 2,
                "vertex_index": 10,
                "scalar": 0.5,
                "has_stop_codon": True,
            }
        ]
    )

    swarm = compute_branch_local_swarm(
        member_table, plot_data, config=SwarmConfig(step_x=0.2, delta_y=0.05)
    )

    assert swarm.height == 1
    row = swarm.to_dicts()[0]
    # Should match trunk x of branch (0, 2)
    edge_x = plot_data.edge_table.filter(
        ((pl.col("source_tree_node_index") == 0) & (pl.col("target_tree_node_index") == 2))
        | ((pl.col("source_tree_node_index") == 2) & (pl.col("target_tree_node_index") == 0))
    )["x_target"][0]
    assert row["x"] == edge_x
    assert row["y"] == 0.5


def test_compute_branch_local_swarm_symmetric_spread():
    plot_data = _build_simple_tree_plot_data()
    # 3 colliding stop codons on branch (0, 2) at identical scalar 0.5
    member_table = pl.DataFrame(
        [
            {
                "source_tree_node_index": 0,
                "target_tree_node_index": 2,
                "vertex_index": 10,
                "scalar": 0.5,
                "has_stop_codon": True,
            },
            {
                "source_tree_node_index": 0,
                "target_tree_node_index": 2,
                "vertex_index": 11,
                "scalar": 0.5,
                "has_stop_codon": True,
            },
            {
                "source_tree_node_index": 0,
                "target_tree_node_index": 2,
                "vertex_index": 12,
                "scalar": 0.5,
                "has_stop_codon": True,
            },
        ]
    )

    swarm = compute_branch_local_swarm(
        member_table, plot_data, config=SwarmConfig(step_x=0.2, delta_y=0.05)
    )

    assert swarm.height == 3
    edge_x = plot_data.edge_table.filter(
        ((pl.col("source_tree_node_index") == 0) & (pl.col("target_tree_node_index") == 2))
        | ((pl.col("source_tree_node_index") == 2) & (pl.col("target_tree_node_index") == 0))
    )["x_target"][0]

    x_values = swarm.get_column("x").to_list()
    # Offsets should be 0, +0.2, -0.2 relative to edge_x
    expected_x = sorted([edge_x, edge_x + 0.2, edge_x - 0.2])
    assert sorted(x_values) == expected_x

    # No vertical jitter: all y must be exactly 0.5
    assert swarm.get_column("y").to_list() == [0.5, 0.5, 0.5]


def test_compute_branch_local_swarm_critical_nodes():
    plot_data = _build_simple_tree_plot_data()
    critical_table = pl.DataFrame(
        [
            {
                "tree_node_index": 0,
                "vertex_index": 0,
                "scalar": 0.0,
                "has_stop_codon": True,
            }
        ]
    )
    empty_members = pl.DataFrame(
        schema={
            "source_tree_node_index": pl.Int64,
            "target_tree_node_index": pl.Int64,
            "vertex_index": pl.Int64,
            "scalar": pl.Float64,
            "has_stop_codon": pl.Boolean,
        }
    )

    swarm = compute_branch_local_swarm(empty_members, plot_data, critical_table=critical_table)

    assert swarm.height == 1
    node_0_coords = plot_data.node_table.filter(pl.col("tree_node_index") == 0).to_dicts()[0]
    row = swarm.to_dicts()[0]
    assert row["x"] == node_0_coords["x"]
    assert row["y"] == node_0_coords["y"]
