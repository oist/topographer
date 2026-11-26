"""Tests for canonical topology presentation primitives."""

from __future__ import annotations

from typing import Any

import pytest

from topographer.examples._cases import CASES, export_case
from topographer.examples._presentation import (
    GraphPlotData,
    build_canonical_persistence_table,
    build_graph_plot_data,
    plot_canonical_persistence_diagrams,
    plot_graph_view,
    plot_scalar_network_view,
)
from topographer.examples.cube import cube_edge_table, cube_node_table
from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology


@pytest.fixture
def cube_network() -> tuple[Graph, ScalarNetwork, dict[str, tuple[float, float]]]:
    nodes = cube_node_table()
    edges = cube_edge_table()
    g = Graph.from_tables(
        nodes,
        edges,
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height", "label", "x", "y"),
        edge_attribute_cols=("distance", "capacity"),
    )
    sn = ScalarNetwork(g, scalar_attr="height")
    coords = {row["node_id"]: (float(row["x"]), float(row["y"])) for row in nodes.to_dicts()}
    return g, sn, coords


def test_build_graph_plot_data_and_views(
    cube_network: tuple[Graph, ScalarNetwork, dict[str, tuple[float, float]]],
) -> None:
    g, sn, coords = cube_network

    # 1. Build plot data
    plot_data = build_graph_plot_data(sn, coordinates=coords)
    assert isinstance(plot_data, GraphPlotData)
    assert plot_data.node_table.height == 8
    assert plot_data.edge_table.height == 12
    assert "height" in plot_data.node_table.columns
    assert "x_source" in plot_data.edge_table.columns

    # 2. Graph view chart (topology only)
    graph_chart = plot_graph_view(plot_data)
    graph_dict = graph_chart.to_dict()
    assert "layer" in graph_dict
    # Layers: edges rule, nodes circle, labels text
    assert len(graph_dict["layer"]) == 3

    # 3. Scalar network view chart (scalar encoded)
    sn_chart = plot_scalar_network_view(plot_data, scalar_attr="height")
    sn_dict = sn_chart.to_dict()
    assert "layer" in sn_dict

    # 4. Edge highlight on scalar network view
    hl_chart = plot_scalar_network_view(
        plot_data,
        scalar_attr="height",
        highlight_edges=[("000", "001"), ("001", "011"), ("011", "010"), ("010", "000")],
    )
    hl_dict = hl_chart.to_dict()
    assert "layer" in hl_dict


def test_canonical_persistence_diagrams(
    cube_network: tuple[Graph, ScalarNetwork, dict[str, tuple[float, float]]],
) -> None:
    _, sn, _ = cube_network
    top = ScalarNetworkTopology(sn)

    comp = top.component_persistence()

    # Component persistence table with pair kind and endpoint labels
    table = build_canonical_persistence_table(comp)
    assert table.height == 3
    assert "persistence_family" not in table.columns
    assert set(table["pair_kind"].unique().to_list()) == {
        "ordinary",
        "relative",
        "extended_positive",
    }
    assert set(table["birth_endpoint"].to_list()) == set(
        v for row in comp.pairs["birth_simplex"].to_list() for v in row
    )

    chart = plot_canonical_persistence_diagrams(table, scalar_domain=(0.0, 1.0))
    assert chart.to_dict(validate=True) is not None


def test_export_example_artifacts(tmp_path: Any) -> None:
    case = CASES["cube"]
    paths = export_case(case, case.build_topology(), tmp_path)

    # 10 tables
    expected_tables = [
        "table_vertices",
        "table_edges",
        "table_summary",
        "table_persistence",
        "table_join_events",
        "table_join_persistence",
        "table_join_qualifying",
        "table_split_events",
        "table_split_persistence",
        "table_split_qualifying",
    ]
    for tbl in expected_tables:
        assert tbl in paths
        assert paths[tbl].is_file()

    # 8 charts in both svg and html (16 chart files)
    expected_charts = [
        "scalar_network",
        "canonical_persistence",
        "join_tree_augmented",
        "join_tree_reduced",
        "join_tree_simplified",
        "split_tree_augmented",
        "split_tree_reduced",
        "split_tree_simplified",
    ]
    for ch in expected_charts:
        assert f"chart_{ch}_svg" in paths
        assert paths[f"chart_{ch}_svg"].is_file()
        assert f"chart_{ch}_html" in paths
        assert paths[f"chart_{ch}_html"].is_file()
