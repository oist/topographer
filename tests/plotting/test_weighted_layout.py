"""Tests for the leaf-count-weighted horizontal layout and tree re-layout."""

from __future__ import annotations

import polars as pl
import pytest
import rustworkx as rx

from topographer.plotting import (
    TreePlotConfig,
    make_tree_layout,
    plot_tree,
    relayout_tree_plot_data,
)


def _tree(scalars: list[float], parents: list[int | None]) -> rx.PyDiGraph:
    """Build a tree with node ``i`` below ``parents[i]`` (``None`` for the root)."""

    graph = rx.PyDiGraph()
    for index, scalar in enumerate(scalars):
        graph.add_node({"vertex_index": index, "scalar": scalar, "event": "regular"})
    for child, parent in enumerate(parents):
        if parent is not None:
            graph.add_edge(parent, child, {"regular_count": 1})
    return graph


def _example(root_kind: str = "minimum") -> rx.PyDiGraph:
    # Root 0; node 1 is a big funnel with three leaves, node 2 a lone leaf.
    scalars = [0.0, 0.5, 1.0, 0.9, 1.2, 1.1]
    if root_kind == "maximum":
        scalars = [-s for s in scalars]
    return _tree(scalars, [None, 0, 0, 1, 1, 1])


def _layout(root_kind: str = "minimum", **kwargs):
    return make_tree_layout(
        _example(root_kind), root_kind=root_kind, horizontal_layout="weighted", **kwargs
    )


def _leaves(plot_data) -> set[int]:
    sources = set(plot_data.edge_table["source_tree_node_index"])
    return set(plot_data.node_table["tree_node_index"]) - sources


@pytest.mark.parametrize("root_kind", ["minimum", "maximum"])
def test_weighted_y_equals_scalar(root_kind: str) -> None:
    data = _layout(root_kind)
    assert data.node_table.filter(pl.col("y") != pl.col("scalar")).is_empty()
    assert data.edge_table.filter(pl.col("y_source") != pl.col("source_scalar")).is_empty()
    assert data.edge_table.filter(pl.col("y_target") != pl.col("target_scalar")).is_empty()


def test_weighted_no_leaf_collisions() -> None:
    data = _layout()
    leaf_x = sorted(
        data.node_table.filter(pl.col("tree_node_index").is_in(list(_leaves(data))))["x"]
    )
    assert len(leaf_x) == 4
    gaps = [b - a for a, b in zip(leaf_x, leaf_x[1:])]
    assert all(gap >= 1 - 1e-12 for gap in gaps)


def test_weighted_parent_within_children_span() -> None:
    data = _layout()
    x = dict(zip(data.node_table["tree_node_index"], data.node_table["x"], strict=True))
    leaves = {1: 3, 2: 1, 3: 1, 4: 1, 5: 1}
    children: dict[int, list[int]] = {}
    for row in data.edge_table.to_dicts():
        children.setdefault(row["source_tree_node_index"], []).append(
            row["target_tree_node_index"]
        )
    leaves[0] = 4
    for parent, kids in children.items():
        values = [x[k] for k in kids]
        assert min(values) <= x[parent] <= max(values)
        expected = sum(leaves[k] * x[k] for k in kids) / sum(leaves[k] for k in kids)
        assert x[parent] == pytest.approx(expected, abs=1e-12)


def test_weighted_order_persistence_then_size() -> None:
    data = _layout()
    x = dict(zip(data.node_table["tree_node_index"], data.node_table["x"], strict=True))
    # Subtree of 1 reaches 1.2 > 1.0 of leaf 2, so it comes first (smaller x).
    assert x[1] < x[2]
    # Within node 1 the leaf with the highest scalar (node 4) is first.
    assert x[4] < x[5] < x[3]

    # Equal reach: the larger leaf count goes first.
    graph = _tree([0.0, 0.5, 0.6, 1.0, 1.0, 1.0], [None, 0, 0, 1, 1, 2])
    tie = make_tree_layout(graph, horizontal_layout="weighted")
    tx = dict(zip(tie.node_table["tree_node_index"], tie.node_table["x"], strict=True))
    assert tx[1] < tx[2]


def test_weighted_subtree_span_proportional() -> None:
    data = _layout()
    x = dict(zip(data.node_table["tree_node_index"], data.node_table["x"], strict=True))
    assert max(x[3], x[4], x[5]) - min(x[3], x[4], x[5]) == pytest.approx(2.0)
    assert x[2] == pytest.approx(3.5) or x[2] == pytest.approx(0.5)


def test_weighted_degenerate() -> None:
    single = make_tree_layout(_tree([0.0], [None]), horizontal_layout="weighted")
    assert single.node_table["x"].to_list() == [0.5]

    pair = make_tree_layout(_tree([0.0, 1.0], [None, 0]), horizontal_layout="weighted")
    assert pair.node_table["x"].to_list() == [0.5, 0.5]

    chain = make_tree_layout(_tree([0.0, 1.0, 2.0], [None, 0, 1]), horizontal_layout="weighted")
    assert set(chain.node_table["x"]) == {0.5}

    star = make_tree_layout(
        _tree([0.0] + [1.0 + i * 1e-3 for i in range(1000)], [None] + [0] * 1000),
        horizontal_layout="weighted",
    )
    leaf_x = sorted(star.node_table.filter(pl.col("tree_node_index") != 0)["x"])
    assert leaf_x == [i + 0.5 for i in range(1000)]
    assert star.node_table.filter(pl.col("tree_node_index") == 0).item(0, "x") == pytest.approx(
        500.0
    )

    equal = make_tree_layout(
        _tree([0.0, 1.0, 1.0, 1.0], [None, 0, 0, 0]), horizontal_layout="weighted"
    )
    assert sorted(equal.node_table["x"])[0] == 0.5
    assert sorted(set(equal.node_table["x"])) == [0.5, 1.5, 2.5]

    depth = 5000
    deep = make_tree_layout(
        _tree([float(i) for i in range(depth)], [None, *range(depth - 1)]),
        horizontal_layout="weighted",
    )
    assert deep.node_table.height == depth
    assert set(deep.node_table["x"]) == {0.5}


def test_relayout_matches_make_tree_layout() -> None:
    graph = _example()
    base = make_tree_layout(graph)
    injected = base.edge_table.with_columns(pl.lit(7).alias("custom_edge_column"))
    from dataclasses import replace

    base = replace(base, edge_table=injected)
    relaid = relayout_tree_plot_data(base)
    direct = make_tree_layout(graph, horizontal_layout="weighted")
    assert relaid.node_table["x"].to_list() == direct.node_table["x"].to_list()
    assert relaid.edge_table["x_source"].to_list() == direct.edge_table["x_source"].to_list()
    assert relaid.edge_table["x_target"].to_list() == direct.edge_table["x_target"].to_list()
    assert relaid.edge_segment_table.equals(direct.edge_segment_table)
    assert relaid.edge_table["custom_edge_column"].to_list() == [7] * 5
    assert relaid.edge_table["branch_vertex_count"].to_list() == (
        direct.edge_table["branch_vertex_count"].to_list()
    )
    assert relaid.node_table["y"].to_list() == base.node_table["y"].to_list()


def test_relayout_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="weighted"):
        relayout_tree_plot_data(make_tree_layout(_example()), horizontal_layout="iplotx")


@pytest.mark.parametrize("edge_shape", ["orthogonal", "direct", "trunk"])
def test_weighted_all_edge_shapes(edge_shape: str) -> None:
    chart = plot_tree(_layout(), config=TreePlotConfig(edge_shape=edge_shape))
    spec = chart.to_dict()
    layers = spec["layer"]
    assert any(
        layer.get("encoding", {}).get("detail", {}).get("field") == "logical_edge_index"
        for layer in layers
    )
    edge_layer = next(
        layer
        for layer in layers
        if layer.get("encoding", {}).get("detail", {}).get("field") == "logical_edge_index"
    )
    assert edge_layer["mark"]["type"] == ("rule" if edge_shape == "direct" else "line")


def test_invalid_horizontal_layout() -> None:
    with pytest.raises(ValueError, match="weighted"):
        make_tree_layout(_example(), horizontal_layout="nope")


def test_relayout_moves_vertex_table_like_make_tree_layout() -> None:
    from topographer import ScalarNetworkTopology
    from topographer.examples import build_cube_scalar_network

    network = build_cube_scalar_network()
    tree = ScalarNetworkTopology(network).split_tree()
    base = make_tree_layout(tree, network=network)
    direct = make_tree_layout(tree, network=network, horizontal_layout="weighted")
    relaid = relayout_tree_plot_data(base)
    assert relaid.vertex_table is not None
    assert relaid.vertex_table.sort("vertex_id").equals(direct.vertex_table.sort("vertex_id"))


def test_relayout_vertex_table_survives_parquet_roundtrip(tmp_path) -> None:
    from dataclasses import replace

    from topographer import ScalarNetworkTopology
    from topographer.examples import build_cube_scalar_network

    network = build_cube_scalar_network()
    tree = ScalarNetworkTopology(network).split_tree()
    base = make_tree_layout(tree, network=network)
    direct = make_tree_layout(tree, network=network, horizontal_layout="weighted")
    tables = {}
    for name in ("node_table", "edge_table", "edge_segment_table", "vertex_table"):
        path = tmp_path / f"{name}.parquet"
        getattr(base, name).write_parquet(path)
        tables[name] = pl.read_parquet(path)
    reloaded = replace(base, **tables)
    relaid = relayout_tree_plot_data(reloaded)
    assert relaid.vertex_table.sort("vertex_id").equals(direct.vertex_table.sort("vertex_id"))
    assert relaid.edge_segment_table.equals(direct.edge_segment_table)


def test_relayout_vertex_table_chain_shared_endpoint() -> None:
    from dataclasses import replace

    base = make_tree_layout(_tree([0.0, 1.0, 2.0, 1.5], [None, 0, 1, 1]))
    chain_vertices = pl.DataFrame(
        {
            "vertex_id": ["a", "b", "c"],
            "x": [base.node_table.item(1, "x")] * 3,
            "y": [1.0, 1.0, 1.0],
            "is_critical": [False, False, False],
        }
    )
    relaid = relayout_tree_plot_data(replace(base, vertex_table=chain_vertices))
    assert relaid.vertex_table.height == 3
