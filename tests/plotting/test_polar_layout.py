"""Tests for the polar coordinate system of tree plots."""

from __future__ import annotations

import math

import pytest
import rustworkx as rx

from topographer.plotting import TreePlotConfig, make_tree_layout, plot_tree
from topographer.plotting._polar import (
    format_tick,
    make_polar_transform,
    polar_edge_path_rows,
    polar_node_rows,
    polar_straight_edge_rows,
    polar_tick_values,
    ray_label_rows,
    ring_rows,
)


def _tree(scalars: list[float], parents: list[int | None]) -> rx.PyDiGraph:
    graph = rx.PyDiGraph()
    for index, scalar in enumerate(scalars):
        graph.add_node({"vertex_index": index, "scalar": scalar, "event": "regular"})
    for child, parent in enumerate(parents):
        if parent is not None:
            graph.add_edge(parent, child, {"regular_count": 1})
    return graph


def _data(root_kind: str = "minimum"):
    scalars = [0.0, 0.5, 1.0, 0.9, 1.2, 1.1]
    if root_kind == "maximum":
        scalars = [2.0 - s for s in scalars]
    return make_tree_layout(
        _tree(scalars, [None, 0, 0, 1, 1, 1]), root_kind=root_kind, horizontal_layout="weighted"
    )


def _config(**kwargs) -> TreePlotConfig:
    return TreePlotConfig(coordinate_system="polar", **kwargs)


def _transform(data, config):
    return make_polar_transform(
        data.node_table, data.edge_table, data.root_node_index, data.root_kind, config
    )


@pytest.mark.parametrize("root_kind", ["minimum", "maximum"])
@pytest.mark.parametrize("fraction", [0.0, 0.25])
def test_polar_radius_equals_scalar(root_kind: str, fraction: float) -> None:
    data = _data(root_kind)
    config = _config(angular_span_deg=180.0, angular_start_deg=0.0, inner_radius_fraction=fraction)
    t = _transform(data, config)
    root = t.root_scalar
    for row in polar_node_rows(data.node_table.to_dicts(), t):
        assert math.hypot(row["x"], row["y"]) == pytest.approx(
            t.inner_radius + abs(row["scalar"] - root), abs=1e-9
        )
    for row in polar_straight_edge_rows(data.edge_table.to_dicts(), t):
        assert math.hypot(row["x_source"], row["y_source"]) == pytest.approx(
            t.inner_radius + abs(row["source_scalar"] - root), abs=1e-9
        )
        assert math.hypot(row["x_target"], row["y_target"]) == pytest.approx(
            t.inner_radius + abs(row["target_scalar"] - root), abs=1e-9
        )
    for shape in ("orthogonal", "trunk"):
        paths = polar_edge_path_rows(data.edge_table.to_dicts(), t, shape, 0.85, 32)
        for row in paths:
            assert math.hypot(row["x"], row["y"]) == pytest.approx(row["radius"], abs=1e-9)
            assert row["radius"] >= t.inner_radius - 1e-9
    arcs = [
        row
        for row in polar_edge_path_rows(data.edge_table.to_dicts(), t, "orthogonal", 0, 32)
        if row["segment_kind"] == "arc"
    ]
    for row in arcs:
        assert row["radius"] == pytest.approx(t.inner_radius + abs(row["source_scalar"] - root))


def test_polar_no_leaf_collisions() -> None:
    data = _data()
    config = _config(angular_span_deg=180.0, angular_start_deg=0.0)
    t = _transform(data, config)
    rows = polar_node_rows(data.node_table.to_dicts(), t)
    internal = set(data.edge_table["source_tree_node_index"])
    angles = sorted(r["angle"] for r in rows if r["tree_node_index"] not in internal)
    assert len(set(angles)) == len(angles) == 4
    gaps = [b - a for a, b in zip(angles, angles[1:], strict=False)]
    assert min(gaps) == pytest.approx(math.radians(180.0) / 4)


def test_polar_parent_within_children_angle() -> None:
    data = _data()
    t = _transform(data, _config(angular_span_deg=180.0, angular_start_deg=0.0))
    angle = {
        r["tree_node_index"]: r["angle"] for r in polar_node_rows(data.node_table.to_dicts(), t)
    }
    children: dict[int, list[int]] = {}
    for row in data.edge_table.to_dicts():
        children.setdefault(row["source_tree_node_index"], []).append(
            row["target_tree_node_index"]
        )
    for parent, kids in children.items():
        values = [angle[k] for k in kids]
        assert min(values) - 1e-12 <= angle[parent] <= max(values) + 1e-12


def test_polar_half_fan_root_bottom() -> None:
    data = _data()
    t = _transform(data, _config(angular_span_deg=180.0, angular_start_deg=0.0))
    rows = polar_node_rows(data.node_table.to_dicts(), t)
    assert all(row["y"] >= -1e-9 for row in rows)
    root = next(r for r in rows if r["tree_node_index"] == data.root_node_index)
    assert (root["x"], root["y"]) == (0.0, 0.0)


def test_polar_rings_at_ticks() -> None:
    data = _data()
    config = _config(
        angular_span_deg=180.0,
        angular_start_deg=0.0,
        scalar_tick_values=(0.0, 0.5, 1.0),
        scalar_tick_format=".1f",
    )
    t = _transform(data, config)
    ticks = polar_tick_values(t, config, data.node_table["scalar"].to_list())
    assert ticks == [0.0, 0.5, 1.0]
    rings = ring_rows(t, ticks, 32)
    radii = sorted({round(r["radius"], 9) for r in rings})
    assert radii == [0.0, 0.5, 1.0]
    labels = [r["label"] for r in ray_label_rows(t, ticks, ".1f")]
    assert labels == ["0.0", "0.5", "1.0"]
    assert format_tick(0.25, "~g") == "0.25"

    spec = plot_tree(data, config=config, interactive=False).to_dict()
    texts = [
        layer
        for layer in spec["layer"]
        if layer["mark"]["type"] == "text" and "label" in str(layer["data"])
    ]
    assert texts


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angular_span_deg": 0.0},
        {"angular_span_deg": 361.0},
        {"inner_radius_fraction": 1.0},
        {"inner_radius_fraction": -0.1},
        {"arc_samples_per_radian": 1},
        {"polar_axis": "none"},
        {"coordinate_system": "spherical"},
    ],
)
def test_polar_config_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        TreePlotConfig(**{"coordinate_system": "polar", **kwargs})


def test_polar_degenerate() -> None:
    single = make_tree_layout(_tree([0.3], [None]), horizontal_layout="weighted")
    pair = make_tree_layout(_tree([0.0, 1.0], [None, 0]), horizontal_layout="weighted")
    flat = make_tree_layout(_tree([1.0, 1.0, 1.0], [None, 0, 0]), horizontal_layout="weighted")
    for data in (single, pair, flat):
        for shape in ("orthogonal", "direct", "trunk"):
            chart = plot_tree(data, config=_config(edge_shape=shape), interactive=False)
            chart.to_dict()
    t = _transform(flat, _config())
    assert t.height == 0.0
    assert (
        len({r["radius"] for r in ring_rows(t, polar_tick_values(t, _config(), [1.0]), 32)}) == 1
    )


@pytest.mark.parametrize("edge_shape", ["orthogonal", "direct", "trunk"])
@pytest.mark.parametrize("span", [180.0, 360.0])
def test_polar_renders_with_edge_layer(edge_shape: str, span: float) -> None:
    config = _config(
        edge_shape=edge_shape,
        width=800,
        height=450,
        angular_span_deg=span,
        angular_start_deg=0.0 if span == 180.0 else 90.0,
    )
    spec = plot_tree(_data(), config=config, interactive=False).to_dict()
    assert any(
        layer.get("encoding", {}).get("detail", {}).get("field") == "logical_edge_index"
        for layer in spec["layer"]
    )
    x_domain = next(
        layer["encoding"]["x"]["scale"]["domain"]
        for layer in spec["layer"]
        if "x" in layer.get("encoding", {})
    )
    y_domain = next(
        layer["encoding"]["y"]["scale"]["domain"]
        for layer in spec["layer"]
        if "y" in layer.get("encoding", {})
    )
    assert (x_domain[1] - x_domain[0]) / 800 == pytest.approx((y_domain[1] - y_domain[0]) / 450)
