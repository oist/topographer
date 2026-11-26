"""Tests for backend-independent tree plot tables and Altair charts."""

from __future__ import annotations

import altair as alt
import polars as pl
import pytest
import rustworkx as rx

from topographer import Graph, ScalarNetwork, ScalarNetworkTopology
from topographer.examples import build_cube_scalar_network
from topographer.plotting import (
    ColorScaleConfig,
    PersistencePlotConfig,
    ReferenceLineConfig,
    TreeAnnotationLayer,
    TreeMarkerConfig,
    TreePlotConfig,
    TreePresentationConfig,
    build_scalar_plot_domain,
    make_tree_layout,
    plot_branch_span,
    plot_extended_persistence,
    plot_persistence,
    plot_tree,
)
from topographer.plotting._data import (
    filter_tree_plot_data,
    select_tree_edges_by_horton_strahler,
)
from topographer.topology import build_tree_metrics
from topographer.topology._persistence import (
    build_tree_branch_span_table,
    build_tree_extended_persistence_pair_table,
)
from topographer.topology._tree_construction import build_join_tree
from topographer.topology._tree_reduction import reduce_merge_tree


def test_build_tree_plot_data_preserves_scalar_geometry_and_generic_metadata() -> None:
    tree = build_join_tree(build_cube_scalar_network(), node_attrs=("label",)).tree

    plot_data = make_tree_layout(tree, node_attrs=("label",))

    assert plot_data.root_node_index == 0
    assert plot_data.scalar_attr == "scalar"
    assert plot_data.node_table.height == 8
    assert plot_data.edge_table.height == 7
    assert plot_data.edge_segment_table.height == 14
    assert plot_data.node_table.filter(pl.col("tree_node_index") == 0).item(0, "y") == 0.0
    assert plot_data.node_table.filter(pl.col("tree_node_index") == 1).item(0, "y") == 0.1
    assert "label" in plot_data.node_table.columns
    assert plot_data.edge_table["scalar_difference"].min() >= 0.0
    assert set(plot_data.edge_segment_table["segment_kind"].unique()) == {"horizontal", "vertical"}
    assert {"x_start", "y_start", "x_end", "y_end"}.issubset(plot_data.edge_segment_table.columns)


def test_plot_tree_returns_interactive_layered_specification() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    config = TreePlotConfig(
        width=700,
        height=360,
        scalar_axis_title="Fitness",
        scalar_tick_count=4,
        show_scalar_grid=False,
        node_size=96,
    )
    layout = make_tree_layout(tree)
    assert layout.node_table.height == 8
    assert layout.edge_segment_table.height == 14

    chart = plot_tree(layout, config=config)
    assert isinstance(chart, (alt.Chart, alt.LayerChart))

    specification = chart.to_dict(validate=True)
    assert specification["width"] == 700
    assert specification["height"] == 360
    assert "layer" in specification
    assert len(specification["layer"]) == 2


def test_tree_marker_table_is_rendered_as_a_separate_topmost_layer() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    plot_data = make_tree_layout(tree)
    star_symbol = "M0,-1L0.6,0.8L-1,-0.3L1,-0.3L-0.6,0.8Z"
    marker_table = pl.DataFrame(
        {
            "x": [0.5],
            "y": [0.4],
            "label": ["reference"],
        }
    )

    base_chart = plot_tree(plot_data)
    assert isinstance(base_chart, (alt.Chart, alt.LayerChart))
    base_specification = base_chart.to_dict(validate=True)

    chart = plot_tree(
        plot_data,
        marker_table=marker_table,
        marker_config=TreeMarkerConfig(
            shape=star_symbol,
            color="#f50b0b",
            size=90,
            opacity=0.8,
            label_field="label",
        ),
        marker_tooltip_attrs=("label",),
    )
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    specification = chart.to_dict(validate=True)
    assert len(specification["layer"]) > len(base_specification["layer"])


def test_tree_plot_can_use_event_terminology() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree

    chart = plot_tree(
        make_tree_layout(tree),
        event_legend_title="Event",
        event_tooltip_title="Event",
    )
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None


def test_tree_event_legend_can_include_a_stop_codon_x_marker() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree

    chart = plot_tree(
        make_tree_layout(tree),
        color_by="event",
        event_legend_values=("maximum", "minimum", "join_saddle", "stop_codon"),
    )
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None


def test_plot_configs_validate_shared_presentation_settings() -> None:
    tree_config = TreePlotConfig()
    persistence_config = PersistencePlotConfig()
    assert (tree_config.width, tree_config.height, tree_config.font_size) == (420, 250, 15)
    assert tree_config.scalar_tick_format == ".2f"
    assert tree_config.scalar_padding_fraction == 0.05
    assert tree_config.node_size == 55
    assert tree_config.edge_width == 1.4
    assert tree_config.show_scalar_grid is False
    assert tree_config.show_scalar_axis_line is False
    assert (persistence_config.width, persistence_config.height, persistence_config.font_size) == (
        250,
        250,
        15,
    )
    assert persistence_config.birth_axis_title == "Birth"
    assert persistence_config.diagonal_width == 1.0
    assert persistence_config.point_size == 65
    assert persistence_config.show_grid is False
    assert persistence_config.show_axis_line is False
    with pytest.raises(ValueError, match="width"):
        TreePlotConfig(width=0)
    with pytest.raises(ValueError, match="axis titles"):
        PersistencePlotConfig(death_axis_title="")
    with pytest.raises(ValueError, match="horizontal_padding"):
        TreePlotConfig(horizontal_padding=-1)
    with pytest.raises(ValueError, match="diagonal_width"):
        PersistencePlotConfig(diagonal_width=0.0)
    with pytest.raises(ValueError, match="opacity"):
        TreeMarkerConfig(opacity=1.1)


def test_tree_presentation_modes_use_structural_and_ordered_annotation_layers() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    plot_data = make_tree_layout(tree)
    stop = TreeAnnotationLayer(
        marker_table=pl.DataFrame({"x": [0.2], "y": [0.2]}),
        marker_config=TreeMarkerConfig(shape="circle", color="#000000"),
    )
    wild_type = TreeAnnotationLayer(
        marker_table=pl.DataFrame({"x": [0.2], "y": [0.2]}),
        marker_config=TreeMarkerConfig(shape="star", color="#f50b0b"),
        role="wild_type",
    )
    annotation = plot_tree(
        plot_data,
        presentation=TreePresentationConfig(
            mode="annotation", annotation_layers=(wild_type, stop)
        ),
    )
    assert isinstance(annotation, (alt.Chart, alt.LayerChart))
    assert annotation.to_dict(validate=True) is not None

    metric = plot_tree(
        plot_data,
        presentation=TreePresentationConfig(mode="metric", annotation_layers=(wild_type,)),
    )
    assert isinstance(metric, (alt.Chart, alt.LayerChart))
    assert metric.to_dict(validate=True) is not None

    stop_legend = plot_tree(
        plot_data,
        presentation=TreePresentationConfig(mode="annotation", annotation_layers=(stop,)),
        event_legend_values=("stop_codon",),
    )
    assert isinstance(stop_legend, (alt.Chart, alt.LayerChart))
    assert stop_legend.to_dict(validate=True) is not None


def test_scalar_plot_domain_rounds_outward_and_includes_zero() -> None:
    assert build_scalar_plot_domain((-1.174803, 1.395792)) == (-1.2, 1.4)
    assert build_scalar_plot_domain((0.2, 0.8)) == (0.0, 0.8)
    assert build_scalar_plot_domain((-1.2, 1.4)) == (-1.2, 1.4)
    assert build_scalar_plot_domain((0.0,)) == (-0.1, 0.1)
    assert build_scalar_plot_domain((2.0,), include_zero=False) == (1.9, 2.1)


@pytest.mark.parametrize("values", ((), (float("nan"),), (float("inf"),)))
def test_scalar_plot_domain_rejects_empty_or_non_finite_values(values) -> None:
    with pytest.raises(ValueError):
        build_scalar_plot_domain(values)


def test_scalar_plot_domain_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError):
        build_scalar_plot_domain((True,))


def test_plot_persistence_uses_shared_configuration() -> None:
    tree = rx.PyDiGraph()
    minimum = tree.add_node({"vertex_index": 0, "scalar": 0.1, "event": "minimum"})
    saddle = tree.add_node({"vertex_index": 1, "scalar": 0.8, "event": "join_saddle"})
    tree.add_edge(minimum, saddle, {})
    table = build_tree_extended_persistence_pair_table(
        tree,
        tree_kind="join",
    )

    chart = plot_persistence(
        table,
        config=PersistencePlotConfig(
            width=360,
            height=300,
            birth_axis_title="Start",
            death_axis_title="End",
            tick_count=3,
            point_size=90,
        ),
    )
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    specification = chart.to_dict(validate=True)

    assert specification["width"] == 360
    assert specification["height"] == 300
    assert len(specification["layer"]) == 2


def test_plot_branch_span_uses_branch_columns_and_tooltip() -> None:
    tree = rx.PyDiGraph()
    minimum = tree.add_node({"vertex_index": 0, "scalar": 0.1, "event": "minimum"})
    saddle = tree.add_node({"vertex_index": 1, "scalar": 0.8, "event": "join_saddle"})
    tree.add_edge(minimum, saddle, {})
    table = build_tree_branch_span_table(
        tree,
        extremum_event="minimum",
        saddle_event="join_saddle",
    )

    chart = plot_branch_span(table)
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None


def test_plot_extended_persistence_uses_ordinary_point_treatment() -> None:
    tree = reduce_merge_tree(build_join_tree(build_cube_scalar_network()).tree)
    table = build_tree_extended_persistence_pair_table(tree, tree_kind="join")

    chart = plot_extended_persistence(table)
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None


def test_plot_persistence_supports_semantic_event_shapes() -> None:
    table = pl.DataFrame(
        {
            "birth_scalar": [1.0, 0.8],
            "death_scalar": [0.5, 0.4],
            "persistence": [0.5, 0.4],
            "source_event": ["maximum", "split_saddle"],
        }
    )

    chart = plot_persistence(table, color_by="source_event", shape_by="source_event")
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None

    saddle_chart = plot_persistence(
        table.with_columns(pl.Series("source_event", ["split_saddle", "join_saddle"])),
        color_by="source_event",
        shape_by="source_event",
    )
    assert isinstance(saddle_chart, (alt.Chart, alt.LayerChart))
    assert saddle_chart.to_dict(validate=True) is not None


def test_persistence_semantic_event_legend_can_include_stop_codon() -> None:
    table = pl.DataFrame(
        {
            "birth_scalar": [1.0, 0.8],
            "death_scalar": [0.5, 0.4],
            "persistence": [0.5, 0.4],
            "source_event": ["maximum", "split_saddle"],
        }
    )

    chart = plot_persistence(
        table,
        color_by="source_event",
        shape_by="source_event",
        semantic_event_legend_values=("maximum", "split_saddle", "stop_codon"),
    )
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None

    with pytest.raises(ValueError, match="requires shape_by"):
        plot_persistence(table, color_by=None, semantic_event_legend_values=("stop_codon",))


def test_persistence_draw_order_renders_high_metric_values_last() -> None:
    table = pl.DataFrame(
        {
            "birth_scalar": [0.8, 0.8, 0.8],
            "death_scalar": [0.4, 0.4, 0.4],
            "persistence": [0.4, 0.4, 0.4],
            "metric": [3, 1, 2],
        }
    )

    ordered = plot_persistence(table, color_by=None, draw_order_by="metric")
    assert isinstance(ordered, (alt.Chart, alt.LayerChart))
    ordered_spec = ordered.to_dict(validate=True)
    assert ordered_spec is not None

    with pytest.raises(KeyError, match="required columns"):
        plot_persistence(table, color_by=None, draw_order_by="missing")
    with pytest.raises(TypeError, match="draw-order column.*numeric"):
        plot_persistence(
            table.with_columns(pl.lit("high").alias("metric")),
            color_by=None,
            draw_order_by="metric",
        )
    with pytest.raises(ValueError, match="draw-order column.*finite"):
        plot_persistence(
            table.with_columns(pl.lit(float("inf")).alias("metric")),
            color_by=None,
            draw_order_by="metric",
        )


def test_event_legends_can_filter_critical_point_marks() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    tree_chart = plot_tree(make_tree_layout(tree), event_legend_filter=True)
    assert isinstance(tree_chart, (alt.Chart, alt.LayerChart))
    assert tree_chart.to_dict(validate=True) is not None

    table = pl.DataFrame(
        {
            "birth_scalar": [1.0, 0.8],
            "death_scalar": [0.5, 0.4],
            "persistence": [0.5, 0.4],
            "source_event": ["maximum", "split_saddle"],
        }
    )
    persistence_chart = plot_persistence(
        table,
        color_by=None,
        shape_by="source_event",
        event_legend_filter=True,
    )
    assert isinstance(persistence_chart, (alt.Chart, alt.LayerChart))
    assert persistence_chart.to_dict(validate=True) is not None

    with pytest.raises(ValueError, match="requires shape_by"):
        plot_persistence(table, color_by=None, event_legend_filter=True)


def test_plot_persistence_rejects_missing_or_nonfinite_coordinates() -> None:
    with pytest.raises(KeyError, match="required columns"):
        plot_persistence(pl.DataFrame({"birth_scalar": [0.1]}))
    with pytest.raises(ValueError, match="finite"):
        plot_persistence(
            pl.DataFrame(
                {
                    "birth_scalar": [0.1],
                    "death_scalar": [float("inf")],
                    "extremum_event": ["minimum"],
                    "persistence": [float("inf")],
                }
            )
        )


def test_build_tree_plot_data_rejects_non_tree_topology() -> None:
    graph = rx.PyGraph()
    for index in range(3):
        graph.add_node({"scalar": float(index)})
    graph.add_edge(0, 1, None)
    graph.add_edge(1, 2, None)
    graph.add_edge(2, 0, None)

    with pytest.raises(ValueError, match="one fewer edge"):
        make_tree_layout(graph)


def test_plot_tree_rejects_unknown_visual_fields() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    plot_data = make_tree_layout(tree)

    with pytest.raises(KeyError, match="color_by"):
        plot_tree(plot_data, color_by="missing")
    with pytest.raises(KeyError, match="tooltip"):
        plot_tree(plot_data, tooltip_attrs=("missing",))
    with pytest.raises(KeyError, match="edge tooltip"):
        plot_tree(plot_data, edge_tooltip_attrs=("missing",))
    with pytest.raises(KeyError, match="coordinate columns"):
        plot_tree(plot_data, marker_table=pl.DataFrame({"x": [0.5]}))
    with pytest.raises(KeyError, match="marker tooltip"):
        plot_tree(
            plot_data,
            marker_table=pl.DataFrame({"x": [0.5], "y": [0.4]}),
            marker_tooltip_attrs=("missing",),
        )
    with pytest.raises(KeyError, match="Marker label column"):
        plot_tree(
            plot_data,
            marker_table=pl.DataFrame({"x": [0.5], "y": [0.4]}),
            marker_config=TreeMarkerConfig(label_field="label"),
        )


def test_tree_edge_tooltips_expose_requested_metrics() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    chart = plot_tree(make_tree_layout(tree), edge_tooltip_attrs=("regular_count",))
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None


def test_tree_plot_uses_connected_orthogonal_paths_or_explicit_direct_edges() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    plot_data = make_tree_layout(tree)
    assert plot_data.edge_segment_table.height == plot_data.edge_table.height * 2

    orthogonal = plot_tree(plot_data)
    assert isinstance(orthogonal, (alt.Chart, alt.LayerChart))
    assert orthogonal.to_dict(validate=True) is not None

    direct = plot_tree(plot_data, config=TreePlotConfig(edge_shape="direct"))
    assert isinstance(direct, (alt.Chart, alt.LayerChart))
    assert direct.to_dict(validate=True) is not None


def test_tree_plot_data_can_use_iplotx_horizontal_coordinates() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree

    plot_data = make_tree_layout(tree, horizontal_layout="iplotx")

    assert plot_data.node_table["x"].n_unique() > 1
    assert plot_data.node_table["y"].to_list() == plot_data.node_table["scalar"].to_list()
    assert plot_data.edge_segment_table.height == 14


def test_minimum_root_is_promoted_to_a_minimum_display_event() -> None:
    tree = rx.PyDiGraph()
    root = tree.add_node({"vertex_index": 0, "scalar": 0.0, "event": "regular"})
    maximum = tree.add_node({"vertex_index": 1, "scalar": 1.0, "event": "maximum"})
    tree.add_edge(root, maximum, {})

    plot_data = make_tree_layout(tree, root_kind="minimum")

    root_row = plot_data.node_table.filter(pl.col("tree_node_index") == root).row(0, named=True)
    maximum_row = plot_data.node_table.filter(pl.col("tree_node_index") == maximum).row(
        0, named=True
    )
    assert root_row["event"] == "regular"
    assert root_row["display_event"] == "minimum"
    assert maximum_row["display_event"] == "maximum"
    assert plot_data.edge_segment_table.height == 2


def test_maximum_root_is_promoted_to_a_maximum_display_event_and_marker() -> None:
    tree = rx.PyDiGraph()
    minimum = tree.add_node({"vertex_index": 0, "scalar": 0.0, "event": "minimum"})
    root = tree.add_node({"vertex_index": 1, "scalar": 1.0, "event": "regular"})
    tree.add_edge(minimum, root, {})

    plot_data = make_tree_layout(tree, root_kind="maximum")

    root_row = plot_data.node_table.filter(pl.col("tree_node_index") == root).row(0, named=True)
    assert root_row["event"] == "regular"
    assert root_row["display_event"] == "maximum"
    assert plot_data.edge_segment_table.height == 2

    chart = plot_tree(plot_data)
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None


def test_tree_filters_horton_orders_and_optional_metadata_overlays() -> None:
    tree = rx.PyDiGraph()
    root = tree.add_node({"vertex_index": 0, "scalar": 0.0, "event": "join_saddle"})
    left = tree.add_node({"vertex_index": 1, "scalar": 0.5, "event": "minimum", "stop": True})
    right = tree.add_node({"vertex_index": 2, "scalar": 0.6, "event": "minimum", "peak": False})
    tree.add_edge(root, left, {"synonymous": True, "regular_count": 1})
    tree.add_edge(root, right, {"synonymous": False, "regular_count": 3})
    plot_data = make_tree_layout(tree, node_attrs=("stop", "peak"), edge_attrs=("synonymous",))

    filtered = filter_tree_plot_data(plot_data, edge_predicate=lambda row: row["synonymous"])
    selected = select_tree_edges_by_horton_strahler(plot_data, minimum=1, maximum=1)

    assert filtered.edge_table.height == 1
    assert selected.edge_table.height == 2
    assert filtered.edge_segment_table.height == 2
    assert selected.edge_segment_table.height == 4

    chart = plot_tree(
        plot_data,
        non_significant_predicate=lambda row: row.get("peak") is False,
        stop_marker_predicate=lambda row: row.get("stop") is True,
        synonymous_edge_predicate=lambda row: row.get("synonymous") is True,
    )
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None


def test_explicit_ticks_count_colour_and_persistence_reference_lines() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    tree_data = make_tree_layout(tree)
    tree_chart = plot_tree(
        tree_data,
        config=TreePlotConfig(
            scalar_domain=(0.0, 1.0),
            scalar_tick_values=(0.0, 0.5, 1.0),
            scalar_tick_format=".1f",
            edge_color_scale=ColorScaleConfig(
                attribute="regular_count",
                scale_type="log",
                one_value=1.0,
                one_color="#111111",
                legend_length=120,
                legend_thickness=11,
            ),
        ),
    )
    assert isinstance(tree_chart, (alt.Chart, alt.LayerChart))
    assert tree_chart.to_dict(validate=True) is not None

    stop_count_chart = plot_persistence(
        pl.DataFrame(
            {
                "birth_scalar": [0.1, 0.2],
                "death_scalar": [0.8, 0.9],
                "stop_codon_count": [0, 3],
            }
        ),
        config=PersistencePlotConfig(
            color_scale=ColorScaleConfig(
                attribute="stop_codon_count",
                scale_type="linear",
                domain=(0.0, 3.0),
            )
        ),
        color_by=None,
        tooltip_attrs=("stop_codon_count",),
    )
    assert isinstance(stop_count_chart, (alt.Chart, alt.LayerChart))
    assert stop_count_chart.to_dict(validate=True) is not None

    ordinal_chart = plot_tree(
        tree_data,
        config=TreePlotConfig(
            edge_color_scale=ColorScaleConfig(
                attribute="horton_strahler_order",
                title="Horton–Strahler\norder",
                scheme="greys",
                scale_type="ordinal",
                categories=(1.0, 2.0),
                colors=("#d9d9d9", "#000000"),
            )
        ),
    )
    assert isinstance(ordinal_chart, (alt.Chart, alt.LayerChart))
    assert ordinal_chart.to_dict(validate=True) is not None

    table = pl.DataFrame(
        {
            "birth_scalar": [0.1],
            "death_scalar": [0.8],
            "persistence": [0.7],
            "extremum_event": ["minimum"],
        }
    )
    persistence_chart = plot_persistence(
        table,
        config=PersistencePlotConfig(
            domain=(0.0, 1.0),
            tick_values=(0.0, 0.5, 1.0),
            tick_format=".1f",
            reference_lines=(ReferenceLineConfig(axis="x", value=0.4, label="cutoff"),),
        ),
    )
    assert isinstance(persistence_chart, (alt.Chart, alt.LayerChart))
    assert persistence_chart.to_dict(validate=True) is not None


def test_color_scale_config_zero_value_and_special_legend_layers() -> None:
    # Validation tests
    with pytest.raises(ValueError, match="zero_value and zero_color must be supplied together"):
        ColorScaleConfig(attribute="count", zero_value=0.0)
    with pytest.raises(ValueError, match="zero_label requires zero_value and zero_color"):
        ColorScaleConfig(attribute="count", zero_label="0")

    scale = ColorScaleConfig(
        attribute="branch_stop_codon_count",
        scheme="plasma",
        domain=(1.0, 10.0),
        scale_type="log",
        zero_value=0.0,
        zero_color="#737373",
        zero_label="0",
        zero_legend_title="Empty branch",
    )

    tree = build_join_tree(build_cube_scalar_network()).tree
    tree_data = make_tree_layout(tree)
    chart = plot_tree(
        tree_data,
        config=TreePlotConfig(edge_color_scale=scale),
    )
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    assert chart.to_dict(validate=True) is not None


@pytest.mark.parametrize("show_nodes", (False, True))
def test_component_charts_compose_with_shared_fitness_and_independent_semantics(
    show_nodes: bool,
) -> None:
    """Callers own domains and scale resolution when composing two charts."""
    tree = reduce_merge_tree(build_join_tree(build_cube_scalar_network()).tree)
    plot_data = make_tree_layout(tree, root_kind="maximum")
    persistence = build_tree_extended_persistence_pair_table(tree, tree_kind="join")
    domain = (-0.1, 1.1)
    ticks = (0.0, 0.5, 1.0)
    tree_panel = plot_tree(
        plot_data,
        config=TreePlotConfig(
            scalar_domain=domain, scalar_tick_values=ticks, scalar_padding_fraction=0.0
        ),
        show_nodes=show_nodes,
        interactive=False,
        include_config=False,
    )
    summary_panel = plot_persistence(
        persistence,
        config=PersistencePlotConfig(domain=domain, tick_values=ticks),
        color_by="pair_kind",
        shape_by="pair_kind",
        event_legend_title="Pair Kind",
        semantic_event_colors={"ordinary": "#2563eb", "extended": "#059669"},
        semantic_event_shapes={"ordinary": "triangle-down", "extended": "diamond"},
        include_config=False,
    )
    specification = (
        alt.hconcat(tree_panel, summary_panel)
        .resolve_scale(x="independent", y="shared", color="independent", shape="independent")
        .to_dict(validate=True)
    )

    assert specification["resolve"]["scale"] == {
        "x": "independent",
        "y": "shared",
        "color": "independent",
        "shape": "independent",
    }
    tree_spec, summary_spec = specification["hconcat"]
    assert "config" not in tree_spec and "config" not in summary_spec
    assert tree_spec["layer"][0]["encoding"]["y"]["scale"]["domain"] == list(domain)
    assert summary_spec["layer"][0]["encoding"]["y"]["scale"]["domain"] == list(domain)
    point_encoding = next(
        layer["encoding"]
        for layer in summary_spec["layer"]
        if "shape" in layer.get("encoding", {})
    )
    assert point_encoding["shape"]["scale"]["range"] == ["diamond", "triangle-down"]
    assert point_encoding["color"]["scale"]["range"] == ["#059669", "#2563eb"]
    assert point_encoding["shape"]["title"] == "Pair Kind"
    assert not specification.get("params")


def test_component_composition_preserves_topmost_wild_type_annotation() -> None:
    tree = reduce_merge_tree(build_join_tree(build_cube_scalar_network()).tree)
    plot_data = make_tree_layout(tree, root_kind="maximum")
    persistence = build_tree_extended_persistence_pair_table(tree, tree_kind="join")
    wild_type = TreeAnnotationLayer(
        marker_table=pl.DataFrame({"x": [0.5], "y": [0.4]}),
        marker_config=TreeMarkerConfig(shape="star", color="#f50b0b"),
        role="wild_type",
    )
    chart = alt.hconcat(
        plot_tree(
            plot_data,
            presentation=TreePresentationConfig(mode="metric", annotation_layers=(wild_type,)),
            show_structural_events=False,
            interactive=False,
            include_config=False,
        ),
        plot_persistence(persistence, include_config=False),
    ).resolve_scale(x="independent", y="shared", color="independent", shape="independent")
    tree_spec = chart.to_dict(validate=True)["hconcat"][0]
    assert len(tree_spec["layer"]) == 2
    assert tree_spec["layer"][-1]["mark"]["shape"] == "star"
    assert tree_spec["layer"][-1]["mark"]["color"] == "#f50b0b"


def test_component_metric_composition_places_color_legend_only_in_summary() -> None:
    """Metric attachment and legend choices remain explicit at the call site."""
    from dataclasses import replace

    tree = reduce_merge_tree(build_join_tree(build_cube_scalar_network()).tree)
    plot_data = make_tree_layout(tree, root_kind="maximum")
    metric = "sequence_count"
    plot_data = replace(
        plot_data,
        edge_table=plot_data.edge_table.with_columns(pl.lit(2).alias(metric)),
        edge_segment_table=plot_data.edge_segment_table.with_columns(pl.lit(2).alias(metric)),
    )
    branch_spans = plot_data.edge_table.select(
        "source_scalar", "target_scalar", metric
    ).with_columns(pl.lit("maximum").alias("source_event"))
    scale = ColorScaleConfig(attribute=metric, scheme="plasma", domain=(1.0, 3.0))
    tree_chart = plot_tree(
        plot_data,
        presentation=TreePresentationConfig(
            mode="metric", metric_color_scale=replace(scale, show_legend=False)
        ),
        interactive=False,
        include_config=False,
    )
    summary_chart = plot_branch_span(
        branch_spans,
        config=PersistencePlotConfig(color_scale=scale),
        tooltip_attrs=(metric,),
        include_config=False,
    )
    spec = (
        alt.hconcat(tree_chart, summary_chart)
        .resolve_scale(x="independent", y="shared", color="independent", shape="independent")
        .to_dict(validate=True)
    )
    tree_colors = [
        layer["encoding"]["color"]
        for layer in spec["hconcat"][0]["layer"]
        if "color" in layer.get("encoding", {})
    ]
    summary_colors = [
        layer["encoding"]["color"]
        for layer in spec["hconcat"][1]["layer"]
        if "color" in layer.get("encoding", {})
    ]
    assert tree_colors and all(color["legend"] is None for color in tree_colors)
    assert any(color.get("legend") is not None for color in summary_colors)


def test_tree_plot_edge_counts_match_tree_metrics() -> None:
    vertices = pl.DataFrame(
        {"variant": [10, 20, 30, 35, 40], "fitness": [0.9, 0.5, 0.8, 0.3, 0.1]}
    )
    edges = pl.DataFrame({"source": [10, 20, 30, 35], "target": [20, 35, 20, 40]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="variant",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness",),
    )
    network = ScalarNetwork(graph, scalar_attr="fitness")
    tree = ScalarNetworkTopology(network).split_tree()

    plot_data = make_tree_layout(tree)
    metrics = build_tree_metrics(tree).branch_table
    expected = {
        (int(row["source_tree_node_index"]), int(row["target_tree_node_index"])): (
            int(row["branch_vertex_count"]),
            int(row["horton_strahler_order"]),
        )
        for row in metrics.to_dicts()
    }
    actual = {
        (int(row["target_tree_node_index"]), int(row["source_tree_node_index"])): (
            int(row["branch_vertex_count"]),
            int(row["horton_strahler_order"]),
        )
        for row in plot_data.edge_table.to_dicts()
    }
    assert actual == expected
    assert len(actual) == 3


def _trunk_rows(fraction: float = 0.85) -> list[dict]:
    from topographer.plotting._tree import _trunk_edge_path_rows

    tree = build_join_tree(build_cube_scalar_network()).tree
    plot_data = make_tree_layout(tree)
    edges = plot_data.edge_table.to_dicts()
    return edges, _trunk_edge_path_rows(edges, fraction)


def test_trunk_edge_rows_preserve_endpoints() -> None:
    edges, rows = _trunk_rows()
    for index, edge in enumerate(edges):
        path = sorted(
            (r for r in rows if r["logical_edge_index"] == index), key=lambda r: r["path_order"]
        )
        assert (path[0]["x"], path[0]["y"]) == (edge["x_target"], edge["y_target"])
        assert (path[-1]["x"], path[-1]["y"]) == (edge["x_source"], edge["y_source"])
        assert path[0]["x"] == path[1]["x"]


def test_trunk_y_values_within_edge_range() -> None:
    edges, rows = _trunk_rows()
    for row in rows:
        edge = edges[row["logical_edge_index"]]
        low, high = sorted((edge["y_source"], edge["y_target"]))
        assert low - 1e-12 <= row["y"] <= high + 1e-12


def test_trunk_chart_edge_layer_is_line_with_logical_edge_detail() -> None:
    tree = build_join_tree(build_cube_scalar_network()).tree
    chart = plot_tree(make_tree_layout(tree), config=TreePlotConfig(edge_shape="trunk"))
    spec = chart.to_dict(validate=True)
    layers = spec.get("layer", [spec])
    assert any(
        (layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]) == "line"
        and layer["encoding"].get("detail", {}).get("field") == "logical_edge_index"
        for layer in layers
    )


def test_invalid_edge_shape_and_trunk_fraction_raise() -> None:
    with pytest.raises(ValueError):
        TreePlotConfig(edge_shape="radial")  # type: ignore[arg-type]
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            TreePlotConfig(edge_shape="trunk", trunk_fraction=bad)
