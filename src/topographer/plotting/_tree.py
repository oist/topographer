"""Altair rendering for tree plot data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import polars as pl

from topographer.plotting._config import (
    TreeAnnotationLayer,
    TreeMarkerConfig,
    TreePlotConfig,
    TreePresentationConfig,
)
from topographer.plotting._data import (
    TreePlotData,
    _resolve_tree_annotation_table,
    filter_tree_plot_data,
)
from topographer.plotting._polar import (
    make_polar_transform,
    polar_axis_layers,
    polar_domains,
    polar_edge_path_rows,
    polar_marker_table,
    polar_node_rows,
    polar_straight_edge_rows,
    polar_tick_values,
)
from topographer.plotting._shared import (
    _color_encoding,
    _event_legend_selection,
    _event_values,
    _semantic_event_shape,
    _special_value_legend_layers,
    _tooltip_title,
    _tree_scalar_axis,
    _tree_scalar_scale,
)


def plot_tree(
    plot_data: TreePlotData,
    *,
    config: TreePlotConfig | None = None,
    presentation: TreePresentationConfig | None = None,
    color_by: str = "display_event",
    tooltip_attrs: Sequence[str] = (),
    edge_tooltip_attrs: Sequence[str] = (),
    edge_predicate: Callable[[dict[str, Any]], bool] | None = None,
    node_predicate: Callable[[dict[str, Any]], bool] | None = None,
    non_significant_predicate: Callable[[dict[str, Any]], bool] | None = None,
    stop_marker_predicate: Callable[[dict[str, Any]], bool] | None = None,
    synonymous_edge_predicate: Callable[[dict[str, Any]], bool] | None = None,
    marker_table: pl.DataFrame | None = None,
    marker_config: TreeMarkerConfig | None = None,
    marker_tooltip_attrs: Sequence[str] = (),
    event_legend_filter: bool = False,
    event_legend_values: Sequence[str] | None = None,
    event_legend_title: str = "Event",
    event_tooltip_title: str = "Event",
    separate_event_layers: bool = True,
    show_nodes: bool = True,
    show_structural_events: bool = True,
    interactive: bool = True,
    include_config: bool = True,
) -> Any:
    """Build an interactive Altair tree chart from :class:`TreePlotData`.

    Altair is imported only when this function is called, keeping the generic
    graph and layout APIs independent of the optional visualization stack.
    ``show_structural_events`` suppresses the subtle critical-event overlay
    when a linked summary panel carries the event encodings instead.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import make_tree_layout, plot_tree
    >>> from topographer.topology import ScalarNetworkTopology
    >>> tree = ScalarNetworkTopology(build_cube_scalar_network()).split_tree()
    >>> type(plot_tree(make_tree_layout(tree))).__name__
    'LayerChart'
    """

    try:
        import altair as alt
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Altair plotting requires the 'plotting' optional dependency."
        ) from error

    legacy_requested = any(
        (
            color_by != "display_event",
            marker_table is not None,
            marker_config is not None,
            bool(marker_tooltip_attrs),
            event_legend_filter,
            event_legend_values is not None,
            non_significant_predicate is not None,
            stop_marker_predicate is not None,
            synonymous_edge_predicate is not None,
            not separate_event_layers,
            not show_nodes,
        )
    )
    if presentation is not None and legacy_requested:
        conflicting_legacy_arguments = any(
            (
                color_by != "display_event",
                marker_table is not None,
                marker_config is not None,
                bool(marker_tooltip_attrs),
                event_legend_filter,
                non_significant_predicate is not None,
                stop_marker_predicate is not None,
                synonymous_edge_predicate is not None,
                not separate_event_layers,
            )
        )
        if conflicting_legacy_arguments:
            raise ValueError(
                "presentation cannot be combined with legacy marker, node, or event-display arguments."
            )
    if color_by not in plot_data.node_table.columns:
        raise KeyError(f"color_by column {color_by!r} is not present in the node table.")
    missing_tooltips = sorted(set(tooltip_attrs).difference(plot_data.node_table.columns))
    if missing_tooltips:
        raise KeyError(f"Unknown node tooltip columns: {missing_tooltips}.")
    missing_edge_tooltips = sorted(
        set(edge_tooltip_attrs).difference(plot_data.edge_table.columns)
    )
    if missing_edge_tooltips:
        raise KeyError(f"Unknown edge tooltip columns: {missing_edge_tooltips}.")
    if marker_table is not None:
        if "vertex_id" in marker_table.columns and "x" not in marker_table.columns:
            marker_table = _resolve_tree_annotation_table(plot_data, marker_table)
        marker_table = _place_overlapping_annotations(
            marker_table, (config or TreePlotConfig()).annotation_overlap_step
        )
        missing_marker_columns = sorted({"x", "y"}.difference(marker_table.columns))
        if missing_marker_columns:
            raise KeyError(
                f"Marker table is missing coordinate columns: {missing_marker_columns}."
            )
        missing_marker_tooltips = sorted(
            set(marker_tooltip_attrs).difference(marker_table.columns)
        )
        if missing_marker_tooltips:
            raise KeyError(f"Unknown marker tooltip columns: {missing_marker_tooltips}.")
        resolved_marker_config = marker_config or TreeMarkerConfig()
        if (
            resolved_marker_config.label_field is not None
            and resolved_marker_config.label_field not in marker_table.columns
        ):
            raise KeyError(
                f"Marker label column {resolved_marker_config.label_field!r} is not present."
            )
    resolved_config = config or TreePlotConfig()
    if presentation is None:
        if resolved_config.edge_color_scale is not None or not show_nodes:
            resolved_presentation = TreePresentationConfig(mode="metric")
        elif legacy_requested:
            resolved_presentation = None
        else:
            resolved_presentation = TreePresentationConfig()
    else:
        resolved_presentation = presentation
    if resolved_presentation is not None and resolved_presentation.metric_color_scale is not None:
        resolved_config = replace(
            resolved_config, edge_color_scale=resolved_presentation.metric_color_scale
        )

    filtered_data = filter_tree_plot_data(
        plot_data, node_predicate=node_predicate, edge_predicate=edge_predicate
    )
    node_rows = filtered_data.node_table.to_dicts()
    edge_rows = filtered_data.edge_table.to_dicts()
    polar = resolved_config.coordinate_system == "polar"
    transform = None
    if polar:
        transform = make_polar_transform(
            plot_data.node_table,
            plot_data.edge_table,
            plot_data.root_node_index,
            plot_data.root_kind,
            resolved_config,
        )
        node_rows = polar_node_rows(node_rows, transform)
        layout_edge_rows = edge_rows
        edge_rows = polar_straight_edge_rows(edge_rows, transform)
        if marker_table is not None:
            marker_table = polar_marker_table(marker_table, transform)
    rendered_node_rows = [
        {
            **row,
            "marker_size": (
                resolved_config.structural_node_size
                if (
                    resolved_config.structural_node_size is not None
                    and str(row["display_event"]) in _STRUCTURAL_EVENT_VALUES
                )
                else resolved_config.node_size
            ),
        }
        for row in node_rows
    ]
    event_values = _event_values(
        (str(row["display_event"]) for row in node_rows), event_legend_values
    )
    if polar and resolved_config.edge_shape in {"orthogonal", "trunk"}:
        rendered_edge_rows = polar_edge_path_rows(
            layout_edge_rows,
            transform,
            resolved_config.edge_shape,
            resolved_config.trunk_fraction,
            resolved_config.arc_samples_per_radian,
        )
    elif resolved_config.edge_shape == "orthogonal":
        rendered_edge_rows = _orthogonal_edge_path_rows(edge_rows)
    elif resolved_config.edge_shape == "trunk":
        rendered_edge_rows = _trunk_edge_path_rows(edge_rows, resolved_config.trunk_fraction)
    else:
        rendered_edge_rows = [
            {**_direct_edge_row(row), "logical_edge_index": index}
            for index, row in enumerate(edge_rows)
        ]
    tooltip_fields = [
        alt.Tooltip("tree_node_index:Q", title="Tree node"),
        alt.Tooltip("vertex_index:Q", title="Source vertex"),
        alt.Tooltip("scalar:Q", title=plot_data.scalar_attr, format=".6g"),
        alt.Tooltip("display_event:N", title=event_tooltip_title),
    ]
    tooltip_fields.extend(
        alt.Tooltip(f"{attr}:N", title=_tooltip_title(attr)) for attr in tooltip_attrs
    )
    edge_tooltip_fields = [
        alt.Tooltip("source_tree_node_index:Q", title="Source tree node"),
        alt.Tooltip("target_tree_node_index:Q", title="Target tree node"),
        alt.Tooltip("source_vertex_index:Q", title="Source vertex"),
        alt.Tooltip("target_vertex_index:Q", title="Target vertex"),
        alt.Tooltip("source_scalar:Q", title="Source fitness", format=".6g"),
        alt.Tooltip("target_scalar:Q", title="Target fitness", format=".6g"),
        alt.Tooltip("scalar_difference:Q", title="Fitness difference", format=".6g"),
    ]
    edge_tooltip_fields.extend(
        alt.Tooltip(f"{attr}:N", title=_tooltip_title(attr)) for attr in edge_tooltip_attrs
    )

    scalar_axis = _tree_scalar_axis(alt, resolved_config)
    scalar_scale = _tree_scalar_scale(alt, resolved_config, filtered_data.node_table)
    horizontal_scale_kwargs: dict[str, Any] = {"padding": resolved_config.horizontal_padding}
    if resolved_config.horizontal_domain is not None:
        horizontal_scale_kwargs["domain"] = list(resolved_config.horizontal_domain)
    horizontal_scale = alt.Scale(**horizontal_scale_kwargs)
    polar_layers: list[Any] = []
    if polar:
        x_domain, y_domain = polar_domains(
            transform, resolved_config.width, resolved_config.height
        )
        horizontal_scale = alt.Scale(domain=x_domain, zero=False, nice=False, padding=0)
        scalar_scale = alt.Scale(domain=y_domain, zero=False, nice=False, padding=0)
        scalar_axis = None
        polar_layers = polar_axis_layers(
            alt,
            transform,
            resolved_config,
            polar_tick_values(
                transform, resolved_config, plot_data.node_table["scalar"].to_list()
            ),
            horizontal_scale,
            scalar_scale,
            resolved_config.scalar_axis_title or plot_data.scalar_attr,
        )
    if resolved_config.edge_shape in {"orthogonal", "trunk"}:
        edges = (
            alt.Chart(alt.Data(values=rendered_edge_rows))
            .mark_line(
                color=resolved_config.edge_color,
                strokeWidth=resolved_config.edge_width,
                strokeJoin="round",
            )
            .encode(
                x=alt.X("x:Q", axis=None, scale=horizontal_scale),
                y=alt.Y(
                    "y:Q",
                    title=resolved_config.scalar_axis_title or plot_data.scalar_attr,
                    scale=scalar_scale,
                    axis=scalar_axis,
                ),
                detail="logical_edge_index:N",
                order="path_order:Q",
                tooltip=edge_tooltip_fields,
            )
        )
    else:
        edges = (
            alt.Chart(alt.Data(values=rendered_edge_rows))
            .mark_rule(color=resolved_config.edge_color, strokeWidth=resolved_config.edge_width)
            .encode(
                x=alt.X("x_start:Q", axis=None, scale=horizontal_scale),
                x2="x_end:Q",
                y=alt.Y(
                    "y_start:Q",
                    title=resolved_config.scalar_axis_title or plot_data.scalar_attr,
                    scale=scalar_scale,
                    axis=scalar_axis,
                ),
                y2="y_end:Q",
                detail="logical_edge_index:N",
                tooltip=edge_tooltip_fields,
            )
        )
    has_metric_legend = resolved_config.edge_color_scale is not None
    if has_metric_legend:
        edges = edges.encode(color=_color_encoding(alt, resolved_config.edge_color_scale))
    nodes = (
        alt.Chart(alt.Data(values=rendered_node_rows))
        .mark_point(color="#000000", opacity=0.9, filled=True)
        .encode(
            x=alt.X("x:Q", axis=None, scale=horizontal_scale),
            y=alt.Y(
                "y:Q",
                title=resolved_config.scalar_axis_title or plot_data.scalar_attr,
                scale=scalar_scale,
                axis=scalar_axis,
            ),
            shape=_semantic_event_shape(
                alt, "display_event", event_values, title=event_legend_title
            ),
            size=alt.Size("marker_size:Q", legend=None, scale=None),
            tooltip=tooltip_fields,
        )
    )
    if color_by != "display_event":
        nodes = nodes.encode(
            color=alt.Color(f"{color_by}:N", title=color_by.replace("_", " ").title())
        )
    event_selection = None
    if event_legend_filter and show_nodes:
        event_selection = _event_legend_selection(alt, "display_event", event_values)
        nodes = nodes.encode(
            opacity=alt.condition(event_selection, alt.value(0.9), alt.value(0.0))
        )
    use_legacy_nodes = resolved_presentation is None and show_nodes
    if not use_legacy_nodes:
        chart = alt.layer(*polar_layers, edges)
    elif separate_event_layers:
        node_layers = [
            nodes.transform_filter(f"datum.display_event === '{event}'") for event in event_values
        ]
        chart = alt.layer(*polar_layers, edges, *node_layers)
    else:
        chart = alt.layer(*polar_layers, edges, nodes)
    if use_legacy_nodes and non_significant_predicate is not None:
        chart = chart + _hollow_node_layer(
            alt, node_rows, non_significant_predicate, resolved_config.node_size
        )
    if use_legacy_nodes and stop_marker_predicate is not None:
        chart = chart + _stop_marker_layer(alt, node_rows, stop_marker_predicate)
    if synonymous_edge_predicate is not None:
        chart = chart + _synonymous_edge_layer(
            alt,
            [_direct_edge_row(row) for row in edge_rows],
            synonymous_edge_predicate,
            resolved_config.edge_width,
        )
    if marker_table is not None and not marker_table.is_empty():
        marker_layers = _tree_marker_layers(
            alt,
            marker_table,
            marker_config or TreeMarkerConfig(),
            marker_tooltip_attrs,
            horizontal_scale,
            scalar_scale,
            scalar_axis,
            resolved_config.scalar_axis_title or plot_data.scalar_attr,
            resolved_config.font_size,
        )
        for marker_layer in marker_layers:
            chart = chart + marker_layer
    if resolved_presentation is not None and show_structural_events:
        critical_layer = _critical_event_layer(
            alt,
            node_rows,
            horizontal_scale,
            scalar_scale,
            scalar_axis,
            resolved_config.scalar_axis_title or plot_data.scalar_attr,
            resolved_config.node_size,
            event_tooltip_title,
            event_legend_title,
            event_legend_values,
        )
        if critical_layer is not None:
            chart = chart + critical_layer
    if resolved_presentation is not None:
        for annotation in _ordered_annotation_layers(resolved_presentation.annotation_layers):
            annotation_table = annotation.marker_table
            if "vertex_id" in annotation_table.columns and "x" not in annotation_table.columns:
                annotation_table = _resolve_tree_annotation_table(plot_data, annotation_table)
            if annotation.role != "wild_type":
                annotation_table = _place_overlapping_annotations(
                    annotation_table, resolved_config.annotation_overlap_step
                )
            if annotation_table.is_empty():
                continue
            if polar:
                annotation_table = polar_marker_table(annotation_table, transform)
            for marker_layer in _tree_marker_layers(
                alt,
                annotation_table,
                annotation.marker_config,
                annotation.tooltip_attrs,
                horizontal_scale,
                scalar_scale,
                scalar_axis,
                resolved_config.scalar_axis_title or plot_data.scalar_attr,
                resolved_config.font_size,
            ):
                chart = chart + marker_layer
        if (
            resolved_presentation.mode == "annotation"
            and event_legend_values is not None
            and not show_structural_events
        ):
            chart = chart + _annotation_event_legend_layer(
                alt, node_rows, event_legend_values, event_legend_title
            )
    if resolved_config.edge_color_scale is not None:
        for special_layer in _special_value_legend_layers(alt, resolved_config.edge_color_scale):
            chart = chart + special_layer
    if event_selection is not None:
        chart = chart.add_params(event_selection)
    properties: dict[str, Any] = {
        "width": resolved_config.width,
        "height": resolved_config.height,
    }
    chart = chart.properties(**properties)
    if include_config:
        chart = (
            chart.configure_axis(
                labelFontSize=resolved_config.font_size,
                titleFontSize=resolved_config.font_size,
            )
            .configure_legend(
                labelFontSize=resolved_config.font_size,
                titleFontSize=resolved_config.font_size,
            )
            .configure_view(strokeWidth=0)
        )
    return chart.interactive() if interactive else chart


def _place_overlapping_annotations(table: pl.DataFrame, step: float | None) -> pl.DataFrame:
    """Spread coincident regular-vertex annotations, preserving critical coordinates."""
    if step is None or step == 0.0 or table.is_empty() or "is_critical" not in table.columns:
        return table
    rows = table.to_dicts()
    groups: dict[tuple[float, float], list[int]] = {}
    for i, row in enumerate(rows):
        if not row["is_critical"]:
            groups.setdefault((float(row["x"]), float(row["y"])), []).append(i)
    for (x, _), indices in groups.items():
        if len(indices) < 2:
            continue
        offsets = [0.0]
        for j in range(1, len(indices)):
            magnitude = ((j + 1) // 2) * step
            offsets.append(magnitude if j % 2 else -magnitude)
        for index, offset in zip(indices, offsets, strict=True):
            rows[index]["x"] = x + offset
    return pl.DataFrame(rows)


def _direct_edge_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one logical edge row to a single direct renderer segment."""

    return {
        **row,
        "segment_index": 0,
        "segment_kind": "direct",
        "x_start": row["x_source"],
        "y_start": row["y_source"],
        "x_end": row["x_target"],
        "y_end": row["y_target"],
    }


_STRUCTURAL_EVENT_VALUES = frozenset(
    {"maximum", "minimum", "split_saddle", "join_saddle", "saddle"}
)


def _critical_event_layer(
    alt: Any,
    rows: list[dict[str, Any]],
    horizontal_scale: Any,
    scalar_scale: Any,
    scalar_axis: Any,
    scalar_title: str,
    size: int,
    event_tooltip_title: str,
    event_legend_title: str,
    additional_legend_values: Sequence[str] | None,
) -> Any | None:
    """Return subtle filled black structural markers and their Event legend."""

    event_rows = [
        {
            **row,
            "marker_size": _structural_marker_size(str(row["display_event"]), size),
        }
        for row in rows
        if row["display_event"] in _STRUCTURAL_EVENT_VALUES
    ]
    observed_values = list(dict.fromkeys(str(row["display_event"]) for row in event_rows))
    event_values = [
        *observed_values,
        *(event for event in additional_legend_values or () if event not in observed_values),
    ]
    if not event_rows:
        return None
    shared_encoding = {
        "x": alt.X("x:Q", axis=None, scale=horizontal_scale),
        "y": alt.Y("y:Q", title=scalar_title, axis=scalar_axis, scale=scalar_scale),
        "tooltip": [
            alt.Tooltip("tree_node_index:Q", title="Tree node"),
            alt.Tooltip("vertex_index:Q", title="Source vertex"),
            alt.Tooltip("scalar:Q", title=scalar_title, format=".6g"),
            alt.Tooltip("display_event:N", title=event_tooltip_title),
        ],
    }
    return (
        alt.Chart(alt.Data(values=event_rows))
        .mark_point(color="#000000", opacity=0.25, filled=True)
        .encode(
            **shared_encoding,
            shape=_semantic_event_shape(
                alt, "display_event", event_values, title=event_legend_title
            ),
            size=alt.Size("marker_size:Q", legend=None, scale=None),
        )
    )


def _structural_marker_size(event: str, size: int) -> int:
    """Return a balanced size for one standalone structural tree marker."""

    if event in {"maximum", "minimum"}:
        return round(size * 0.7)
    if event in {"split_saddle", "join_saddle", "saddle"}:
        return round(size * 0.85)
    return size


def _ordered_annotation_layers(
    layers: Sequence[TreeAnnotationLayer],
) -> tuple[TreeAnnotationLayer, ...]:
    """Keep declared annotation order while forcing wild type to the top."""

    return tuple(layer for layer in layers if layer.role == "annotation") + tuple(
        layer for layer in layers if layer.role == "wild_type"
    )


def _annotation_event_legend_layer(
    alt: Any,
    rows: list[dict[str, Any]],
    event_values: Sequence[str],
    event_legend_title: str,
) -> Any:
    """Return an invisible data layer that supplies an annotation Event legend."""

    return (
        alt.Chart(alt.Data(values=rows))
        .transform_filter("false")
        .mark_point(color="#000000", filled=True)
        .encode(
            shape=_semantic_event_shape(
                alt, "display_event", event_values, title=event_legend_title
            ),
        )
    )


def _hollow_node_layer(
    alt: Any,
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    node_size: int,
) -> Any:
    """Return the optional hollow-node overlay for arbitrary metadata rules."""

    return (
        alt.Chart(alt.Data(values=[row for row in rows if predicate(row)]))
        .mark_circle(size=node_size * 1.5, filled=False, strokeWidth=2)
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None))
    )


def _stop_marker_layer(
    alt: Any, rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> Any:
    """Return an opt-in X-marker overlay without assuming a biology schema."""

    return (
        alt.Chart(alt.Data(values=[row for row in rows if predicate(row)]))
        .mark_text(text="×", fontSize=18, fontWeight="bold")
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None))
    )


def _synonymous_edge_layer(
    alt: Any,
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    edge_width: float,
) -> Any:
    """Return an opt-in green edge overlay for arbitrary metadata rules."""

    return (
        alt.Chart(alt.Data(values=[row for row in rows if predicate(row)]))
        .mark_rule(color="#2ca02c", strokeWidth=edge_width * 1.8)
        .encode(
            x=alt.X("x_start:Q", axis=None),
            x2="x_end:Q",
            y=alt.Y("y_start:Q", axis=None),
            y2="y_end:Q",
        )
    )


def _vega_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert opaque stable IDs and metadata to serializable display values."""
    return [
        {
            key: value if value is None or type(value) in {str, int, float, bool} else str(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def _tree_marker_layers(
    alt: Any,
    marker_table: pl.DataFrame,
    config: TreeMarkerConfig,
    tooltip_attrs: Sequence[str],
    horizontal_scale: Any,
    scalar_scale: Any,
    scalar_axis: Any,
    scalar_title: str,
    tree_font_size: int,
) -> tuple[Any, ...]:
    """Return point and optional text layers using the tree's existing scales."""

    tooltips = [alt.Tooltip("y:Q", title=scalar_title, format=".6g")]
    tooltips.extend(alt.Tooltip(f"{attr}:N", title=_tooltip_title(attr)) for attr in tooltip_attrs)
    shared_encoding = {
        "x": alt.X("x:Q", axis=None, scale=horizontal_scale),
        "y": alt.Y(
            "y:Q",
            title=scalar_title,
            axis=scalar_axis,
            scale=scalar_scale,
        ),
    }
    point = (
        alt.Chart(alt.Data(values=_vega_safe_rows(marker_table.to_dicts())))
        .mark_point(
            shape=config.shape,
            color=config.color,
            size=config.size,
            opacity=config.opacity,
            filled=config.filled,
        )
        .encode(
            **shared_encoding,
            tooltip=tooltips,
        )
    )
    if config.label_field is None:
        return (point,)
    label = (
        alt.Chart(alt.Data(values=_vega_safe_rows(marker_table.to_dicts())))
        .mark_text(
            align="left",
            baseline="middle",
            color=config.label_color,
            dx=config.label_dx,
            fontSize=config.label_font_size or tree_font_size,
        )
        .encode(
            **shared_encoding,
            text=alt.Text(f"{config.label_field}:N"),
        )
    )
    return point, label


def _orthogonal_edge_path_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return connected three-point elbow paths, one path per logical edge."""

    paths: list[dict[str, Any]] = []
    for logical_edge_index, edge in enumerate(edge_rows):
        common = {**edge, "logical_edge_index": logical_edge_index}
        paths.extend(
            [
                {**common, "path_order": 0, "x": edge["x_target"], "y": edge["y_target"]},
                {**common, "path_order": 1, "x": edge["x_target"], "y": edge["y_source"]},
                {**common, "path_order": 2, "x": edge["x_source"], "y": edge["y_source"]},
            ]
        )
    return paths


def _trunk_edge_path_rows(
    edge_rows: list[dict[str, Any]], fraction: float
) -> list[dict[str, Any]]:
    """Return three-point trunk paths: a vertical trunk at the child, then a diagonal link.

    The trunk rises from the child to ``fraction`` of the way to the saddle; the
    link joins that point to the saddle. All y values lie between the endpoints.
    """

    paths: list[dict[str, Any]] = []
    for logical_edge_index, edge in enumerate(edge_rows):
        common = {**edge, "logical_edge_index": logical_edge_index}
        y_trunk = edge["y_target"] + fraction * (edge["y_source"] - edge["y_target"])
        paths.extend(
            [
                {
                    **common,
                    "path_order": 0,
                    "segment_kind": "trunk",
                    "x": edge["x_target"],
                    "y": edge["y_target"],
                },
                {
                    **common,
                    "path_order": 1,
                    "segment_kind": "trunk",
                    "x": edge["x_target"],
                    "y": y_trunk,
                },
                {
                    **common,
                    "path_order": 2,
                    "segment_kind": "link",
                    "x": edge["x_source"],
                    "y": edge["y_source"],
                },
            ]
        )
    return paths
