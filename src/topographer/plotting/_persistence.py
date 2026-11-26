"""Altair rendering for persistence tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

import polars as pl

from topographer.plotting._config import (
    PersistencePlotConfig,
    ReferenceLineConfig,
    TreeMarkerConfig,
)
from topographer.plotting._shared import (
    _color_encoding,
    _event_legend_selection,
    _persistence_axis,
    _semantic_event_color,
    _semantic_event_shape,
    _special_value_legend_layers,
    _tooltip_title,
)


def plot_persistence(
    persistence_table: pl.DataFrame,
    *,
    config: PersistencePlotConfig | None = None,
    birth_column: str = "birth_scalar",
    death_column: str = "death_scalar",
    color_by: str | None = "extremum_event",
    shape_by: str | None = None,
    tooltip_attrs: Sequence[str] = ("persistence",),
    event_legend_filter: bool = False,
    draw_order_by: str | None = None,
    semantic_event_legend_values: Sequence[str] | None = None,
    event_legend_title: str = "Event",
    semantic_event_colors: Mapping[str, str] | None = None,
    semantic_event_shapes: Mapping[str, str] | None = None,
    endpoint_markers: bool = False,
    saddle_event_column: str = "saddle_event",
    separate_event_layers: bool = True,
    interaction_selection: Any | None = None,
    interaction_predicate: Any | None = None,
    interaction_source_selection: Any | None = None,
    include_config: bool = True,
    show_birth_axis: bool = True,
    annotation_table: pl.DataFrame | None = None,
    annotation_vertex_id_column: str = "vertex_id",
    summary_vertex_id_columns: Sequence[str] | None = None,
    annotation_config: TreeMarkerConfig | None = None,
) -> Any:
    """Build an Altair persistence diagram from a generic Polars table.

    Only finite numeric birth and death columns are required. A table may carry
    arbitrary additional columns for colour and tooltips, so this renderer can
    serve persistence results from merge trees or another future topology
    algorithm. ``separate_event_layers`` renders one point layer per
    semantic event value. With ``endpoint_markers=True``, every pair is
    rendered as its saddle Y marker behind its extremum marker.
    ``annotation_table`` contains unique public vertex IDs and application
    metadata. By default, annotations match extrema through public birth,
    extremum, or source IDs; matching vertices receive the pair coordinates.
    ``summary_vertex_id_columns`` explicitly selects other endpoint fields.
    Branch-span charts additionally place matching target annotations on
    the diagonal.     ``annotation_config`` controls these generic markers.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import plot_persistence
    >>> from topographer.topology import ScalarNetworkTopology
    >>> pairs = ScalarNetworkTopology(build_cube_scalar_network()).split_tree().persistence()
    >>> type(plot_persistence(pairs)).__name__
    'LayerChart'
    """

    try:
        import altair as alt
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Altair plotting requires the 'plotting' optional dependency."
        ) from error

    if color_by == "extremum_event" and color_by not in persistence_table.columns:
        if "birth_event" in persistence_table.columns:
            color_by = "birth_event"
    if separate_event_layers and shape_by is None and color_by is not None:
        shape_by = color_by
    _require_table_columns(persistence_table, (birth_column, death_column), "persistence table")
    if color_by is not None:
        _require_table_columns(persistence_table, (color_by,), "persistence table")
    if shape_by is not None:
        _require_table_columns(persistence_table, (shape_by,), "persistence table")
    if event_legend_filter and shape_by is None:
        raise ValueError("event_legend_filter requires shape_by.")
    if semantic_event_legend_values is not None and shape_by is None:
        raise ValueError("semantic_event_legend_values requires shape_by.")
    if endpoint_markers:
        _require_table_columns(persistence_table, (saddle_event_column,), "persistence table")
    if draw_order_by is not None:
        _require_table_columns(persistence_table, (draw_order_by,), "persistence table")
    _require_table_columns(persistence_table, tooltip_attrs, "persistence table")

    resolved_config = config or PersistencePlotConfig()
    rows = persistence_table.to_dicts()
    _validate_persistence_rows(rows, birth_column, death_column)
    if draw_order_by is not None:
        _validate_draw_order_rows(rows, draw_order_by)
        rows.sort(
            key=lambda row: (
                float(row[draw_order_by]) if row[draw_order_by] is not None else -math.inf
            )
        )

    annotation_rows: list[dict[str, Any]] = []
    if annotation_table is not None:
        _require_table_columns(
            annotation_table, (annotation_vertex_id_column,), "annotation table"
        )
        annotations = {
            row[annotation_vertex_id_column]: row for row in annotation_table.to_dicts()
        }
        if len(annotations) != annotation_table.height:
            raise ValueError("Annotation vertex IDs must be unique.")
        candidate_columns = tuple(
            summary_vertex_id_columns
            or (
                "extremum_vertex_id",
                "birth_vertex_id",
                "source_vertex_id",
            )
        )
        matching_columns = [
            column for column in candidate_columns if column in persistence_table.columns
        ]
        if not matching_columns:
            raise KeyError(
                "Persistence table needs public endpoint vertex ID columns for annotations."
            )
        for row in rows:
            for column in matching_columns:
                vertex_id = row.get(column)
                if vertex_id not in annotations:
                    continue
                is_target = column in {"saddle_vertex_id", "target_vertex_id"}
                annotation_rows.append(
                    {
                        **annotations[vertex_id],
                        "x": row[death_column],
                        "y": row[death_column] if is_target else row[birth_column],
                    }
                )

    if annotation_rows:
        annotation_rows = list(
            {
                (row[annotation_vertex_id_column], row["x"], row["y"]): row
                for row in annotation_rows
            }.values()
        )
    domain = list(resolved_config.domain or _persistence_domain(rows, birth_column, death_column))
    tooltip_fields = [
        alt.Tooltip(f"{birth_column}:Q", title=resolved_config.birth_axis_title, format=".6g"),
        alt.Tooltip(f"{death_column}:Q", title=resolved_config.death_axis_title, format=".6g"),
    ]
    tooltip_fields.extend(
        alt.Tooltip(f"{attr}:N", title=_tooltip_title(attr)) for attr in tooltip_attrs
    )

    axis = _persistence_axis(alt, resolved_config, domain)
    x_encoding = alt.X(
        f"{death_column}:Q",
        title=resolved_config.death_axis_title,
        scale=alt.Scale(domain=domain, zero=False),
        axis=axis,
    )
    y_encoding = alt.Y(
        f"{birth_column}:Q",
        title=resolved_config.birth_axis_title,
        scale=alt.Scale(domain=domain, zero=False),
        axis=axis if show_birth_axis else None,
    )
    diagonal = (
        alt.Chart(
            alt.Data(
                values=[
                    {death_column: domain[0], birth_column: domain[0]},
                    {death_column: domain[1], birth_column: domain[1]},
                ]
            )
        )
        .mark_line(color="#737373", strokeWidth=resolved_config.diagonal_width)
        .encode(
            x=alt.X(
                f"{death_column}:Q",
                title=resolved_config.death_axis_title,
                axis=axis,
                scale=alt.Scale(domain=domain, zero=False),
            ),
            y=alt.Y(
                f"{birth_column}:Q",
                title=resolved_config.birth_axis_title,
                axis=axis if show_birth_axis else None,
                scale=alt.Scale(domain=domain, zero=False),
            ),
        )
    )
    point_rows = (
        _persistence_endpoint_rows(rows, saddle_event_column) if endpoint_markers else rows
    )
    global_extremum_rows = (
        [row for row in point_rows if _is_global_extremum_self_point(row)]
        if not endpoint_markers
        else []
    )
    event_field = "critical_event" if endpoint_markers else color_by
    shape_field = "critical_event" if endpoint_markers else shape_by
    semantic_event_values = (
        list(semantic_event_legend_values)
        if semantic_event_legend_values is not None
        else sorted(
            {
                str(row["critical_event"] if endpoint_markers else row[shape_by])
                for row in point_rows
            }
        )
        if shape_field is not None
        else []
    )
    points = (
        alt.Chart(alt.Data(values=point_rows))
        .mark_point(
            color="#737373",
            size=resolved_config.point_size,
            opacity=0.7,
            filled=True,
        )
        .encode(
            x=x_encoding,
            y=y_encoding,
            tooltip=tooltip_fields,
        )
    )
    if resolved_config.color_scale is not None:
        _require_table_columns(
            persistence_table, (resolved_config.color_scale.attribute,), "persistence table"
        )
        points = points.encode(color=_color_encoding(alt, resolved_config.color_scale))
    elif event_field is not None and event_field != shape_field:
        points = points.encode(
            color=alt.Color(f"{event_field}:N", title=event_field.replace("_", " ").title())
        )
    elif semantic_event_colors and shape_field is not None:
        points = points.encode(
            color=_semantic_event_color(
                alt,
                shape_field,
                semantic_event_values,
                title=event_legend_title,
                overrides=semantic_event_colors,
            )
        )
    event_selection = None
    if shape_field is not None:
        points = points.encode(
            shape=_semantic_event_shape(
                alt,
                shape_field,
                semantic_event_values,
                title=event_legend_title,
                overrides=semantic_event_shapes,
            )
        )
    if event_legend_filter:
        event_values = sorted({str(row[shape_field]) for row in point_rows})
        event_selection = _event_legend_selection(alt, shape_field, event_values)
        points = points.encode(
            opacity=alt.condition(event_selection, alt.value(0.7), alt.value(0.0))
        )
    if interaction_selection is not None:
        selected = (
            interaction_predicate if interaction_predicate is not None else interaction_selection
        )
        points = points.encode(
            opacity=alt.condition(selected, alt.value(1.0), alt.value(0.2)),
            size=alt.condition(
                selected,
                alt.value(resolved_config.point_size * 2),
                alt.value(resolved_config.point_size),
            ),
        )
    if interaction_source_selection is not None:
        points = points.add_params(interaction_source_selection)
    ordinary_points = (
        points.transform_filter(
            "datum.is_global_extremum_self_point !== true && "
            "datum.is_global_minimum_self_point !== true"
        )
        if global_extremum_rows
        else points
    )
    metricless_points = None
    if resolved_config.color_scale is not None and any(
        row.get(resolved_config.color_scale.attribute) is None for row in point_rows
    ):
        metric_attribute = resolved_config.color_scale.attribute
        metricless_predicate = f"datum.{metric_attribute} === null"
        metricless_points = (
            ordinary_points.transform_filter(metricless_predicate)
            .mark_point(
                color="#000000",
                size=resolved_config.point_size,
                opacity=0.7,
                filled=True,
            )
            .encode(color=alt.value("#000000"))
        )
        ordinary_points = ordinary_points.transform_filter(f"!({metricless_predicate})")
    global_extremum_points: list[Any] = []
    root_event_field = shape_field or event_field
    for event in dict.fromkeys(
        str(row[root_event_field]) for row in global_extremum_rows if root_event_field is not None
    ):
        root_rows_are_metricless = resolved_config.color_scale is not None and any(
            row.get(resolved_config.color_scale.attribute) is None
            and str(row[root_event_field]) == event
            for row in global_extremum_rows
        )
        root_points = points.transform_filter(
            "(datum.is_global_extremum_self_point === true || "
            "datum.is_global_minimum_self_point === true) && "
            f"datum.{root_event_field} === '{event}'"
        ).mark_point(
            size=resolved_config.point_size,
            opacity=0.7,
            filled=True,
            clip=False,
            dx=-4 if event == "maximum" else 4,
            dy=4 if event == "maximum" else -4,
        )
        if root_rows_are_metricless:
            root_points = root_points.encode(color=alt.value("#000000"))
        elif resolved_config.color_scale is None:
            root_points = root_points.encode(color=alt.value("#737373"))
        global_extremum_points.append(root_points)
    if endpoint_markers:
        event_layers = [diagonal]
        layer_filters = (
            ("datum.endpoint_role === 'saddle'", resolved_config.point_size * 2),
            ("datum.critical_event === 'minimum'", resolved_config.point_size),
            ("datum.critical_event === 'maximum'", resolved_config.point_size),
        )
        for predicate, size in layer_filters:
            if any(_matches_persistence_layer(row, predicate) for row in point_rows):
                event_layers.append(
                    ordinary_points.transform_filter(predicate).encode(size=alt.value(size))
                )
        chart = alt.layer(*event_layers)
    elif separate_event_layers and shape_field is not None:
        event_layers = [diagonal]
        visible_metric_legend = True
        for event in semantic_event_values:
            has_ordinary_event = any(
                str(row[shape_field]) == event and not _is_global_extremum_self_point(row)
                for row in point_rows
            )
            if not has_ordinary_event:
                continue
            event_layer = ordinary_points.transform_filter(f"datum.{shape_field} === '{event}'")
            if resolved_config.color_scale is not None and not visible_metric_legend:
                event_layer = event_layer.encode(
                    color=_color_encoding(
                        alt, replace(resolved_config.color_scale, show_legend=False)
                    )
                )
            event_layers.append(event_layer)
            visible_metric_legend = False
        chart = alt.layer(*event_layers)
    else:
        chart = diagonal + ordinary_points
    for reference_line in resolved_config.reference_lines:
        chart = chart + _reference_line_layer(alt, reference_line, domain)
    if metricless_points is not None:
        chart = chart + metricless_points
    for root_points in global_extremum_points:
        chart = chart + root_points
    if annotation_rows:
        marker = annotation_config or TreeMarkerConfig()
        annotation_layer = (
            alt.Chart(
                alt.Data(
                    values=[
                        {
                            key: value
                            if value is None or type(value) in {str, int, float, bool}
                            else str(value)
                            for key, value in row.items()
                        }
                        for row in annotation_rows
                    ]
                )
            )
            .mark_point(
                shape=marker.shape,
                color=marker.color,
                size=marker.size,
                opacity=marker.opacity,
                filled=marker.filled,
            )
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=domain, zero=False), axis=None),
                y=alt.Y("y:Q", scale=alt.Scale(domain=domain, zero=False), axis=None),
                tooltip=[
                    alt.Tooltip(f"{field}:N", title=_tooltip_title(field))
                    for field in (annotation_table.columns if annotation_table is not None else ())
                ],
            )
        )
        chart = chart + annotation_layer
    if event_selection is not None:
        chart = chart.add_params(event_selection)
    chart = chart.properties(width=resolved_config.width, height=resolved_config.height)
    if resolved_config.color_scale is not None:
        for special_layer in _special_value_legend_layers(alt, resolved_config.color_scale):
            chart = chart + special_layer
        chart = _position_summary_legends(
            alt,
            chart,
            summary_width=resolved_config.width,
            font_size=resolved_config.font_size,
        )
    if not include_config:
        return chart
    return (
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


def _position_summary_legends(
    alt: Any,
    chart: Any,
    *,
    summary_width: int,
    font_size: int,
) -> Any:
    """Stack the Event guide above the metric guide beside a summary chart."""

    specification = chart.to_dict(validate=True)
    specification.pop("$schema", None)
    specification.pop("config", None)
    event_count = max(
        (
            len(domain)
            for encoding in _chart_encodings(specification)
            if isinstance((shape := encoding.get("shape")), dict)
            and isinstance((scale := shape.get("scale")), dict)
            and isinstance((domain := scale.get("domain")), list)
        ),
        default=0,
    )
    legend_x = summary_width + 20
    metric_legend_y = round(font_size * 1.2) * (event_count + 1) + 20
    current_metric_y = metric_legend_y
    for encoding in _chart_encodings(specification):
        shape = encoding.get("shape")
        if isinstance(shape, dict) and isinstance(shape.get("legend"), dict):
            shape["legend"].update(orient="none", legendX=legend_x, legendY=0)
        color = encoding.get("color")
        if isinstance(color, dict) and isinstance(color.get("legend"), dict):
            color["legend"].update(
                orient="none",
                legendX=legend_x,
                legendY=current_metric_y,
            )
            grad_len = color["legend"].get("gradientLength", 40)
            current_metric_y += grad_len + round(font_size * 3.5)
    return alt.LayerChart.from_dict(specification)


def _chart_encodings(specification: Any) -> list[dict[str, Any]]:
    """Return every encoding mapping nested in an Altair specification."""

    encodings: list[dict[str, Any]] = []
    if isinstance(specification, dict):
        encoding = specification.get("encoding")
        if isinstance(encoding, dict):
            encodings.append(encoding)
        for composition_key in ("layer", "hconcat", "vconcat", "concat", "spec"):
            encodings.extend(_chart_encodings(specification.get(composition_key)))
    elif isinstance(specification, list):
        for child_specification in specification:
            encodings.extend(_chart_encodings(child_specification))
    return encodings


def _is_global_extremum_self_point(row: dict[str, Any]) -> bool:
    """Return whether a row is a current or legacy display-only root point."""

    return bool(
        row.get("is_global_extremum_self_point") is True
        or row.get("is_global_minimum_self_point") is True
    )


def plot_branch_spans(
    branch_span_table: pl.DataFrame,
    *,
    config: PersistencePlotConfig | None = None,
    tooltip_attrs: Sequence[str] = ("branch_span",),
    event_legend_filter: bool = False,
    draw_order_by: str | None = None,
    semantic_event_legend_values: Sequence[str] | None = None,
    event_legend_title: str = "Event",
    endpoint_markers: bool = False,
    separate_event_layers: bool = False,
    include_config: bool = True,
    show_source_axis: bool = True,
    annotation_table: pl.DataFrame | None = None,
    annotation_config: TreeMarkerConfig | None = None,
) -> Any:
    """Build an Altair scalar-pair diagram for branch-span rows.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import plot_branch_spans
    >>> from topographer.topology import ScalarNetworkTopology
    >>> spans = ScalarNetworkTopology(build_cube_scalar_network()).split_tree().spans()
    >>> type(plot_branch_spans(spans)).__name__
    'LayerChart'
    """

    if {"source_scalar", "target_scalar"}.issubset(branch_span_table.columns):
        source_column = "source_scalar"
        target_column = "target_scalar"
        event_column = "source_event"
        target_event_column = "target_event"
    elif {"extremum_scalar", "saddle_scalar"}.issubset(branch_span_table.columns):
        source_column = "extremum_scalar"
        target_column = "saddle_scalar"
        event_column = "extremum_event"
        target_event_column = "saddle_event"
    else:
        raise KeyError(
            "branch span table must contain source/target or extremum/saddle scalar columns."
        )
    return plot_persistence(
        branch_span_table,
        config=config,
        birth_column=source_column,
        death_column=target_column,
        color_by=event_column,
        shape_by=(
            event_column
            if separate_event_layers
            or event_legend_filter
            or semantic_event_legend_values is not None
            else None
        ),
        tooltip_attrs=tooltip_attrs,
        event_legend_filter=event_legend_filter,
        draw_order_by=draw_order_by,
        semantic_event_legend_values=semantic_event_legend_values,
        event_legend_title=event_legend_title,
        endpoint_markers=endpoint_markers,
        saddle_event_column=target_event_column,
        separate_event_layers=separate_event_layers,
        include_config=include_config,
        show_birth_axis=show_source_axis,
        annotation_table=annotation_table,
        annotation_config=annotation_config,
        summary_vertex_id_columns=(
            ("source_vertex_id", "target_vertex_id")
            if "source_vertex_id" in branch_span_table.columns
            else ("extremum_vertex_id", "saddle_vertex_id")
        ),
    )


plot_branch_span = plot_branch_spans


def plot_ordinary_persistence(
    persistence_table: pl.DataFrame,
    *,
    config: PersistencePlotConfig | None = None,
    tooltip_attrs: Sequence[str] = ("persistence", "pair_kind"),
    event_legend_filter: bool = False,
    draw_order_by: str | None = None,
    semantic_event_legend_values: Sequence[str] | None = None,
    event_legend_title: str = "Event",
    include_config: bool = True,
    show_birth_axis: bool = True,
    annotation_table: pl.DataFrame | None = None,
    annotation_config: TreeMarkerConfig | None = None,
) -> Any:
    """Build an Altair diagram from ordinary H0 persistence rows.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import plot_ordinary_persistence
    >>> from topographer.topology import ScalarNetworkTopology
    >>> pairs = ScalarNetworkTopology(build_cube_scalar_network()).split_tree().persistence()
    >>> type(plot_ordinary_persistence(pairs)).__name__
    'LayerChart'
    """

    # The renderer is intentionally shared with extended persistence so both
    # pair kinds retain the same diagram geometry and visual treatment.

    return _plot_explicit_persistence(
        persistence_table,
        config=config,
        tooltip_attrs=tooltip_attrs,
        event_legend_filter=event_legend_filter,
        draw_order_by=draw_order_by,
        semantic_event_legend_values=semantic_event_legend_values,
        event_legend_title=event_legend_title,
        include_config=include_config,
        show_birth_axis=show_birth_axis,
        annotation_table=annotation_table,
        annotation_config=annotation_config,
    )


def plot_extended_persistence(
    persistence_table: pl.DataFrame,
    *,
    config: PersistencePlotConfig | None = None,
    tooltip_attrs: Sequence[str] = ("persistence", "pair_kind"),
    event_legend_filter: bool = False,
    draw_order_by: str | None = None,
    semantic_event_legend_values: Sequence[str] | None = None,
    event_legend_title: str = "Event",
    include_config: bool = True,
    show_birth_axis: bool = True,
    annotation_table: pl.DataFrame | None = None,
    annotation_config: TreeMarkerConfig | None = None,
) -> Any:
    """Build an Altair diagram from extended H0 persistence rows.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import plot_extended_persistence
    >>> from topographer.topology import ScalarNetworkTopology
    >>> pairs = ScalarNetworkTopology(build_cube_scalar_network()).split_tree().persistence()
    >>> type(plot_extended_persistence(pairs)).__name__
    'LayerChart'
    """

    return _plot_explicit_persistence(
        persistence_table,
        config=config,
        tooltip_attrs=tooltip_attrs,
        event_legend_filter=event_legend_filter,
        draw_order_by=draw_order_by,
        semantic_event_legend_values=semantic_event_legend_values,
        event_legend_title=event_legend_title,
        include_config=include_config,
        show_birth_axis=show_birth_axis,
        annotation_table=annotation_table,
        annotation_config=annotation_config,
    )


def _plot_explicit_persistence(
    persistence_table: pl.DataFrame,
    *,
    config: PersistencePlotConfig | None,
    tooltip_attrs: Sequence[str],
    event_legend_filter: bool,
    draw_order_by: str | None,
    semantic_event_legend_values: Sequence[str] | None,
    event_legend_title: str,
    include_config: bool,
    show_birth_axis: bool,
    annotation_table: pl.DataFrame | None,
    annotation_config: TreeMarkerConfig | None,
) -> Any:
    """Render explicit ordinary or extended persistence tables uniformly."""

    return plot_persistence(
        persistence_table,
        config=config,
        color_by="birth_event",
        shape_by="birth_event",
        tooltip_attrs=tooltip_attrs,
        event_legend_filter=event_legend_filter,
        draw_order_by=draw_order_by,
        semantic_event_legend_values=semantic_event_legend_values,
        event_legend_title=event_legend_title,
        include_config=include_config,
        show_birth_axis=show_birth_axis,
        annotation_table=annotation_table,
        annotation_config=annotation_config,
    )


def _persistence_endpoint_rows(
    rows: Sequence[dict[str, Any]], saddle_event_column: str
) -> list[dict[str, Any]]:
    """Expand every persistence pair into co-located extremum and saddle marks."""

    endpoint_rows: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        pair_id = str(
            row.get(
                "pair_id",
                f"{row.get('extremum_tree_node_index', position)}:"
                f"{row.get('saddle_tree_node_index', position)}",
            )
        )
        endpoint_rows.extend(
            (
                {
                    **row,
                    "pair_id": pair_id,
                    "endpoint_role": "saddle",
                    "critical_event": str(row[saddle_event_column]),
                    "critical_vertex_index": int(row["saddle_vertex_index"]),
                },
                {
                    **row,
                    "pair_id": pair_id,
                    "endpoint_role": "extremum",
                    "critical_event": str(row["extremum_event"]),
                    "critical_vertex_index": int(row["extremum_vertex_index"]),
                },
            )
        )
    return endpoint_rows


def _matches_persistence_layer(row: dict[str, Any], predicate: str) -> bool:
    """Return whether an endpoint row belongs to one explicit event mark layer."""

    if "endpoint_role" in predicate:
        return row["endpoint_role"] == "saddle"
    return row["critical_event"] == predicate.split("'")[1]


def _reference_line_layer(alt: Any, config: ReferenceLineConfig, domain: list[float]) -> Any:
    """Build a labelled reference line on a persistence diagram."""

    values = [{"x": config.value, "y": domain[0], "x2": config.value, "y2": domain[1]}]
    if config.axis == "y":
        values = [{"x": domain[0], "y": config.value, "x2": domain[1], "y2": config.value}]
    line = (
        alt.Chart(alt.Data(values=values))
        .mark_rule(color=config.color, strokeDash=list(config.dash))
        .encode(x=alt.X("x:Q", axis=None), x2="x2:Q", y=alt.Y("y:Q", axis=None), y2="y2:Q")
    )
    if config.label is None:
        return line
    label = (
        alt.Chart(alt.Data(values=values))
        .mark_text(text=config.label, align="left", dx=4, dy=-4)
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None))
    )
    return line + label


def _require_table_columns(table: pl.DataFrame, columns: Sequence[str], name: str) -> None:
    """Validate required input columns for an Altair table renderer."""

    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}.")


def _validate_persistence_rows(
    rows: list[dict[str, Any]], birth_column: str, death_column: str
) -> None:
    """Require finite numeric birth and death values before chart serialization."""

    for row_index, row in enumerate(rows):
        for column in (birth_column, death_column):
            value = row[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Persistence row {row_index} column {column!r} must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError(f"Persistence row {row_index} column {column!r} must be finite.")


def _validate_draw_order_rows(rows: list[dict[str, Any]], column: str) -> None:
    """Require finite numeric values or nulls when a persistence draw order is requested."""

    for row_index, row in enumerate(rows):
        value = row[column]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Persistence row {row_index} draw-order column {column!r} must be numeric."
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"Persistence row {row_index} draw-order column {column!r} must be finite."
            )


def _persistence_domain(
    rows: list[dict[str, Any]], birth_column: str, death_column: str
) -> list[float]:
    """Return a shared non-degenerate birth/death scale domain."""

    values = [float(row[column]) for row in rows for column in (birth_column, death_column)]
    if not values:
        return [0.0, 1.0]
    lower, upper = min(values), max(values)
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        return [lower - padding, upper + padding]
    return [lower, upper]
