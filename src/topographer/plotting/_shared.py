"""Shared Altair encodings for tree and persistence renderers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import polars as pl

from topographer.plotting._config import ColorScaleConfig, PersistencePlotConfig, TreePlotConfig

_FILLED_X_SYMBOL = (
    "M-1,-0.55L-0.55,-1L0,-0.45L0.55,-1L1,-0.55L0.45,0L1,0.55L0.55,1L0,0.45"
    "L-0.55,1L-1,0.55L-0.45,0Z"
)
_SPLIT_SADDLE_SYMBOL = "M -0.126,1 L 0.126,1 L 0.22,0.25 L 0.866,-0.35 L 0.74,-0.57 L 0,-0.15 L -0.74,-0.57 L -0.866,-0.35 L -0.22,0.25 Z"
_JOIN_SADDLE_SYMBOL = "M -0.126,-1 L 0.126,-1 L 0.22,-0.25 L 0.866,0.35 L 0.74,0.57 L 0,0.15 L -0.74,0.57 L -0.866,0.35 L -0.22,-0.25 Z"
_MULTI_SADDLE_SYMBOL = (
    "M 0.420,0.000 L 0.866,0.500 L 0.210,0.364 L 0.000,1.000 L -0.210,0.364 L -0.866,0.500 "
    "L -0.420,0.000 L -0.866,-0.500 L -0.210,-0.364 L -0.000,-1.000 L 0.210,-0.364 L 0.866,-0.500 Z"
)

_SEMANTIC_EVENT_SHAPES = {
    "maximum": "triangle-up",
    "minimum": "triangle-down",
    "split_saddle": _SPLIT_SADDLE_SYMBOL,
    "join_saddle": _JOIN_SADDLE_SYMBOL,
    "multi_saddle": _MULTI_SADDLE_SYMBOL,
    "saddle": _JOIN_SADDLE_SYMBOL,
    "regular": "circle",
    "stop_codon": _FILLED_X_SYMBOL,
}

_SEMANTIC_EVENT_COLORS = {
    "maximum": "#4c78a8",
    "minimum": "#000000",
    "split_saddle": "#f58518",
    "join_saddle": "#f58518",
    "multi_saddle": "#f58518",
    "saddle": "#f58518",
    "regular": "#737373",
    "stop_codon": "#000000",
}


def _semantic_event_color_value(event: str, overrides: Mapping[str, str] | None = None) -> str:
    """Return the stable display color for one semantic event."""

    return (overrides or {}).get(event, _SEMANTIC_EVENT_COLORS.get(event, "#737373"))


def _semantic_event_shape(
    alt: Any,
    field: str,
    values: Sequence[str],
    *,
    title: str = "Event",
    show_legend: bool = True,
    overrides: Mapping[str, str] | None = None,
) -> Any:
    """Return semantic extrema/saddle marker shapes with a circle fallback."""

    return alt.Shape(
        f"{field}:N",
        title=title,
        legend=(
            alt.Legend(
                title=title,
                values=values,
                labelExpr="replace(datum.label, '_', ' ')",
            )
            if show_legend
            else None
        ),
        scale=alt.Scale(
            domain=values,
            range=[
                (overrides or {}).get(value, _SEMANTIC_EVENT_SHAPES.get(value, "circle"))
                for value in values
            ],
        ),
    )


def _semantic_event_color(
    alt: Any,
    field: str,
    values: Sequence[str],
    *,
    title: str = "Event",
    show_legend: bool = True,
    overrides: Mapping[str, str] | None = None,
) -> Any:
    """Return the shared colour half of a persistence critical-point legend."""

    return alt.Color(
        f"{field}:N",
        title=title,
        legend=(
            alt.Legend(
                title=title,
                values=values,
                labelExpr="replace(datum.label, '_', ' ')",
            )
            if show_legend
            else None
        ),
        scale=alt.Scale(
            domain=values,
            range=[_semantic_event_color_value(value, overrides) for value in values],
        ),
    )


def _event_legend_selection(alt: Any, field: str, values: Sequence[str]) -> Any:
    """Return a legend-bound event selector initially displaying every event.

    A regular legend click isolates one event, while shift-click toggles
    additional event classes.  With every class toggled off, critical-point
    marks disappear but any unfiltered structural layers remain visible.
    """

    return alt.selection_point(
        name=f"{field}_visibility",
        fields=[field],
        bind="legend",
        toggle=True,
        empty=False,
        value=[{field: value} for value in values],
    )


def _event_values(observed: Iterable[str], requested: Sequence[str] | None) -> list[str]:
    """Return requested event order followed by any remaining observed events."""

    observed_values = list(dict.fromkeys(str(value) for value in observed))
    if requested is None:
        return sorted(observed_values)
    requested_values = list(dict.fromkeys(str(value) for value in requested))
    return [
        *requested_values,
        *(value for value in observed_values if value not in requested_values),
    ]


def _tree_scalar_axis(alt: Any, config: TreePlotConfig) -> Any:
    """Build the shared scalar axis while preserving explicit ticks."""

    kwargs: dict[str, Any] = {
        "format": config.scalar_tick_format,
        "grid": config.show_scalar_grid,
        "domain": config.show_scalar_axis_line,
    }
    if config.scalar_tick_values is None:
        kwargs["tickCount"] = config.scalar_tick_count
    else:
        kwargs["values"] = list(config.scalar_tick_values)
    return alt.Axis(**kwargs)


def _tree_scalar_scale(alt: Any, config: TreePlotConfig, node_table: pl.DataFrame) -> Any:
    """Build a scalar scale with consistent top and bottom presentation padding."""

    kwargs: dict[str, Any] = {"zero": False}
    domain = config.scalar_domain
    if domain is None:
        domain = (
            float(node_table.get_column("scalar").min()),
            float(node_table.get_column("scalar").max()),
        )
    lower, upper = domain
    span = upper - lower
    padding = max(span * config.scalar_padding_fraction, 0.05 if span == 0.0 else 0.0)
    kwargs["domain"] = [lower - padding, upper + padding]
    return alt.Scale(**kwargs)


def _persistence_axis(alt: Any, config: PersistencePlotConfig, domain: list[float]) -> Any:
    """Build one persistence axis with labels retained when its line is hidden."""

    kwargs: dict[str, Any] = {
        "format": config.tick_format,
        "grid": config.show_grid,
        "domain": config.show_axis_line,
    }
    kwargs["values"] = list(config.tick_values or _persistence_tick_values(domain))
    return alt.Axis(**kwargs)


def _persistence_tick_values(domain: list[float]) -> tuple[float, ...]:
    """Return the required minimum, zero, and maximum persistence ticks."""

    return tuple(sorted({float(domain[0]), 0.0, float(domain[1])}))


def _color_encoding(alt: Any, config: ColorScaleConfig) -> Any:
    """Build an Altair continuous colour encoding from a reusable config."""

    scale_kwargs: dict[str, Any] = {"scheme": config.scheme, "type": config.scale_type}
    if config.domain is not None:
        scale_kwargs["domain"] = list(config.domain)
    if config.categories is not None:
        scale_kwargs["domain"] = list(config.categories)
    if config.colors is not None:
        scale_kwargs["range"] = list(config.colors)
    scale = alt.Scale(**scale_kwargs)
    title = config.title or config.attribute.replace("_", " ").title()
    legend = alt.Legend(
        title=title.split("\n"),
        orient=config.legend_orient,
        gradientLength=config.legend_length,
        gradientThickness=config.legend_thickness,
    )
    field_type = "N" if config.scale_type == "ordinal" else "Q"
    encoding = alt.Color(
        f"{config.attribute}:{field_type}",
        scale=scale,
        legend=legend if config.show_legend else None,
    )
    if config.zero_value is not None and config.one_value is not None:
        return alt.condition(
            f"datum.{config.attribute} === {config.zero_value}",
            alt.value(config.zero_color),
            alt.condition(
                f"datum.{config.attribute} === {config.one_value}",
                alt.value(config.one_color),
                encoding,
            ),
        )
    if config.zero_value is not None:
        return alt.condition(
            f"datum.{config.attribute} === {config.zero_value}",
            alt.value(config.zero_color),
            encoding,
        )
    if config.one_value is not None:
        return alt.condition(
            f"datum.{config.attribute} === {config.one_value}",
            alt.value(config.one_color),
            encoding,
        )
    return encoding


def _special_value_legend_layers(alt: Any, config: ColorScaleConfig) -> list[Any]:
    """Return auxiliary invisible legend layers for out-of-palette values."""

    if not config.show_legend:
        return []
    layers = []
    if config.zero_value is not None:
        label = config.zero_label or str(int(config.zero_value))
        title = (config.zero_legend_title or "Zero count").split("\n")
        dummy = (
            alt.Chart(alt.Data(values=[{"special_label": label}]))
            .transform_filter("false")
            .mark_point(color=config.zero_color, filled=True, size=50)
            .encode(
                color=alt.Color(
                    "special_label:N",
                    scale=alt.Scale(domain=[label], range=[config.zero_color]),
                    legend=alt.Legend(title=title, orient=config.legend_orient),
                )
            )
        )
        layers.append(dummy)
    if config.one_value is not None and (
        config.one_label is not None or config.one_legend_title is not None
    ):
        label = config.one_label or str(int(config.one_value))
        title = (config.one_legend_title or "Count = 1").split("\n")
        dummy = (
            alt.Chart(alt.Data(values=[{"special_label": label}]))
            .transform_filter("false")
            .mark_point(color=config.one_color, filled=True, size=50)
            .encode(
                color=alt.Color(
                    "special_label:N",
                    scale=alt.Scale(domain=[label], range=[config.one_color]),
                    legend=alt.Legend(title=title, orient=config.legend_orient),
                )
            )
        )
        layers.append(dummy)
    return layers


def _tooltip_title(attribute: str) -> str:
    """Return readable labels for common biological plotting fields."""

    titles = {
        "dna_sequence": "DNA sequence",
        "aa_sequence": "AA sequence",
        "source_dna_sequence": "Birth DNA sequence",
        "target_dna_sequence": "Death DNA sequence",
        "source_aa_sequence": "Birth AA sequence",
        "target_aa_sequence": "Death AA sequence",
        "dna_hamming": "DNA Hamming",
        "aa_hamming": "AA Hamming",
        "horton_strahler_order": "HS order",
        "lineage_vertex_count": "Lineage vertex count",
        "lineage_stop_codon_count": "Lineage stop-codon count",
        "branch_vertex_count": "Branch vertex count",
        "branch_stop_codon_count": "Branch stop-codon count",
        "synonymous_endpoints": "Synonymous endpoints",
    }
    return titles.get(attribute, attribute.replace("_", " ").title())
