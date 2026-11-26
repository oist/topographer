"""Reusable, immutable presentation settings for Altair topology plots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
import math
from numbers import Real
from typing import Literal

import polars as pl

__all__ = [
    "ColorScaleConfig",
    "PersistencePlotConfig",
    "ReferenceLineConfig",
    "TreeAnnotationLayer",
    "TreeMarkerConfig",
    "TreePlotConfig",
    "TreePresentationConfig",
    "build_scalar_plot_domain",
    "count_color_scale",
]


def build_scalar_plot_domain(
    values: Iterable[Real], *, decimal_places: int = 1, include_zero: bool = True
) -> tuple[float, float]:
    """Return an outward-rounded scalar domain suitable for related plots.

    The lower bound is rounded toward negative infinity and the upper bound
    toward positive infinity. By default, the result also contains zero so
    separate and linked scalar plots can share meaningful reference ticks.

    Examples
    --------
    >>> from topographer.plotting import build_scalar_plot_domain
    >>> build_scalar_plot_domain([0.2, 0.9])
    (0.0, 0.9)
    """

    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
        raise TypeError("decimal_places must be a non-negative integer.")
    if decimal_places < 0:
        raise ValueError("decimal_places must be a non-negative integer.")
    if not isinstance(include_zero, bool):
        raise TypeError("include_zero must be a boolean.")

    numeric_values: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Scalar plot domain values must be finite real numbers.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Scalar plot domain values must be finite real numbers.")
        numeric_values.append(numeric)
    if not numeric_values:
        raise ValueError("Scalar plot domain values must not be empty.")

    quantum = Decimal(1).scaleb(-decimal_places)
    lower = Decimal(str(min(numeric_values))).quantize(quantum, rounding=ROUND_FLOOR)
    upper = Decimal(str(max(numeric_values))).quantize(quantum, rounding=ROUND_CEILING)
    if include_zero:
        lower = min(lower, Decimal(0))
        upper = max(upper, Decimal(0))
    if lower == upper:
        lower -= quantum
        upper += quantum
    return float(lower), float(upper)


@dataclass(frozen=True, slots=True)
class ColorScaleConfig:
    """How a numeric edge or point attribute is mapped to colour.

    ``one_color`` is useful for a special count value: that value is shown in
    a distinct colour, while the remaining values use the selected continuous
    palette.

    Examples
    --------
    >>> from topographer.plotting import ColorScaleConfig
    >>> ColorScaleConfig(attribute="scalar").scheme
    'plasma'
    """

    attribute: str
    title: str | None = None
    scheme: str = "plasma"
    domain: tuple[float, float] | None = None
    scale_type: Literal["linear", "log", "ordinal"] = "linear"
    categories: tuple[float, ...] | None = None
    colors: tuple[str, ...] | None = None
    legend_orient: str = "right"
    legend_length: int = 180
    legend_thickness: int = 14
    show_legend: bool = True
    one_value: float | None = None
    one_color: str | None = None
    one_label: str | None = None
    one_legend_title: str | None = None
    zero_value: float | None = None
    zero_color: str | None = None
    zero_label: str | None = None
    zero_legend_title: str | None = None

    def __post_init__(self) -> None:
        if not self.attribute.strip():
            raise ValueError("Color-scale attribute must be a nonempty string.")
        if not self.scheme.strip():
            raise ValueError("Color-scale scheme must be a nonempty string.")
        if self.scale_type not in {"linear", "log", "ordinal"}:
            raise ValueError("scale_type must be 'linear', 'log', or 'ordinal'.")
        _validate_positive_int(self.legend_length, "legend_length")
        _validate_positive_int(self.legend_thickness, "legend_thickness")
        if (self.one_value is None) != (self.one_color is None):
            raise ValueError("one_value and one_color must be supplied together.")
        if (self.zero_value is None) != (self.zero_color is None):
            raise ValueError("zero_value and zero_color must be supplied together.")
        if self.one_label is not None and self.one_value is None:
            raise ValueError("one_label requires one_value and one_color.")
        if self.zero_label is not None and self.zero_value is None:
            raise ValueError("zero_label requires zero_value and zero_color.")
        if self.domain is not None and self.domain[0] >= self.domain[1]:
            raise ValueError("Color-scale domain must be an increasing pair.")
        if self.categories is not None and not self.categories:
            raise ValueError("categories must not be empty when provided.")
        if self.categories is not None and tuple(sorted(set(self.categories))) != self.categories:
            raise ValueError("categories must be unique and increasing.")
        if self.colors is not None and not self.colors:
            raise ValueError("colors must not be empty when provided.")
        if (
            self.categories is not None
            and self.colors is not None
            and len(self.categories) != len(self.colors)
        ):
            raise ValueError("categories and colors must have the same length.")
        if self.scale_type == "ordinal" and self.domain is not None:
            raise ValueError("ordinal colour scales use categories, not domain.")


@dataclass(frozen=True, slots=True)
class ReferenceLineConfig:
    """A labelled horizontal or vertical reference line for a scalar chart.

    Examples
    --------
    >>> from topographer.plotting import ReferenceLineConfig
    >>> ReferenceLineConfig(axis="y", value=0.5).color
    '#737373'
    """

    axis: Literal["x", "y"]
    value: float
    label: str | None = None
    color: str = "#737373"
    dash: tuple[int, ...] = (4, 4)
    title: str | None = None

    def __post_init__(self) -> None:
        if self.axis not in {"x", "y"}:
            raise ValueError("Reference-line axis must be 'x' or 'y'.")
        if not self.color:
            raise ValueError("Reference-line color must be nonempty.")


def count_color_scale(
    attribute: str = "branch_vertex_count",
    *,
    title: str | None = None,
    scheme: str = "plasma",
    one_color: str = "#4a4a4a",
    domain: tuple[float, float] | None = None,
    legend_orient: str = "right",
    legend_length: int = 180,
    legend_thickness: int = 14,
) -> ColorScaleConfig:
    """Return a log colour scale that paints a count of 1 in its own colour.

    Examples
    --------
    >>> from topographer.plotting import count_color_scale
    >>> count_color_scale().scale_type
    'log'
    """
    """Return the standard exact-one-plus-logarithmic branch-count scale."""

    return ColorScaleConfig(
        attribute=attribute,
        title=title,
        scheme=scheme,
        domain=domain,
        scale_type="log",
        legend_orient=legend_orient,
        legend_length=legend_length,
        legend_thickness=legend_thickness,
        one_value=1.0,
        one_color=one_color,
    )


@dataclass(frozen=True, slots=True)
class TreePlotConfig:
    """Shared visual settings for interactive scalar-tree charts.

    ``edge_shape`` sets how each edge joins its child to its saddle: ``"orthogonal"`` draws a
    vertical segment at the child and a horizontal one at the saddle (an elbow), ``"direct"``
    draws one straight segment, and ``"trunk"`` draws a vertical trunk at the child that rises
    ``trunk_fraction`` of the way to the saddle, then a diagonal link to the saddle. The y values
    always come from the endpoint scalars, so the scalar axis is the same for every shape.
    ``trunk_fraction`` must satisfy ``0 < trunk_fraction <= 1`` and is used only by ``"trunk"``.

    ``coordinate_system="polar"`` draws the tree as a fan: the scalar maps exactly to the radius
    (``inner_radius_fraction * max radius`` plus the distance from the root scalar), x maps to
    the angle over ``angular_span_deg`` starting at ``angular_start_deg`` (degrees, counter-
    clockwise from the +x axis; use ``0`` with span ``180`` for a half-fan with the root at the
    bottom centre), and the y axis is replaced by concentric rings and tick labels along the
    ray at the start angle (``polar_axis``). Orthogonal edges become a radial segment plus an
    arc sampled at ``arc_samples_per_radian``. Polar works with any ``horizontal_layout``; the
    weighted one spreads the wedge by leaf count. Tick labels use a Python format spec, so
    d3-only specs fall back to ``.6g`` there.

    Examples
    --------
    >>> from topographer.plotting import TreePlotConfig
    >>> TreePlotConfig().edge_shape
    'orthogonal'
    """

    width: int = 420
    height: int = 250
    font_size: int = 15
    scalar_axis_title: str | None = None
    scalar_tick_count: int = 6
    scalar_tick_values: tuple[float, ...] | None = None
    scalar_tick_format: str = ".2f"
    scalar_domain: tuple[float, float] | None = None
    show_scalar_grid: bool = False
    show_scalar_axis_line: bool = False
    horizontal_padding: int = 18
    horizontal_domain: tuple[float, float] | None = None
    scalar_padding_fraction: float = 0.05
    node_size: int = 55
    structural_node_size: int | None = None
    edge_width: float = 1.4
    edge_color: str = "#737373"
    edge_shape: Literal["orthogonal", "direct", "trunk"] = "orthogonal"
    trunk_fraction: float = 0.85
    edge_color_scale: ColorScaleConfig | None = None
    annotation_overlap_step: float | None = None
    coordinate_system: Literal["cartesian", "polar"] = "cartesian"
    angular_span_deg: float = 360.0
    angular_start_deg: float = 90.0
    inner_radius_fraction: float = 0.0
    polar_axis: Literal["rings", "ray", "both"] = "both"
    arc_samples_per_radian: int = 32

    def __post_init__(self) -> None:
        _validate_positive_int(self.width, "width")
        _validate_positive_int(self.height, "height")
        _validate_positive_int(self.font_size, "font_size")
        _validate_positive_int(self.scalar_tick_count, "scalar_tick_count")
        _validate_positive_int(self.node_size, "node_size")
        if self.structural_node_size is not None:
            _validate_positive_int(self.structural_node_size, "structural_node_size")
        if self.horizontal_padding < 0:
            raise ValueError("horizontal_padding must be non-negative.")
        _validate_domain(self.horizontal_domain, "horizontal_domain")
        if self.scalar_padding_fraction < 0.0:
            raise ValueError("scalar_padding_fraction must be non-negative.")
        if self.annotation_overlap_step is not None and self.annotation_overlap_step < 0.0:
            raise ValueError("annotation_overlap_step must be non-negative.")
        if self.edge_width <= 0.0:
            raise ValueError("edge_width must be positive.")
        if not self.edge_color:
            raise ValueError("edge_color must be a nonempty color string.")
        if self.edge_shape not in {"orthogonal", "direct", "trunk"}:
            raise ValueError("edge_shape must be 'orthogonal', 'direct' or 'trunk'.")
        if not 0.0 < self.trunk_fraction <= 1.0:
            raise ValueError("trunk_fraction must satisfy 0 < trunk_fraction <= 1.")
        _validate_domain(self.scalar_domain, "scalar_domain")
        _validate_tick_values(self.scalar_tick_values, "scalar_tick_values")
        if self.coordinate_system not in {"cartesian", "polar"}:
            raise ValueError("coordinate_system must be 'cartesian' or 'polar'.")
        if not 0.0 < self.angular_span_deg <= 360.0:
            raise ValueError("angular_span_deg must satisfy 0 < angular_span_deg <= 360.")
        if not 0.0 <= self.inner_radius_fraction < 1.0:
            raise ValueError("inner_radius_fraction must satisfy 0 <= fraction < 1.")
        if self.polar_axis not in {"rings", "ray", "both"}:
            raise ValueError("polar_axis must be 'rings', 'ray' or 'both'.")
        if self.arc_samples_per_radian < 2:
            raise ValueError("arc_samples_per_radian must be at least 2.")


@dataclass(frozen=True, slots=True)
class TreeMarkerConfig:
    """Presentation settings for a table-driven tree marker overlay.

    Examples
    --------
    >>> from topographer.plotting import TreeMarkerConfig
    >>> TreeMarkerConfig().shape
    'circle'
    """

    shape: str = "circle"
    color: str = "#4c78a8"
    size: int = 65
    opacity: float = 0.9
    filled: bool = True
    label_field: str | None = None
    label_color: str = "#000000"
    label_dx: float = 8.0
    label_font_size: int | None = None

    def __post_init__(self) -> None:
        if not self.shape.strip():
            raise ValueError("Tree marker shape must be a nonempty string.")
        if not self.color.strip():
            raise ValueError("Tree marker color must be a nonempty string.")
        _validate_positive_int(self.size, "size")
        if isinstance(self.opacity, bool) or not 0.0 <= self.opacity <= 1.0:
            raise ValueError("Tree marker opacity must be between 0 and 1.")
        if self.label_field is not None and not self.label_field.strip():
            raise ValueError("Tree marker label_field must be nonempty when provided.")
        if not self.label_color.strip():
            raise ValueError("Tree marker label color must be a nonempty string.")
        if self.label_font_size is not None:
            _validate_positive_int(self.label_font_size, "label_font_size")


@dataclass(frozen=True, slots=True)
class TreeAnnotationLayer:
    """One immutable application-marker layer over a tree chart.

    ``marker_table`` is a Polars data frame so the renderer can remain generic:
    supply ``vertex_id`` rows for automatic mapping through plot data built
    from a merge tree and its network, or explicit ``x``/``y`` coordinates.
    Optional tooltip attributes describe the original annotation rows.
    A ``role`` of ``"wild_type"`` is always rendered after ordinary annotation
    layers, preserving its visibility when it overlaps another marker.

    Examples
    --------
    >>> from topographer.plotting import TreeAnnotationLayer
    >>> TreeAnnotationLayer.__name__
    'TreeAnnotationLayer'

    """

    marker_table: pl.DataFrame
    marker_config: TreeMarkerConfig = field(default_factory=TreeMarkerConfig)
    tooltip_attrs: tuple[str, ...] = ()
    role: Literal["annotation", "wild_type"] = "annotation"

    def __post_init__(self) -> None:
        missing_columns = sorted(
            (
                {"x", "y"} if "vertex_id" not in self.marker_table.columns else {"vertex_id"}
            ).difference(self.marker_table.columns)
        )
        if missing_columns:
            raise ValueError(
                f"Tree annotation marker table is missing required columns: {missing_columns}."
            )
        missing_tooltips = sorted(set(self.tooltip_attrs).difference(self.marker_table.columns))
        if missing_tooltips:
            raise ValueError(f"Unknown tree annotation tooltip columns: {missing_tooltips}.")
        if self.marker_config.label_field not in {None, *self.marker_table.columns}:
            raise ValueError(
                f"Tree annotation label column {self.marker_config.label_field!r} is not present."
            )
        if self.role not in {"annotation", "wild_type"}:
            raise ValueError("Tree annotation role must be 'annotation' or 'wild_type'.")


@dataclass(frozen=True, slots=True)
class TreePresentationConfig:
    """Visual role, metric encoding, and annotation layers for a tree chart.

    ``baseline`` renders neutral structural critical-event markers;
    ``annotation`` adds supplied application-marker layers; and ``metric``
    colour-encodes branches through ``metric_color_scale`` while retaining
    supplied annotations such as the wild type. The renderer can suppress
    structural marks for linked summary compositions.

    Examples
    --------
    >>> from topographer.plotting import TreePresentationConfig
    >>> TreePresentationConfig(mode="baseline").mode
    'baseline'
    """

    mode: Literal["baseline", "annotation", "metric"] = "baseline"
    annotation_layers: tuple[TreeAnnotationLayer, ...] = ()
    metric_color_scale: ColorScaleConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"baseline", "annotation", "metric"}:
            raise ValueError(
                "Tree presentation mode must be 'baseline', 'annotation', or 'metric'."
            )


@dataclass(frozen=True, slots=True)
class PersistencePlotConfig:
    """Shared visual settings for persistence diagrams and related charts.

    Examples
    --------
    >>> from topographer.plotting import PersistencePlotConfig
    >>> PersistencePlotConfig().birth_axis_title
    'Birth'
    """

    width: int = 250
    height: int = 250
    font_size: int = 15
    birth_axis_title: str = "Birth"
    death_axis_title: str = "Death"
    tick_count: int = 6
    tick_values: tuple[float, ...] | None = None
    tick_format: str = ".1f"
    domain: tuple[float, float] | None = None
    point_size: int = 65
    diagonal_width: float = 1.0
    show_grid: bool = False
    show_axis_line: bool = False
    color_scale: ColorScaleConfig | None = None
    reference_lines: tuple[ReferenceLineConfig, ...] = ()

    def __post_init__(self) -> None:
        _validate_positive_int(self.width, "width")
        _validate_positive_int(self.height, "height")
        _validate_positive_int(self.font_size, "font_size")
        _validate_positive_int(self.tick_count, "tick_count")
        _validate_positive_int(self.point_size, "point_size")
        if self.diagonal_width <= 0.0:
            raise ValueError("diagonal_width must be positive.")
        if not self.birth_axis_title or not self.death_axis_title:
            raise ValueError("Persistence axis titles must be nonempty strings.")
        _validate_domain(self.domain, "domain")
        _validate_tick_values(self.tick_values, "tick_values")


def _validate_positive_int(value: int, name: str) -> None:
    """Reject non-positive or boolean presentation dimensions and counts."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_domain(value: tuple[float, float] | None, name: str) -> None:
    """Validate an optional increasing numeric scale domain."""

    if value is not None and value[0] >= value[1]:
        raise ValueError(f"{name} must be an increasing pair.")


def _validate_tick_values(value: tuple[float, ...] | None, name: str) -> None:
    """Reject empty explicit tick collections."""

    if value is not None and not value:
        raise ValueError(f"{name} must not be empty when provided.")
