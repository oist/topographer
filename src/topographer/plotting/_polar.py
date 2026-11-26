"""Polar (radial) coordinates for scalar trees: fitness maps exactly to radius."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import polars as pl

from topographer.plotting._config import TreePlotConfig

_MAX_ARC_POINTS = 64  # per edge arc only; rings are never clipped


@dataclass(frozen=True, slots=True)
class PolarTransform:
    """Affine map from tree layout ``(x, scalar)`` to Cartesian polar coordinates.

    The angle is an affine image of x: a leaf at x sits in the middle of its slot of
    ``leaf_gap`` x-units, and the whole x extent plus one slot spans ``span_deg``. The radius is
    ``inner_radius + sign * (scalar - root_scalar)``, so equal scalars share a circle.

    Examples
    --------
    >>> from topographer.plotting._polar import PolarTransform
    >>> t = PolarTransform(
    ...     x_min=0.0, x_extent=3.0, leaf_gap=1.0, span_deg=180.0, start_deg=0.0,
    ...     inner_radius=0.0, root_scalar=0.0, sign=1.0, height=2.0,
    ... )
    >>> t.radius(1.5)
    1.5
    """

    x_min: float
    x_extent: float
    leaf_gap: float
    span_deg: float
    start_deg: float
    inner_radius: float
    root_scalar: float
    sign: float
    height: float

    @property
    def outer_radius(self) -> float:
        return self.inner_radius + self.height

    def radius(self, scalar: float) -> float:
        return self.inner_radius + self.sign * (scalar - self.root_scalar)

    def angle(self, x: float) -> float:
        """Angle in radians."""
        width = self.x_extent + self.leaf_gap
        fraction = (x - self.x_min + 0.5 * self.leaf_gap) / width
        return math.radians(self.start_deg + self.span_deg * fraction)

    def point(self, x: float, scalar: float) -> tuple[float, float]:
        return self.cartesian(self.radius(scalar), self.angle(x))

    @staticmethod
    def cartesian(radius: float, angle: float) -> tuple[float, float]:
        return radius * math.cos(angle), radius * math.sin(angle)


def make_polar_transform(
    node_table: pl.DataFrame,
    edge_table: pl.DataFrame,
    root_node_index: int,
    root_kind: str,
    config: TreePlotConfig,
) -> PolarTransform:
    """Build the transform from laid-out tables and the polar settings of ``config``."""

    x = node_table["x"].to_list()
    scalars = node_table["scalar"].to_list()
    root_scalar = float(
        node_table.filter(pl.col("tree_node_index") == root_node_index).item(0, "scalar")
    )
    sign = 1.0 if root_kind == "minimum" else -1.0
    internal = (
        set(edge_table["source_tree_node_index"].to_list())
        if "source_tree_node_index" in edge_table.columns
        else set()
    )
    leaf_x = sorted(
        {
            float(value)
            for index, value in zip(node_table["tree_node_index"], x, strict=True)
            if index not in internal
        }
    )
    gaps = [b - a for a, b in zip(leaf_x, leaf_x[1:], strict=False) if b > a]
    leaf_gap = min(gaps) if gaps else 1.0
    domain = config.scalar_domain or (min(scalars), max(scalars))
    heights = [sign * (value - root_scalar) for value in (*scalars, *domain)]
    height = max(max(heights), 0.0)
    fraction = config.inner_radius_fraction
    return PolarTransform(
        x_min=float(min(x)),
        x_extent=float(max(x) - min(x)),
        leaf_gap=float(leaf_gap),
        span_deg=config.angular_span_deg,
        start_deg=config.angular_start_deg,
        inner_radius=fraction * height / (1.0 - fraction),
        root_scalar=root_scalar,
        sign=sign,
        height=height,
    )


def polar_node_rows(rows: list[dict[str, Any]], transform: PolarTransform) -> list[dict[str, Any]]:
    """Return node rows with Cartesian ``x``/``y`` and polar ``radius``/``angle`` columns."""

    out = []
    for row in rows:
        x, y = transform.point(float(row["x"]), float(row["scalar"]))
        out.append(
            {
                **row,
                "x": x,
                "y": y,
                "radius": transform.radius(float(row["scalar"])),
                "angle": transform.angle(float(row["x"])),
            }
        )
    return out


def polar_straight_edge_rows(
    rows: list[dict[str, Any]], transform: PolarTransform
) -> list[dict[str, Any]]:
    """Return edge rows with Cartesian chord endpoints and polar endpoint columns."""

    out = []
    for row in rows:
        xs, ys = transform.point(float(row["x_source"]), float(row["source_scalar"]))
        xt, yt = transform.point(float(row["x_target"]), float(row["target_scalar"]))
        out.append(
            {
                **row,
                "x_source": xs,
                "y_source": ys,
                "x_target": xt,
                "y_target": yt,
                "radius_source": transform.radius(float(row["source_scalar"])),
                "radius_target": transform.radius(float(row["target_scalar"])),
                "angle_source": transform.angle(float(row["x_source"])),
                "angle_target": transform.angle(float(row["x_target"])),
            }
        )
    return out


def polar_marker_table(table: pl.DataFrame, transform: PolarTransform) -> pl.DataFrame:
    """Move marker ``x``/``y`` (layout x, scalar) to Cartesian polar coordinates."""

    if table.is_empty():
        return table
    rows = []
    for row in table.to_dicts():
        x, y = transform.point(float(row["x"]), float(row["y"]))
        rows.append({**row, "x": x, "y": y})
    return pl.DataFrame(rows)


def _arc_count(delta_angle: float, samples_per_radian: int, cap: int | None = None) -> int:
    count = max(math.ceil(abs(delta_angle) * samples_per_radian) + 1, 2)
    return count if cap is None else min(count, cap)


def polar_edge_path_rows(
    rows: list[dict[str, Any]],
    transform: PolarTransform,
    edge_shape: str,
    trunk_fraction: float,
    samples_per_radian: int,
) -> list[dict[str, Any]]:
    """Return polyline path rows for ``orthogonal`` (radial plus arc) and ``trunk`` edges."""

    paths: list[dict[str, Any]] = []
    for index, edge in enumerate(rows):
        common = {**edge, "logical_edge_index": index}
        theta_c = transform.angle(float(edge["x_target"]))
        theta_s = transform.angle(float(edge["x_source"]))
        r_c = transform.radius(float(edge["y_target"]))
        r_s = transform.radius(float(edge["y_source"]))
        if edge_shape == "orthogonal":
            count = _arc_count(theta_s - theta_c, samples_per_radian, _MAX_ARC_POINTS)
            vertices = [(r_c, theta_c, "radial")]
            vertices += [
                (r_s, theta_c + (theta_s - theta_c) * k / (count - 1), "arc") for k in range(count)
            ]
        else:
            r_trunk = r_c + trunk_fraction * (r_s - r_c)
            vertices = [
                (r_c, theta_c, "trunk"),
                (r_trunk, theta_c, "trunk"),
                (r_s, theta_s, "link"),
            ]
        for order, (radius, angle, kind) in enumerate(vertices):
            x, y = transform.cartesian(radius, angle)
            paths.append(
                {
                    **common,
                    "path_order": order,
                    "segment_kind": kind,
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "angle": angle,
                }
            )
    return paths


def nice_ticks(lower: float, upper: float, count: int) -> list[float]:
    """Return about ``count`` round tick values covering ``[lower, upper]``.

    Examples
    --------
    >>> from topographer.plotting._polar import nice_ticks
    >>> nice_ticks(0.0, 1.0, 5)
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    """

    if upper <= lower:
        return [lower]
    raw = (upper - lower) / max(count, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1.0, 2.0, 5.0, 10.0) if m * magnitude >= raw)
    first = math.ceil(lower / step - 1e-9)
    last = math.floor(upper / step + 1e-9)
    return [round(k * step, 12) for k in range(first, last + 1)]


def polar_tick_values(
    transform: PolarTransform, config: TreePlotConfig, scalars: Sequence[float]
) -> list[float]:
    """Tick scalars whose radius lies in the plotted radial range."""

    if config.scalar_tick_values is not None:
        ticks = [float(v) for v in config.scalar_tick_values]
    else:
        domain = config.scalar_domain or (min(scalars), max(scalars))
        ticks = nice_ticks(float(domain[0]), float(domain[1]), config.scalar_tick_count)
    return [
        t
        for t in ticks
        if -1e-12 <= transform.sign * (t - transform.root_scalar) <= transform.height + 1e-12
    ]


def format_tick(value: float, spec: str) -> str:
    """Format a tick with a Python format spec, falling back to ``.6g`` for d3-only specs."""

    try:
        return format(value, spec)
    except ValueError:
        return format(value, ".6g")


def ring_rows(
    transform: PolarTransform, ticks: Sequence[float], samples_per_radian: int
) -> list[dict[str, Any]]:
    """Return sampled concentric ring polylines, one ring per tick, over the angular wedge."""

    rows: list[dict[str, Any]] = []
    start = math.radians(transform.start_deg)
    span = math.radians(transform.span_deg)
    count = _arc_count(span, samples_per_radian)
    for ring, tick in enumerate(ticks):
        radius = transform.radius(tick)
        for k in range(count):
            x, y = transform.cartesian(radius, start + span * k / (count - 1))
            rows.append(
                {
                    "ring_index": ring,
                    "path_order": k,
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "scalar": tick,
                }
            )
    return rows


def ray_label_rows(
    transform: PolarTransform, ticks: Sequence[float], spec: str
) -> list[dict[str, Any]]:
    """Return tick labels placed along the ray at the start angle."""

    angle = math.radians(transform.start_deg)
    rows = []
    for tick in ticks:
        radius = transform.radius(tick)
        x, y = transform.cartesian(radius, angle)
        rows.append(
            {"x": x, "y": y, "radius": radius, "scalar": tick, "label": format_tick(tick, spec)}
        )
    return rows


def polar_domains(
    transform: PolarTransform, width: int, height: int
) -> tuple[list[float], list[float]]:
    """Return x and y scale domains with equal units, framing the wedge in the slot."""

    outer = transform.outer_radius if transform.outer_radius > 0.0 else 1.0
    start = transform.start_deg
    end = start + transform.span_deg
    quarter_turns = range(math.ceil(start / 90.0), math.floor(end / 90.0) + 1)
    angles = [start, end, *(90.0 * k for k in quarter_turns)]
    points = [(0.0, 0.0)]
    for radius in (outer, transform.inner_radius):
        points += [
            (radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)))
            for a in angles
        ]
    margin = 0.08 * outer
    x_lo = min(p[0] for p in points) - margin
    x_hi = max(p[0] for p in points) + margin
    y_lo = min(p[1] for p in points) - margin
    y_hi = max(p[1] for p in points) + margin
    unit = max((x_hi - x_lo) / width, (y_hi - y_lo) / height)
    x_pad = (width * unit - (x_hi - x_lo)) / 2.0
    y_pad = (height * unit - (y_hi - y_lo)) / 2.0
    return [x_lo - x_pad, x_hi + x_pad], [y_lo - y_pad, y_hi + y_pad]


def polar_axis_layers(
    alt: Any,
    transform: PolarTransform,
    config: TreePlotConfig,
    ticks: Sequence[float],
    x_scale: Any,
    y_scale: Any,
    title: str,
) -> list[Any]:
    """Return the ring and ray-label layers selected by ``config.polar_axis``."""

    def encoded(chart: Any) -> Any:
        return chart.encode(
            x=alt.X("x:Q", axis=None, scale=x_scale), y=alt.Y("y:Q", axis=None, scale=y_scale)
        )

    layers: list[Any] = []
    if config.polar_axis in {"rings", "both"}:
        rows = ring_rows(transform, ticks, config.arc_samples_per_radian)
        if rows:
            layers.append(
                encoded(
                    alt.Chart(alt.Data(values=rows)).mark_line(
                        color="#d4d4d4", strokeWidth=0.8, strokeDash=[3, 3]
                    )
                ).encode(detail="ring_index:N", order="path_order:Q")
            )
    if config.polar_axis in {"ray", "both"}:
        labels = ray_label_rows(transform, ticks, config.scalar_tick_format)
        if labels:
            layers.append(
                encoded(
                    alt.Chart(alt.Data(values=labels)).mark_text(
                        align="center",
                        baseline="top",
                        dy=4,
                        color="#525252",
                        fontSize=config.font_size,
                    )
                ).encode(text=alt.Text("label:N"))
            )
        mid = transform.cartesian(transform.outer_radius * 0.5, math.radians(transform.start_deg))
        layers.append(
            encoded(
                alt.Chart(alt.Data(values=[{"x": mid[0], "y": mid[1], "label": title}])).mark_text(
                    align="center",
                    baseline="top",
                    dy=config.font_size + 12,
                    angle=(-transform.start_deg) % 360.0,
                    color="#525252",
                    fontSize=config.font_size,
                )
            ).encode(text=alt.Text("label:N"))
        )
    return layers
