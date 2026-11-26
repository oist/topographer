"""Fitness histogram data and Altair rendering."""

from __future__ import annotations

from bisect import bisect_right
from decimal import Decimal
import math
from numbers import Integral
from typing import Any

import polars as pl


def compute_fitness_histogram_table(
    table: pl.DataFrame,
    *,
    scalar_column: str = "scalar",
    bin_count: int = 100,
) -> pl.DataFrame:
    """Count rows in equal-width bins spanning the observed scalar range.

    Parameters
    ----------
    table
        Input rows, one scalar observation per row.
    scalar_column
        Numeric column to histogram.
    bin_count
        Number of equal-width bins (default 100).

    Returns
    -------
    polars.DataFrame
        One row per bin with ``bin_min``, ``bin_max``, and integer ``count``.
        Bins are left-closed and right-open except the final bin, which includes
        the observed maximum.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.plotting import compute_fitness_histogram_table
    >>> bins = compute_fitness_histogram_table(pl.DataFrame({"scalar": [0.0, 0.5, 1.0]}), bin_count=2)
    >>> bins["count"].to_list()
    [1, 2]

    Raises
    ------
    KeyError
        If ``scalar_column`` is missing.
    ValueError
        If values are empty, null, non-numeric, non-finite, constant, or cannot
        be represented with the requested bin count.
    """
    if scalar_column not in table.columns:
        raise KeyError(f"table is missing scalar column: {scalar_column!r}")
    if isinstance(bin_count, bool) or not isinstance(bin_count, Integral) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer.")
    if table.is_empty():
        raise ValueError("table must contain at least one scalar value.")

    values = table.get_column(scalar_column).to_list()
    if any(value is None for value in values):
        raise ValueError("scalar values must not be null.")
    try:
        scalars = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("scalar values must be numeric and finite.") from error
    if any(not math.isfinite(value) for value in scalars):
        raise ValueError("scalar values must be finite.")
    minimum, maximum = min(scalars), max(scalars)
    if minimum == maximum:
        raise ValueError("scalar values must span a non-zero range.")

    decimal_min = Decimal(str(minimum))
    decimal_span = Decimal(str(maximum)) - decimal_min
    edges = [float(decimal_min + decimal_span * i / bin_count) for i in range(bin_count)] + [
        maximum
    ]
    if any(not math.isfinite(edge) for edge in edges) or any(
        a >= b for a, b in zip(edges, edges[1:])
    ):
        raise ValueError("scalar range cannot be represented by the requested bins.")
    counts = [0] * bin_count
    for value in scalars:
        index = min(bisect_right(edges, value) - 1, bin_count - 1)
        counts[index] += 1

    return pl.DataFrame(
        {
            "bin_min": edges[:-1],
            "bin_max": edges[1:],
            "count": counts,
        },
        schema={"bin_min": pl.Float64, "bin_max": pl.Float64, "count": pl.Int64},
    )


def plot_fitness_histogram(
    table: pl.DataFrame,
    *,
    scalar_column: str = "scalar",
    bin_count: int = 100,
    scalar_domain: tuple[float, float] | None = None,
    count_domain: tuple[int, int] | None = None,
    scalar_scale: Any | None = None,
    scalar_axis: Any | bool | None = None,
    count_axis: Any | None = None,
    scalar_title: str = "Fitness",
    count_title: str = "Vertex count",
    color: str = "#bdbdbd",
    width: int = 140,
    height: int = 250,
    font_size: int = 15,
    include_config: bool = True,
) -> Any:
    """Plot touching horizontal count bars with the scalar increasing upward.

    ``scalar_domain`` and ``count_domain`` control display scales independently
    of the observed range used to construct bins. This lets separate histograms
    retain their own bin edges while sharing axes in a composed chart.

    Parameters
    ----------
    table
        Input rows, one scalar observation per row.
    scalar_column
        Numeric column to histogram.
    bin_count
        Number of equal-width bins (default 100).
    scalar_domain
        Optional finite increasing display bounds for the scalar axis.
    count_domain
        Optional display bounds for counts; must start at zero and include
        every bin count. When omitted, the maximum observed bin count is used.
    scalar_scale
        Optional Altair scale override for the scalar axis.
    scalar_axis
        Optional Altair axis override for the scalar axis; pass ``False`` to
        hide it.
    count_axis
        Optional Altair axis override for the count axis.
    scalar_title, count_title
        Axis titles used when the corresponding axis override is omitted.
    color
        Bar color.
    width, height
        Chart dimensions in pixels.
    font_size
        Axis label and title font size when configuration is enabled.
    include_config
        Whether to attach axis and view configuration to the chart.

    Returns
    -------
    alt.Chart
        Horizontal count histogram.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.plotting import plot_fitness_histogram
    >>> chart = plot_fitness_histogram(pl.DataFrame({"scalar": [0.0, 0.5, 1.0]}), bin_count=2)
    >>> chart.to_dict(validate=True)["mark"]["type"]
    'bar'

    Raises
    ------
    ValueError
        If either display domain is invalid.
    """
    try:
        import altair as alt
    except ImportError as error:
        raise ImportError(
            "Altair plotting requires the 'plotting' optional dependency."
        ) from error

    bins = compute_fitness_histogram_table(table, scalar_column=scalar_column, bin_count=bin_count)
    max_count = int(bins["count"].max())
    if scalar_domain is not None:
        if (
            len(scalar_domain) != 2
            or not all(math.isfinite(float(value)) for value in scalar_domain)
            or scalar_domain[0] >= scalar_domain[1]
        ):
            raise ValueError("scalar_domain must contain finite increasing bounds.")
    if scalar_scale is None:
        scalar_scale = (
            alt.Scale(domain=list(scalar_domain), zero=False)
            if scalar_domain
            else alt.Scale(zero=False)
        )
    if scalar_axis is None:
        scalar_axis = alt.Axis(title=scalar_title, grid=False)
    if count_axis is None:
        count_axis = alt.Axis(title=count_title, grid=False)
    if count_domain is not None and (
        len(count_domain) != 2
        or not all(math.isfinite(float(value)) for value in count_domain)
        or count_domain[0] != 0
        or count_domain[1] < max_count
    ):
        raise ValueError("count_domain must start at zero and include every bin count.")
    count_scale = (
        alt.Scale(domain=list(count_domain), zero=False)
        if count_domain
        else alt.Scale(domain=[0, max_count], zero=False)
    )

    scalar_axis_value = None if scalar_axis is False else scalar_axis
    chart = (
        alt.Chart(bins)
        .mark_bar(color=color, binSpacing=0, orient="horizontal")
        .encode(
            y=alt.Y("bin_min:Q", scale=scalar_scale, axis=scalar_axis_value),
            y2=alt.Y2("bin_max:Q"),
            x=alt.X("count:Q", scale=count_scale, axis=count_axis),
            x2=alt.X2(datum=0),
        )
        .properties(width=width, height=height)
    )
    if not include_config:
        return chart
    return chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size).configure_view(
        strokeWidth=0
    )
