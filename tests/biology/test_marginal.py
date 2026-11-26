"""Tests for fitness histograms and composable charts."""

from __future__ import annotations

import math

import polars as pl
import pytest

from topographer.examples import build_cube_scalar_network
from topographer.plotting import (
    compute_fitness_histogram_table,
    make_tree_layout,
    plot_branch_spans,
    plot_fitness_histogram,
    plot_persistence,
    plot_tree,
)
from topographer.topology import ScalarNetworkTopology


def test_histogram_counts_rows_in_equal_width_observed_range_bins() -> None:
    bins = compute_fitness_histogram_table(
        pl.DataFrame({"scalar": [0.0, 0.25, 0.5, 0.75, 1.0]}), bin_count=4
    )
    assert bins.columns == ["bin_min", "bin_max", "count"]
    assert bins["count"].to_list() == [1, 1, 1, 2]
    assert bins["bin_min"][0] == 0
    assert bins["bin_max"][-1] == 1
    assert bins["count"].sum() == 5
    edge_bins = compute_fitness_histogram_table(
        pl.DataFrame({"scalar": [0.0, 0.29, 1.0]}), bin_count=100
    )
    assert edge_bins["count"].sum() == 3
    assert edge_bins["count"][29] == 1
    assert edge_bins.height == 100
    repeated_decimal = compute_fitness_histogram_table(
        pl.DataFrame({"scalar": [0.0, 0.1, 0.2, 0.3]}), bin_count=3
    )
    assert repeated_decimal["count"].to_list() == [1, 1, 2]
    for index, edge in enumerate(repeated_decimal["bin_min"].to_list()[1:], start=1):
        assigned = compute_fitness_histogram_table(
            pl.DataFrame({"scalar": [0.0, math.nextafter(edge, -math.inf), edge, 0.3]}),
            bin_count=3,
        )["count"].to_list()
        assert assigned[index - 1] == (2 if index == 1 else 1)
        assert assigned[index] == (2 if index == 2 else 1)
    assert compute_fitness_histogram_table(pl.DataFrame({"scalar": [0.0, 0.2, 0.4]})).height == 100
    with pytest.raises(KeyError, match="scalar column"):
        compute_fitness_histogram_table(pl.DataFrame({"other": [0.0, 1.0]}))


@pytest.mark.parametrize(
    "values", [[], [1.0, 1.0], [1.0, None], [1.0, float("nan")], [1.0, float("inf")]]
)
def test_histogram_rejects_invalid_scalar_data(values: list[float | None]) -> None:
    if not values:
        table = pl.DataFrame(schema={"scalar": pl.Float64})
    else:
        table = pl.DataFrame({"scalar": values}, strict=False)
    with pytest.raises(ValueError):
        compute_fitness_histogram_table(table)


@pytest.mark.parametrize("bin_count", [0, -1, 1.5, True])
def test_histogram_rejects_invalid_bin_count(bin_count: float | bool) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        compute_fitness_histogram_table(pl.DataFrame({"scalar": [0.0, 1.0]}), bin_count=bin_count)


def test_histogram_plot_uses_count_bars_and_separate_display_domains() -> None:
    table = pl.DataFrame({"raw_fitness": [0.1, 0.3, 0.6, 0.8]})
    chart = plot_fitness_histogram(
        table,
        scalar_column="raw_fitness",
        bin_count=4,
        scalar_domain=(0, 1),
        count_domain=(0, 5),
        include_config=False,
    )
    spec = chart.to_dict(validate=True)
    assert spec["mark"]["type"] == "bar"
    assert spec["mark"]["binSpacing"] == 0
    assert spec["encoding"]["y"]["field"] == "bin_min"
    assert spec["encoding"]["y2"]["field"] == "bin_max"
    assert spec["encoding"]["x"]["field"] == "count"
    assert spec["encoding"]["x"]["scale"]["domain"] == [0, 5]
    assert "color" not in spec["encoding"]
    assert spec["encoding"]["x2"] == {"datum": 0}


def test_histogram_titles_axis_overrides_and_domain_validation() -> None:
    import altair as alt

    table = pl.DataFrame({"scalar": [0.0, 0.5, 1.0]})
    custom = plot_fitness_histogram(
        table,
        bin_count=2,
        scalar_title="Raw fitness",
        count_title="Sequences",
        include_config=False,
    ).to_dict(validate=True)
    assert custom["encoding"]["y"]["axis"]["title"] == "Raw fitness"
    assert custom["encoding"]["x"]["axis"]["title"] == "Sequences"

    overridden = plot_fitness_histogram(
        table,
        scalar_axis=alt.Axis(title="Provided fitness"),
        count_axis=alt.Axis(title="Provided count"),
        include_config=False,
    ).to_dict(validate=True)
    assert overridden["encoding"]["y"]["axis"]["title"] == "Provided fitness"
    assert overridden["encoding"]["x"]["axis"]["title"] == "Provided count"
    hidden = plot_fitness_histogram(table, scalar_axis=False, include_config=False).to_dict(
        validate=True
    )
    assert hidden["encoding"]["y"]["axis"] is None

    for kwargs in (
        {"scalar_domain": (1.0, 0.0)},
        {"scalar_domain": (0.0, float("inf"))},
        {"count_domain": (1, 10)},
        {"count_domain": (0, float("nan"))},
        {"count_domain": (0, 0)},
    ):
        with pytest.raises(ValueError):
            plot_fitness_histogram(table, **kwargs)


def test_independently_binned_populations_can_share_display_axes() -> None:
    import altair as alt

    first = pl.DataFrame({"scalar": [0.0, 0.2, 0.4]})
    second = pl.DataFrame({"scalar": [0.1, 0.5, 0.9]})
    chart = alt.hconcat(
        plot_fitness_histogram(
            first, bin_count=3, scalar_domain=(0, 1), count_domain=(0, 3), include_config=False
        ),
        plot_fitness_histogram(
            second, bin_count=5, scalar_domain=(0, 1), count_domain=(0, 3), include_config=False
        ),
    ).resolve_scale(y="shared", x="shared")
    spec = chart.to_dict(validate=True)
    assert spec["resolve"]["scale"] == {"x": "shared", "y": "shared"}
    datasets = spec["datasets"]
    left, right = [datasets[item["data"]["name"]] for item in spec["hconcat"]]
    assert len(left) == 3
    assert len(right) == 5

    same_bins = plot_fitness_histogram(
        first, bin_count=3, scalar_domain=(-1, 2), count_domain=(0, 10), include_config=False
    ).to_dict(validate=True)
    same_name = same_bins["data"]["name"]
    assert spec["datasets"][spec["hconcat"][0]["data"]["name"]] == spec["datasets"][same_name]


def test_individual_charts_compose_horizontally() -> None:
    import altair as alt

    network = build_cube_scalar_network()
    tree = ScalarNetworkTopology(network).join_tree()
    tree_data = make_tree_layout(tree, network=network)
    persistence = pl.DataFrame(
        {
            "birth_scalar": [1.0, 0.0],
            "death_scalar": [0.2, 0.0],
            "persistence": [0.8, 0.0],
            "extremum_event": ["maximum", "minimum"],
        }
    )
    histogram = pl.DataFrame({"scalar": [0.1, 0.2, 0.5, 0.9]})
    chart = alt.hconcat(
        plot_fitness_histogram(histogram, scalar_domain=(0.0, 1.0), include_config=False),
        plot_tree(tree_data, include_config=False),
        plot_persistence(persistence, color_by=None, include_config=False),
    ).resolve_scale(y="shared", x="independent")
    spec = chart.to_dict(validate=True)
    assert len(spec["hconcat"]) == 3
    assert spec["resolve"]["scale"]["y"] == "shared"
    assert spec["resolve"]["scale"]["x"] == "independent"


def test_generic_summary_annotation_table_selects_endpoint_ids() -> None:
    p_table = pl.DataFrame(
        {
            "birth_scalar": [1.0, 0.8],
            "death_scalar": [0.2, 0.4],
            "persistence": [0.8, 0.4],
            "birth_vertex_id": [5, 6],
            "saddle_vertex_id": [3, 4],
        }
    )
    annotations = pl.DataFrame({"vertex_id": [5, 3], "has_stop_codon": [True, True]})
    persistence_spec = plot_persistence(
        p_table,
        color_by=None,
        annotation_table=annotations,
        summary_vertex_id_columns=("birth_vertex_id",),
    ).to_dict(validate=True)
    marker_rows = [
        value
        for layer in persistence_spec["layer"]
        for value in layer.get("data", {}).get("values", [])
        if "vertex_id" in value
    ]
    assert [row["vertex_id"] for row in marker_rows] == [5]

    spans = pl.DataFrame(
        {
            "source_scalar": [1.0, 0.8],
            "target_scalar": [0.2, 0.4],
            "branch_span": [0.8, 0.4],
            "source_vertex_id": [5, 6],
            "target_vertex_id": [3, 4],
            "source_event": ["maximum", "maximum"],
            "target_event": ["join_saddle", "join_saddle"],
        }
    )
    span_spec = plot_branch_spans(spans, annotation_table=annotations).to_dict(validate=True)
    marker_rows = [
        value
        for layer in span_spec["layer"]
        for value in layer.get("data", {}).get("values", [])
        if "vertex_id" in value
    ]
    assert {row["vertex_id"] for row in marker_rows} == {5, 3}
