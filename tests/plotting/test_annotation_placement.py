"""Generic annotation placement keeps scalar coordinates exact."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.plotting._tree import _place_overlapping_annotations


def test_annotation_overlap_spreads_only_horizontal_coordinates() -> None:
    table = pl.DataFrame(
        {
            "vertex_id": ["a", "b", "c"],
            "is_critical": [False, False, True],
            "x": [0.5, 0.5, 0.5],
            "y": [0.4, 0.4, 0.4],
        }
    )

    placed = _place_overlapping_annotations(table, 0.1)

    assert placed.get_column("x").to_list() == [0.5, 0.6, 0.5]
    assert placed.get_column("y").to_list() == [0.4, 0.4, 0.4]
    assert placed.get_column("vertex_id").to_list() == ["a", "b", "c"]


def test_persistence_annotations_match_public_vertex_ids() -> None:
    from topographer.plotting import plot_persistence

    summary = pl.DataFrame(
        {
            "vertex_id": ["peak-a", "peak-b"],
            "birth_vertex_id": ["peak-a", "peak-b"],
            "birth_scalar": [0.2, 0.4],
            "death_scalar": [0.8, 0.9],
        }
    )
    annotations = pl.DataFrame({"vertex_id": ["peak-b"], "label": ["selected"]})

    spec = plot_persistence(
        summary,
        color_by=None,
        annotation_table=annotations,
        tooltip_attrs=(),
        include_config=False,
    ).to_dict()

    marker_layer = spec["layer"][-1]
    assert marker_layer["data"]["values"] == [
        {"vertex_id": "peak-b", "label": "selected", "x": 0.9, "y": 0.4}
    ]


@pytest.mark.parametrize("plotter_name", ["ordinary", "extended"])
def test_specialized_persistence_plotters_forward_generic_annotations(plotter_name: str) -> None:
    from topographer.plotting import (
        TreeMarkerConfig,
        plot_extended_persistence,
        plot_ordinary_persistence,
    )

    table = pl.DataFrame(
        {
            "birth_scalar": [0.2],
            "death_scalar": [0.8],
            "birth_event": ["maximum"],
            "extremum_vertex_id": ["peak"],
            "persistence": [0.6],
        }
    )
    annotations = pl.DataFrame({"vertex_id": ["peak"], "label": ["keep"]})
    plotter = (
        plot_ordinary_persistence if plotter_name == "ordinary" else plot_extended_persistence
    )

    spec = plotter(
        table,
        tooltip_attrs=(),
        annotation_table=annotations,
        annotation_config=TreeMarkerConfig(shape="diamond", size=123),
        include_config=False,
    ).to_dict()

    marker = spec["layer"][-1]
    assert marker["data"]["values"] == [{"vertex_id": "peak", "label": "keep", "x": 0.8, "y": 0.2}]
    assert marker["mark"]["shape"] == "diamond"
    assert marker["mark"]["size"] == 123
