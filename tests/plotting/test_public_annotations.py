"""Public annotation workflows preserve biological identity and scalar position."""

from __future__ import annotations

from uuid import UUID

import polars as pl
import pytest

from topographer import Graph, ScalarNetwork, ScalarNetworkTopology
from topographer.plotting import (
    TreeAnnotationLayer,
    TreePresentationConfig,
    make_tree_layout,
    plot_branch_spans,
    plot_persistence,
    plot_tree,
)


def _network(ids: list[object]) -> ScalarNetwork:
    # Preserve opaque identities instead of letting Polars infer list-valued IDs.
    id_series = (
        pl.Series("id", ids, dtype=pl.Object) if isinstance(ids[0], UUID) else pl.Series("id", ids)
    )
    vertices = pl.DataFrame({"id": id_series, "fitness": [1.0, 0.7, 0.4, 0.9, 0.0]})
    source = [ids[i] for i in (0, 1, 2, 3)]
    target = [ids[i] for i in (1, 2, 4, 2)]
    if isinstance(ids[0], UUID):
        edges = pl.DataFrame(
            {
                "source": pl.Series(source, dtype=pl.Object),
                "target": pl.Series(target, dtype=pl.Object),
            }
        )
    else:
        edges = pl.DataFrame({"source": source, "target": target})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness",),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def _inline_rows(spec: object) -> list[dict]:
    if isinstance(spec, dict):
        rows = spec.get("data", {}).get("values", [])
        return list(rows) + [row for value in spec.values() for row in _inline_rows(value)]
    if isinstance(spec, list):
        return [row for value in spec for row in _inline_rows(value)]
    return []


@pytest.mark.parametrize("ids", [[10, 20, 30, 40, 50], [UUID(int=i) for i in range(1, 6)]])
def test_tree_annotations_resolve_regular_public_identity_at_exact_fitness(
    ids: list[object],
) -> None:
    network = _network(ids)
    tree = ScalarNetworkTopology(network).split_tree()
    layout = make_tree_layout(tree, network=network)
    assert layout.scalar_attr == "fitness"
    assert set(layout.vertex_table["vertex_id"].to_list()) == set(ids)
    annotations = pl.DataFrame(
        {
            "vertex_id": pl.Series([ids[1]], dtype=pl.Object)
            if isinstance(ids[1], UUID)
            else [ids[1]],
            "annotation_note": ["regular sequence"],
        }
    )
    chart = plot_tree(
        layout,
        presentation=TreePresentationConfig(
            mode="annotation",
            annotation_layers=(
                TreeAnnotationLayer(annotations, tooltip_attrs=("annotation_note",)),
            ),
        ),
    )
    rows = [
        row
        for row in _inline_rows(chart.to_dict())
        if row.get("annotation_note") == "regular sequence"
    ]
    assert len(rows) == 1
    assert rows[0]["y"] == 0.7
    assert rows[0]["is_critical"] is False


def test_tree_annotations_reject_unknown_and_duplicate_identities() -> None:
    network = _network(["a", "regular", "saddle", "b", "root"])
    layout = make_tree_layout(ScalarNetworkTopology(network).split_tree(), network=network)
    with pytest.raises(KeyError, match="Unknown annotation"):
        plot_tree(layout, marker_table=pl.DataFrame({"vertex_id": ["absent"]}))
    with pytest.raises(ValueError, match="unique"):
        plot_tree(layout, marker_table=pl.DataFrame({"vertex_id": ["regular", "regular"]}))


def test_summary_annotations_preserve_extremum_and_saddle_scopes() -> None:
    annotations = pl.DataFrame(
        {"vertex_id": ["peak", "saddle"], "annotation_note": ["peak stop", "saddle stop"]}
    )
    pairs = pl.DataFrame(
        {
            "birth_scalar": [0.9],
            "death_scalar": [0.2],
            "birth_vertex_id": ["peak"],
            "saddle_vertex_id": ["saddle"],
        }
    )
    rows = [
        row
        for row in _inline_rows(
            plot_persistence(
                pairs, annotation_table=annotations, color_by=None, tooltip_attrs=()
            ).to_dict()
        )
        if "annotation_note" in row
    ]
    assert [(row["annotation_note"], row["x"], row["y"]) for row in rows] == [
        ("peak stop", 0.2, 0.9)
    ]
    spans = pl.DataFrame(
        {
            "source_scalar": [0.9],
            "target_scalar": [0.2],
            "source_vertex_id": ["peak"],
            "target_vertex_id": ["saddle"],
            "source_event": ["maximum"],
        }
    )
    rows = [
        row
        for row in _inline_rows(
            plot_branch_spans(spans, annotation_table=annotations, tooltip_attrs=()).to_dict()
        )
        if "annotation_note" in row
    ]
    assert {(row["annotation_note"], row["x"], row["y"]) for row in rows} == {
        ("peak stop", 0.2, 0.9),
        ("saddle stop", 0.2, 0.2),
    }
