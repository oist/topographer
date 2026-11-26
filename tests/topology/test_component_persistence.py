"""Tests for component persistence derivation from join and split trees."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
from topographer.topology import PersistenceResult, ScalarNetworkTopology


def test_y_shaped_tree_component_persistence() -> None:
    # Root v0 (0.0), branch saddle v1 (0.5), leaves v2 (0.8) and v3 (0.9)
    vertices = pl.DataFrame({"id": ["v0", "v1", "v2", "v3"], "val": [0.0, 0.5, 0.8, 0.9]})
    edges = pl.DataFrame({"src": ["v0", "v1", "v1"], "dst": ["v1", "v2", "v3"]})
    g = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    sn = ScalarNetwork(g, scalar_attr="val")
    top = ScalarNetworkTopology(sn)

    comp_res = top.component_persistence()
    assert isinstance(comp_res, PersistenceResult)
    # Join tree has 0 ordinary pairs (only global).
    # Split tree has 1 ordinary pair (v2 dying at saddle v1, val 0.8 -> 0.5, persistence 0.3) -> becomes relative H1 pair!
    # Global pair: v0 (0.0) -> v3 (0.9) -> extended_positive H0.
    # Total: 2 rows.
    df = comp_res.pairs
    assert len(df) == 2

    # Verify rows
    relative_row = df.filter(pl.col("pair_kind") == "relative").to_dicts()[0]
    assert relative_row["homology_dimension"] == 1
    assert tuple(relative_row["birth_simplex"]) == ("v2",)
    assert tuple(relative_row["death_simplex"]) == ("v1",)
    assert relative_row["birth_scalar"] == 0.8
    assert relative_row["death_scalar"] == 0.5
    assert pytest.approx(relative_row["persistence"]) == 0.3

    global_row = df.filter(pl.col("pair_kind") == "extended_positive").to_dicts()[0]
    assert global_row["homology_dimension"] == 0
    assert tuple(global_row["birth_simplex"]) == ("v0",)
    assert tuple(global_row["death_simplex"]) == ("v3",)
    assert global_row["birth_scalar"] == 0.0
    assert global_row["death_scalar"] == 0.9
    assert pytest.approx(global_row["persistence"]) == 0.9


def test_tied_vertices_component_persistence() -> None:
    # 3 vertices with same scalar 1.0: v0 -- v1 -- v2
    vertices = pl.DataFrame({"id": ["v0", "v1", "v2"], "val": [1.0, 1.0, 1.0]})
    edges = pl.DataFrame({"src": ["v0", "v1"], "dst": ["v1", "v2"]})
    g = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    sn = ScalarNetwork(g, scalar_attr="val")
    top = ScalarNetworkTopology(sn)

    comp_res = top.component_persistence()
    assert isinstance(comp_res, PersistenceResult)
    # Global pair must exist and have persistence 0.0
    global_rows = comp_res.pairs.filter(pl.col("pair_kind") == "extended_positive")
    assert len(global_rows) == 1
    row = global_rows.to_dicts()[0]
    assert row["birth_scalar"] == 1.0
    assert row["death_scalar"] == 1.0
    assert row["persistence"] == 0.0
    # Original values preserved
    assert all(comp_res.pairs["birth_scalar"] == 1.0)
    assert all(comp_res.pairs["death_scalar"] == 1.0)


def test_global_pair_ties_follow_join_sweep() -> None:
    """With tied global extrema the global pair equals the join tree's extended row."""
    # Path e-a-d-b-c with f = 0.0, 1.0, 0.5, 1.0, 0.0. Native identity order is
    # a, b, c, d, e, so the first tied minimum swept is c and the last tied maximum is b.
    vertices = pl.DataFrame({"id": ["e", "a", "d", "b", "c"], "val": [0.0, 1.0, 0.5, 1.0, 0.0]})
    edges = pl.DataFrame({"src": ["e", "a", "d", "b"], "dst": ["a", "d", "b", "c"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    network = ScalarNetwork(graph, scalar_attr="val")
    topology = ScalarNetworkTopology(network)

    global_row = (
        topology.component_persistence()
        .pairs.filter(pl.col("pair_kind") == "extended_positive")
        .row(0, named=True)
    )
    assert tuple(global_row["birth_simplex"]) == ("c",)
    assert tuple(global_row["death_simplex"]) == ("b",)

    join_extended = (
        topology.join_tree()
        .persistence()
        .filter(pl.col("pair_kind") == "extended")
        .row(0, named=True)
    )
    vertex_ids = network.vertex_ids
    assert vertex_ids[join_extended["birth_vertex_index"]] == "c"
    assert vertex_ids[join_extended["death_vertex_index"]] == "b"
