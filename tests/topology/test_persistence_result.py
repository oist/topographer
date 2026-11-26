"""Tests for PersistenceResult schema and basic 1-vertex / 1-edge persistence."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
from topographer.topology import PersistenceResult, ScalarNetworkTopology


def test_persistence_result_schema_and_immutability() -> None:
    pairs = pl.DataFrame(
        {
            "pair_id": [0],
            "homology_dimension": [0],
            "pair_kind": ["extended_positive"],
            "birth_simplex": [["v0"]],
            "death_simplex": [["v0"]],
            "birth_scalar": [1.0],
            "death_scalar": [1.0],
            "persistence": [0.0],
        },
        schema={
            "pair_id": pl.Int64,
            "homology_dimension": pl.Int64,
            "pair_kind": pl.String,
            "birth_simplex": pl.List(pl.String),
            "death_simplex": pl.List(pl.String),
            "birth_scalar": pl.Float64,
            "death_scalar": pl.Float64,
            "persistence": pl.Float64,
        },
    )
    res = PersistenceResult(pairs=pairs)

    assert res.pairs.shape == (1, 8)

    # Immutability check
    with pytest.raises((AttributeError, TypeError)):
        res.pairs = None  # type: ignore[misc]


def test_one_vertex_persistence() -> None:
    vertices = pl.DataFrame({"id": ["v0"], "val": [42.0]})
    edges = pl.DataFrame({"src": [], "dst": []}, schema={"src": pl.String, "dst": pl.String})
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
    assert len(comp_res.pairs) == 1

    row = comp_res.pairs.to_dicts()[0]
    assert row["pair_id"] == 0
    assert row["homology_dimension"] == 0
    assert row["pair_kind"] == "extended_positive"
    assert tuple(row["birth_simplex"]) == ("v0",)
    assert tuple(row["death_simplex"]) == ("v0",)
    assert row["birth_scalar"] == 42.0
    assert row["death_scalar"] == 42.0
    assert row["persistence"] == 0.0


def test_one_edge_persistence() -> None:
    vertices = pl.DataFrame({"id": ["v0", "v1"], "val": [10.0, 20.0]})
    edges = pl.DataFrame({"src": ["v0"], "dst": ["v1"]})
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
    assert len(comp_res.pairs) == 1
    row = comp_res.pairs.to_dicts()[0]
    assert row["pair_id"] == 0
    assert row["homology_dimension"] == 0
    assert row["pair_kind"] == "extended_positive"
    assert tuple(row["birth_simplex"]) == ("v0",)
    assert tuple(row["death_simplex"]) == ("v1",)
    assert row["birth_scalar"] == 10.0
    assert row["death_scalar"] == 20.0
    assert row["persistence"] == 10.0
