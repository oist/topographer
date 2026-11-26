"""Contracts for the canonical 3D cube tutorial assets."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from topographer.examples._cases import CASES, export_case


def test_cube_tutorial_artifacts(tmp_path: Path) -> None:
    case = CASES["cube"]
    paths = export_case(case, case.build_topology(), tmp_path)

    expected_tables = {
        "table_vertices",
        "table_edges",
        "table_summary",
        "table_persistence",
        "table_join_events",
        "table_join_persistence",
        "table_join_qualifying",
        "table_split_events",
        "table_split_persistence",
        "table_split_qualifying",
    }
    for tbl in expected_tables:
        assert tbl in paths
        assert paths[tbl].is_file()

    # Verify mathematical properties on exported data
    split_persistence = pl.read_csv(paths["table_split_persistence"])
    assert split_persistence.filter(pl.col("pair_kind") == "extended").height == 1
    assert split_persistence.filter(pl.col("pair_kind") == "ordinary").height == 1

    join_persistence = pl.read_csv(paths["table_join_persistence"])
    assert join_persistence.filter(pl.col("pair_kind") == "extended").height == 1
    assert join_persistence.filter(pl.col("pair_kind") == "ordinary").height == 1

    vertices = pl.read_csv(paths["table_vertices"])
    edges = pl.read_csv(paths["table_edges"])
    assert vertices.height == 8
    assert edges.height == 12
    # Cycle rank: 12 - 8 + 1 = 5
    cycle_count = edges.height - vertices.height + 1
    assert cycle_count == 5
