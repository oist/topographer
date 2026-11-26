"""Contracts for the canonical 4D hypercube tutorial assets and docs asset generation."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from topographer.examples._cases import CASES, export_case
from topographer.examples._docs import write_all_docs_assets


def test_hypercube_tutorial_artifacts(tmp_path: Path) -> None:
    case = CASES["hypercube"]
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

    split_persistence = pl.read_csv(
        paths["table_split_persistence"],
        schema_overrides={"birth_label": pl.String, "death_label": pl.String},
    )
    assert split_persistence.filter(pl.col("pair_kind") == "extended").height == 1
    assert split_persistence.filter(pl.col("pair_kind") == "ordinary").height == 3
    assert (
        split_persistence.filter(pl.col("birth_label") == "0111").item(0, "persistence")
        == 0.27941176470588236
    )

    join_persistence = pl.read_csv(
        paths["table_join_persistence"],
        schema_overrides={"birth_label": pl.String, "death_label": pl.String},
    )
    assert join_persistence.filter(pl.col("pair_kind") == "extended").height == 1
    assert join_persistence.filter(pl.col("pair_kind") == "ordinary").height == 1

    vertices = pl.read_csv(paths["table_vertices"])
    edges = pl.read_csv(paths["table_edges"])
    assert vertices.height == 16
    assert edges.height == 32
    cycle_count = edges.height - vertices.height + 1
    assert cycle_count == 17


@pytest.mark.slow
def test_write_all_docs_assets_generates_complete_galleries(tmp_path: Path) -> None:
    pytest.importorskip("marimo")
    assets = write_all_docs_assets(tmp_path)
    assert "simplification" in assets.chart_paths
    assert len(assets.cube_paths) >= 20
    assert len(assets.hypercube_paths) >= 20
