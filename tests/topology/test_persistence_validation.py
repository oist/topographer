"""Tests verifying that network persistence rejects non-simple graphs."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology


def test_self_loop_rejection_for_persistence() -> None:
    # Graph with self-loop: v0 -> v0
    v_df = pl.DataFrame({"id": ["v0", "v1"], "val": [0.0, 1.0]})
    e_df = pl.DataFrame({"src": ["v0", "v0"], "dst": ["v1", "v0"]})
    g = Graph.from_tables(
        v_df,
        e_df,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    # ScalarNetwork can be formed (general scalar-decorated graph)
    sn = ScalarNetwork(g, scalar_attr="val")
    top = ScalarNetworkTopology(sn)

    # Rejection happens upon requesting network persistence
    with pytest.raises(
        ValueError,
        match=r"^Network extended persistence requires a simple graph without self-loops\.$",
    ):
        top.component_persistence()

    with pytest.raises(ValueError, match="without self-loops"):
        top.persistence_diagram("sublevel")


def test_parallel_edges_multigraph_rejection_for_persistence() -> None:
    # Multigraph with parallel edges between v0 and v1
    v_df = pl.DataFrame({"id": ["v0", "v1"], "val": [0.0, 1.0]})
    e_df = pl.DataFrame(
        {
            "src": ["v0", "v0"],
            "dst": ["v1", "v1"],
            "eid": ["e1", "e2"],
        }
    )
    g = Graph.from_tables(
        v_df,
        e_df,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        edge_id_col="eid",
        vertex_attribute_cols=("val",),
        multigraph=True,
    )
    sn = ScalarNetwork(g, scalar_attr="val")
    top = ScalarNetworkTopology(sn)

    # Rejection happens upon requesting network persistence
    with pytest.raises(
        ValueError,
        match=r"^Network extended persistence requires a simple graph without parallel edges\.$",
    ):
        top.component_persistence()

    with pytest.raises(ValueError, match="without parallel edges"):
        top.persistence_diagram("superlevel")


def test_simple_graph_allowed_for_persistence() -> None:
    # Simple graph without self-loops or parallel edges
    v_df = pl.DataFrame({"id": ["v0", "v1"], "val": [0.0, 1.0]})
    e_df = pl.DataFrame({"src": ["v0"], "dst": ["v1"]})
    g = Graph.from_tables(
        v_df,
        e_df,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    sn = ScalarNetwork(g, scalar_attr="val")
    top = ScalarNetworkTopology(sn)

    comp_res = top.component_persistence()
    assert len(comp_res.pairs) == 1
