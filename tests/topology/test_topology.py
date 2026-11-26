"""Tests for ScalarNetworkTopology, MergeTree, and MergeTreeMapping."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
from topographer.topology import MergeTree, MergeTreeMapping, ScalarNetworkTopology


def _make_sample_network() -> ScalarNetwork:
    # 0 (0.0) -- 1 (0.2) -- 2 (0.9) -- 4 (0.3) -- 3 (0.1)
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2", "v3", "v4"],
            "val": [0.0, 0.2, 0.9, 0.1, 0.3],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1", "v3", "v4"],
            "dst": ["v1", "v2", "v4", "v2"],
        }
    )
    g = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    return ScalarNetwork(g, scalar_attr="val")


def test_scalar_network_topology_lazy_and_cached() -> None:
    sn = _make_sample_network()
    top = ScalarNetworkTopology(sn)

    assert top._join_tree is None
    assert top._split_tree is None

    join = top.join_tree()
    assert isinstance(join, MergeTree)
    assert join.kind == "join"
    # Repeated call returns same object
    assert top.join_tree() is join

    split = top.split_tree()
    assert isinstance(split, MergeTree)
    assert split.kind == "split"
    assert top.split_tree() is split


def test_merge_tree_spans_and_persistence() -> None:
    sn = _make_sample_network()
    top = ScalarNetworkTopology(sn)
    join = top.join_tree()

    spans = join.spans()
    assert not spans.is_empty()
    pers = join.persistence()
    assert not pers.is_empty()

    mapping = join.mapping
    assert isinstance(mapping, MergeTreeMapping)
    assert not mapping.critical_vertices.is_empty()


def test_merge_tree_simplification() -> None:
    sn = _make_sample_network()
    top = ScalarNetworkTopology(sn)
    join = top.join_tree()

    simplified = join.simplify(persistence_threshold=0.15)
    assert simplified.original is join
    assert join.simplifications[0.15] is simplified

    # Cannot simplify already simplified tree
    with pytest.raises(ValueError, match="already simplified"):
        simplified.simplify(0.1)


def test_merge_tree_augmented_graph() -> None:
    sn = _make_sample_network()
    top = ScalarNetworkTopology(sn)

    join = top.join_tree()
    split = top.split_tree()

    # 1. augmented_graph contains every vertex from the scalar network (5 vertices)
    assert join.augmented_graph.num_nodes() == 5
    assert split.augmented_graph.num_nodes() == 5

    # 2. graph remains reduced
    assert join.graph.num_nodes() < join.augmented_graph.num_nodes()
    assert split.graph.num_nodes() < split.augmented_graph.num_nodes()

    # 3. Defensive copy: mutating returned augmented_graph does not affect MergeTree
    aug = join.augmented_graph
    aug.add_node({"dummy": 123})
    assert join.augmented_graph.num_nodes() == 5

    # 4. Cached calls retain identical augmented and reduced content
    assert top.join_tree().augmented_graph.num_nodes() == 5
    assert top.split_tree().augmented_graph.num_nodes() == 5
