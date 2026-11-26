"""Tests for the example workspaces built on ScalarNetworkTopology."""

from __future__ import annotations

from topographer.examples._presentation import build_canonical_persistence_table
from topographer.examples.cube import (
    CUBE_SIMPLIFICATION_THRESHOLD,
    build_cube_scalar_network,
    build_cube_topology,
)
from topographer.examples.hamming import build_hamming_scalar_network, build_hamming_topology


def test_example_networks_are_valid() -> None:
    cube = build_cube_scalar_network()
    assert cube.vertex_count == 8
    assert cube.edge_count == 12
    assert cube.scalar_attr == "height"

    hypercube = build_hamming_scalar_network(4)
    assert hypercube.vertex_count == 16
    assert hypercube.edge_count == 32
    assert hypercube.scalar_attr == "scalar"


def test_cube_workspace_has_both_trees_and_component_persistence() -> None:
    topology = build_cube_topology()

    assert topology.join_tree().kind == "join"
    assert topology.split_tree().kind == "split"
    pairs = build_canonical_persistence_table(topology.component_persistence())
    assert pairs.height == 3  # component pairs only


def test_cube_simplification_of_workspace_trees() -> None:
    topology = build_cube_topology()
    threshold = CUBE_SIMPLIFICATION_THRESHOLD

    join = topology.join_tree()
    join_pairs = join.persistence().filter(
        (join.persistence()["pair_kind"] == "ordinary")
        & (join.persistence()["persistence"] <= threshold)
    )
    assert join_pairs.height == 0
    assert join.simplify(threshold).graph.num_nodes() == join.graph.num_nodes()

    split = topology.split_tree()
    split_pairs = split.persistence().filter(
        (split.persistence()["pair_kind"] == "ordinary")
        & (split.persistence()["persistence"] <= threshold)
    )
    assert split_pairs.height == 1
    assert split.simplify(threshold).graph.num_nodes() < split.graph.num_nodes()


def test_hypercube_workspace_trees_are_cached() -> None:
    topology = build_hamming_topology(4)
    assert topology.join_tree() is topology.join_tree()
    assert topology.split_tree().persistence().height == 4
