"""Compatibility checks for filtration-based merge tree construction."""

import random

from hypothesis import given, settings
from hypothesis import strategies as st
import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology
from topographer.topology._merge_tree import construct_oriented_merge_tree


@pytest.mark.parametrize(("filtration", "kind"), [("sublevel", "join"), ("superlevel", "split")])
@settings(max_examples=30, derandomize=True, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_filtration_matches_existing_construction_on_random_graphs(filtration, kind, seed):
    rng = random.Random(seed)
    size = rng.randint(2, 9)
    values = [rng.randrange(5) / 4 for _ in range(size)]
    edges = [(i, rng.randrange(i)) for i in range(1, size)]
    edges += [
        (i, j) for i in range(size) for j in range(i) if (i, j) not in edges and rng.random() < 0.2
    ]
    graph = Graph.from_tables(
        pl.DataFrame({"id": list(range(size)), "scalar": values}),
        pl.DataFrame({"src": [a for a, _ in edges], "dst": [b for _, b in edges]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    network = ScalarNetwork(graph, scalar_attr="scalar")
    expected = construct_oriented_merge_tree(network, kind=kind)
    topology = ScalarNetworkTopology(network)
    actual = topology.compute_merge_tree(filtration, tie_break="vertex_index")
    assert (topology.join_tree() if kind == "join" else topology.split_tree()) is actual
    assert actual.kind == expected.kind
    assert actual.events.equals(expected.events)
    assert actual.persistence().equals(expected.persistence())
    for actual_graph, expected_graph in (
        (actual.graph, expected.graph),
        (actual.augmented_graph, expected.augmented_graph),
    ):
        assert list(actual_graph.nodes()) == list(expected_graph.nodes())
        assert list(actual_graph.weighted_edge_list()) == list(expected_graph.weighted_edge_list())


@pytest.mark.parametrize(
    ("filtration", "wrapper"), [("sublevel", "join_tree"), ("superlevel", "split_tree")]
)
@pytest.mark.parametrize("wrapper_first", [True, False])
def test_filtration_and_wrapper_share_cache(filtration, wrapper, wrapper_first):
    from topographer.examples import build_cube_scalar_network

    topology = ScalarNetworkTopology(build_cube_scalar_network())
    calls = (getattr(topology, wrapper), lambda: topology.compute_merge_tree(filtration))
    first, second = calls if wrapper_first else calls[::-1]
    assert first() is second()


def test_filtration_rejects_unknown_options():
    from topographer.examples import build_cube_scalar_network

    topology = ScalarNetworkTopology(build_cube_scalar_network())
    with pytest.raises(ValueError, match="filtration"):
        topology.compute_merge_tree("unknown")


@pytest.mark.parametrize("filtration", ["sublevel", "superlevel"])
def test_filtration_rejects_unknown_tie_break(filtration):
    from topographer.examples import build_cube_scalar_network

    topology = ScalarNetworkTopology(build_cube_scalar_network())
    with pytest.raises(ValueError, match="tie_break"):
        topology.compute_merge_tree(filtration, tie_break="unknown")


def test_join_and_split_trees_are_filtration_trees():
    from topographer.examples import build_cube_scalar_network

    topology = ScalarNetworkTopology(build_cube_scalar_network())
    assert topology.join_tree() is topology.compute_merge_tree("sublevel")
    assert topology.split_tree() is topology.compute_merge_tree("superlevel")
