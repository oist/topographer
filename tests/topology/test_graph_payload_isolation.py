"""Tests that graph accessors and constructors isolate payload dicts from callers."""

from __future__ import annotations

from typing import Any

import pytest
import rustworkx as rx

from topographer.graph import ScalarNetwork
from topographer.topology import MergeTree, ScalarNetworkTopology


def _join_tree(network: ScalarNetwork) -> MergeTree:
    return ScalarNetworkTopology(network).join_tree()


def _payloads(graph: rx.PyDiGraph | rx.PyGraph) -> tuple[list[Any], list[Any]]:
    nodes = [dict(graph[idx]) for idx in graph.node_indices()]
    edges = [dict(graph.get_edge_data_by_index(idx)) for idx in graph.edge_indices()]
    # Edge order from separate tree builds is not guaranteed, so compare order-free.
    return nodes, sorted(edges, key=repr)


def _rebuild(graph: rx.PyDiGraph) -> rx.PyDiGraph:
    """Rebuild a directed graph with fresh payload dicts (indices assume no holes)."""
    rebuilt = rx.PyDiGraph()
    for idx in graph.node_indices():
        rebuilt.add_node(dict(graph[idx]))
    for u, v, data in graph.weighted_edge_list():
        rebuilt.add_edge(u, v, dict(data))
    return rebuilt


def _scribble(graph: rx.PyDiGraph | rx.PyGraph) -> None:
    """Mutate every node and edge payload dict in place."""
    for idx in graph.node_indices():
        graph[idx]["scalar"] = -99.0
        graph[idx]["injected"] = True
    for idx in graph.edge_indices():
        graph.get_edge_data_by_index(idx)["weight"] = -99.0
        graph.get_edge_data_by_index(idx)["injected"] = True


@pytest.mark.parametrize("accessor", ["graph", "augmented_graph"])
def test_merge_tree_graph_accessors_isolate_edge_payloads(
    single_saddle_diamond: ScalarNetwork, accessor: str
) -> None:
    tree = _join_tree(single_saddle_diamond)
    before = _payloads(getattr(tree, accessor))

    leaked = getattr(tree, accessor)
    for idx in leaked.edge_indices():
        leaked.get_edge_data_by_index(idx)["weight"] = -99.0

    assert _payloads(getattr(tree, accessor)) == before


@pytest.mark.parametrize("accessor", ["graph", "augmented_graph"])
def test_merge_tree_graph_accessors_isolate_node_payloads(
    single_saddle_diamond: ScalarNetwork, accessor: str
) -> None:
    tree = _join_tree(single_saddle_diamond)
    before = _payloads(getattr(tree, accessor))

    leaked = getattr(tree, accessor)
    for idx in leaked.node_indices():
        leaked[idx]["scalar"] = -99.0

    assert _payloads(getattr(tree, accessor)) == before


def test_merge_tree_constructor_isolates_input_graph(
    single_saddle_diamond: ScalarNetwork,
) -> None:
    source = _join_tree(single_saddle_diamond)
    graph = _rebuild(source.graph)
    augmented = _rebuild(source.augmented_graph)

    with pytest.warns(DeprecationWarning, match="constructor arguments"):
        tree = MergeTree(
            kind=source.kind,
            graph=graph,
            augmented_graph=augmented,
            event_table=source.events,
            mapping=source.mapping,
            scalar_attr=source.scalar_attr,
        )
    before_reduced = _payloads(tree.graph)
    before_augmented = _payloads(tree.augmented_graph)
    before_spans = tree.spans()
    before_persistence = tree.persistence()

    _scribble(graph)
    _scribble(augmented)

    after_reduced = tree.graph
    after_augmented = tree.augmented_graph
    assert _payloads(after_reduced) == before_reduced
    assert _payloads(after_augmented) == before_augmented
    assert tree.spans().equals(before_spans)
    assert tree.persistence().equals(before_persistence)


def test_legacy_constructor_provenance_remains_supported(
    single_saddle_diamond: ScalarNetwork,
) -> None:
    source = ScalarNetworkTopology(single_saddle_diamond).join_tree()
    with pytest.warns(DeprecationWarning, match="constructor arguments"):
        variant = MergeTree(
            kind=source.kind,
            graph=source.graph,
            augmented_graph=source.augmented_graph,
            event_table=source.events,
            mapping=source.mapping,
            scalar_attr=source.scalar_attr,
            original=source,
            persistence_threshold=0.1,
        )
    assert variant.original is source
    with pytest.raises(ValueError, match="already simplified"):
        variant.simplify(0.1)


def test_merge_tree_results_unchanged_after_accessor_mutation(
    single_saddle_diamond: ScalarNetwork,
) -> None:
    pristine = _join_tree(single_saddle_diamond)
    expected_spans = pristine.spans()
    expected_persistence = pristine.persistence()

    # Mutate before any result is cached so corrupted payloads would feed the computation.
    tree = _join_tree(single_saddle_diamond)
    _scribble(tree.graph)
    _scribble(tree.augmented_graph)

    assert tree.spans().equals(expected_spans)
    assert tree.persistence().equals(expected_persistence)


def test_simplified_tree_unchanged_after_accessor_mutation(
    bifurcating_two_peaks: ScalarNetwork,
) -> None:
    expected = _join_tree(bifurcating_two_peaks).simplify(0.35)
    expected_spans = expected.spans()
    expected_augmented = _payloads(expected.augmented_graph)

    tree = _join_tree(bifurcating_two_peaks)
    _scribble(tree.graph)
    _scribble(tree.augmented_graph)
    simplified = tree.simplify(0.35)
    _scribble(simplified.graph)
    _scribble(simplified.augmented_graph)

    assert simplified.spans().equals(expected_spans)
    # The variant shares its augmented graph with the original; both stay pristine.
    assert _payloads(simplified.augmented_graph) == expected_augmented
    assert _payloads(tree.augmented_graph) == expected_augmented
