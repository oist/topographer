"""Independent structural validation for merge-tree sweeps."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import rustworkx as rx

from topographer.topology._tree_construction import (
    MergeTreeResult,
    _select_root_node_index,
    build_join_tree,
    build_split_tree,
)

Edge = tuple[int, int]


@st.composite
def _scalar_graph_cases(
    draw: st.DrawFn,
) -> tuple[list[float], list[Edge], bool, bool]:
    """Generate bounded scalar graphs with ties, cycles, and optional index gaps."""

    node_count = draw(st.integers(min_value=1, max_value=7))
    values = draw(
        st.lists(
            st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]),
            min_size=node_count,
            max_size=node_count,
        )
    )
    possible_edges = list(
        (source, target)
        for source in range(node_count)
        for target in range(source + 1, node_count)
    )
    if possible_edges:
        edges = list(
            draw(
                st.sets(
                    st.sampled_from(possible_edges),
                    max_size=min(len(possible_edges), node_count * 2),
                )
            )
        )
    else:
        edges = []
    return values, sorted(edges), draw(st.booleans()), draw(st.booleans())


def _build_graph(
    values: list[float],
    edges: Iterable[Edge],
    *,
    directed: bool = False,
) -> rx.PyGraph | rx.PyDiGraph:
    """Build a small scalar graph without relying on package fixtures."""

    graph: rx.PyGraph | rx.PyDiGraph = rx.PyDiGraph() if directed else rx.PyGraph()
    for value in values:
        graph.add_node({"scalar": value})
    for source, target in edges:
        graph.add_edge(source, target, None)
    return graph


def _build_generated_graph(
    values: list[float],
    logical_edges: list[Edge],
    *,
    directed: bool,
    sparse_indices: bool,
    reverse_edges: bool = False,
    duplicate_edges: bool = False,
) -> rx.PyGraph | rx.PyDiGraph:
    """Build generated cases with optional removed nodes and parallel arcs."""

    graph: rx.PyGraph | rx.PyDiGraph = rx.PyDiGraph() if directed else rx.PyGraph()
    node_by_logical_index: dict[int, int] = {}
    removed_nodes: list[int] = []
    for logical_index, value in enumerate(values):
        node_by_logical_index[logical_index] = int(graph.add_node({"scalar": value}))
        if sparse_indices and logical_index < len(values) - 1:
            removed_nodes.append(int(graph.add_node({"scalar": value})))
    for node in removed_nodes:
        graph.remove_node(node)

    ordered_edges = list(reversed(logical_edges)) if reverse_edges else logical_edges
    for logical_source, logical_target in ordered_edges:
        source = node_by_logical_index[logical_source]
        target = node_by_logical_index[logical_target]
        if reverse_edges:
            source, target = target, source
        graph.add_edge(source, target, None)
        if duplicate_edges:
            graph.add_edge(source, target, None)
    return graph


def _component_partition(nodes: Iterable[int], edges: Iterable[Edge]) -> frozenset[frozenset[int]]:
    """Compute weak components from scratch using a plain depth-first search."""

    remaining = set(nodes)
    adjacency = {node: set() for node in remaining}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)

    components: set[frozenset[int]] = set()
    while remaining:
        stack = [min(remaining)]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            remaining.discard(node)
            stack.extend(adjacency[node] - component)
        components.add(frozenset(component))
    return frozenset(components)


def _graph_edges(graph: Any) -> list[Edge]:
    return [
        tuple(map(int, graph.get_edge_endpoints_by_index(edge_index)))
        for edge_index in graph.edge_indices()
    ]


def _tree_edges_by_vertex(result: MergeTreeResult) -> list[Edge]:
    return [
        (
            int(result.tree.get_node_data(source)["vertex_index"]),
            int(result.tree.get_node_data(target)["vertex_index"]),
        )
        for source, target in result.tree.edge_list()
    ]


def _assert_sweep_matches_oracle(
    graph: rx.PyGraph | rx.PyDiGraph,
    result: MergeTreeResult,
) -> None:
    """Check every filtration prefix against independent connectivity."""

    node_indices = [int(node) for node in graph.node_indices()]
    values = {node: float(graph.get_node_data(node)["scalar"]) for node in node_indices}
    input_edges = _graph_edges(graph)
    tree_edges = _tree_edges_by_vertex(result)
    ordered_nodes = sorted(
        node_indices,
        key=lambda node: (values[node] if result.ascending else -values[node], node),
    )

    assert result.tree.num_nodes() == len(node_indices)
    assert result.event_table.height == len(node_indices)
    assert list(result.event_table["vertex_index"]) == ordered_nodes
    assert rx.is_directed_acyclic_graph(result.tree)

    active: list[int] = []
    for row, node in zip(result.event_table.to_dicts(), ordered_nodes, strict=True):
        prior_components = _component_partition(active, input_edges)
        touched_components = {
            component
            for component in prior_components
            if any(
                (source == node and target in component)
                or (target == node and source in component)
                for source, target in input_edges
            )
        }
        active_neighbor_components = len(touched_components)
        if active_neighbor_components == 0:
            expected_event = "minimum" if result.ascending else "maximum"
        elif active_neighbor_components == 1:
            expected_event = "regular"
        else:
            expected_event = "join_saddle" if result.ascending else "split_saddle"

        assert row == {
            "vertex_index": node,
            "event": expected_event,
            "active_neighbor_components": active_neighbor_components,
        }
        active.append(node)
        assert _component_partition(active, input_edges) == _component_partition(
            active, tree_edges
        )

    component_count = len(_component_partition(node_indices, input_edges))
    assert result.tree.num_edges() == len(node_indices) - component_count
    assert len(_component_partition(node_indices, tree_edges)) == component_count

    for source, target in result.tree.edge_list():
        source_scalar = float(result.tree.get_node_data(source)["scalar"])
        target_scalar = float(result.tree.get_node_data(target)["scalar"])
        if result.ascending:
            assert source_scalar <= target_scalar
        else:
            assert source_scalar >= target_scalar


@pytest.mark.parametrize(
    ("values", "edges"),
    [
        ([0.5], []),
        ([0.0, 0.4, 1.0], [(0, 1), (1, 2)]),
        ([0.0, 0.2, 0.9, 0.1, 0.3], [(0, 1), (1, 2), (3, 4), (4, 2)]),
        ([0.0, 0.1, 0.2, 1.0], [(0, 3), (1, 3), (2, 3)]),
        ([0.0, 0.3, 0.4, 1.0], [(0, 1), (0, 2), (1, 3), (2, 3)]),
        ([0.0, 0.3, 0.1, 0.4], [(0, 1), (2, 3)]),
        ([0.0, 0.5], [(0, 1), (0, 1)]),
        ([0.0, 0.0, 0.0], [(0, 1), (1, 2)]),
        ([0.0, 0.0, 0.0, 0.0], [(0, 3), (1, 3), (2, 3)]),
    ],
    ids=[
        "singleton",
        "monotone-path",
        "y-join",
        "multiway-saddle",
        "diamond-cycle",
        "disconnected-paths",
        "parallel-edges",
        "constant-path",
        "tied-plateau",
    ],
)
def test_canonical_graph_sweeps_match_brute_force_connectivity(
    values: list[float], edges: list[Edge]
) -> None:
    graph = _build_graph(values, edges)
    if len(_component_partition(graph.node_indices(), edges)) != 1:
        with pytest.raises(ValueError, match="weakly connected"):
            build_join_tree(graph, scalar_attr="scalar")
        return

    _assert_sweep_matches_oracle(graph, build_join_tree(graph, scalar_attr="scalar"))
    _assert_sweep_matches_oracle(graph, build_split_tree(graph, scalar_attr="scalar"))


def test_y_split_has_hand_verified_critical_events() -> None:
    graph = _build_graph(
        [1.0, 0.8, 0.1, 0.9, 0.7],
        [(0, 1), (1, 2), (3, 4), (4, 2)],
    )

    result = build_split_tree(graph, scalar_attr="scalar")

    assert result.event_table.to_dicts() == [
        {"vertex_index": 0, "event": "maximum", "active_neighbor_components": 0},
        {"vertex_index": 3, "event": "maximum", "active_neighbor_components": 0},
        {"vertex_index": 1, "event": "regular", "active_neighbor_components": 1},
        {"vertex_index": 4, "event": "regular", "active_neighbor_components": 1},
        {"vertex_index": 2, "event": "split_saddle", "active_neighbor_components": 2},
    ]


def test_exact_ties_use_node_index_as_symbolic_perturbation() -> None:
    graph = _build_graph(
        [0.0, 0.0, 0.0, 0.0],
        [(0, 3), (1, 3), (2, 3)],
    )

    result = build_join_tree(graph, scalar_attr="scalar")

    assert result.event_table["vertex_index"].to_list() == [0, 1, 2, 3]
    assert result.event_table["event"].to_list() == [
        "minimum",
        "minimum",
        "minimum",
        "join_saddle",
    ]
    assert all(
        float(result.tree.get_node_data(node)["scalar"]) == 0.0
        for node in result.tree.node_indices()
    )


@given(_scalar_graph_cases())
@settings(max_examples=150, deadline=None)
def test_generated_join_and_split_sweeps_match_brute_force_connectivity(
    case: tuple[list[float], list[Edge], bool, bool],
) -> None:
    values, edges, directed, sparse_indices = case
    graph = _build_generated_graph(
        values,
        edges,
        directed=directed,
        sparse_indices=sparse_indices,
    )
    if len(_component_partition(graph.node_indices(), _graph_edges(graph))) != 1:
        return

    _assert_sweep_matches_oracle(graph, build_join_tree(graph, scalar_attr="scalar"))
    _assert_sweep_matches_oracle(graph, build_split_tree(graph, scalar_attr="scalar"))


@given(_scalar_graph_cases())
@settings(max_examples=100, deadline=None)
def test_join_split_scalar_negation_duality(
    case: tuple[list[float], list[Edge], bool, bool],
) -> None:
    values, edges, directed, sparse_indices = case
    graph = _build_generated_graph(
        values,
        edges,
        directed=directed,
        sparse_indices=sparse_indices,
    )
    mirrored_graph = _build_generated_graph(
        [-value for value in values],
        edges,
        directed=directed,
        sparse_indices=sparse_indices,
    )
    if len(_component_partition(graph.node_indices(), _graph_edges(graph))) != 1:
        return

    join_result = build_join_tree(graph, scalar_attr="scalar")
    mirrored_split_result = build_split_tree(mirrored_graph, scalar_attr="scalar")
    mirrored_events = {
        "minimum": "maximum",
        "regular": "regular",
        "join_saddle": "split_saddle",
    }

    assert (
        join_result.event_table["vertex_index"].to_list()
        == mirrored_split_result.event_table["vertex_index"].to_list()
    )
    assert [
        mirrored_events[event] for event in join_result.event_table["event"].to_list()
    ] == mirrored_split_result.event_table["event"].to_list()
    assert sorted(_tree_edges_by_vertex(join_result)) == sorted(
        _tree_edges_by_vertex(mirrored_split_result)
    )


@given(_scalar_graph_cases())
@settings(max_examples=100, deadline=None)
def test_edge_order_orientation_and_multiplicity_do_not_change_topology(
    case: tuple[list[float], list[Edge], bool, bool],
) -> None:
    values, edges, _directed, sparse_indices = case
    forward_graph = _build_generated_graph(
        values,
        edges,
        directed=False,
        sparse_indices=sparse_indices,
    )
    reversed_multigraph = _build_generated_graph(
        values,
        edges,
        directed=True,
        sparse_indices=sparse_indices,
        reverse_edges=True,
        duplicate_edges=True,
    )
    if len(_component_partition(forward_graph.node_indices(), _graph_edges(forward_graph))) != 1:
        return

    for builder in (build_join_tree, build_split_tree):
        forward_result = builder(forward_graph, scalar_attr="scalar")
        reversed_result = builder(reversed_multigraph, scalar_attr="scalar")
        assert forward_result.event_table.to_dicts() == reversed_result.event_table.to_dicts()
        assert sorted(_tree_edges_by_vertex(forward_result)) == sorted(
            _tree_edges_by_vertex(reversed_result)
        )


_TIED_EXTREMA_VALUES = [1.0, 0.5, 1.0, 0.0, 0.0]
_TIED_EXTREMA_EDGES = [(0, 1), (1, 2), (3, 1), (4, 1)]


@pytest.mark.parametrize(
    ("build_tree", "root_kind"),
    [(build_join_tree, "maximum"), (build_split_tree, "minimum")],
)
def test_select_tree_root_agrees_with_sweep_root_for_tied_global_extrema(
    build_tree: Any, root_kind: str
) -> None:
    graph = _build_graph(_TIED_EXTREMA_VALUES, _TIED_EXTREMA_EDGES)
    result = build_tree(graph, scalar_attr="scalar")

    selected_root = _select_root_node_index(result.tree, "scalar", root_kind=root_kind)

    selected_vertex = result.tree.get_node_data(selected_root)["vertex_index"]
    sweep_vertex = result.tree.get_node_data(result.root_node_index)["vertex_index"]
    assert selected_vertex == sweep_vertex
