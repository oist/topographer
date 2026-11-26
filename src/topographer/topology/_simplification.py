"""Merge-tree simplification by ordinary-persistence leaf pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import rustworkx as rx

from topographer.topology._tree_construction import (
    _graph_edge_data,
    _graph_edge_endpoints,
    _graph_node_data,
    _graph_node_indices,
)

__all__ = ["build_induced_subtree", "prune_leaf_branches_by_ordinary_persistence"]


@dataclass(frozen=True, slots=True)
class _PrunedTree:
    """Pruned tree graph and statistics returned to :meth:`MergeTree.simplify`.

    Attributes
    ----------
    tree : rustworkx.PyDiGraph
        The pruned tree; edges keep the direction of the input tree.
    stats : dict
        Counts describing the operation (input and output sizes, pruned branches).
    kept_nodes : frozenset[int]
        Node indices of the input tree that survive the pruning.
    """

    tree: rx.PyDiGraph
    stats: dict[str, Any]
    kept_nodes: frozenset[int]


def prune_leaf_branches_by_ordinary_persistence(
    tree: Any,
    *,
    max_persistence: float,
    tree_kind: str,
) -> _PrunedTree:
    """Prune finite elder-rule branches at or below a persistence threshold.

    This operation uses ordinary H0 persistence pairs. The elder root lineage is never removed because it has
    no finite ordinary-persistence death.


    """

    if max_persistence < 0.0:
        raise ValueError("max_persistence must be non-negative.")

    from topographer.topology._persistence import build_tree_ordinary_persistence_pair_table

    pair_table = build_tree_ordinary_persistence_pair_table(tree, tree_kind=tree_kind)
    node_indices = _graph_node_indices(tree)
    edge_endpoints = [
        _graph_edge_endpoints(tree, edge_index) for edge_index in tree.edge_indices()
    ]
    successors = {index: [] for index in node_indices}
    for source, target in edge_endpoints:
        successors[source].append(target)
    for targets in successors.values():
        targets.sort()

    removed_nodes: set[int] = set()
    pruned_branches: list[dict[str, Any]] = []
    for row in pair_table.sort(
        ["persistence", "birth_tree_node_index"], descending=[False, False]
    ).to_dicts():
        persistence = float(row["persistence"])
        if persistence > max_persistence:
            continue
        birth = int(row["birth_tree_node_index"])
        death = int(row["death_tree_node_index"])
        path = _directed_path_to_node(successors, birth, death)
        removed_nodes.update(path[:-1])
        pruned_branches.append(
            {
                "birth_tree_node_index": birth,
                "death_tree_node_index": death,
                "persistence": persistence,
                "removed_node_count": len(path) - 1,
            }
        )

    kept_nodes = frozenset(node_indices) - removed_nodes
    pruned_tree = build_induced_subtree(tree, kept_nodes)

    return _PrunedTree(
        tree=pruned_tree,
        stats={
            "input_node_count": len(node_indices),
            "input_edge_count": len(edge_endpoints),
            "node_count": pruned_tree.num_nodes(),
            "edge_count": pruned_tree.num_edges(),
            "pruned_branch_count": len(pruned_branches),
            "pruned_node_count": len(removed_nodes),
            "pruned_branches": pruned_branches,
        },
        kept_nodes=kept_nodes,
    )


def build_induced_subtree(tree: Any, kept_nodes: Any) -> rx.PyDiGraph:
    """Return the subgraph of ``tree`` induced by ``kept_nodes``, with copied payloads.

    Surviving nodes are renumbered in ascending order of their index in ``tree`` and edges keep
    the edge order and direction of ``tree``. An edge survives when both endpoints survive.

    Examples
    --------
    >>> import rustworkx as rx
    >>> from topographer.topology._simplification import build_induced_subtree
    >>> tree = rx.PyDiGraph()
    >>> _ = tree.add_nodes_from([{"k": 0}, {"k": 1}, {"k": 2}])
    >>> _ = tree.add_edge(0, 2, {"w": 1.0})
    >>> _ = tree.add_edge(1, 2, {"w": 2.0})
    >>> sub = build_induced_subtree(tree, {1, 2})
    >>> sub.nodes(), sub.weighted_edge_list()
    ([{'k': 1}, {'k': 2}], WeightedEdgeList[(0, 1, {'w': 2.0})])
    """
    kept = set(kept_nodes)
    subtree = rx.PyDiGraph()
    node_mapping: dict[int, int] = {}
    for node_index in _graph_node_indices(tree):
        if node_index in kept:
            node_mapping[node_index] = subtree.add_node(dict(_graph_node_data(tree, node_index)))
    for edge_index in tree.edge_indices():
        source, target = _graph_edge_endpoints(tree, edge_index)
        if source in kept and target in kept:
            subtree.add_edge(
                node_mapping[source],
                node_mapping[target],
                dict(_graph_edge_data(tree, edge_index)),
            )
    return subtree


def _directed_path_to_node(successors: dict[int, list[int]], start: int, stop: int) -> list[int]:
    """Follow a merge-tree lineage from a birth to its paired saddle."""

    path = [start]
    current = start
    while current != stop:
        targets = successors[current]
        if len(targets) != 1:
            raise ValueError("Persistence pair does not follow one merge-tree lineage.")
        current = targets[0]
        path.append(current)
    return path
