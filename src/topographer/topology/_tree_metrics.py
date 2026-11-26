"""Sequence-free vertex counts and Horton-Strahler orders on rooted merge trees."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import operator
from typing import Any, TypeVar

import polars as pl

from topographer.topology._merge_tree import MergeTree

T = TypeVar("T")

_BRANCH_SCHEMA = {
    "source_tree_node_index": pl.Int64,
    "target_tree_node_index": pl.Int64,
    "branch_vertex_count": pl.Int64,
    "lineage_vertex_count": pl.Int64,
    "horton_strahler_order": pl.Int64,
}
_CRITICAL_SCHEMA = {
    "tree_node_index": pl.Int64,
    "lineage_vertex_count": pl.Int64,
    "horton_strahler_order": pl.Int64,
}


@dataclass(frozen=True)
class TreeMetrics:
    """Vertex counts and Horton-Strahler orders for a merge tree.

    ``branch_table`` has one row per tree branch. ``source_tree_node_index`` is
    the leaf-side vertex where the branch starts in scan direction and
    ``target_tree_node_index`` is the saddle (or root) where it ends. This is
    the reverse of the orientation of ``MergeTree.graph`` edges, which point
    from root to leaves.

    ``critical_node_table`` describes the component after each tree node has
    been added; the root row holds the whole-tree total.

    ``persistence_table`` is ``MergeTree.persistence()`` plus
    ``lineage_vertex_count`` and ``horton_strahler_order``.

    Examples
    --------
    >>> from topographer.topology import TreeMetrics
    >>> TreeMetrics.__name__
    'TreeMetrics'
    """

    branch_table: pl.DataFrame
    critical_node_table: pl.DataFrame
    persistence_table: pl.DataFrame


@dataclass(frozen=True)
class _RootedTree:
    """Rooted tree with deterministic children and a child-before-parent order."""

    root: int
    children: dict[int, list[int]]
    parent: dict[int, int | None]
    order: list[int]
    regular_count: dict[int, int]


@dataclass(frozen=True)
class _FoldResult:
    """Per-node results of :func:`_fold_tree`.

    ``branch_local`` and ``branch_lineage`` are keyed by the branch's start
    vertex (the child side of the rooted edge). ``post_node`` is the aggregate
    of the component after the node itself has been added.
    """

    branch_local: dict[int, Any]
    branch_lineage: dict[int, Any]
    post_node: dict[int, Any]
    horton_strahler: dict[int, int]


def _rooted_tree(
    root: int, children: dict[int, list[int]], regular_count: dict[int, int]
) -> _RootedTree:
    """Build a rooted tree, sorting children and computing a postorder."""

    sorted_children = {node: sorted(kids) for node, kids in children.items()}
    parent: dict[int, int | None] = {root: None}
    order: list[int] = []
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            order.append(node)
            continue
        stack.append((node, True))
        for child in reversed(sorted_children.get(node, [])):
            parent[child] = node
            stack.append((child, False))
    return _RootedTree(root, sorted_children, parent, order, dict(regular_count))


def _rooted_merge_tree(tree: MergeTree) -> _RootedTree:
    """Root a ``MergeTree`` using its root-to-leaf edges and ``regular_count`` payloads."""

    graph = tree.graph
    nodes = [int(node) for node in graph.node_indices()]
    roots = [node for node in nodes if graph.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Merge tree must have exactly one root, found {len(roots)}.")
    children: dict[int, list[int]] = {node: [] for node in nodes}
    regular_count: dict[int, int] = {}
    for parent, child, payload in graph.weighted_edge_list():
        children[int(parent)].append(int(child))
        regular_count[int(child)] = int(dict(payload).get("regular_count", 0))
    rooted = _rooted_tree(roots[0], children, regular_count)
    if len(rooted.order) != len(nodes):
        raise ValueError("Merge tree must be connected and acyclic.")
    return rooted


def _fold_tree(
    rooted: _RootedTree,
    *,
    vertex_value: Callable[[int], T],
    regular_value: Callable[[int], T],
    combine: Callable[[T, T], T],
    progress_callback: Callable[[int, int], None] | None = None,
) -> _FoldResult:
    """Aggregate vertex values from the leaves to the root.

    ``vertex_value(node)`` is the value of one critical vertex and
    ``regular_value(child)`` the value of the regular vertices on the edge
    above ``child``. ``combine`` must not mutate its arguments.
    """

    branch_local: dict[int, T] = {}
    branch_lineage: dict[int, T] = {}
    post_node: dict[int, T] = {}
    horton_strahler: dict[int, int] = {}
    total = len(rooted.order)
    if progress_callback is not None:
        progress_callback(0, total)
    for position, node in enumerate(rooted.order, start=1):
        kids = rooted.children[node]
        aggregate = vertex_value(node)
        for child in kids:
            aggregate = combine(aggregate, branch_lineage[child])
        post_node[node] = aggregate
        orders = sorted((horton_strahler[child] for child in kids), reverse=True)
        if not orders:
            horton_strahler[node] = 1
        else:
            horton_strahler[node] = orders[0] + int(len(orders) > 1 and orders[0] == orders[1])
        if node != rooted.root:
            v = vertex_value(node)
            r = regular_value(node)
            branch_local[node] = combine(v, r)
            branch_lineage[node] = combine(aggregate, r)
        if progress_callback is not None:
            progress_callback(position, total)
    return _FoldResult(branch_local, branch_lineage, post_node, horton_strahler)


def _elder_key(tree_kind: str, scalar: float, vertex_index: int) -> tuple[float, int]:
    return (scalar, vertex_index) if tree_kind == "join" else (-scalar, vertex_index)


def _losing_children(
    rooted: _RootedTree,
    *,
    tree_kind: str,
    scalar: dict[int, float],
    vertex: dict[int, int],
) -> dict[int, int]:
    """Map each losing component's birth node to the child branch that carries it."""

    representative: dict[int, int] = {}
    losing: dict[int, int] = {}
    for node in rooted.order:
        kids = rooted.children[node]
        if not kids:
            representative[node] = node
            continue
        elder = min(
            kids,
            key=lambda child: _elder_key(
                tree_kind, scalar[representative[child]], vertex[representative[child]]
            ),
        )
        for child in kids:
            if child != elder:
                losing[representative[child]] = child
        representative[node] = representative[elder]
    return losing


def _pair_value(
    row: dict[str, Any],
    *,
    losing: dict[int, int],
    branch_lineage: dict[int, T],
    post_node: dict[int, T],
) -> T:
    """Return the component aggregate a persistence row refers to."""

    if row["pair_kind"] == "extended":
        return post_node[int(row["death_tree_node_index"])]
    return branch_lineage[losing[int(row["birth_tree_node_index"])]]


def build_tree_metrics(tree: MergeTree) -> TreeMetrics:
    """Compute branch and lineage vertex counts and Horton-Strahler orders.

    A branch counts the critical vertex it starts at (in scan direction) plus
    its regular vertices, and excludes the saddle it ends at, so a leaf branch
    counts one. The lineage count of a branch is its own count plus the lineage
    counts of the branches below its starting vertex. The root vertex is never
    on any branch; only the root row of ``critical_node_table`` holds the
    whole-tree total. Ordinary persistence rows carry the losing pre-saddle
    component and extended rows the root component.

    Counts follow each tree edge's ``regular_count`` as given. On trees
    returned by ``MergeTree.simplify`` the root total can therefore be below
    the number of vertices in the network.

    Parameters
    ----------
    tree
        Merge tree whose branches, critical nodes and persistence pairs are
        counted.

    Returns
    -------
    TreeMetrics
        Branch, critical-node and persistence tables with vertex counts and
        Horton-Strahler orders.

    Raises
    ------
    TypeError
        If ``tree`` is not a ``MergeTree``.
    ValueError
        If the tree has multiple roots or is disconnected.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.topology import ScalarNetworkTopology, build_tree_metrics
    >>> tree = ScalarNetworkTopology(build_cube_scalar_network()).split_tree()
    >>> build_tree_metrics(tree).branch_table.height > 0
    True
    """

    if not isinstance(tree, MergeTree):
        raise TypeError("tree must be a MergeTree.")
    rooted = _rooted_merge_tree(tree)
    fold = _fold_tree(
        rooted,
        vertex_value=lambda _node: 1,
        regular_value=rooted.regular_count.__getitem__,
        combine=operator.add,
    )
    graph = tree.graph
    scalar = {n: float(dict(graph.get_node_data(n))["scalar"]) for n in rooted.order}
    vertex = {n: int(dict(graph.get_node_data(n)).get("vertex_index", n)) for n in rooted.order}
    losing = _losing_children(rooted, tree_kind=tree.kind, scalar=scalar, vertex=vertex)

    branch_rows = [
        {
            "source_tree_node_index": node,
            "target_tree_node_index": rooted.parent[node],
            "branch_vertex_count": fold.branch_local[node],
            "lineage_vertex_count": fold.branch_lineage[node],
            "horton_strahler_order": fold.horton_strahler[node],
        }
        for node in sorted(fold.branch_local)
    ]
    critical_rows = [
        {
            "tree_node_index": node,
            "lineage_vertex_count": fold.post_node[node],
            "horton_strahler_order": fold.horton_strahler[node],
        }
        for node in sorted(fold.post_node)
    ]
    persistence = tree.persistence()
    pair_values = [
        _pair_value(
            row, losing=losing, branch_lineage=fold.branch_lineage, post_node=fold.post_node
        )
        for row in persistence.to_dicts()
    ]
    pair_orders = [
        fold.horton_strahler[
            int(row["death_tree_node_index"])
            if row["pair_kind"] == "extended"
            else losing[int(row["birth_tree_node_index"])]
        ]
        for row in persistence.to_dicts()
    ]
    persistence_table = persistence.with_columns(
        pl.Series("lineage_vertex_count", pair_values, dtype=pl.Int64),
        pl.Series("horton_strahler_order", pair_orders, dtype=pl.Int64),
    )
    return TreeMetrics(
        branch_table=pl.DataFrame(branch_rows, schema=_BRANCH_SCHEMA),
        critical_node_table=pl.DataFrame(critical_rows, schema=_CRITICAL_SCHEMA),
        persistence_table=persistence_table,
    )
