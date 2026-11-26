"""Reusable tree layout helpers for graphs with scalar vertex functions."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from topographer.topology import select_extremum_root


@dataclass(frozen=True, slots=True)
class TreeLayoutResult:
    """Container for an iplotx-ready rooted tree and its metadata."""

    tree: Any
    root_node_index: int
    node_data_by_index: dict[int, dict[str, Any]]
    node_data_by_name: dict[str, dict[str, Any]]
    scalar_by_name: dict[str, float]
    regular_count_by_edge_name: dict[tuple[str, str], float]
    scalar_min: float
    scalar_max: float
    root_kind: str


def _graph_node_indices(graph: Any) -> list[int]:
    """Return integer node indices from a supported graph object."""

    if hasattr(graph, "node_indices"):
        return [int(node_index) for node_index in graph.node_indices()]
    raise TypeError(f"Unsupported graph type: {type(graph).__name__}")


def _graph_node_data(graph: Any, node_index: int) -> dict[str, Any]:
    """Return node metadata as a plain dictionary."""

    if hasattr(graph, "get_node_data"):
        return dict(graph.get_node_data(node_index))
    raise TypeError(f"Unsupported graph type: {type(graph).__name__}")


def _graph_edge_list(graph: Any) -> list[tuple[int, int, dict[str, Any]]]:
    """Return graph edges together with any available edge payloads."""

    if hasattr(graph, "edge_indices") and hasattr(graph, "get_edge_endpoints_by_index"):
        edges: list[tuple[int, int, dict[str, Any]]] = []
        for edge_index in graph.edge_indices():
            source_index, target_index = graph.get_edge_endpoints_by_index(edge_index)
            edge_data = graph.get_edge_data_by_index(edge_index)
            edges.append((int(source_index), int(target_index), dict(edge_data or {})))
        return edges
    if hasattr(graph, "edge_list"):
        return [(int(source), int(target), {}) for source, target in graph.edge_list()]
    raise TypeError(f"Unsupported graph type: {type(graph).__name__}")


def _resolve_scalar_key(graph: Any, scalar_attr: str | None = None) -> str:
    """Resolve the scalar attribute key on a graph."""
    if scalar_attr is not None:
        return scalar_attr
    if hasattr(graph, "scalar_attr"):
        return str(graph.scalar_attr)
    node_indices = _graph_node_indices(graph)
    if node_indices:
        data = _graph_node_data(graph, node_indices[0])
        if "scalar" in data:
            return "scalar"
    return "scalar"


def _tree_scalar_bounds(graph: Any, scalar_attr: str | None = None) -> tuple[float, float]:
    """Return min and max scalar values on a graph."""
    scalar_key = _resolve_scalar_key(graph, scalar_attr)
    node_indices = _graph_node_indices(graph)
    if not node_indices:
        raise ValueError("Graph must contain at least one node.")
    scalars = [float(_graph_node_data(graph, idx)[scalar_key]) for idx in node_indices]
    return min(scalars), max(scalars)


def select_tree_root_node_index(
    graph: Any, scalar_attr: str | None = None, root_kind: str = "minimum"
) -> int:
    """Return the deterministic root node index of a scalar-annotated tree graph.

    Parameters
    ----------
    graph
        Tree graph whose nodes carry the scalar attribute.
    scalar_attr
        Node attribute holding the scalar value; resolved from the graph when omitted.
    root_kind
        ``"minimum"`` or ``"maximum"``: root at the lowest or highest scalar node.

    Returns
    -------
    int
        Index of the root node; ties resolve deterministically.

    Raises
    ------
    ValueError
        If the graph is empty or ``root_kind`` is not ``"minimum"`` or ``"maximum"``.

    Examples
    --------
    >>> import rustworkx as rx
    >>> from topographer.plotting import select_tree_root_node_index
    >>> tree = rx.PyDiGraph()
    >>> _ = tree.add_nodes_from([{"scalar": 0.0}, {"scalar": 1.0}])
    >>> select_tree_root_node_index(tree, "scalar")
    0
    >>> select_tree_root_node_index(tree, "scalar", "maximum")
    1
    """
    scalar_key = _resolve_scalar_key(graph, scalar_attr)
    node_indices = _graph_node_indices(graph)
    if not node_indices:
        raise ValueError("Graph must contain at least one node.")
    scalars = {idx: float(_graph_node_data(graph, idx)[scalar_key]) for idx in node_indices}
    return select_extremum_root(node_indices, scalars, root_kind)


def _tree_children(
    node_indices: Sequence[int],
    edge_records: Sequence[tuple[int, int, dict[str, Any]]],
) -> dict[int, list[int]]:
    """Build an undirected adjacency map for a tree-like graph."""

    adjacency: dict[int, set[int]] = {node_index: set() for node_index in node_indices}
    for source, target, _edge_data in edge_records:
        adjacency[source].add(target)
        adjacency[target].add(source)
    return {node_index: sorted(children) for node_index, children in adjacency.items()}


def _order_children_by_subtree_extremum(
    children_by_parent: dict[int, list[int]],
    root: int,
    scalar_by_index: dict[int, float],
    root_kind: str,
    size_key: str = "nodes",
) -> None:
    """Sort every child list in place, farthest subtree extremum first.

    The key is the scalar extremum farthest from the root over the child's whole
    subtree (the child included): the maximum for ``root_kind="minimum"`` (split
    trees), highest first, and the minimum for ``root_kind="maximum"`` (join
    trees), lowest first. Ties go to the subtree with more tree nodes, then to the
    lower node index. The count covers tree nodes only, not the regular vertices
    carried on edges. ``size_key="leaves"`` counts leaves (a leaf counts one)
    instead of tree nodes for the tie-break. Subtree values are computed
    iteratively, children before parents.
    """

    if size_key not in {"nodes", "leaves"}:
        raise ValueError("size_key must be 'nodes' or 'leaves'.")

    # Height away from the root: the root's own direction is always "down".
    sign = 1.0 if root_kind == "minimum" else -1.0

    preorder: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        preorder.append(node)
        stack.extend(children_by_parent[node])

    subtree_reach: dict[int, float] = {}
    subtree_size: dict[int, int] = {}
    for node in reversed(preorder):
        children = children_by_parent[node]
        own_height = sign * scalar_by_index[node]
        subtree_reach[node] = max([own_height, *(subtree_reach[c] for c in children)])
        if size_key == "leaves":
            subtree_size[node] = sum(subtree_size[c] for c in children) if children else 1
        else:
            subtree_size[node] = 1 + sum(subtree_size[c] for c in children)

    for children in children_by_parent.values():
        children.sort(key=lambda idx: (-subtree_reach[idx], -subtree_size[idx], idx))


def build_iplotx_tree_layout(
    graph: Any,
    *,
    scalar_attr: str | None = None,
    root_kind: str = "minimum",
    node_attrs: Sequence[str] = (),
    horizontal_layout: str = "iplotx",
) -> TreeLayoutResult:
    """Build an iterative iplotx-ready rooted tree from a graph.

        Tree nodes always receive generic ``vertex_index`` and ``event`` metadata.
        Additional source metadata is copied only when named in ``node_attrs``.
        This prevents the generic layout adapter from assuming sequence fields or
        another scientific domain.

    Parameters
    ----------
    graph
        A connected tree-like Rustworkx graph.
    scalar_attr
        Scalar attribute on the graph nodes.
    root_kind
        Root direction, ``minimum`` or ``maximum``.
    node_attrs
        Additional metadata copied onto the tree nodes.
    horizontal_layout
        Child order, which sets the leaf order of the horizontal coordinates.
        ``iplotx`` (default) puts the child with the lowest scalar first.
        ``subtree_extremum`` puts first the child whose whole subtree reaches
        farthest from the root, chosen by ``root_kind``. With ``minimum`` (split
        trees) the key is the maximum scalar over the subtree, highest first, so
        the global maximum is the leftmost leaf. With ``maximum`` (join trees)
        the key is the minimum scalar over the subtree, lowest first, so the
        global minimum is the leftmost leaf. Ties go to the larger subtree, then
        to the lower node index. Subtree size counts tree nodes only; regular
        vertices carried on edges (the ``regular_count`` edge attribute) are not
        counted. The first child gets the smallest x.


    """

    if horizontal_layout not in {"iplotx", "subtree_extremum"}:
        raise ValueError("horizontal_layout must be 'iplotx' or 'subtree_extremum'.")
    node_indices = _graph_node_indices(graph)
    if not node_indices:
        raise ValueError("Tree must contain at least one node.")

    node_data_by_index = {idx: _graph_node_data(graph, idx) for idx in node_indices}
    edge_records = _graph_edge_list(graph)
    scalar_key = _resolve_scalar_key(graph, scalar_attr=scalar_attr)
    scalar_by_index = {idx: float(node_data_by_index[idx][scalar_key]) for idx in node_indices}

    scalar_min, scalar_max = _tree_scalar_bounds(graph, scalar_attr=scalar_attr)
    root = select_tree_root_node_index(graph, scalar_attr=scalar_attr, root_kind=root_kind)

    children_by_parent = {idx: [] for idx in node_indices}
    parent_by_child: dict[int, int | None] = {root: None}
    bfs = deque([root])

    adjacency = _tree_children(node_indices, edge_records)
    edge_payload_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for source, target, edge_data in edge_records:
        edge_payload_by_pair[(source, target)] = edge_data
        edge_payload_by_pair[(target, source)] = edge_data
    while bfs:
        parent = bfs.popleft()
        for child in adjacency[parent]:
            if child in parent_by_child:
                continue
            parent_by_child[child] = parent
            children_by_parent[parent].append(child)
            bfs.append(child)

    if len(parent_by_child) != len(node_indices):
        raise ValueError("Tree root traversal must reach all nodes.")
    if horizontal_layout == "subtree_extremum":
        _order_children_by_subtree_extremum(children_by_parent, root, scalar_by_index, root_kind)
    else:
        for parent in children_by_parent:
            children_by_parent[parent].sort(key=lambda idx: (scalar_by_index[idx], idx))

    regular_count_by_edge_name: dict[tuple[str, str], float] = {}
    tree_nodes_by_index: dict[int, Any] = {}

    SimpleTree = None
    try:
        from iplotx.ingest.providers.tree.simple import SimpleTree as _SimpleTree

        SimpleTree = _SimpleTree
    except Exception as exc:  # pragma: no cover - import error should fail at call time
        raise RuntimeError("iplotx is required to build tree layouts.") from exc

    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        node_idx, visited = stack.pop()
        if not visited:
            stack.append((node_idx, True))
            for child_idx in reversed(children_by_parent[node_idx]):
                stack.append((child_idx, False))
            continue

        tree_node = SimpleTree()
        tree_node.name = str(node_idx)
        tree_node.scalar = float(scalar_by_index[node_idx])
        tree_node.children = [
            tree_nodes_by_index[child_idx] for child_idx in children_by_parent[node_idx]
        ]
        if parent_by_child[node_idx] is None:
            tree_node.branch_length = 0.0
            tree_node.regular_count = 0.0
        else:
            parent_idx = int(parent_by_child[node_idx])
            edge_payload = edge_payload_by_pair.get((parent_idx, node_idx), {})
            tree_node.branch_length = abs(
                float(scalar_by_index[node_idx] - scalar_by_index[parent_idx])
            )
            tree_node.regular_count = float(edge_payload.get("regular_count", 0.0))
            regular_count_by_edge_name[(str(parent_idx), str(node_idx))] = float(
                edge_payload.get("regular_count", 0.0)
            )

        node_payload = node_data_by_index[node_idx]
        for key in ("vertex_index", "event", *node_attrs):
            if key in node_payload:
                setattr(tree_node, key, node_payload.get(key))

        tree_nodes_by_index[node_idx] = tree_node

    tree = tree_nodes_by_index[root]

    node_data_by_name = {
        str(node_idx): dict(payload) for node_idx, payload in node_data_by_index.items()
    }
    scalar_by_name = {
        name: float(payload[scalar_key]) for name, payload in node_data_by_name.items()
    }
    return TreeLayoutResult(
        tree=tree,
        root_node_index=int(root),
        node_data_by_index=node_data_by_index,
        node_data_by_name=node_data_by_name,
        scalar_by_name=scalar_by_name,
        regular_count_by_edge_name=regular_count_by_edge_name,
        scalar_min=float(scalar_min),
        scalar_max=float(scalar_max),
        root_kind=root_kind,
    )


def rooted_tree_horizontal_positions(root: Any) -> dict[Any, float]:
    """Place tree nodes horizontally the way IPlotX's vertical rooted layout does.

    Leaves receive ``0, 1, 2, ...`` in postorder visiting order (children in
    stored order) and every internal node sits at the mean of its children.
    The traversal uses an explicit stack, so tree depth is unbounded.

    Examples
    --------
    >>> class Node:
    ...     def __init__(self, children):
    ...         self.children = children
    >>> leaves = [Node([]), Node([])]
    >>> root = Node(leaves)
    >>> rooted_tree_horizontal_positions(root)[root]
    0.5

    """

    positions: dict[Any, float] = {}
    next_leaf_x = 0
    stack: list[tuple[Any, bool]] = [(root, False)]
    while stack:
        node, children_done = stack.pop()
        children = list(node.children)
        if not children:
            positions[node] = float(next_leaf_x)
            next_leaf_x += 1
        elif children_done:
            positions[node] = float(np.mean([positions[child] for child in children]))
        else:
            stack.append((node, True))
            # Reversed so the first child is popped, and therefore visited, first.
            stack.extend((child, False) for child in reversed(children))
    return positions


def extract_iplotx_tree_coordinates(layout: TreeLayoutResult) -> pl.DataFrame:
    """Extract stable horizontal coordinates for a rooted tree.

    This reproduces IPlotX's vertical rooted-tree layout iteratively: leaves are
    numbered in postorder and internal nodes sit at the mean of their children.
    IPlotX's own ``SimpleTree`` traversal recurses once per tree level, and real
    merge trees are deeper than any practical recursion limit.  The returned
    ``y`` coordinate intentionally remains the original scalar value, rather
    than IPlotX's branch-length coordinate, so an Altair renderer can retain a
    scientifically meaningful vertical axis.


    """

    positions = rooted_tree_horizontal_positions(layout.tree)
    rows = [
        {
            "tree_node_index": int(node.name),
            "x": float(x),
            "y": float(layout.scalar_by_name[str(node.name)]),
        }
        for node, x in positions.items()
    ]
    return pl.DataFrame(rows).sort("tree_node_index")


def hierarchical_tree_layout(
    graph: Any,
    node_data_by_idx: dict[int, dict[str, Any]],
) -> dict[int, tuple[float, float]]:
    """Create a deterministic hierarchical tree layout with iterative traversal."""

    import networkx as nx

    node_indices = _graph_node_indices(graph)
    if not node_indices:
        return {}

    scalar_values = {idx: float(node_data_by_idx[int(idx)]["scalar"]) for idx in node_indices}
    root_idx = min(node_indices, key=lambda idx: (scalar_values[idx], idx))

    min_scalar = min(scalar_values.values())
    max_scalar = max(scalar_values.values())
    scalar_range = max_scalar - min_scalar if max_scalar > min_scalar else 1.0
    y_positions = {
        node_idx: (scalar - min_scalar) / scalar_range
        for node_idx, scalar in scalar_values.items()
    }

    nx_graph = nx.Graph()
    for node_index in node_indices:
        nx_graph.add_node(int(node_index))
    for source, target, _edge_data in _graph_edge_list(graph):
        nx_graph.add_edge(int(source), int(target))

    parents: dict[int, int | None] = {root_idx: None}
    bfs = deque([root_idx])
    children_by_parent = {idx: [] for idx in node_indices}
    while bfs:
        parent = bfs.popleft()
        for child in sorted(
            nx_graph.neighbors(parent), key=lambda idx: (scalar_values[int(idx)], int(idx))
        ):
            child = int(child)
            if child in parents:
                continue
            parents[child] = parent
            children_by_parent[parent].append(child)
            bfs.append(child)

    if len(parents) != len(node_indices):
        raise ValueError(
            "Hierarchical tree layout requires a connected tree-like graph; "
            f"reached {len(parents)} of {len(node_indices)} nodes from root {root_idx}."
        )

    x_positions: dict[int, float] = {}
    next_leaf_x = 0.0
    stack: list[tuple[int, bool]] = [(root_idx, False)]
    while stack:
        node_idx, visited = stack.pop()
        if not visited:
            stack.append((node_idx, True))
            for child_idx in reversed(children_by_parent[node_idx]):
                stack.append((child_idx, False))
            continue
        children = children_by_parent[node_idx]
        if not children:
            x_positions[node_idx] = next_leaf_x
            next_leaf_x += 1.0
        else:
            x_positions[node_idx] = sum(x_positions[child] for child in children) / len(children)

    if x_positions:
        min_x = min(x_positions.values())
        max_x = max(x_positions.values())
        x_range = max_x - min_x if max_x > min_x else 1.0
        x_positions = {node: (x - min_x) / x_range * 10.0 for node, x in x_positions.items()}

    return {
        node_idx: (x_positions.get(node_idx, 0.0), y_positions[node_idx])
        for node_idx in node_indices
    }


def layout_positions_to_rows(
    graph: Any,
    positions: dict[int, tuple[float, float]],
    node_data_by_idx: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert graph layout positions into edge/node row dictionaries."""

    edge_rows: list[dict[str, Any]] = []
    if hasattr(graph, "weighted_edge_list"):
        weighted_edges = graph.weighted_edge_list()
    else:
        weighted_edges = [(source, target, {}) for source, target in graph.edge_list()]

    for source, target, edge_data in weighted_edges:
        sx, sy = positions[int(source)]
        tx, ty = positions[int(target)]
        source_data = node_data_by_idx[int(source)]
        target_data = node_data_by_idx[int(target)]
        edge_payload = edge_data if isinstance(edge_data, dict) else {}
        edge_rows.append(
            {
                "source": int(source),
                "target": int(target),
                "x_source": float(sx),
                "y_source": float(sy),
                "x_target": float(tx),
                "y_target": float(ty),
                "source_vertex": int(source_data["vertex_index"]),
                "target_vertex": int(target_data["vertex_index"]),
                "source_event": str(source_data.get("event", "regular")),
                "target_event": str(target_data.get("event", "regular")),
                "source_scalar": float(source_data["scalar"]),
                "target_scalar": float(target_data["scalar"]),
                "scalar_difference": float(edge_payload.get("scalar_difference", 0.0)),
                "source_dna_sequence": source_data.get("dna_sequence"),
                "target_dna_sequence": target_data.get("dna_sequence"),
                "source_aa_sequence": source_data.get("aa_sequence"),
                "target_aa_sequence": target_data.get("aa_sequence"),
                "regular_count": int(edge_payload.get("regular_count", 0)),
                "dna_hamming": (
                    int(edge_payload["dna_hamming"])
                    if edge_payload.get("dna_hamming") is not None
                    else None
                ),
                "aa_hamming": (
                    int(edge_payload["aa_hamming"])
                    if edge_payload.get("aa_hamming") is not None
                    else None
                ),
            }
        )

    node_rows: list[dict[str, Any]] = []
    for node_idx, (x_pos, y_pos) in positions.items():
        node_data = node_data_by_idx[int(node_idx)]
        node_rows.append(
            {
                "tree_node_index": int(node_idx),
                "vertex_index": int(node_data["vertex_index"]),
                "scalar": float(node_data["scalar"]),
                "event": str(node_data.get("event", "regular")),
                "x_pos": float(x_pos),
                "y_pos": float(y_pos),
            }
        )

    return edge_rows, node_rows
