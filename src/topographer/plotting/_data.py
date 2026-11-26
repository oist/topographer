"""Backend-independent data preparation for tree plots."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math
import operator
from typing import Any

import polars as pl

from topographer.topology import _fold_tree, _rooted_tree, select_extremum_root


@dataclass(frozen=True, slots=True)
class TreePlotData:
    """Stable table inputs for a tree plot, independent of its rendering backend.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import make_tree_layout
    >>> from topographer.topology import ScalarNetworkTopology
    >>> tree = ScalarNetworkTopology(build_cube_scalar_network()).split_tree()
    >>> make_tree_layout(tree).node_table.height
    4
    """

    node_table: pl.DataFrame
    edge_table: pl.DataFrame
    edge_segment_table: pl.DataFrame
    root_node_index: int
    scalar_attr: str
    root_kind: str
    vertex_table: pl.DataFrame | None = None


def make_tree_layout(
    tree: Any,
    *,
    scalar_attr: str | None = None,
    network: Any | None = None,
    root_kind: str | None = None,
    node_attrs: Sequence[str] = (),
    edge_attrs: Sequence[str] = (),
    horizontal_layout: str = "leaf_order",
    progress_callback: Callable[[int, int], None] | None = None,
) -> TreePlotData:
    """Lay out a connected tree as deterministic Polars node and edge tables.

        The horizontal position follows a leaf-order traversal by default or the
        IPlotX's vertical rooted-tree placement (postorder leaf numbering,
        internal nodes at the mean of their children, computed iteratively)
        when ``horizontal_layout='iplotx'``. ``horizontal_layout='subtree_extremum'``
        uses the same placement but orders the children at every node by the
        subtree extremum farthest from the root, chosen by ``root_kind``. On a
        split tree (root ``minimum``) the key is the maximum scalar over each
        child's whole subtree, highest first, so the global maximum is the
        leftmost leaf. On a join tree (root ``maximum``) the key is the minimum
        scalar over the subtree, lowest first, so the global minimum is the
        leftmost leaf. The vertical
        position always preserves the requested scalar value. ``node_attrs`` and
        ``edge_attrs`` copy optional metadata columns into the respective tables
        without interpreting their scientific meaning. A ``MergeTree`` supplies
        its declared scalar and root direction automatically. Supplying its
        ``ScalarNetwork`` also creates an inspectable ``vertex_table`` that maps
        every original public vertex identity to exact scalar coordinates and
        its branch or critical-node position.

    Parameters
    ----------
    tree
        A merge tree, or an existing tree-like graph for explicit layouts.
    scalar_attr
        Scalar attribute for a tree-like graph. Merge trees use their declared
        scalar attribute; their normalized internal scalar remains private.
    network
        Source scalar network for identity-based annotation placement.
    root_kind
        Root direction, ``minimum`` or ``maximum``. Inferred for merge trees.
    node_attrs
        Additional metadata columns copied to the node table.
    edge_attrs
        Additional metadata columns copied to the edge table.
    horizontal_layout
        ``weighted``: every subtree owns an x interval as wide as its leaf count, leaves
        sit one unit apart at interval centres, and each internal node sits at the
        leaf-weighted mean of its children's x (so a large funnel's saddle stays over its
        mass). Children are ordered by the farthest subtree extremum, then by leaf count,
        then by node index; the x coordinates are not rescaled to ``[0, 1]``.
        ``leaf_order`` and ``iplotx`` are the plain coordinate generators. ``subtree_extremum``:
        ``iplotx`` coordinates with the children of every node ordered by the
        subtree extremum farthest from the root, leftmost first. With a
        ``minimum`` root (split trees) that is the maximum scalar over the child's
        whole subtree, highest first; with a ``maximum`` root (join trees) it is the
        minimum scalar over the subtree, lowest first. The global maximum of a split
        tree and the global minimum of a join tree are therefore the leftmost leaf.
        Ties go to the larger subtree, then to the lower node index. Subtree size
        counts tree nodes only; regular vertices carried on edges (the
        ``regular_count`` edge attribute) are not counted. ``leaf_order`` and
        ``iplotx`` put the child with the lowest scalar first.
    progress_callback
        Optional callback receiving completed and total layout row counts.

    Returns
    -------
    TreePlotData
        Node, edge, segment, and optional public-vertex location tables.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import make_tree_layout
    >>> from topographer.topology import ScalarNetworkTopology
    >>> tree = ScalarNetworkTopology(build_cube_scalar_network()).split_tree()
    >>> make_tree_layout(tree).root_kind
    'minimum'
    """

    merge_tree = tree if hasattr(tree, "mapping") and hasattr(tree, "graph") else None
    display_scalar_attr = scalar_attr
    if merge_tree is not None:
        display_scalar_attr = scalar_attr or merge_tree.scalar_attr
        scalar_attr = "scalar"
        root_kind = root_kind or ("maximum" if merge_tree.kind == "join" else "minimum")
        tree = merge_tree.graph
    scalar_attr = scalar_attr or "scalar"
    root_kind = root_kind or "minimum"
    node_indices = _node_indices(tree)
    if not node_indices:
        raise ValueError("Tree plotting requires at least one node.")
    if root_kind not in {"minimum", "maximum"}:
        raise ValueError("root_kind must be 'minimum' or 'maximum'.")

    requested_node_attrs = _normalize_attrs(node_attrs, "node")
    requested_edge_attrs = _normalize_attrs(edge_attrs, "edge")
    if horizontal_layout not in {"leaf_order", "iplotx", "subtree_extremum", "weighted"}:
        raise ValueError(
            "horizontal_layout must be 'leaf_order', 'iplotx', 'subtree_extremum' or 'weighted'."
        )
    node_payloads = {index: _node_payload(tree, index) for index in node_indices}
    scalars = _scalar_values(node_payloads, scalar_attr)
    adjacency, edge_payloads = _tree_adjacency(tree, node_indices)
    _validate_tree_topology(adjacency, edge_payloads, node_indices)

    root = select_extremum_root(node_indices, scalars, root_kind)
    parents, children = _root_tree(adjacency, scalars, root)
    x_positions = _leaf_order_positions(children, root)
    if horizontal_layout == "weighted":
        from topographer.plotting._weighted import weighted_tree_positions

        x_positions = weighted_tree_positions(children, root, scalars, root_kind)
    if horizontal_layout in {"iplotx", "subtree_extremum"}:
        from topographer.plotting._layout import (
            build_iplotx_tree_layout,
            extract_iplotx_tree_coordinates,
        )

        coordinate_table = extract_iplotx_tree_coordinates(
            build_iplotx_tree_layout(
                tree,
                scalar_attr=scalar_attr,
                root_kind=root_kind,
                horizontal_layout=horizontal_layout,
            )
        )
        x_positions = {row["tree_node_index"]: row["x"] for row in coordinate_table.to_dicts()}
    rooted = _rooted_tree(
        root,
        children,
        {
            child: int(edge_payloads[frozenset((parent, child))].get("regular_count", 0))
            for child, parent in parents.items()
            if parent is not None
        },
    )
    fold = _fold_tree(
        rooted,
        vertex_value=lambda _node: 1,
        regular_value=rooted.regular_count.__getitem__,
        combine=operator.add,
    )

    node_rows: list[dict[str, Any]] = []
    total_rows = len(node_indices) * 2
    _report_progress(progress_callback, 0, total_rows)
    for position, node_index in enumerate(sorted(node_indices), start=1):
        payload = node_payloads[node_index]
        row = {
            "tree_node_index": int(node_index),
            "vertex_index": int(payload.get("vertex_index", node_index)),
            "scalar": float(scalars[node_index]),
            "event": str(payload.get("event", "regular")),
            "display_event": root_kind
            if node_index == root
            else str(payload.get("event", "regular")),
            "x": float(x_positions[node_index]),
            "y": float(scalars[node_index]),
        }
        if "vertex_id" in payload:
            row["vertex_id"] = str(payload["vertex_id"])
        if "label" in payload:
            row["label"] = str(payload["label"])
        for attr in requested_node_attrs:
            if attr in payload and attr not in row:
                row[attr] = payload[attr]
        node_rows.append(row)
        _report_progress(progress_callback, position, total_rows)

    edge_rows: list[dict[str, Any]] = []
    for position, child_index in enumerate(sorted(node_indices), start=len(node_indices) + 1):
        parent_index = parents[child_index]
        if parent_index is None:
            continue
        payload = edge_payloads[frozenset((parent_index, child_index))]
        row = {
            "source_tree_node_index": int(parent_index),
            "target_tree_node_index": int(child_index),
            "source_vertex_index": int(
                node_payloads[parent_index].get("vertex_index", parent_index)
            ),
            "target_vertex_index": int(
                node_payloads[child_index].get("vertex_index", child_index)
            ),
            "x_source": float(x_positions[parent_index]),
            "y_source": float(scalars[parent_index]),
            "x_target": float(x_positions[child_index]),
            "y_target": float(scalars[child_index]),
            "source_scalar": float(scalars[parent_index]),
            "target_scalar": float(scalars[child_index]),
            "scalar_difference": abs(float(scalars[child_index] - scalars[parent_index])),
            "regular_count": int(payload.get("regular_count", 0)),
            "branch_vertex_count": int(fold.branch_local[child_index]),
            "horton_strahler_order": int(fold.horton_strahler[child_index]),
        }
        for attr in requested_edge_attrs:
            if attr in payload:
                row[attr] = payload[attr]
        edge_rows.append(row)
        _report_progress(progress_callback, position, total_rows)

    segment_rows = _orthogonal_edge_segments(edge_rows)

    vertex_table = (
        _make_vertex_table(merge_tree, network, node_rows, edge_rows)
        if merge_tree is not None and network is not None
        else None
    )
    return TreePlotData(
        node_table=pl.DataFrame(node_rows),
        edge_table=pl.DataFrame(edge_rows),
        edge_segment_table=pl.DataFrame(segment_rows),
        root_node_index=int(root),
        scalar_attr=display_scalar_attr or scalar_attr,
        root_kind=root_kind,
        vertex_table=vertex_table,
    )


build_tree_plot_data = make_tree_layout


def relayout_tree_plot_data(
    plot_data: TreePlotData, *, horizontal_layout: str = "weighted"
) -> TreePlotData:
    """Recompute the horizontal placement of an existing :class:`TreePlotData`.

    The tree is rebuilt from ``edge_table`` (``source_tree_node_index`` is the parent,
    ``target_tree_node_index`` the child, rooted at ``root_node_index``) and the scalars in
    ``node_table``. Only x changes: the node x, the edge ``x_source``/``x_target``, the orthogonal
    ``edge_segment_table`` and the ``vertex_table`` x are regenerated, and every other column,
    including extra metadata such as ``branch_vertex_count``, is kept. A branch vertex follows its
    edge's child node; one that cannot be matched to a single edge by its previous x and scalar
    raises ``ValueError``. Only ``horizontal_layout="weighted"`` is supported.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import make_tree_layout, relayout_tree_plot_data
    >>> from topographer.topology import ScalarNetworkTopology
    >>> tree = ScalarNetworkTopology(build_cube_scalar_network()).split_tree()
    >>> relayout_tree_plot_data(make_tree_layout(tree), horizontal_layout="weighted").node_table.height
    4
    """

    if horizontal_layout != "weighted":
        raise ValueError("relayout_tree_plot_data supports only horizontal_layout='weighted'.")
    from topographer.plotting._weighted import weighted_tree_positions

    node_rows = plot_data.node_table.to_dicts()
    edge_rows = plot_data.edge_table.to_dicts()
    scalars = {int(row["tree_node_index"]): float(row["scalar"]) for row in node_rows}
    children: dict[int, list[int]] = {node: [] for node in scalars}
    for row in edge_rows:
        children[int(row["source_tree_node_index"])].append(int(row["target_tree_node_index"]))
    x_positions = weighted_tree_positions(
        children, int(plot_data.root_node_index), scalars, plot_data.root_kind
    )
    old_edge_rows = edge_rows
    node_rows_old = node_rows
    node_rows = [{**row, "x": x_positions[int(row["tree_node_index"])]} for row in node_rows]
    edge_rows = [
        {
            **row,
            "x_source": x_positions[int(row["source_tree_node_index"])],
            "x_target": x_positions[int(row["target_tree_node_index"])],
        }
        for row in edge_rows
    ]
    vertex_table = plot_data.vertex_table
    if vertex_table is not None:
        vertex_table = _relayout_vertex_table(
            vertex_table, node_rows_old, x_positions, old_edge_rows, edge_rows
        )
    return replace(
        plot_data,
        node_table=pl.DataFrame(node_rows, schema=plot_data.node_table.schema),
        edge_table=pl.DataFrame(edge_rows, schema=plot_data.edge_table.schema),
        edge_segment_table=_segment_table_like(
            _orthogonal_edge_segments(edge_rows), plot_data.edge_segment_table
        ),
        vertex_table=vertex_table,
    )


def _segment_table_like(rows: list[dict[str, Any]], template: pl.DataFrame) -> pl.DataFrame:
    """Build a segment table, keeping the dtypes of the columns ``template`` already has."""

    table = pl.DataFrame(rows)
    cast = {name: dtype for name, dtype in template.schema.items() if name in table.columns}
    return table.cast(cast).select(template.columns)


def _relayout_vertex_table(
    vertex_table: pl.DataFrame,
    old_node_rows: list[dict[str, Any]],
    new_node_x: dict[int, float],
    old_edge_rows: list[dict[str, Any]],
    new_edge_rows: list[dict[str, Any]],
) -> pl.DataFrame:
    """Move vertex-table x with the node or edge each row was placed on.

    A critical vertex follows its node by ``vertex_id`` when the node table carries one, else by
    its coordinates. A branch vertex has no stored edge key, so it follows the single edge whose
    child x and scalar range contain it; coordinates match to 9 decimals, so a table written to
    and read back from disk still matches.
    """

    def key(value: float) -> float:
        return round(float(value), 9)

    nodes_by_id = {
        str(row["vertex_id"]): int(row["tree_node_index"])
        for row in old_node_rows
        if row.get("vertex_id") is not None
    }
    nodes_by_place: dict[tuple[float, float], set[int]] = {}
    for row in old_node_rows:
        place = (key(row["x"]), key(row["scalar"]))
        nodes_by_place.setdefault(place, set()).add(int(row["tree_node_index"]))
    edges_by_x: dict[float, list[int]] = {}
    for position, row in enumerate(old_edge_rows):
        edges_by_x.setdefault(key(row["x_target"]), []).append(position)
    rows = vertex_table.to_dicts()
    for row in rows:
        y = float(row["y"])
        if row["is_critical"]:
            if str(row["vertex_id"]) in nodes_by_id:
                nodes = {nodes_by_id[str(row["vertex_id"])]}
            else:
                nodes = nodes_by_place.get((key(row["x"]), key(y)), set())
            new_values = {new_node_x[node] for node in nodes}
        else:
            candidates = [
                position
                for position in edges_by_x.get(key(row["x"]), ())
                if min(old_edge_rows[position]["y_target"], old_edge_rows[position]["y_source"])
                <= y
                <= max(old_edge_rows[position]["y_target"], old_edge_rows[position]["y_source"])
            ]
            # A vertex at a shared endpoint of a chain belongs to the edge whose child is nearest.
            nearest = min(
                (abs(y - old_edge_rows[p]["y_target"]) for p in candidates), default=None
            )
            new_values = {
                float(new_edge_rows[p]["x_target"])
                for p in candidates
                if abs(y - old_edge_rows[p]["y_target"]) == nearest
            }
        if len(new_values) != 1:
            raise ValueError(
                "Cannot relayout a vertex_table row that matches no single node or edge."
            )
        row["x"] = new_values.pop()
    return pl.DataFrame(rows, schema=vertex_table.schema)


def _make_vertex_table(
    tree: Any, network: Any, node_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]]
) -> pl.DataFrame:
    """Map all stable network vertex IDs to their plotted branch or node coordinates."""
    vertex_ids = network.vertex_ids
    graph = network.graph
    id_by_index = dict(zip(graph.node_indices(), vertex_ids, strict=True))
    scalar_attr = network.scalar_attr
    network_scalar = {
        id_by_index[index]: float(graph.get_node_data(index)[scalar_attr])
        for index in graph.node_indices()
    }
    nodes = {int(row["tree_node_index"]): row for row in node_rows}
    branch_x = {
        (int(row["source_tree_node_index"]), int(row["target_tree_node_index"])): float(
            row["x_target"]
        )
        for row in edge_rows
    }
    mapping = tree.mapping
    rows: list[dict[str, Any]] = []
    for row in mapping.critical_vertices.to_dicts():
        index = int(row["vertex_index"])
        node = nodes.get(int(row["tree_node_index"]))
        if node is None or index not in id_by_index:
            raise ValueError("MergeTree critical vertices do not match the ScalarNetwork.")
        vertex_id = id_by_index[index]
        scalar = float(row["scalar"])
        if network_scalar.get(vertex_id) != scalar or (
            node.get("vertex_id") is not None and str(vertex_id) != str(node["vertex_id"])
        ):
            raise ValueError(
                "MergeTree mapping and ScalarNetwork do not agree on vertex identity or scalar."
            )
        rows.append({"vertex_id": vertex_id, "x": node["x"], "y": scalar, "is_critical": True})
    for row in mapping.branch_vertices.to_dicts():
        index = int(row["vertex_index"])
        key = (int(row["source_tree_node_index"]), int(row["target_tree_node_index"]))
        if key not in branch_x or index not in id_by_index:
            raise ValueError("MergeTree branch vertices do not match the ScalarNetwork.")
        vertex_id = id_by_index[index]
        scalar = float(row["scalar"])
        if network_scalar.get(vertex_id) != scalar:
            raise ValueError("MergeTree mapping and ScalarNetwork do not agree on vertex scalars.")
        rows.append(
            {"vertex_id": vertex_id, "x": branch_x[key], "y": scalar, "is_critical": False}
        )
    if len({row["vertex_id"] for row in rows}) != len(rows):
        raise ValueError("Tree mapping contains duplicate public vertex IDs.")
    vertex_values = [row["vertex_id"] for row in rows]
    vertex_series = (
        pl.Series("vertex_id", vertex_values)
        if not vertex_values
        or all(type(value) in {str, int, float, bool} for value in vertex_values)
        else pl.Series("vertex_id", vertex_values, dtype=pl.Object)
    )
    table = (
        pl.DataFrame(
            {
                "vertex_id": vertex_series,
                "x": [row["x"] for row in rows],
                "y": [row["y"] for row in rows],
                "is_critical": [row["is_critical"] for row in rows],
            }
        )
        if rows
        else pl.DataFrame(
            schema={
                "vertex_id": pl.Object,
                "x": pl.Float64,
                "y": pl.Float64,
                "is_critical": pl.Boolean,
            }
        )
    )
    if table.height != len(vertex_ids):
        raise ValueError("MergeTree mapping does not cover every ScalarNetwork vertex.")
    return table


def _resolve_tree_annotation_table(
    plot_data: TreePlotData, annotation_table: pl.DataFrame, *, vertex_id_column: str = "vertex_id"
) -> pl.DataFrame:
    """Resolve generic annotation rows keyed by public vertex IDs to tree coordinates."""
    if vertex_id_column not in annotation_table.columns:
        raise KeyError(f"Annotation table must contain {vertex_id_column!r}.")
    if plot_data.vertex_table is None:
        raise ValueError(
            "ID-keyed annotations require tree plot data built with a MergeTree and ScalarNetwork."
        )
    annotation_ids = annotation_table.get_column(vertex_id_column).to_list()
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("Annotation vertex IDs must be unique.")
    lookup = {row["vertex_id"]: row for row in plot_data.vertex_table.to_dicts()}
    rows = []
    unknown = []
    for row in annotation_table.to_dicts():
        location = lookup.get(row[vertex_id_column])
        if location is None:
            unknown.append(row[vertex_id_column])
        else:
            rows.append(
                {
                    **row,
                    "x": location["x"],
                    "y": location["y"],
                    "is_critical": location["is_critical"],
                }
            )
    if unknown:
        raise KeyError(f"Unknown annotation vertex IDs: {unknown}.")
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                **annotation_table.schema,
                "x": pl.Float64,
                "y": pl.Float64,
                "is_critical": pl.Boolean,
            }
        )
    )


def _report_progress(
    callback: Callable[[int, int], None] | None, completed: int, total: int
) -> None:
    """Report roughly one thousand plot-table progress increments when requested."""

    if callback is None:
        return
    step = max(1, total // 1000)
    if completed == 0 or completed == total or completed % step == 0:
        callback(completed, total)


def _node_indices(tree: Any) -> list[int]:
    """Return Rustworkx-style node indices from a tree-like graph."""

    if not hasattr(tree, "node_indices"):
        raise TypeError(f"Unsupported tree type: {type(tree).__name__}.")
    return sorted(int(index) for index in tree.node_indices())


def _node_payload(tree: Any, node_index: int) -> dict[str, Any]:
    """Return one node payload as a plain mapping."""

    if not hasattr(tree, "get_node_data"):
        raise TypeError(f"Unsupported tree type: {type(tree).__name__}.")
    payload = tree.get_node_data(node_index)
    if not isinstance(payload, dict):
        raise TypeError(f"Tree node {node_index} payload must be a dictionary.")
    return dict(payload)


def _scalar_values(node_payloads: dict[int, dict[str, Any]], scalar_attr: str) -> dict[int, float]:
    """Read finite scalar values from all tree nodes."""

    values: dict[int, float] = {}
    for node_index, payload in node_payloads.items():
        if scalar_attr not in payload:
            raise KeyError(f"Tree node {node_index} is missing scalar attribute {scalar_attr!r}.")
        value = payload[scalar_attr]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Tree node {node_index} scalar attribute {scalar_attr!r} must be numeric."
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"Tree node {node_index} scalar attribute {scalar_attr!r} must be finite."
            )
        values[node_index] = float(value)
    return values


def _tree_adjacency(
    tree: Any, node_indices: list[int]
) -> tuple[dict[int, list[int]], dict[frozenset[int], dict[str, Any]]]:
    """Build undirected adjacency and payload lookup for a tree-like graph."""

    if not hasattr(tree, "edge_indices") or not hasattr(tree, "get_edge_endpoints_by_index"):
        raise TypeError(f"Unsupported tree type: {type(tree).__name__}.")
    adjacency = {node_index: [] for node_index in node_indices}
    edge_payloads: dict[frozenset[int], dict[str, Any]] = {}
    for edge_index in tree.edge_indices():
        source, target = tree.get_edge_endpoints_by_index(edge_index)
        source, target = int(source), int(target)
        if source == target:
            raise ValueError("Tree plotting does not support self-loop edges.")
        edge_key = frozenset((source, target))
        if edge_key in edge_payloads:
            raise ValueError("Tree plotting does not support parallel edges.")
        edge_payload = tree.get_edge_data_by_index(edge_index)
        if edge_payload is not None and not isinstance(edge_payload, dict):
            raise TypeError(f"Tree edge {edge_index} payload must be a dictionary or None.")
        adjacency[source].append(target)
        adjacency[target].append(source)
        edge_payloads[edge_key] = dict(edge_payload or {})
    return adjacency, edge_payloads


def _validate_tree_topology(
    adjacency: dict[int, list[int]],
    edge_payloads: dict[frozenset[int], dict[str, Any]],
    node_indices: list[int],
) -> None:
    """Require a connected, acyclic undirected topology for hierarchical layout."""

    if len(edge_payloads) != len(node_indices) - 1:
        raise ValueError("Tree plotting requires exactly one fewer edge than vertices.")
    reached: set[int] = set()
    queue: deque[int] = deque([node_indices[0]])
    while queue:
        node_index = queue.popleft()
        if node_index in reached:
            continue
        reached.add(node_index)
        queue.extend(adjacency[node_index])
    if len(reached) != len(node_indices):
        raise ValueError("Tree plotting requires a connected tree.")


def _root_tree(
    adjacency: dict[int, list[int]], scalars: dict[int, float], root: int
) -> tuple[dict[int, int | None], dict[int, list[int]]]:
    """Orient undirected adjacency from a chosen root with deterministic children."""

    parents: dict[int, int | None] = {root: None}
    children = {node_index: [] for node_index in adjacency}
    queue: deque[int] = deque([root])
    while queue:
        node_index = queue.popleft()
        for neighbor in sorted(adjacency[node_index], key=lambda index: (scalars[index], index)):
            if neighbor in parents:
                continue
            parents[neighbor] = node_index
            children[node_index].append(neighbor)
            queue.append(neighbor)
    return parents, children


def _leaf_order_positions(children: dict[int, list[int]], root: int) -> dict[int, float]:
    """Assign normalized horizontal coordinates from an iterative postorder traversal."""

    raw_positions: dict[int, float] = {}
    next_leaf_position = 0.0
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        node_index, visited = stack.pop()
        if not visited:
            stack.append((node_index, True))
            for child_index in reversed(children[node_index]):
                stack.append((child_index, False))
            continue
        if not children[node_index]:
            raw_positions[node_index] = next_leaf_position
            next_leaf_position += 1.0
        else:
            raw_positions[node_index] = sum(
                raw_positions[child] for child in children[node_index]
            ) / len(children[node_index])

    minimum = min(raw_positions.values())
    maximum = max(raw_positions.values())
    if minimum == maximum:
        return {node_index: 0.5 for node_index in raw_positions}
    return {
        node_index: (position - minimum) / (maximum - minimum)
        for node_index, position in raw_positions.items()
    }


def _orthogonal_edge_segments(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Encode each logical tree edge as its vertical and horizontal segments."""

    segments: list[dict[str, Any]] = []
    for logical_edge_index, edge in enumerate(edge_rows):
        common = {**edge, "logical_edge_index": logical_edge_index}
        segments.append(
            {
                **common,
                "segment_index": 0,
                "segment_kind": "vertical",
                "x_start": edge["x_target"],
                "y_start": edge["y_target"],
                "x_end": edge["x_target"],
                "y_end": edge["y_source"],
            }
        )
        segments.append(
            {
                **common,
                "segment_index": 1,
                "segment_kind": "horizontal",
                "x_start": edge["x_target"],
                "y_start": edge["y_source"],
                "x_end": edge["x_source"],
                "y_end": edge["y_source"],
            }
        )
    return segments


def filter_tree_plot_data(
    plot_data: TreePlotData,
    *,
    node_predicate: Callable[[dict[str, Any]], bool] | None = None,
    edge_predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> TreePlotData:
    """Return a filtered view of plot tables without changing the source tree.

    Filtering is deliberately table-based: nodes remain available to preserve
    their coordinates, while matching logical edges and their geometry are
    selected for display.


    """

    node_rows = plot_data.node_table.to_dicts()
    edge_rows = plot_data.edge_table.to_dicts()
    kept_nodes = [row for row in node_rows if node_predicate is None or node_predicate(row)]
    kept_edges = [row for row in edge_rows if edge_predicate is None or edge_predicate(row)]
    segments = _orthogonal_edge_segments(kept_edges)
    return TreePlotData(
        node_table=pl.DataFrame(kept_nodes, schema=plot_data.node_table.schema),
        edge_table=pl.DataFrame(kept_edges, schema=plot_data.edge_table.schema),
        edge_segment_table=pl.DataFrame(segments, schema=plot_data.edge_segment_table.schema),
        root_node_index=plot_data.root_node_index,
        scalar_attr=plot_data.scalar_attr,
        root_kind=plot_data.root_kind,
    )


def _with_global_root_self_point(
    table: pl.DataFrame,
    tree: Any,
    *,
    root_node_index: int,
    root_event: str,
    label_attr: str = "label",
) -> pl.DataFrame:
    """Return diagram data with one display-only root-extremum diagonal point."""

    if root_event not in {"minimum", "maximum"}:
        raise ValueError("root_event must be 'minimum' or 'maximum'.")

    source_columns = {"source_scalar", "target_scalar", "source_event", "target_event"}
    extremum_columns = {
        "extremum_scalar",
        "saddle_scalar",
        "extremum_event",
        "saddle_event",
    }
    if source_columns.issubset(table.columns):
        names = {
            "source_index": "source_tree_node_index",
            "target_index": "target_tree_node_index",
            "source_vertex": "source_vertex_index",
            "target_vertex": "target_vertex_index",
            "source_label": "source_label",
            "target_label": "target_label",
            "source_scalar": "source_scalar",
            "target_scalar": "target_scalar",
            "source_event": "source_event",
            "target_event": "target_event",
        }
    elif extremum_columns.issubset(table.columns):
        names = {
            "source_index": "extremum_tree_node_index",
            "target_index": "saddle_tree_node_index",
            "source_vertex": "extremum_vertex_index",
            "target_vertex": "saddle_vertex_index",
            "source_label": "extremum_label",
            "target_label": "saddle_label",
            "source_scalar": "extremum_scalar",
            "target_scalar": "saddle_scalar",
            "source_event": "extremum_event",
            "target_event": "saddle_event",
        }
    else:
        raise KeyError("Root self-point data requires a supported branch-span table.")

    root = dict(tree.get_node_data(root_node_index))
    scalar = float(root["scalar"])
    vertex_index = int(root.get("vertex_index", root_node_index))
    label = str(root.get(label_attr, vertex_index))
    augmented = table.with_columns(pl.lit(False).alias("is_global_extremum_self_point"))
    row = {column: None for column in augmented.columns}
    row.update(
        {
            names["source_index"]: int(root_node_index),
            names["target_index"]: int(root_node_index),
            names["source_vertex"]: vertex_index,
            names["target_vertex"]: vertex_index,
            names["source_label"]: label,
            names["target_label"]: label,
            names["source_scalar"]: scalar,
            names["target_scalar"]: scalar,
            names["source_event"]: root_event,
            names["target_event"]: root_event,
            "branch_span": 0.0,
            "is_global_extremum_self_point": True,
        }
    )
    self_point = pl.DataFrame([row], schema=augmented.schema)
    return pl.concat((augmented, self_point))


def _with_global_minimum_self_point(
    table: pl.DataFrame,
    tree: Any,
    *,
    root_node_index: int,
    label_attr: str = "label",
) -> pl.DataFrame:
    """Compatibility wrapper for the former minimum-only internal helper."""

    return _with_global_root_self_point(
        table,
        tree,
        root_node_index=root_node_index,
        root_event="minimum",
        label_attr=label_attr,
    )


def select_tree_edges_by_horton_strahler(
    plot_data: TreePlotData,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> TreePlotData:
    """Return a non-mutating view restricted to an inclusive order interval."""

    if minimum is not None and minimum < 1:
        raise ValueError("minimum Horton–Strahler order must be at least one.")
    if maximum is not None and maximum < 1:
        raise ValueError("maximum Horton–Strahler order must be at least one.")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum Horton–Strahler order cannot exceed maximum.")

    return filter_tree_plot_data(
        plot_data,
        edge_predicate=lambda row: (
            (minimum is None or row["horton_strahler_order"] >= minimum)
            and (maximum is None or row["horton_strahler_order"] <= maximum)
        ),
    )


def _normalize_attrs(attrs: Sequence[str], location: str) -> tuple[str, ...]:
    """Validate optional metadata columns requested by a table consumer."""

    normalized = tuple(attrs)
    if any(not isinstance(attr, str) or not attr.strip() for attr in normalized):
        raise TypeError(f"{location} attribute names must be nonempty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{location} attribute names must not be repeated.")
    return normalized
