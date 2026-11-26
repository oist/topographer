"""Domain-independent merge-tree construction for scalar networks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl
import rustworkx as rx

from topographer.graph import ScalarNetwork

__all__ = [
    "MergeTreeResult",
    "MergeTreeSweepTrace",
    "build_join_tree",
    "build_merge_tree",
    "build_merge_tree_sweep_trace",
    "build_split_tree",
    "resolve_scalar_key",
    "select_extremum_root",
    "tree_scalar_bounds",
]


@dataclass(frozen=True, slots=True)
class MergeTreeResult:
    """Container for a deterministic merge-tree sweep on a connected graph."""

    tree: rx.PyDiGraph
    event_table: pl.DataFrame
    scalar_key: str
    ascending: bool

    @property
    def tree_kind(self) -> str:
        """Return the explicit filtration name for this construction result."""

        return "join" if self.ascending else "split"

    @property
    def root_node_index(self) -> int:
        """Return the unique directed sink of the connected merge tree."""

        sinks = [
            int(node_index)
            for node_index in self.tree.node_indices()
            if not list(self.tree.out_edges(node_index))
        ]
        if len(sinks) != 1:
            raise ValueError("A merge-tree result must have exactly one directed root.")
        return sinks[0]


@dataclass(frozen=True, slots=True)
class MergeTreeSweepTrace:
    """Inspectable states from a deterministic merge-tree construction sweep.

    The tables describe a vertex-filtration sweep without mutating either the
    input graph or the returned merge tree. ``step_table`` has one row per
    activated vertex. ``component_table`` records every active vertex at every
    step and the extremum whose component it belongs to. ``graph_edge_table``
    and ``tree_edge_table`` record the first step at which each input or tree
    edge becomes visible.

    """

    result: MergeTreeResult
    step_table: pl.DataFrame
    component_table: pl.DataFrame
    graph_edge_table: pl.DataFrame
    tree_node_table: pl.DataFrame
    tree_edge_table: pl.DataFrame


def _make_tree_node_payload(
    *,
    vertex_index: int,
    scalar: float,
    event: str,
    source_node_data: dict[str, Any],
    node_attrs: tuple[str, ...],
) -> dict[str, Any]:
    """Build a generic merge-tree node payload with requested source attributes."""

    payload = {
        "vertex_index": int(vertex_index),
        "scalar": float(scalar),
        "event": event,
    }
    for attr in node_attrs:
        if attr in source_node_data:
            payload[attr] = source_node_data[attr]
    return payload


def _make_merge_tree_edge_payload(
    *,
    source_scalar: float,
    target_scalar: float,
    merge_scalar: float,
    from_repr: int,
    to_saddle: int,
) -> dict[str, Any]:
    """Build the domain-independent payload for one merge-tree edge."""

    scalar_difference = abs(target_scalar - source_scalar)
    reciprocal_weight = 1.0 / max(scalar_difference, 1e-12)
    return {
        "source_scalar": source_scalar,
        "target_scalar": target_scalar,
        "scalar_difference": scalar_difference,
        "weight": reciprocal_weight,
        "merge_scalar": merge_scalar,
        "from_repr": int(from_repr),
        "to_saddle": int(to_saddle),
    }


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


def resolve_scalar_key(graph: Any, scalar_attr: str | None = None) -> str:
    """Validate and return an explicitly requested vertex scalar attribute.

    For a :class:`~topographer.graph.ScalarNetwork`, the scalar may be
    omitted only when exactly one vertex scalar attribute was declared.
    Bare Rustworkx graphs always require ``scalar_attr``.


    """

    _graph, scalar_key = _resolve_scalar_graph(graph, scalar_attr=scalar_attr)
    return scalar_key


def tree_scalar_bounds(graph: Any, scalar_attr: str | None = None) -> tuple[float, float]:
    """Return the minimum and maximum scalar values on a graph."""

    resolved_graph, scalar_key = _resolve_scalar_graph(graph, scalar_attr=scalar_attr)
    scalar_values = [
        float(_graph_node_data(resolved_graph, node_index)[scalar_key])
        for node_index in _graph_node_indices(resolved_graph)
    ]
    if not scalar_values:
        raise ValueError("Graph must contain at least one node.")
    return min(scalar_values), max(scalar_values)


def select_extremum_root(
    node_indices: Sequence[int],
    scalars: Mapping[int, float],
    root_kind: str,
) -> int:
    """Return the tree root among tied global extrema using the sweep's tie-break.

    Exact scalar ties follow the sweep's symbolic order, so the selected root is the
    last tied extremum swept. This matches ``MergeTreeResult.root_node_index`` for a
    join tree with ``"maximum"`` and a split tree with ``"minimum"``.

    Examples
    --------
    >>> from topographer.topology._tree_construction import select_extremum_root
    >>> select_extremum_root([0, 1, 2], {0: 1.0, 1: 0.0, 2: 1.0}, "maximum")
    2
    >>> select_extremum_root([0, 1, 2], {0: 0.0, 1: 1.0, 2: 0.0}, "minimum")
    2

    """

    if not node_indices:
        raise ValueError("Graph must contain at least one node.")
    if root_kind == "maximum":
        return max(node_indices, key=lambda idx: (scalars[idx], int(idx)))
    if root_kind == "minimum":
        return min(node_indices, key=lambda idx: (scalars[idx], -int(idx)))
    raise ValueError("root_kind must be 'minimum' or 'maximum'.")


def _select_root_node_index(
    graph: Any,
    scalar_attr: str | None = None,
    *,
    root_kind: str = "minimum",
) -> int:
    """Return the deterministic root node for a tree-like graph."""

    resolved_graph, scalar_key = _resolve_scalar_graph(graph, scalar_attr=scalar_attr)
    node_indices = _graph_node_indices(resolved_graph)
    if not node_indices:
        raise ValueError("Graph must contain at least one node.")

    scalar_values = {
        node_index: float(_graph_node_data(resolved_graph, node_index)[scalar_key])
        for node_index in node_indices
    }
    return select_extremum_root(node_indices, scalar_values, root_kind)


class _DisjointSet:
    """Minimal union-find with path compression and union by rank."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def add(self, value: int) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> int:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return root_left

        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            self.parent[root_left] = root_right
            return root_right
        if rank_right < rank_left:
            self.parent[root_right] = root_left
            return root_left

        self.parent[root_right] = root_left
        self.rank[root_left] += 1
        return root_left


def _node_neighbors(graph: Any, node_index: int) -> set[int]:
    """Return undirected-style neighbors for both directed and undirected graphs."""

    if hasattr(graph, "predecessor_indices") and hasattr(graph, "successor_indices"):
        return set(graph.predecessor_indices(node_index)) | set(
            graph.successor_indices(node_index)
        )
    if hasattr(graph, "neighbors"):
        return set(graph.neighbors(node_index))
    raise TypeError(f"Unsupported graph type: {type(graph).__name__}")


def _graph_edge_endpoints(graph: Any, edge_index: int) -> tuple[int, int]:
    """Return the integer endpoints for an indexed edge."""

    if hasattr(graph, "get_edge_endpoints_by_index"):
        source, target = graph.get_edge_endpoints_by_index(edge_index)
        return int(source), int(target)
    raise TypeError(f"Unsupported graph type: {type(graph).__name__}")


def _graph_edge_data(graph: Any, edge_index: int) -> dict[str, Any]:
    """Return edge metadata as a plain dictionary."""

    if hasattr(graph, "get_edge_data_by_index"):
        return dict(graph.get_edge_data_by_index(edge_index) or {})
    raise TypeError(f"Unsupported graph type: {type(graph).__name__}")


def _source_edge_attributes(graph: Any) -> dict[tuple[int, int], dict[str, Any]]:
    """Index source edge payloads so sweep edges can retain selected attributes."""

    if not hasattr(graph, "edge_indices"):
        return {}
    attributes: dict[tuple[int, int], dict[str, Any]] = {}
    for edge_index in graph.edge_indices():
        source, target = _graph_edge_endpoints(graph, edge_index)
        payload = _graph_edge_data(graph, edge_index)
        attributes[(source, target)] = payload
        attributes.setdefault((target, source), payload)
    return attributes


def _resolve_scalar_graph(graph: Any, scalar_attr: str | None) -> tuple[Any, str]:
    """Resolve a validated Rustworkx graph and its explicit vertex scalar name."""

    if isinstance(graph, ScalarNetwork):
        if scalar_attr is None:
            resolved_graph, scalar_key = graph.graph, graph.scalar_attr
            _validate_merge_tree_domain(resolved_graph)
            return resolved_graph, scalar_key
        if scalar_attr != graph.scalar_attr:
            raise KeyError(
                f"{scalar_attr!r} is not a declared ScalarNetwork vertex scalar attribute."
            )
        resolved_graph = graph.graph
        _validate_merge_tree_domain(resolved_graph)
        return resolved_graph, scalar_attr

    if scalar_attr is None:
        raise ValueError(
            "scalar_attr is required for bare Rustworkx graphs; construct a ScalarNetwork "
            "to declare a default vertex scalar."
        )
    raw_graph = graph.graph if hasattr(graph, "graph") else graph
    _validate_merge_tree_domain(raw_graph)
    return raw_graph, scalar_attr


def _validate_merge_tree_domain(graph: rx.PyGraph | rx.PyDiGraph) -> None:
    """Reject graph topology that has no single merge-tree root."""

    if graph.num_nodes() == 0:
        return
    for edge_index in graph.edge_indices():
        source, target = graph.get_edge_endpoints_by_index(edge_index)
        if source == target:
            raise ValueError("Merge-tree construction does not support self-loops.")
    component_count = (
        rx.number_weakly_connected_components(graph)
        if isinstance(graph, rx.PyDiGraph)
        else rx.number_connected_components(graph)
    )
    if component_count != 1:
        raise ValueError("Merge-tree construction requires a weakly connected graph.")


def _normalize_requested_attrs(attrs: Sequence[str] | None, location: str) -> tuple[str, ...]:
    """Validate unique, nonempty metadata attribute names for propagation."""

    normalized = tuple(attrs or ())
    if any(not isinstance(attr, str) or not attr.strip() for attr in normalized):
        raise TypeError(f"{location} attribute names must be nonempty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{location} attribute names must not be repeated.")
    return normalized


def _report_progress(
    callback: Callable[[int, int], None] | None, completed: int, total: int
) -> None:
    """Report roughly one thousand deterministic progress increments when requested."""

    if callback is None:
        return
    step = max(1, total // 1000)
    if completed == 0 or completed == total or completed % step == 0:
        callback(completed, total)


def build_merge_tree_sweep_trace(
    graph: Any,
    scalar_attr: str | None = None,
    *,
    ascending: bool = True,
    node_attrs: Sequence[str] | None = None,
    edge_attrs: Sequence[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MergeTreeSweepTrace:
    """Build a merge tree together with deterministic construction-sweep tables.

    Parameters mirror :func:`build_merge_tree`.  The trace uses the same
    symbolic node-index ordering for exact scalar ties.  A component is named
    by its birth extremum. At a merge it retains the elder extremum: the lower
    minimum for an ascending join sweep and the higher maximum for a descending
    split sweep, with node index as the tie breaker.


    """

    result = build_merge_tree(
        graph,
        scalar_attr=scalar_attr,
        ascending=ascending,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        progress_callback=progress_callback,
    )
    resolved_graph, scalar_key = _resolve_scalar_graph(graph, scalar_attr=scalar_attr)
    node_indices = sorted(int(node_index) for node_index in resolved_graph.node_indices())
    scalar_values = {
        node_index: float(resolved_graph.get_node_data(node_index)[scalar_key])
        for node_index in node_indices
    }
    ordered_nodes = sorted(
        node_indices,
        key=lambda node_index: (
            scalar_values[node_index] if ascending else -scalar_values[node_index],
            node_index,
        ),
    )
    event_by_vertex = {int(row["vertex_index"]): row for row in result.event_table.to_dicts()}
    graph_edges = _trace_graph_edges(resolved_graph)
    adjacency = {node_index: [] for node_index in node_indices}
    for edge in graph_edges:
        adjacency[edge["source_vertex_index"]].append(edge["target_vertex_index"])
        adjacency[edge["target_vertex_index"]].append(edge["source_vertex_index"])

    tree_node_by_vertex = {
        int(result.tree.get_node_data(tree_node_index)["vertex_index"]): int(tree_node_index)
        for tree_node_index in result.tree.node_indices()
    }
    tree_edges_by_target: dict[int, list[dict[str, int]]] = {}
    for tree_edge_index in result.tree.edge_indices():
        source, target = result.tree.get_edge_endpoints_by_index(tree_edge_index)
        source_vertex = int(result.tree.get_node_data(source)["vertex_index"])
        target_vertex = int(result.tree.get_node_data(target)["vertex_index"])
        tree_edges_by_target.setdefault(target_vertex, []).append(
            {
                "tree_edge_index": int(tree_edge_index),
                "source_tree_node_index": int(source),
                "target_tree_node_index": int(target),
                "source_vertex_index": source_vertex,
                "target_vertex_index": target_vertex,
            }
        )

    union_find = _DisjointSet()
    active_nodes: set[int] = set()
    component_owner: dict[int, int] = {}
    step_rows: list[dict[str, int | float | str]] = []
    component_rows: list[dict[str, int | bool]] = []
    graph_edge_rows: list[dict[str, int]] = []
    tree_edge_rows: list[dict[str, int]] = []

    for step_index, node_index in enumerate(ordered_nodes):
        union_find.add(node_index)
        active_neighbor_roots = {
            union_find.find(neighbor)
            for neighbor in adjacency[node_index]
            if neighbor in active_nodes
        }
        neighbor_owners = sorted(
            (component_owner[root] for root in active_neighbor_roots),
            key=lambda owner: _trace_elder_key(owner, scalar_values, ascending),
        )
        owner_by_active_vertex = {
            active_node: component_owner[union_find.find(active_node)]
            for active_node in active_nodes
        }
        event = event_by_vertex[node_index]
        new_graph_edges = [
            edge
            for edge in graph_edges
            if node_index in (edge["source_vertex_index"], edge["target_vertex_index"])
            and (
                edge["source_vertex_index"] in active_nodes
                or edge["target_vertex_index"] in active_nodes
            )
        ]

        if not active_neighbor_roots:
            component_owner[union_find.find(node_index)] = node_index
            owner = node_index
        else:
            owner = neighbor_owners[0]
            for root in active_neighbor_roots:
                combined_root = union_find.union(node_index, root)
                component_owner[combined_root] = owner
            component_owner[union_find.find(node_index)] = owner

        active_nodes.add(node_index)
        for edge in new_graph_edges:
            graph_edge_rows.append({**edge, "activation_step": step_index})

        for tree_edge in tree_edges_by_target.get(node_index, []):
            tree_edge_rows.append(
                {
                    **tree_edge,
                    "activation_step": step_index,
                    "component_vertex_index": owner_by_active_vertex.get(
                        tree_edge["source_vertex_index"], owner
                    ),
                }
            )

        active_component_count = len(
            {component_owner[union_find.find(active)] for active in active_nodes}
        )
        step_rows.append(
            {
                "step_index": step_index,
                "vertex_index": node_index,
                "scalar": scalar_values[node_index],
                "event": str(event["event"]),
                "active_neighbor_components": int(event["active_neighbor_components"]),
                "new_graph_edge_count": len(new_graph_edges),
                "active_vertex_count": len(active_nodes),
                "active_component_count": active_component_count,
                "component_vertex_index": owner,
            }
        )
        for active_node in sorted(active_nodes):
            component_rows.append(
                {
                    "step_index": step_index,
                    "vertex_index": active_node,
                    "component_vertex_index": component_owner[union_find.find(active_node)],
                    "is_new": active_node == node_index,
                }
            )

    activation_step_by_vertex = {
        vertex_index: step for step, vertex_index in enumerate(ordered_nodes)
    }
    tree_node_rows = [
        {
            "tree_node_index": tree_node_index,
            "vertex_index": vertex_index,
            "activation_step": activation_step_by_vertex[vertex_index],
        }
        for vertex_index, tree_node_index in sorted(tree_node_by_vertex.items())
    ]
    return MergeTreeSweepTrace(
        result=result,
        step_table=pl.DataFrame(step_rows),
        component_table=pl.DataFrame(component_rows),
        graph_edge_table=pl.DataFrame(
            graph_edge_rows,
            schema={
                "graph_edge_index": pl.Int64,
                "source_vertex_index": pl.Int64,
                "target_vertex_index": pl.Int64,
                "activation_step": pl.Int64,
            },
        ).sort("graph_edge_index"),
        tree_node_table=pl.DataFrame(tree_node_rows).sort("tree_node_index"),
        tree_edge_table=pl.DataFrame(
            tree_edge_rows,
            schema={
                "tree_edge_index": pl.Int64,
                "source_tree_node_index": pl.Int64,
                "target_tree_node_index": pl.Int64,
                "source_vertex_index": pl.Int64,
                "target_vertex_index": pl.Int64,
                "activation_step": pl.Int64,
                "component_vertex_index": pl.Int64,
            },
        ).sort("tree_edge_index"),
    )


def _trace_elder_key(
    vertex_index: int, scalar_values: dict[int, float], ascending: bool
) -> tuple[float, int]:
    """Return the deterministic elder-rule ordering key for a sweep component."""

    scalar = scalar_values[vertex_index]
    return (scalar if ascending else -scalar, vertex_index)


def _trace_graph_edges(graph: Any) -> list[dict[str, int]]:
    """Return source graph edges with stable Rustworkx edge indices."""

    return [
        {
            "graph_edge_index": int(edge_index),
            "source_vertex_index": int(graph.get_edge_endpoints_by_index(edge_index)[0]),
            "target_vertex_index": int(graph.get_edge_endpoints_by_index(edge_index)[1]),
        }
        for edge_index in graph.edge_indices()
    ]


def build_merge_tree(
    graph: Any,
    scalar_attr: str | None = None,
    *,
    ascending: bool = True,
    node_attrs: Sequence[str] | None = None,
    edge_attrs: Sequence[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MergeTreeResult:
    """Build an augmented merge tree from an explicit vertex-scalar sweep.

    ``graph`` may be a validated :class:`ScalarNetwork` or a bare Rustworkx
    graph. Bare graphs require ``scalar_attr``; a ScalarNetwork supplies the
    default only when it declares exactly one vertex scalar attribute.

    ``node_attrs`` and ``edge_attrs`` optionally copy named source metadata
    into the result. The topology core does not interpret these attributes.
    ``edge_attrs`` optionally copies named attributes from the source graph edge
    used to create each merge-tree edge.

    With ``ascending=True`` this constructs the sublevel-set join tree; with
    ``ascending=False`` it constructs the decreasing-superlevel split tree.
    A vertex touching zero, one, or multiple active components is recorded as
    an extremum, regular point, or merge saddle, respectively. Edge scalar
    attributes do not participate in this vertex-filtration sweep.

    Exact scalar ties use the Rustworkx node index as a fixed symbolic
    perturbation. This affects only sweep order: stored scalar values and edge
    spans remain unchanged. Directed inputs use weak (orientation-independent)
    adjacency. Disconnected input is rejected by merge-tree domain validation;
    the public ScalarNetwork tree workspace therefore always has one root.


    """

    resolved_graph, scalar_key = _resolve_scalar_graph(graph, scalar_attr=scalar_attr)
    node_indices = [int(node_index) for node_index in resolved_graph.node_indices()]
    if not node_indices:
        raise ValueError("Merge-tree construction requires at least one vertex.")

    scalar_values = {
        node_index: float(resolved_graph.get_node_data(node_index)[scalar_key])
        for node_index in node_indices
    }
    # The node index is an infinitesimal deterministic perturbation for exact
    # scalar ties; it is never added to the user-provided scalar value.
    ordered_nodes = sorted(
        node_indices,
        key=lambda node_index: (
            scalar_values[node_index] if ascending else -scalar_values[node_index],
            int(node_index),
        ),
    )

    union_find = _DisjointSet()
    active_nodes: set[int] = set()
    component_repr: dict[int, int] = {}
    component_front: dict[int, int] = {}
    tree_node_by_graph_node: dict[int, int] = {}
    merge_tree = rx.PyDiGraph()
    merge_events: list[dict[str, int | str]] = []
    requested_node_attrs = _normalize_requested_attrs(node_attrs, "node")
    requested_edge_attrs = _normalize_requested_attrs(edge_attrs, "edge")
    source_edge_data = _source_edge_attributes(resolved_graph) if requested_edge_attrs else {}

    def ensure_tree_node(graph_node: int, event_kind: str) -> int:
        if graph_node in tree_node_by_graph_node:
            return tree_node_by_graph_node[graph_node]
        source_node_data = dict(resolved_graph.get_node_data(graph_node))
        tree_node_by_graph_node[graph_node] = merge_tree.add_node(
            _make_tree_node_payload(
                vertex_index=int(graph_node),
                scalar=float(scalar_values[graph_node]),
                event=event_kind,
                source_node_data=source_node_data,
                node_attrs=requested_node_attrs,
            )
        )
        return tree_node_by_graph_node[graph_node]

    def add_augmented_edge(source_graph_node: int, target_graph_node: int) -> None:
        source_tree_node = tree_node_by_graph_node[source_graph_node]
        target_tree_node = tree_node_by_graph_node[target_graph_node]
        source_scalar = float(scalar_values[source_graph_node])
        target_scalar = float(scalar_values[target_graph_node])
        edge_payload = _make_merge_tree_edge_payload(
            source_scalar=source_scalar,
            target_scalar=target_scalar,
            merge_scalar=target_scalar,
            from_repr=int(source_graph_node),
            to_saddle=int(target_graph_node),
        )
        source_payload = source_edge_data.get((source_graph_node, target_graph_node), {})
        for attr in requested_edge_attrs:
            if attr in source_payload:
                edge_payload[attr] = source_payload[attr]
        merge_tree.add_edge(
            source_tree_node,
            target_tree_node,
            edge_payload,
        )

    extremum_label = "minimum" if ascending else "maximum"
    saddle_label = "join_saddle" if ascending else "split_saddle"

    _report_progress(progress_callback, 0, len(ordered_nodes))
    for position, node_index in enumerate(ordered_nodes, start=1):
        union_find.add(node_index)
        active_nodes.add(node_index)

        active_neighbor_roots = {
            union_find.find(neighbor)
            for neighbor in _node_neighbors(resolved_graph, node_index)
            if neighbor in active_nodes
        }

        if not active_neighbor_roots:
            component_root = union_find.find(node_index)
            component_repr[component_root] = node_index
            component_front[component_root] = node_index
            ensure_tree_node(node_index, extremum_label)
            merge_events.append(
                {
                    "vertex_index": int(node_index),
                    "event": extremum_label,
                    "active_neighbor_components": 0,
                }
            )
            _report_progress(progress_callback, position, len(ordered_nodes))
            continue

        if len(active_neighbor_roots) == 1:
            root = next(iter(active_neighbor_roots))
            ensure_tree_node(node_index, "regular")
            source_graph_node = component_front.get(root, component_repr.get(root, root))
            add_augmented_edge(int(source_graph_node), int(node_index))
            new_root = union_find.union(node_index, root)
            component_repr[new_root] = component_repr.get(root, root)
            component_front[new_root] = node_index
            merge_events.append(
                {
                    "vertex_index": int(node_index),
                    "event": "regular",
                    "active_neighbor_components": 1,
                }
            )
            _report_progress(progress_callback, position, len(ordered_nodes))
            continue

        ensure_tree_node(node_index, saddle_label)
        roots_sorted = sorted(
            active_neighbor_roots,
            key=lambda root: (
                scalar_values[component_repr.get(root, root)],
                int(component_repr.get(root, root)),
            ),
        )

        survivor = roots_sorted[0]
        for root in roots_sorted:
            source_graph_node = component_front.get(root, component_repr.get(root, root))
            add_augmented_edge(int(source_graph_node), int(node_index))
            if root != survivor:
                survivor = union_find.union(survivor, root)

        survivor = union_find.union(node_index, survivor)
        component_repr[survivor] = node_index
        component_front[survivor] = node_index
        merge_events.append(
            {
                "vertex_index": int(node_index),
                "event": saddle_label,
                "active_neighbor_components": len(active_neighbor_roots),
            }
        )

        _report_progress(progress_callback, position, len(ordered_nodes))

    event_table = pl.DataFrame(merge_events)
    return MergeTreeResult(
        tree=merge_tree,
        event_table=event_table,
        scalar_key=scalar_key,
        ascending=ascending,
    )


def build_join_tree(
    graph: Any,
    scalar_attr: str | None = None,
    *,
    node_attrs: Sequence[str] | None = None,
    edge_attrs: Sequence[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MergeTreeResult:
    """Build the augmented join tree of the sublevel filtration.

    Vertices are swept from low to high scalar value. Components are born at
    minima and meet at join saddles.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.topology import build_join_tree
    >>> network = build_cube_scalar_network()
    >>> result = build_join_tree(network.graph, scalar_attr="height")
    >>> result.tree.num_nodes()
    8
    """

    return build_merge_tree(
        graph,
        scalar_attr=scalar_attr,
        ascending=True,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        progress_callback=progress_callback,
    )


def build_split_tree(
    graph: Any,
    scalar_attr: str | None = None,
    *,
    node_attrs: Sequence[str] | None = None,
    edge_attrs: Sequence[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MergeTreeResult:
    """Build the augmented split tree of the superlevel filtration.

    Vertices are swept from high to low scalar value. Components are born at
    maxima and meet at split saddles; reversing the viewpoint gives the usual
    splitting interpretation.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.topology import build_split_tree
    >>> network = build_cube_scalar_network()
    >>> result = build_split_tree(network.graph, scalar_attr="height")
    >>> result.tree.num_nodes()
    8
    """

    return build_merge_tree(
        graph,
        scalar_attr=scalar_attr,
        ascending=False,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        progress_callback=progress_callback,
    )
