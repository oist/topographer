"""Folder format of a saved MergeTree (private): write and read without recomputation.

Layout of a folder (format ``"topographer.merge_tree"``, version 1)::

    nodes.parquet            row k is reduced-tree node k: the node payload columns, in payload key order
    edges.parquet            row j is edge j: source_vertex_index, target_vertex_index + payload
    augmented_nodes.parquet  same convention for the augmented tree (all n vertices)
    augmented_edges.parquet
    events.parquet           the event table, in sweep order
    simplifications.parquet  threshold, vertex_index: surviving reduced-tree nodes per threshold
    manifest.json

Rustworkx node indices are never stored: nodes are keyed by ``vertex_index`` (unique within a
tree) and row order rebuilds the same node and edge indices.
"""

from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

import polars as pl
import rustworkx as rx

from topographer.graph._archive import (
    MERGE_TREE_FORMAT,
    create_folder,
    make_column,
    read_manifest,
    read_table,
    require_keys,
    write_manifest,
)

if TYPE_CHECKING:
    from topographer.topology._merge_tree import MergeTree

NODES_TABLE = "nodes.parquet"
EDGES_TABLE = "edges.parquet"
AUGMENTED_NODES_TABLE = "augmented_nodes.parquet"
AUGMENTED_EDGES_TABLE = "augmented_edges.parquet"
EVENTS_TABLE = "events.parquet"
SIMPLIFICATIONS_TABLE = "simplifications.parquet"

_SOURCE = "source_vertex_index"
_TARGET = "target_vertex_index"
_KEY = "vertex_index"
_EVENT_LABELS = {
    "join": {"minimum", "join_saddle", "regular"},
    "split": {"maximum", "split_saddle", "regular"},
}
_MANIFEST_KEYS = (
    "kind",
    "scalar_attr",
    "vertex_count",
    "tree_node_count",
    "simplification_thresholds",
)


# --- write ---------------------------------------------------------------------------------------


def write_tree(tree: MergeTree, path: str | os.PathLike[str]) -> None:
    """Write an original tree and its cached simplifications to a new folder."""
    if tree.original is not None:
        raise ValueError(
            "Cannot write a simplified tree; write the original tree, which stores its "
            "simplifications (tree.original.write(path))."
        )
    nodes, edges = _graph_to_tables(tree._graph, "reduced tree")
    aug_nodes, aug_edges = _graph_to_tables(tree._augmented_graph, "augmented tree")
    thresholds = list(tree._simplifications)
    simplifications = _simplification_table(tree)
    fields = {
        "kind": tree.kind,
        "scalar_attr": tree.scalar_attr,
        "vertex_count": aug_nodes.height,
        "tree_node_count": nodes.height,
        "simplification_thresholds": thresholds,
    }
    folder = create_folder(path)
    try:
        nodes.write_parquet(folder / NODES_TABLE)
        edges.write_parquet(folder / EDGES_TABLE)
        aug_nodes.write_parquet(folder / AUGMENTED_NODES_TABLE)
        aug_edges.write_parquet(folder / AUGMENTED_EDGES_TABLE)
        tree.events.write_parquet(folder / EVENTS_TABLE)
        simplifications.write_parquet(folder / SIMPLIFICATIONS_TABLE)
        write_manifest(folder, MERGE_TREE_FORMAT, fields)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _graph_to_tables(graph: rx.PyDiGraph, what: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Turn a tree graph into node and edge tables keyed by ``vertex_index``."""
    node_indices = list(graph.node_indices())
    if node_indices != list(range(len(node_indices))):
        raise ValueError(f"The {what} node indices are not contiguous.")
    payloads = [graph.get_node_data(i) for i in node_indices]
    keys = [payload.get(_KEY) for payload in payloads]
    if any(type(key) is not int for key in keys) or len(set(keys)) != len(keys):
        raise ValueError(f"The {what} nodes do not have unique integer {_KEY!r} values.")
    node_columns = [
        make_column(name, _values(payloads, name, what), f"{what} node")
        for name in _payload_names(payloads)
    ]

    edge_indices = list(graph.edge_indices())
    endpoints = [graph.get_edge_endpoints_by_index(i) for i in edge_indices]
    edge_payloads = [graph.get_edge_data_by_index(i) for i in edge_indices]
    edge_columns = [
        _int_column(_SOURCE, [keys[u] for u, _ in endpoints]),
        _int_column(_TARGET, [keys[v] for _, v in endpoints]),
    ]
    for name in _payload_names(edge_payloads):
        edge_columns.append(make_column(name, _values(edge_payloads, name, what), f"{what} edge"))
    return pl.DataFrame(node_columns), pl.DataFrame(edge_columns)


def _int_column(name: str, values: list[int]) -> pl.Series:
    return pl.Series(name, values, dtype=pl.Int64)


def _payload_names(payloads: list[dict[str, Any]]) -> list[str]:
    """Payload keys in first-seen order; downstream tables follow the payload key order."""
    names: dict[str, None] = {}
    for payload in payloads:
        names.update(dict.fromkeys(payload))
    return list(names)


def _values(payloads: list[dict[str, Any]], name: str, what: str) -> list[Any]:
    values = []
    for payload in payloads:
        if name not in payload:
            raise TypeError(f"Cannot save the {what}: payload key {name!r} is missing on some.")
        values.append(payload[name])
    return values


def _simplification_table(tree: MergeTree) -> pl.DataFrame:
    thresholds: list[float] = []
    keys: list[int] = []
    for threshold, variant in tree._simplifications.items():
        graph = variant._graph
        kept = [int(graph.get_node_data(i)[_KEY]) for i in graph.node_indices()]
        thresholds.extend([float(threshold)] * len(kept))
        keys.extend(kept)
    return pl.DataFrame(
        [
            pl.Series("threshold", thresholds, dtype=pl.Float64),
            pl.Series(_KEY, keys, dtype=pl.Int64),
        ]
    )


# --- read ----------------------------------------------------------------------------------------


def read_tree(path: str | os.PathLike[str]) -> MergeTree:
    """Read a tree folder; nothing is swept, reduced or re-oriented except simplifications."""
    from topographer.topology._mapping import MergeTreeMapping
    from topographer.topology._merge_tree import _build_merge_tree
    from topographer.topology._tree_location import build_tree_location_tables

    folder = Path(path)
    manifest = read_manifest(folder, MERGE_TREE_FORMAT)
    require_keys(manifest, _MANIFEST_KEYS)
    kind, vertex_count, node_count, thresholds = _manifest_values(manifest)

    nodes = read_table(folder, NODES_TABLE, [_KEY, "event"])
    edges = read_table(folder, EDGES_TABLE, [_SOURCE, _TARGET])
    aug_nodes = read_table(folder, AUGMENTED_NODES_TABLE, [_KEY, "event"])
    aug_edges = read_table(folder, AUGMENTED_EDGES_TABLE, [_SOURCE, _TARGET])
    events = read_table(folder, EVENTS_TABLE, [_KEY, "event", "active_neighbor_components"])
    simplifications = read_table(folder, SIMPLIFICATIONS_TABLE, ["threshold", _KEY])

    if nodes.height != node_count:
        raise ValueError(
            f"The tree_node_count in the manifest ({node_count}) does not match the node table "
            f"({nodes.height} rows)."
        )
    if aug_nodes.height != vertex_count:
        raise ValueError(
            f"The vertex_count in the manifest ({vertex_count}) does not match the augmented "
            f"node table ({aug_nodes.height} rows)."
        )
    graph = _tables_to_graph(nodes, edges, "reduced tree", kind)
    augmented = _tables_to_graph(aug_nodes, aug_edges, "augmented tree", kind)
    _check_vertex_cover(aug_nodes[_KEY].to_list(), vertex_count, "augmented tree")
    _check_vertex_cover(events[_KEY].to_list(), vertex_count, "events")
    if events.height != vertex_count:
        raise ValueError("The events table must have one row per vertex.")
    _check_event_labels(events["event"].to_list(), kind, "events")
    reduced_keys = set(nodes[_KEY].to_list())
    if not reduced_keys <= set(aug_nodes[_KEY].to_list()):
        raise ValueError("Reduced-tree vertices are not all in the augmented tree.")
    groups = _simplification_groups(simplifications, thresholds, reduced_keys)

    loc_tables = build_tree_location_tables(graph)
    mapping = MergeTreeMapping(
        branch_vertices=loc_tables.branch_member_table,
        critical_vertices=loc_tables.critical_node_table,
    )
    tree = _build_merge_tree(
        kind=kind,
        graph=graph,
        augmented_graph=augmented,
        event_table=events,
        mapping=mapping,
        scalar_attr=manifest["scalar_attr"],
    )
    node_by_key = {key: index for index, key in enumerate(nodes[_KEY].to_list())}
    for threshold, keys in groups.items():
        try:
            tree._build_simplification({node_by_key[key] for key in keys}, threshold)
        except ValueError as error:
            raise ValueError(
                f"The stored simplification at threshold {threshold!r} is not a rooted tree: "
                f"{error}"
            ) from None
    return tree


def _manifest_values(manifest: dict[str, Any]) -> tuple[str, int, int, list[float]]:
    kind = manifest["kind"]
    if kind not in _EVENT_LABELS:
        raise ValueError(f"The manifest kind must be 'join' or 'split', got {kind!r}.")
    if not isinstance(manifest["scalar_attr"], str):
        raise ValueError("The manifest scalar_attr must be a string.")
    counts = []
    for key in ("vertex_count", "tree_node_count"):
        value = manifest[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"The manifest {key} must be a positive integer, got {value!r}.")
        counts.append(value)
    thresholds = manifest["simplification_thresholds"]
    if not isinstance(thresholds, list) or any(
        isinstance(t, bool) or not isinstance(t, (int, float)) for t in thresholds
    ):
        raise ValueError("The manifest simplification_thresholds must be a list of numbers.")
    return kind, counts[0], counts[1], [float(t) for t in thresholds]


def _tables_to_graph(
    nodes: pl.DataFrame, edges: pl.DataFrame, what: str, kind: str
) -> rx.PyDiGraph:
    """Rebuild a tree graph in row order and check that it is a rooted tree."""
    for label, table in (("node", nodes), ("edge", edges)):
        for column in table.columns:
            if table[column].null_count() > 0:
                raise ValueError(f"The {what} {label} table has nulls in column {column!r}.")
    keys = nodes[_KEY].to_list()
    if len(set(keys)) != len(keys):
        raise ValueError(f"The {what} node table has duplicate {_KEY!r} values.")
    _check_event_labels(nodes["event"].to_list(), kind, f"{what} nodes")
    graph = rx.PyDiGraph()
    for row in nodes.iter_rows(named=True):
        graph.add_node(dict(row))
    index_by_key = {key: index for index, key in enumerate(keys)}
    payload_columns = [c for c in edges.columns if c not in (_SOURCE, _TARGET)]
    for row in edges.iter_rows(named=True):
        try:
            source, target = index_by_key[row[_SOURCE]], index_by_key[row[_TARGET]]
        except KeyError as error:
            raise ValueError(
                f"The {what} edge table refers to unknown vertex_index {error.args[0]!r}."
            ) from None
        graph.add_edge(source, target, {name: row[name] for name in payload_columns})
    _check_rooted_tree(graph, what)
    return graph


def _check_rooted_tree(graph: rx.PyDiGraph, what: str) -> None:
    """Require ``n - 1`` edges, one root and edges that point from the root to the leaves."""
    count = graph.num_nodes()
    if graph.num_edges() != count - 1:
        raise ValueError(
            f"The {what} must have {count - 1} edges for {count} nodes; found {graph.num_edges()}."
        )
    roots = [i for i in graph.node_indices() if graph.in_degree(i) == 0]
    if len(roots) != 1:
        raise ValueError(f"The {what} must have exactly one root; found {len(roots)}.")
    if any(graph.in_degree(i) != 1 for i in graph.node_indices() if i != roots[0]):
        raise ValueError(f"The {what} edges must point from the root to the leaves.")
    seen = {roots[0]}
    queue = deque(roots)
    while queue:
        for child in graph.successor_indices(queue.popleft()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    if len(seen) != count:
        raise ValueError(f"The {what} is not connected from its root.")


def _check_vertex_cover(keys: list[int], vertex_count: int, what: str) -> None:
    if sorted(keys) != list(range(vertex_count)):
        raise ValueError(
            f"The {what} vertex_index values must cover 0..{vertex_count - 1} exactly once."
        )


def _check_event_labels(labels: list[str], kind: str, what: str) -> None:
    unknown = sorted(set(labels) - _EVENT_LABELS[kind])
    if unknown:
        raise ValueError(f"The {what} hold event labels {unknown} that a {kind} tree cannot have.")


def _simplification_groups(
    table: pl.DataFrame, manifest_thresholds: list[float], reduced_keys: set[int]
) -> dict[float, list[int]]:
    for column in table.columns:
        if table[column].null_count() > 0:
            raise ValueError(f"The simplification table has nulls in column {column!r}.")
    groups: dict[float, list[int]] = {}
    for threshold, key in zip(table["threshold"].to_list(), table[_KEY].to_list(), strict=True):
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"Simplification thresholds must be positive, got {threshold!r}.")
        if key not in reduced_keys:
            raise ValueError(
                f"The simplification at threshold {threshold!r} keeps vertex_index {key!r}, "
                "which is not a node of the reduced tree."
            )
        groups.setdefault(threshold, []).append(key)
    for threshold, keys in groups.items():
        if len(set(keys)) != len(keys):
            raise ValueError(f"The simplification at threshold {threshold!r} repeats a node.")
    if sorted(groups) != sorted(manifest_thresholds) or len(set(manifest_thresholds)) != len(
        manifest_thresholds
    ):
        raise ValueError(
            "The simplification_thresholds in the manifest do not match the simplification "
            f"table: {sorted(manifest_thresholds)} versus {sorted(groups)}."
        )
    return groups
