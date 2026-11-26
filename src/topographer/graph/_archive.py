"""Shared on-disk helpers for saveable objects (private).

A saved object is a folder: Parquet tables plus a ``manifest.json`` that is written last, so a
folder without a manifest is an unfinished write and is refused on read. The manifest carries a
``format`` string (one per kind of folder, for example ``"topographer.scalar_network"``) and an
integer ``format_version``; both are checked exactly. No pickles, and no Rustworkx indices are
ever stored as identities: tables hold the native vertex and edge identities.

The helpers are small and independent so that other folder kinds (trees, workspaces) can reuse
them with their own ``format`` string.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
import json
import math
import os
from pathlib import Path
from typing import Any

import polars as pl
import rustworkx as rx

from topographer.graph._adapters import from_tables
from topographer.graph._core import Graph

MANIFEST_NAME = "manifest.json"
VERTICES_TABLE = "vertices.parquet"
EDGES_TABLE = "edges.parquet"
SCALAR_NETWORK_FORMAT = "topographer.scalar_network"
MERGE_TREE_FORMAT = "topographer.merge_tree"
WORKSPACE_FORMAT = "topographer.workspace"

# Which ``read`` to call for each kind of folder; appended to the wrong-format message.
_READ_HINTS = {
    SCALAR_NETWORK_FORMAT: "ScalarNetwork.read",
    MERGE_TREE_FORMAT: "MergeTree.read",
    WORKSPACE_FORMAT: "ScalarNetworkTopology.read (its network is in the 'network' subfolder)",
}

_FORMAT_VERSION = 1
_SCALAR_TYPES = (str, int, float, bool)
_RESERVED_COLUMNS = ("source_id", "target_id")


@dataclass(frozen=True)
class NetworkTables:
    """Vertex and edge tables of a network plus what a manifest needs to describe them."""

    vertices: pl.DataFrame
    edges: pl.DataFrame
    vertex_id_attr: str
    edge_id_attr: str | None
    vertex_attributes: list[str]
    edge_attributes: list[str]
    multigraph: bool

    def manifest_fields(self) -> dict[str, Any]:
        """Return the manifest entries that describe these tables."""
        return {
            "vertex_id_attr": self.vertex_id_attr,
            "edge_id_attr": self.edge_id_attr,
            "multigraph": self.multigraph,
            "vertex_count": self.vertices.height,
            "edge_count": self.edges.height,
            "vertex_attributes": list(self.vertex_attributes),
            "edge_attributes": list(self.edge_attributes),
        }


# --- folder and manifest -------------------------------------------------------------------------


def create_folder(path: str | os.PathLike[str]) -> Path:
    """Create the new folder at ``path``; raise ``FileExistsError`` if it exists."""
    folder = Path(path)
    folder.mkdir(parents=False, exist_ok=False)
    return folder


def write_manifest(folder: Path, format_name: str, fields: dict[str, Any]) -> None:
    """Write ``manifest.json`` (sorted keys, indent 2) with the format, version and ``fields``."""
    manifest = {
        "format": format_name,
        "format_version": _FORMAT_VERSION,
        "topographer_version": _topographer_version(),
        **fields,
    }
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest, sort_keys=True, indent=2))


def read_manifest(path: str | os.PathLike[str], format_name: str) -> dict[str, Any]:
    """Read and check the manifest of the folder at ``path``.

    Raises
    ------
    ValueError
        If the folder or manifest is missing or unreadable, ``format`` is not exactly
        ``format_name`` (the message names the format found), or ``format_version`` is not 1.
    """
    folder = Path(path)
    if not folder.is_dir():
        raise ValueError(f"Folder {str(folder)!r} not found.")
    manifest_path = folder / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            f"Folder {str(folder)!r} has no {MANIFEST_NAME}; it was not written completely."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError as error:
        raise ValueError(f"Cannot read {MANIFEST_NAME} in {str(folder)!r}: {error}") from None
    if not isinstance(manifest, dict):
        raise ValueError(f"{MANIFEST_NAME} in {str(folder)!r} is not a JSON object.")
    found = manifest.get("format")
    if found != format_name:
        message = f"Unsupported folder format {found!r}; expected {format_name!r}."
        if found in _READ_HINTS:
            message += f" This folder was written for {_READ_HINTS[found]}."
        raise ValueError(message)
    found_version = manifest.get("format_version")
    if type(found_version) is not int or found_version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format_version {found_version!r}; expected {_FORMAT_VERSION}."
        )
    return manifest


def require_keys(manifest: dict[str, Any], keys: Sequence[str]) -> None:
    """Raise ``ValueError`` naming the first of ``keys`` that the manifest lacks."""
    for key in keys:
        if key not in manifest:
            raise ValueError(f"{MANIFEST_NAME} is missing the key {key!r}.")


def read_table(folder: Path, name: str, columns: Sequence[str]) -> pl.DataFrame:
    """Read table ``name`` from ``folder`` and require every column in ``columns``."""
    table_path = folder / name
    if not table_path.is_file():
        raise ValueError(f"Table {name!r} not found in {str(folder)!r}.")
    table = pl.read_parquet(table_path)
    for column in columns:
        if column not in table.columns:
            raise ValueError(f"Column {column!r} missing from table {name!r}.")
    return table


def _topographer_version() -> str:
    try:
        return _distribution_version("topographer")
    except PackageNotFoundError:  # pragma: no cover - not installed
        return "unknown"


# --- write-time validation -----------------------------------------------------------------------


def check_identities(values: Sequence[Any], what: str) -> None:
    """Require finite ``str``, ``int``, ``float`` or ``bool`` identities of one exact type.

    ``what`` names the identities in messages (``"vertex"`` or ``"edge"``). Types are compared
    exactly, so ``bool`` never matches ``int``.

    Raises
    ------
    TypeError
        If an identity is not a finite scalar or the types are mixed.
    """
    for value in values:
        if type(value) not in _SCALAR_TYPES or (type(value) is float and not math.isfinite(value)):
            raise TypeError(
                f"Saving requires finite scalar string/integer/float/bool {what} identities; "
                f"got {value!r}."
            )
    types = {type(value) for value in values}
    if len(types) > 1:
        raise TypeError(
            f"Saving requires one {what} identity type; got {sorted(t.__name__ for t in types)}."
        )


def check_column_name(name: str) -> None:
    """Reject names that collide with the reserved edge table columns."""
    if name in _RESERVED_COLUMNS:
        raise TypeError(f"Attribute name {name!r} is reserved for the edge table; rename it.")


def make_column(name: str, values: list[Any], what: str, *, allow_null: bool = False) -> pl.Series:
    """Build a Polars column that stores ``values`` losslessly, or raise ``TypeError``.

    Values must be ``str``, ``int``, ``float`` or ``bool`` of one exact type; anything else
    (including objects that Polars could only store as ``pl.Object``) is refused. With
    ``allow_null`` (attribute columns) ``None`` values are stored as Parquet nulls and the
    type is taken from the non-null values; an all-null column is stored as ``Float64`` nulls
    and restored as all ``None``.
    """
    if allow_null:
        types = {type(value) for value in values if value is not None}
        if not types:
            return pl.Series(name, [None] * len(values), dtype=pl.Float64)
    else:
        types = {type(value) for value in values}
    if len(types) > 1 or not types <= set(_SCALAR_TYPES):
        names = sorted(t.__name__ for t in types)
        raise TypeError(
            f"Cannot save {what} attribute {name!r}: values must be one of str, int, float "
            f"or bool, all of one type; got {names}."
        )
    try:
        series = pl.Series(name, values)
    except (TypeError, OverflowError, pl.exceptions.PolarsError) as error:
        raise TypeError(f"Cannot save {what} attribute {name!r}: {error}") from None
    if series.dtype == pl.Object:
        raise TypeError(f"Cannot save {what} attribute {name!r}: it would be stored as an object.")
    return series


# --- graph to tables and back --------------------------------------------------------------------


def graph_to_tables(
    graph: rx.PyGraph,
    *,
    vertex_ids: Sequence[Any],
    vertex_id_attr: str,
    edge_id_attr: str | None,
    edge_scalar_attr: str | None = None,
) -> NetworkTables:
    """Turn a network's Rustworkx graph into vertex and edge tables with native identities.

    Node ``i`` must be the vertex ``vertex_ids[i]`` (the canonical position), and its payload
    must hold the identity under ``vertex_id_attr``. Vertex rows follow node order; edge rows
    follow edge-index order with ``source_id`` and ``target_id`` taken from ``vertex_ids``. A
    generated ``"edge_id"`` payload entry (no declared ``edge_id_attr``) is not stored.
    Everything is validated before any table is returned.

    Raises
    ------
    TypeError
        For identities that are not finite scalars of one type, attribute columns that are not
        one scalar type, attributes missing on some rows, or reserved attribute names.
    """
    check_identities(vertex_ids, "vertex")
    for name in (vertex_id_attr, edge_id_attr):
        if name is not None:
            check_column_name(name)
    node_payloads = [graph.get_node_data(i) for i in range(len(vertex_ids))]
    edge_indices = list(graph.edge_indices())
    edge_payloads = [graph.get_edge_data_by_index(i) for i in edge_indices]

    skipped = {vertex_id_attr}
    vertex_names = _attribute_names(node_payloads, skipped)
    skipped_edge = {edge_id_attr} if edge_id_attr is not None else {"edge_id"}
    edge_names = _attribute_names(edge_payloads, skipped_edge)
    if edge_scalar_attr is not None and edge_scalar_attr not in edge_names:
        edge_names = sorted([*edge_names, edge_scalar_attr])
    for name in (*vertex_names, *edge_names):
        check_column_name(name)

    vertex_columns = [make_column(vertex_id_attr, list(vertex_ids), "vertex identity")]
    vertex_columns += [
        make_column(name, _column_values(node_payloads, name, "vertex"), "vertex", allow_null=True)
        for name in vertex_names
    ]
    edge_columns = _edge_columns(
        graph, edge_indices, edge_payloads, vertex_ids, edge_id_attr, edge_names, vertex_columns[0]
    )
    return NetworkTables(
        vertices=pl.DataFrame(vertex_columns),
        edges=pl.DataFrame(edge_columns),
        vertex_id_attr=vertex_id_attr,
        edge_id_attr=edge_id_attr,
        vertex_attributes=list(vertex_names),
        edge_attributes=list(edge_names),
        multigraph=bool(getattr(graph, "multigraph", False)),
    )


def _attribute_names(payloads: list[Any], skipped: set[str | None]) -> list[str]:
    names: set[str] = set()
    for payload in payloads:
        names.update(key for key in payload if key not in skipped)
    return sorted(names)


def _column_values(payloads: list[Any], name: str, what: str) -> list[Any]:
    values = []
    for payload in payloads:
        if name not in payload:
            raise TypeError(f"Cannot save {what} attribute {name!r}: missing on some {what}s.")
        values.append(payload[name])
    return values


def _edge_columns(
    graph: rx.PyGraph,
    edge_indices: list[int],
    edge_payloads: list[Any],
    vertex_ids: Sequence[Any],
    edge_id_attr: str | None,
    edge_names: list[str],
    id_column: pl.Series,
) -> list[pl.Series]:
    endpoints = [graph.get_edge_endpoints_by_index(i) for i in edge_indices]
    columns = [
        _endpoint_column("source_id", [vertex_ids[u] for u, _ in endpoints], id_column),
        _endpoint_column("target_id", [vertex_ids[v] for _, v in endpoints], id_column),
    ]
    if edge_id_attr is not None:
        edge_ids = _column_values(edge_payloads, edge_id_attr, "edge")
        check_identities(edge_ids, "edge")
        columns.append(make_column(edge_id_attr, edge_ids, "edge identity"))
    for name in edge_names:
        if edge_payloads:
            columns.append(
                make_column(
                    name, _column_values(edge_payloads, name, "edge"), "edge", allow_null=True
                )
            )
        else:
            columns.append(pl.Series(name, [], dtype=pl.Float64))
    return columns


def _endpoint_column(name: str, values: list[Any], id_column: pl.Series) -> pl.Series:
    if values:
        return make_column(name, values, "edge endpoint")
    return pl.Series(name, [], dtype=id_column.dtype)


def check_counts(manifest: dict[str, Any], vertices: pl.DataFrame, edges: pl.DataFrame) -> None:
    """Require the table heights to equal ``vertex_count`` and ``edge_count`` of the manifest."""
    for key, height, label in (
        ("vertex_count", vertices.height, "vertex"),
        ("edge_count", edges.height, "edge"),
    ):
        if manifest[key] != height:
            raise ValueError(
                f"The {label} count in the manifest ({manifest[key]!r}) does not match the "
                f"{label} table ({height} rows)."
            )


NETWORK_MANIFEST_KEYS = (
    "scalar_attr",
    "edge_scalar_attr",
    "vertex_id_attr",
    "edge_id_attr",
    "multigraph",
    "vertex_count",
    "edge_count",
    "vertex_attributes",
    "edge_attributes",
)


def read_network_graph(folder: Path, manifest: dict[str, Any]) -> Graph:
    """Rebuild the ``Graph`` of a network folder through ``Graph.from_tables``.

    Checks the manifest keys, the tables, their columns and the counts first; everything else
    (unique ids, nulls, endpoints, simple versus multigraph) is validated by ``from_tables``.
    """
    require_keys(manifest, NETWORK_MANIFEST_KEYS)
    id_attr = manifest["vertex_id_attr"]
    edge_id_attr = manifest["edge_id_attr"]
    vertex_attrs = list(manifest["vertex_attributes"])
    edge_attrs = list(manifest["edge_attributes"])
    vertices = read_table(folder, VERTICES_TABLE, [id_attr, *vertex_attrs])
    edge_required = [
        "source_id",
        "target_id",
        *([edge_id_attr] if edge_id_attr else []),
        *edge_attrs,
    ]
    edges = read_table(folder, EDGES_TABLE, edge_required)
    check_counts(manifest, vertices, edges)
    _require_non_null(vertices, [id_attr, manifest["scalar_attr"]], "vertex")
    _require_non_null(
        edges,
        [
            "source_id",
            "target_id",
            *([edge_id_attr] if edge_id_attr else []),
            *([manifest["edge_scalar_attr"]] if manifest["edge_scalar_attr"] else []),
        ],
        "edge",
    )
    return from_tables(
        Graph,
        vertices,
        edges,
        vertex_id_col=id_attr,
        source_id_col="source_id",
        target_id_col="target_id",
        edge_id_col=edge_id_attr,
        vertex_attribute_cols=vertex_attrs,
        edge_attribute_cols=edge_attrs,
        multigraph=bool(manifest["multigraph"]),
        allow_null_attributes=True,
    )


def _require_non_null(table: pl.DataFrame, columns: list[str], what: str) -> None:
    """Require identity, endpoint and declared scalar columns to hold no nulls."""
    for column in columns:
        if column in table.columns and table[column].null_count() > 0:
            raise ValueError(f"The {what} column {column!r} must not contain null values.")
