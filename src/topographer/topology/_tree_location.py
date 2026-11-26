"""Branch membership and critical node location tables for reduced merge trees."""

from __future__ import annotations

from dataclasses import dataclass
import json

import polars as pl
import rustworkx as rx

__all__ = [
    "TreeLocationTables",
    "build_tree_location_tables",
]


@dataclass(frozen=True, slots=True)
class TreeLocationTables:
    """Normalized partition of all represented vertices in an unsimplified tree."""

    branch_member_table: pl.DataFrame
    critical_node_table: pl.DataFrame


def build_tree_location_tables(tree: rx.PyDiGraph) -> TreeLocationTables:
    """Return normalized location tables partitioning regular and critical vertices.

    For a reduced, unsimplified merge tree, regular vertices contracted into each
    edge are exposed in ``branch_member_table`` with their branch lane endpoints
    and vertical scalar coordinates. Critical vertices (extrema and saddles)
    are exposed in ``critical_node_table``.

    Parameters
    ----------
    tree
        A reduced, unsimplified merge tree whose edges carry ``vertex_chain_json``.

    Returns
    -------
    TreeLocationTables
        Container holding ``branch_member_table`` and ``critical_node_table``.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.topology import (
    ...     build_join_tree,
    ...     build_tree_location_tables,
    ...     reduce_merge_tree,
    ... )
    >>> network = build_cube_scalar_network()
    >>> reduced = reduce_merge_tree(build_join_tree(network.graph, scalar_attr="height").tree)
    >>> build_tree_location_tables(reduced).critical_node_table.height
    4
    """

    branch_rows: list[dict[str, object]] = []
    edges = sorted((int(s), int(t)) for s, t in tree.edge_list())

    for source, target in edges:
        edge_data = dict(tree.get_edge_data(source, target) or {})
        chain_json = edge_data.get("vertex_chain_json")
        if not chain_json:
            continue
        try:
            chain = json.loads(chain_json)
        except (ValueError, TypeError):
            continue

        if not isinstance(chain, list) or len(chain) < 3:
            continue

        # chain[0] is source critical node, chain[-1] is target critical node.
        # Intermediate elements chain[1:-1] are the regular contracted vertices.
        for node in chain[1:-1]:
            if not isinstance(node, dict):
                continue
            vertex_index = node.get("vertex_index")
            scalar = node.get("scalar")
            if vertex_index is None or scalar is None:
                continue

            row: dict[str, object] = {
                "source_tree_node_index": source,
                "target_tree_node_index": target,
                "vertex_index": int(vertex_index),
                "scalar": float(scalar),
            }
            for key, value in node.items():
                if key not in {
                    "source_tree_node_index",
                    "target_tree_node_index",
                    "vertex_index",
                    "scalar",
                    "event",
                    "type",
                }:
                    row[key] = value
            branch_rows.append(row)

    critical_rows: list[dict[str, object]] = []
    for node_index in sorted(int(idx) for idx in tree.node_indices()):
        node_data = dict(tree.get_node_data(node_index) or {})
        vertex_index = node_data.get("vertex_index", node_index)
        scalar = node_data.get("scalar", 0.0)
        event = node_data.get("event", node_data.get("type", "regular"))

        row = {
            "tree_node_index": node_index,
            "vertex_index": int(vertex_index),
            "scalar": float(scalar),
            "event": str(event),
        }
        for key, value in node_data.items():
            if key not in {
                "tree_node_index",
                "vertex_index",
                "scalar",
                "event",
                "type",
            }:
                row[key] = value
        critical_rows.append(row)

    branch_table = (
        pl.DataFrame(branch_rows)
        if branch_rows
        else pl.DataFrame(
            schema={
                "source_tree_node_index": pl.Int64,
                "target_tree_node_index": pl.Int64,
                "vertex_index": pl.Int64,
                "scalar": pl.Float64,
            }
        )
    )
    critical_table = (
        pl.DataFrame(critical_rows)
        if critical_rows
        else pl.DataFrame(
            schema={
                "tree_node_index": pl.Int64,
                "vertex_index": pl.Int64,
                "scalar": pl.Float64,
                "event": pl.String,
            }
        )
    )

    if not branch_table.is_empty():
        sort_cols = [
            c
            for c in ["source_tree_node_index", "target_tree_node_index", "scalar", "vertex_index"]
            if c in branch_table.columns
        ]
        branch_table = branch_table.sort(sort_cols)

    if not critical_table.is_empty():
        critical_table = critical_table.sort("tree_node_index")

    return TreeLocationTables(
        branch_member_table=branch_table,
        critical_node_table=critical_table,
    )
