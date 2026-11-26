"""MergeTreeMapping for partition of original graph vertices to tree branches and critical vertices."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class MergeTreeMapping:
    """Immutable mapping assigning original graph vertices to branches or critical vertices.

    Examples
    --------
    >>> from topographer.topology import MergeTreeMapping
    >>> issubclass(MergeTreeMapping, object)
    True
    """

    branch_vertices: pl.DataFrame
    critical_vertices: pl.DataFrame

    def __post_init__(self) -> None:
        # Ensure deterministic sort
        if not self.branch_vertices.is_empty():
            sort_cols = [
                c
                for c in [
                    "source_tree_node_index",
                    "target_tree_node_index",
                    "scalar",
                    "vertex_index",
                ]
                if c in self.branch_vertices.columns
            ]
            if sort_cols:
                object.__setattr__(self, "branch_vertices", self.branch_vertices.sort(sort_cols))
        if not self.critical_vertices.is_empty():
            if "tree_node_index" in self.critical_vertices.columns:
                object.__setattr__(
                    self, "critical_vertices", self.critical_vertices.sort("tree_node_index")
                )
