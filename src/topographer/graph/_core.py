"""Core Graph implementation wrapping Rustworkx graph by composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl
import rustworkx as rx

from topographer.graph._adapters import (
    from_igraph,
    from_networkx,
    from_rustworkx,
    from_tables,
)
from topographer.graph._payloads import _copy_payload


class Graph:
    """Concrete Graph class wrapping an internal Rustworkx graph by composition.

    Examples
    --------
    >>> from topographer.graph import Graph
    >>> issubclass(Graph, object)
    True
    """

    def __init__(
        self,
        rx_graph: rx.PyGraph | rx.PyDiGraph,
        *,
        vertex_ids: Sequence[Any],
        vertex_id_to_index: dict[Any, int],
        index_to_vertex_id: dict[int, Any],
        vertex_attr_names: set[str],
        edge_attr_names: set[str],
        vertex_id_attr: str = "id",
        edge_ids: Sequence[Any] | None = None,
        edge_id_to_index: dict[Any, int] | None = None,
        index_to_edge_id: dict[int, Any] | None = None,
        edge_id_attr: str | None = None,
    ) -> None:
        self._graph = rx_graph
        self._vertex_ids = list(vertex_ids)
        self._vertex_id_to_index = dict(vertex_id_to_index)
        self._index_to_vertex_id = dict(index_to_vertex_id)
        self._vertex_id_attr = str(vertex_id_attr)

        # Validate vertex identities
        if not self._vertex_ids:
            # Empty graph is allowed
            pass
        else:
            first_type = type(self._vertex_ids[0])
            for vid in self._vertex_ids:
                if type(vid) is not first_type:
                    raise TypeError("Mixed vertex identity types are not permitted.")
                try:
                    hash(vid)
                except TypeError:
                    raise TypeError(f"Vertex identity {vid!r} is not hashable.") from None

        # Exclude vertex_id_attr from ordinary vertex_attr_names
        self._vertex_attr_names = set(vertex_attr_names) - {self._vertex_id_attr}

        # Edge state setup
        self._edge_id_attr = str(edge_id_attr) if edge_id_attr is not None else None
        self._edge_attr_names = (
            set(edge_attr_names) - {self._edge_id_attr}
            if self._edge_id_attr is not None
            else set(edge_attr_names) - {"edge_id"}
        )

        num_edges = self._graph.num_edges()
        if edge_ids is not None:
            if len(edge_ids) != num_edges:
                raise ValueError(
                    f"edge_ids length ({len(edge_ids)}) does not match graph edge count ({num_edges})."
                )
            if len(set(edge_ids)) != len(edge_ids):
                raise ValueError("Duplicate edge identities found.")
            self._edge_ids = list(edge_ids)
            self._edge_id_to_index = (
                dict(edge_id_to_index)
                if edge_id_to_index is not None
                else {eid: idx for idx, eid in enumerate(self._edge_ids)}
            )
            self._index_to_edge_id = (
                dict(index_to_edge_id)
                if index_to_edge_id is not None
                else {idx: eid for idx, eid in enumerate(self._edge_ids)}
            )
        else:
            # Simple graph without declared edge identity: derive ordered endpoint pairs
            derived_edge_ids = []
            derived_edge_id_to_index = {}
            derived_index_to_edge_id = {}
            seen_pairs: set[Any] = set()
            is_directed = isinstance(self._graph, rx.PyDiGraph)
            is_multigraph = getattr(self._graph, "multigraph", False)

            # In rustworkx, edge_indices() gives indices
            for e_idx in self._graph.edge_indices():
                u_idx, v_idx = self._graph.get_edge_endpoints_by_index(e_idx)
                u_vid = self._index_to_vertex_id[u_idx]
                v_vid = self._index_to_vertex_id[v_idx]
                if is_directed:
                    pair = (u_vid, v_vid)
                else:
                    pair = (u_vid, v_vid) if u_vid <= v_vid else (v_vid, u_vid)

                if is_multigraph and pair in seen_pairs:
                    raise ValueError(
                        f"Multigraph contains parallel edge between {pair} without explicit edge identities."
                    )
                seen_pairs.add(pair)
                derived_edge_ids.append(pair)
                derived_edge_id_to_index[pair] = e_idx
                derived_index_to_edge_id[e_idx] = pair

            self._edge_ids = derived_edge_ids
            self._edge_id_to_index = derived_edge_id_to_index
            self._index_to_edge_id = derived_index_to_edge_id

    @property
    def directed(self) -> bool:
        """Whether the graph is directed."""
        return isinstance(self._graph, rx.PyDiGraph)

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the graph."""
        return len(self._vertex_ids)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self._graph.num_edges()

    @property
    def vertex_ids(self) -> list[Any]:
        """List of vertex identities in canonical sorted order."""
        return list(self._vertex_ids)

    @property
    def edge_ids(self) -> list[Any]:
        """List of edge identities."""
        return list(self._edge_ids)

    @property
    def vertex_id_attr(self) -> str:
        """Name of the declared vertex identity attribute/column."""
        return self._vertex_id_attr

    @property
    def edge_id_attr(self) -> str | None:
        """Name of the declared edge identity attribute/column, or None if generated."""
        return self._edge_id_attr

    @property
    def vertex_attr_names(self) -> set[str]:
        """Set of ordinary vertex attribute names, excluding vertex identity."""
        return set(self._vertex_attr_names)

    @property
    def edge_attr_names(self) -> set[str]:
        """Set of ordinary edge attribute names, excluding edge identity."""
        return set(self._edge_attr_names)

    @property
    def graph(self) -> rx.PyGraph | rx.PyDiGraph:
        """Return a defensive copy of the internal Rustworkx graph with self-describing payloads."""
        is_directed = isinstance(self._graph, rx.PyDiGraph)
        is_multi = getattr(self._graph, "multigraph", False)
        new_rx: rx.PyGraph | rx.PyDiGraph = (
            rx.PyDiGraph(multigraph=is_multi) if is_directed else rx.PyGraph(multigraph=is_multi)
        )

        old_to_new_idx: dict[int, int] = {}
        # Ensure nodes carry self-describing payloads including vertex identity
        for vid in self._vertex_ids:
            old_idx = self._vertex_id_to_index[vid]
            old_data = self._graph.get_node_data(old_idx)
            payload = _copy_payload(old_data) if isinstance(old_data, dict) else {}
            payload[self._vertex_id_attr] = vid
            new_idx = new_rx.add_node(payload)
            old_to_new_idx[old_idx] = new_idx

        # Edges
        edge_id_key = self._edge_id_attr if self._edge_id_attr is not None else "edge_id"
        for e_idx in self._graph.edge_indices():
            u_old, v_old = self._graph.get_edge_endpoints_by_index(e_idx)
            old_data = self._graph.get_edge_data_by_index(e_idx)
            payload = _copy_payload(old_data) if isinstance(old_data, dict) else {}
            eid = self._index_to_edge_id[e_idx]
            payload[edge_id_key] = eid
            new_rx.add_edge(old_to_new_idx[u_old], old_to_new_idx[v_old], payload)

        return new_rx

    def neighbors(self, vertex_id: Any) -> list[Any]:
        """Return canonical-ordered list of neighbor vertex identities."""
        if vertex_id not in self._vertex_id_to_index:
            raise KeyError(f"Vertex identity {vertex_id!r} not found in graph.")
        node_idx = self._vertex_id_to_index[vertex_id]
        nbr_indices = self._graph.neighbors(node_idx)
        nbr_ids = [self._index_to_vertex_id[idx] for idx in nbr_indices]
        return sorted(nbr_ids)

    def connected_components(self) -> list[list[Any]]:
        """Return connected components ordered by decreasing size, then canonical identity."""
        if self.directed:
            raw_comps = rx.weakly_connected_components(self._graph)
        else:
            raw_comps = rx.connected_components(self._graph)

        comps: list[list[Any]] = []
        for comp in raw_comps:
            comp_ids = sorted([self._index_to_vertex_id[idx] for idx in comp])
            comps.append(comp_ids)

        # Sort components by decreasing size (-len), then by the first element's identity
        comps.sort(key=lambda c: (-len(c), c[0] if c else None))
        return comps

    def add_vertex_attributes(
        self,
        table: pl.DataFrame,
        *,
        vertex_id_col: str,
        attribute_cols: Sequence[str] | None = None,
    ) -> None:
        """Add vertex attributes atomically with exact coverage validation."""
        if vertex_id_col not in table.columns:
            raise ValueError(f"Vertex ID column {vertex_id_col!r} not in table.")

        if attribute_cols is None:
            raise ValueError("attribute_cols must be explicitly specified.")

        cols_to_add = list(attribute_cols)
        if not cols_to_add:
            raise ValueError("attribute_cols cannot be empty.")

        for col in cols_to_add:
            if col == vertex_id_col or col == self._vertex_id_attr:
                raise ValueError(
                    f"Identity column {col!r} cannot be selected as an ordinary attribute."
                )
            if col in self._vertex_attr_names:
                raise ValueError(f"Attribute {col!r} already exists on vertices.")
            if col not in table.columns:
                raise ValueError(f"Attribute column {col!r} missing from attribute table.")
            if table[col].null_count() > 0:
                raise ValueError(f"Attribute column {col!r} contains null values.")

        # Check coverage
        table_ids = set(table[vertex_id_col].to_list())
        graph_ids = set(self._vertex_ids)
        if table_ids != graph_ids:
            raise ValueError(
                f"Attribute table IDs do not match graph vertex IDs exactly. "
                f"Missing: {len(graph_ids - table_ids)}, Extra: {len(table_ids - graph_ids)}"
            )

        # Apply atomically to existing nodes
        row_dicts = {
            row[vertex_id_col]: {c: row[c] for c in cols_to_add}
            for row in table.iter_rows(named=True)
        }
        for vid, attrs in row_dicts.items():
            idx = self._vertex_id_to_index[vid]
            payload = self._graph.get_node_data(idx)
            payload.update(attrs)
        self._vertex_attr_names.update(cols_to_add)

    def add_edge_attributes(
        self,
        table: pl.DataFrame,
        *,
        source_id_col: str | None = None,
        target_id_col: str | None = None,
        edge_id_col: str | None = None,
        attribute_cols: Sequence[str] | None = None,
    ) -> None:
        """Add edge attributes atomically with exact coverage validation."""
        if attribute_cols is None:
            raise ValueError("attribute_cols must be explicitly specified.")

        cols_to_add = list(attribute_cols)
        if not cols_to_add:
            raise ValueError("attribute_cols cannot be empty.")

        for col in cols_to_add:
            if col in self._edge_attr_names:
                raise ValueError(f"Attribute {col!r} already exists on edges.")
            if col not in table.columns:
                raise ValueError(f"Attribute column {col!r} missing from table.")
            if table[col].null_count() > 0:
                raise ValueError(f"Attribute column {col!r} contains null values.")

        # Determine matching form
        if edge_id_col is not None:
            if source_id_col is not None or target_id_col is not None:
                raise ValueError(
                    "Specify either edge_id_col or (source_id_col, target_id_col), not both."
                )
            if edge_id_col not in table.columns:
                raise ValueError(f"edge_id_col {edge_id_col!r} not in table.")
            for col in cols_to_add:
                if col == edge_id_col:
                    raise ValueError(
                        f"Edge identity column {col!r} cannot be selected as an attribute."
                    )

            # Validate coverage
            table_eids = table[edge_id_col].to_list()
            if len(set(table_eids)) != len(table_eids):
                raise ValueError("Duplicate edge identities in attribute table.")
            if set(table_eids) != set(self._edge_ids):
                raise ValueError("Attribute table edge IDs do not match graph edge IDs exactly.")

            row_map = {
                row[edge_id_col]: {c: row[c] for c in cols_to_add}
                for row in table.iter_rows(named=True)
            }
            for eid, attrs in row_map.items():
                e_idx = self._edge_id_to_index[eid]
                payload = self._graph.get_edge_data_by_index(e_idx)
                payload.update(attrs)

        else:
            if source_id_col is None or target_id_col is None:
                raise ValueError(
                    "Must specify either edge_id_col or both source_id_col and target_id_col."
                )
            if getattr(self._graph, "multigraph", False):
                raise ValueError(
                    "Multigraph edge attribute updates require matching by edge_id_col."
                )
            if source_id_col not in table.columns or target_id_col not in table.columns:
                raise ValueError("source_id_col or target_id_col not in table.")
            for col in cols_to_add:
                if col in (source_id_col, target_id_col):
                    raise ValueError(
                        f"Endpoint column {col!r} cannot be selected as an attribute."
                    )

            is_directed = self.directed
            # Build endpoint pairs from table
            pairs: list[Any] = []
            row_map = {}
            for row in table.iter_rows(named=True):
                u = row[source_id_col]
                v = row[target_id_col]
                pair = (u, v) if is_directed else ((u, v) if u <= v else (v, u))
                if pair in row_map:
                    raise ValueError(f"Duplicate edge endpoint pair {pair} in attribute table.")
                row_map[pair] = {c: row[c] for c in cols_to_add}
                pairs.append(pair)

            if set(row_map.keys()) != set(self._edge_ids):
                raise ValueError("Attribute table edges do not match graph edges exactly.")

            for pair, attrs in row_map.items():
                e_idx = self._edge_id_to_index[pair]
                payload = self._graph.get_edge_data_by_index(e_idx)
                payload.update(attrs)

        self._edge_attr_names.update(cols_to_add)

    @classmethod
    def from_rustworkx(
        cls,
        graph: rx.PyGraph | rx.PyDiGraph,
        *,
        vertex_id_attr: str,
        edge_id_attr: str | None = None,
        vertex_attribute_cols: Sequence[str] = (),
        edge_attribute_cols: Sequence[str] = (),
    ) -> Graph:
        """Create a Graph from a Rustworkx PyGraph or PyDiGraph."""
        return from_rustworkx(
            cls,
            graph,
            vertex_id_attr=vertex_id_attr,
            edge_id_attr=edge_id_attr,
            vertex_attribute_cols=vertex_attribute_cols,
            edge_attribute_cols=edge_attribute_cols,
        )

    @classmethod
    def from_networkx(
        cls,
        graph: Any,
        *,
        vertex_id_attr: str = "id",
        edge_id_attr: str | None = None,
        vertex_attribute_cols: Sequence[str] = (),
        edge_attribute_cols: Sequence[str] = (),
    ) -> Graph:
        """Create a Graph from a NetworkX Graph or DiGraph."""
        return from_networkx(
            cls,
            graph,
            vertex_id_attr=vertex_id_attr,
            edge_id_attr=edge_id_attr,
            vertex_attribute_cols=vertex_attribute_cols,
            edge_attribute_cols=edge_attribute_cols,
        )

    @classmethod
    def from_igraph(
        cls,
        graph: Any,
        *,
        vertex_id_attr: str,
        edge_id_attr: str | None = None,
        vertex_attribute_cols: Sequence[str] = (),
        edge_attribute_cols: Sequence[str] = (),
    ) -> Graph:
        """Create a Graph from an igraph Graph."""
        return from_igraph(
            cls,
            graph,
            vertex_id_attr=vertex_id_attr,
            edge_id_attr=edge_id_attr,
            vertex_attribute_cols=vertex_attribute_cols,
            edge_attribute_cols=edge_attribute_cols,
        )

    @classmethod
    def from_tables(
        cls,
        vertex_table: pl.DataFrame,
        edge_table: pl.DataFrame,
        *,
        vertex_id_col: str,
        source_id_col: str,
        target_id_col: str,
        edge_id_col: str | None = None,
        vertex_attribute_cols: Sequence[str] = (),
        edge_attribute_cols: Sequence[str] = (),
        directed: bool = False,
        multigraph: bool = False,
    ) -> Graph:
        """Create a Graph from Polars vertex and edge tables."""
        return from_tables(
            cls,
            vertex_table,
            edge_table,
            vertex_id_col=vertex_id_col,
            source_id_col=source_id_col,
            target_id_col=target_id_col,
            edge_id_col=edge_id_col,
            vertex_attribute_cols=vertex_attribute_cols,
            edge_attribute_cols=edge_attribute_cols,
            directed=directed,
            multigraph=multigraph,
        )
