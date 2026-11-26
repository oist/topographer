"""SequenceGraph constrained specialization of Graph."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl
import rustworkx as rx

from topographer.graph._adapters import _canonical_vertex_ids
from topographer.graph._core import Graph


class SequenceGraph(Graph):
    """Constrained Graph specialization with variant ID identity and biological sequence.

    Examples
    --------
    >>> from topographer.graph import SequenceGraph
    >>> issubclass(SequenceGraph, object)
    True
    """

    def __init__(
        self,
        vertex_table: pl.DataFrame,
        edge_table: pl.DataFrame,
        *,
        vertex_id_col: str,
        sequence_col: str,
        source_id_col: str,
        target_id_col: str,
        alphabet: Any,
        vertex_attribute_cols: Sequence[str] = (),
        edge_attribute_cols: Sequence[str] = (),
    ) -> None:
        if vertex_id_col not in vertex_table.columns:
            raise ValueError(f"vertex_id_col {vertex_id_col!r} not in vertex_table.")
        if sequence_col not in vertex_table.columns:
            raise ValueError(f"sequence_col {sequence_col!r} not in vertex_table.")
        if source_id_col not in edge_table.columns or target_id_col not in edge_table.columns:
            raise ValueError("source_id_col or target_id_col not in edge_table.")

        # Check nulls and duplicates
        if vertex_table[vertex_id_col].null_count() > 0:
            raise ValueError(f"Nulls in {vertex_id_col!r}.")
        if vertex_table[sequence_col].null_count() > 0:
            raise ValueError(f"Nulls in {sequence_col!r}.")

        raw_vids = vertex_table[vertex_id_col].to_list()
        if len(set(raw_vids)) != len(raw_vids):
            raise ValueError("Duplicate vertex IDs found.")

        raw_seqs = vertex_table[sequence_col].to_list()
        if len(set(raw_seqs)) != len(raw_seqs):
            raise ValueError("Duplicate sequences found.")

        # Alphabet and length validation
        if raw_seqs:
            seq_len = len(raw_seqs[0])
            if seq_len == 0:
                raise ValueError("Sequence cannot be empty.")
            for s in raw_seqs:
                if len(s) != seq_len:
                    raise ValueError("All sequences must have common nonzero length.")
                alphabet.validate_sequence(s)

        canonical_vids = _canonical_vertex_ids(raw_vids)

        new_rx = rx.PyGraph(multigraph=False)
        vertex_id_to_index: dict[Any, int] = {}
        index_to_vertex_id: dict[int, Any] = {}
        seq_to_vid: dict[str, Any] = {}

        v_lookup = {row[vertex_id_col]: row for row in vertex_table.iter_rows(named=True)}

        for vid in canonical_vids:
            row = v_lookup[vid]
            seq = row[sequence_col]
            seq_to_vid[seq] = vid
            payload = {vertex_id_col: vid, sequence_col: seq}
            for attr in vertex_attribute_cols:
                payload[attr] = row[attr]
            idx = new_rx.add_node(payload)
            vertex_id_to_index[vid] = idx
            index_to_vertex_id[idx] = vid

        # Undirected simple adjacency only: reject self loops and multigraph
        seen_pairs: set[tuple[Any, Any]] = set()
        for row in edge_table.iter_rows(named=True):
            src = row[source_id_col]
            dst = row[target_id_col]
            if src == dst:
                raise ValueError(f"Self loops not permitted: ({src}, {dst})")
            if src not in vertex_id_to_index or dst not in vertex_id_to_index:
                raise ValueError(f"Unknown edge endpoint ({src}, {dst})")
            pair = (min(src, dst), max(src, dst))
            if pair in seen_pairs:
                raise ValueError(f"Duplicate undirected edge {pair}")
            seen_pairs.add(pair)
            u_idx = vertex_id_to_index[src]
            v_idx = vertex_id_to_index[dst]
            payload = {attr: row[attr] for attr in edge_attribute_cols}
            new_rx.add_edge(u_idx, v_idx, payload)

        super().__init__(
            new_rx,
            vertex_ids=canonical_vids,
            vertex_id_to_index=vertex_id_to_index,
            index_to_vertex_id=index_to_vertex_id,
            vertex_attr_names=set(vertex_attribute_cols) | {sequence_col},
            edge_attr_names=set(edge_attribute_cols),
            vertex_id_attr=vertex_id_col,
        )
        self._alphabet = alphabet
        self._sequence_col = sequence_col
        self._source_id_col = source_id_col
        self._target_id_col = target_id_col
        self._seq_to_vid = seq_to_vid

    @property
    def alphabet(self) -> Any:
        return self._alphabet

    def vertex_id_for_sequence(self, sequence: str) -> Any:
        """Lookup variant ID for sequence."""
        if sequence not in self._seq_to_vid:
            raise KeyError(f"Sequence {sequence!r} not found in graph.")
        return self._seq_to_vid[sequence]

    def _export_current_tables(
        self,
        retained_vids: set[Any],
        retained_edge_pairs: set[tuple[Any, Any]],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Construct current vertex and edge tables from authoritative graph state."""
        v_rows = []
        for vid in self._vertex_ids:
            if vid in retained_vids:
                idx = self._vertex_id_to_index[vid]
                data = self._graph.get_node_data(idx)
                row = {self._vertex_id_attr: vid, self._sequence_col: data[self._sequence_col]}
                for attr in self.vertex_attr_names:
                    if attr != self._sequence_col:
                        row[attr] = data[attr]
                v_rows.append(row)

        e_rows = []
        for e_idx in self._graph.edge_indices():
            u_idx, v_idx = self._graph.get_edge_endpoints_by_index(e_idx)
            u_vid = self._index_to_vertex_id[u_idx]
            v_vid = self._index_to_vertex_id[v_idx]
            pair = (min(u_vid, v_vid), max(u_vid, v_vid))
            if pair in retained_edge_pairs:
                data = self._graph.get_edge_data_by_index(e_idx)
                row = {self._source_id_col: u_vid, self._target_id_col: v_vid}
                for attr in self.edge_attr_names:
                    row[attr] = data[attr]
                e_rows.append(row)

        v_table = (
            pl.DataFrame(v_rows)
            if v_rows
            else pl.DataFrame(schema={self._vertex_id_attr: pl.Utf8, self._sequence_col: pl.Utf8})
        )
        e_table = (
            pl.DataFrame(e_rows)
            if e_rows
            else pl.DataFrame(schema={self._source_id_col: pl.Utf8, self._target_id_col: pl.Utf8})
        )
        return v_table, e_table

    def induced_subgraph(self, vertex_ids: Sequence[Any]) -> SequenceGraph:
        """Return a new SequenceGraph induced by the given vertex IDs."""
        v_set = set(vertex_ids)
        unknown = v_set - set(self._vertex_ids)
        if unknown:
            raise KeyError(f"Unknown vertex IDs: {unknown}")

        # Retained edges: both endpoints in v_set
        retained_edge_pairs: set[tuple[Any, Any]] = set()
        for e_idx in self._graph.edge_indices():
            u_idx, v_idx = self._graph.get_edge_endpoints_by_index(e_idx)
            u_vid = self._index_to_vertex_id[u_idx]
            v_vid = self._index_to_vertex_id[v_idx]
            if u_vid in v_set and v_vid in v_set:
                retained_edge_pairs.add((min(u_vid, v_vid), max(u_vid, v_vid)))

        v_table, e_table = self._export_current_tables(v_set, retained_edge_pairs)
        v_attrs = [c for c in self.vertex_attr_names if c != self._sequence_col]
        e_attrs = list(self.edge_attr_names)

        return SequenceGraph(
            v_table,
            e_table,
            vertex_id_col=self._vertex_id_attr,
            sequence_col=self._sequence_col,
            source_id_col=self._source_id_col,
            target_id_col=self._target_id_col,
            alphabet=self._alphabet,
            vertex_attribute_cols=v_attrs,
            edge_attribute_cols=e_attrs,
        )

    def incident_subgraph(self, core_vertex_ids: Sequence[Any]) -> SequenceGraph:
        """Return new SequenceGraph retaining edges incident to at least one core vertex."""
        core_set = set(core_vertex_ids)
        unknown = core_set - set(self._vertex_ids)
        if unknown:
            raise KeyError(f"Unknown core vertex IDs: {unknown}")

        # Retained edges: incident to at least one core vertex (core-core and core-fringe)
        # Fringe-fringe edges are excluded!
        retained_edge_pairs: set[tuple[Any, Any]] = set()
        retained_vids: set[Any] = set(core_set)

        for e_idx in self._graph.edge_indices():
            u_idx, v_idx = self._graph.get_edge_endpoints_by_index(e_idx)
            u_vid = self._index_to_vertex_id[u_idx]
            v_vid = self._index_to_vertex_id[v_idx]
            if u_vid in core_set or v_vid in core_set:
                retained_edge_pairs.add((min(u_vid, v_vid), max(u_vid, v_vid)))
                retained_vids.add(u_vid)
                retained_vids.add(v_vid)

        v_table, e_table = self._export_current_tables(retained_vids, retained_edge_pairs)
        v_attrs = [c for c in self.vertex_attr_names if c != self._sequence_col]
        e_attrs = list(self.edge_attr_names)

        return SequenceGraph(
            v_table,
            e_table,
            vertex_id_col=self._vertex_id_attr,
            sequence_col=self._sequence_col,
            source_id_col=self._source_id_col,
            target_id_col=self._target_id_col,
            alphabet=self._alphabet,
            vertex_attribute_cols=v_attrs,
            edge_attribute_cols=e_attrs,
        )

    def largest_connected_component(self) -> SequenceGraph:
        """Return new SequenceGraph for the unique largest connected component."""
        comps = self.connected_components()
        if not comps:
            raise ValueError("Graph is empty.")
        if len(comps) > 1 and len(comps[0]) == len(comps[1]):
            raise ValueError(
                f"Tie for largest connected component among components of size {len(comps[0])}."
            )
        return self.induced_subgraph(comps[0])
