"""HammingGraph constrained specialization of Graph."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl
import rustworkx as rx

from topographer.graph._core import Graph


def _hamming_distance(s1: str, s2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(s1, s2, strict=True))


class HammingGraph(Graph):
    """Constrained Graph specialization whose vertices are unique sequences.

    Examples
    --------
    >>> from topographer.graph import HammingGraph
    >>> issubclass(HammingGraph, object)
    True
    """

    def __init__(
        self,
        vertex_table: pl.DataFrame,
        *,
        sequence_col: str,
        alphabet: Any,
        hamming_radius: int = 1,
        vertex_attribute_cols: Sequence[str] = (),
    ) -> None:
        if type(hamming_radius) is not int or isinstance(hamming_radius, bool):
            raise TypeError("hamming_radius must be an integer.")
        if hamming_radius < 1:
            raise ValueError("hamming_radius must be positive.")

        if sequence_col not in vertex_table.columns:
            raise ValueError(f"sequence_col {sequence_col!r} not in table.")
        if vertex_table[sequence_col].null_count() > 0:
            raise ValueError("Null values in sequence column.")

        raw_seqs = vertex_table[sequence_col].to_list()
        if len(set(raw_seqs)) != len(raw_seqs):
            raise ValueError("Duplicate sequences found in table.")

        if not raw_seqs:
            raise ValueError("Empty sequence table.")

        # Check common length
        seq_len = len(raw_seqs[0])
        if seq_len == 0:
            raise ValueError("Sequences cannot be empty string.")
        for s in raw_seqs:
            if len(s) != seq_len:
                raise ValueError("All sequences must have one common nonzero length.")
            alphabet.validate_sequence(s)

        # Lexicographical sort by alphabet order or string order
        canonical_vids = sorted(raw_seqs)

        new_rx = rx.PyGraph(multigraph=False)
        vertex_id_to_index: dict[str, int] = {}
        index_to_vertex_id: dict[int, str] = {}

        v_lookup = {row[sequence_col]: row for row in vertex_table.iter_rows(named=True)}

        for seq in canonical_vids:
            row = v_lookup[seq]
            payload = {"sequence": seq}
            for attr in vertex_attribute_cols:
                payload[attr] = row[attr]
            idx = new_rx.add_node(payload)
            vertex_id_to_index[seq] = idx
            index_to_vertex_id[idx] = seq

        # Pairwise Hamming distance <= hamming_radius
        n = len(canonical_vids)
        for i in range(n):
            s1 = canonical_vids[i]
            idx1 = vertex_id_to_index[s1]
            for j in range(i + 1, n):
                s2 = canonical_vids[j]
                d = _hamming_distance(s1, s2)
                if 1 <= d <= hamming_radius:
                    idx2 = vertex_id_to_index[s2]
                    new_rx.add_edge(idx1, idx2, {"distance": d})

        super().__init__(
            new_rx,
            vertex_ids=canonical_vids,
            vertex_id_to_index=vertex_id_to_index,
            index_to_vertex_id=index_to_vertex_id,
            vertex_attr_names=set(vertex_attribute_cols),
            edge_attr_names={"distance"},
            vertex_id_attr="sequence",
        )
        self._alphabet = alphabet
        self._hamming_radius = hamming_radius

    @property
    def alphabet(self) -> Any:
        return self._alphabet

    @property
    def hamming_radius(self) -> int:
        return self._hamming_radius
