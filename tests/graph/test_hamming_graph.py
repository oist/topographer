"""Tests for topographer.graph.HammingGraph."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.biology import DNA_ALPHABET
from topographer.graph import HammingGraph


def test_hamming_graph_basic() -> None:
    # 2-bit DNA sequences
    df = pl.DataFrame(
        {
            "seq": ["AA", "AC", "CA", "TT"],
            "fitness": [0.1, 0.2, 0.3, 0.4],
        }
    )
    graph = HammingGraph(
        df,
        sequence_col="seq",
        alphabet=DNA_ALPHABET,
        hamming_radius=1,
        vertex_attribute_cols=("fitness",),
    )
    assert graph.vertex_count == 4
    # AA connects to AC (dist 1) and CA (dist 1). TT is isolated (dist 2 from AA).
    assert graph.neighbors("AA") == ["AC", "CA"]
    assert graph.neighbors("TT") == []
    assert graph.vertex_ids == ["AA", "AC", "CA", "TT"]


def test_hamming_graph_radius_2() -> None:
    df = pl.DataFrame(
        {
            "seq": ["AA", "TT"],
        }
    )
    # Radius 1: no edge
    g1 = HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, hamming_radius=1)
    assert g1.edge_count == 0

    # Radius 2: edge exists
    g2 = HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, hamming_radius=2)
    assert g2.edge_count == 1
    assert g2.neighbors("AA") == ["TT"]


def test_hamming_graph_invalid_inputs() -> None:
    # Differing lengths
    df_diff = pl.DataFrame({"seq": ["A", "AA"]})
    with pytest.raises(ValueError, match="common nonzero length"):
        HammingGraph(df_diff, sequence_col="seq", alphabet=DNA_ALPHABET)

    # Undeclared symbols
    df_undec = pl.DataFrame({"seq": ["AX"]})
    with pytest.raises(ValueError, match="undeclared symbols"):
        HammingGraph(df_undec, sequence_col="seq", alphabet=DNA_ALPHABET)

    # Duplicate sequence
    df_dup = pl.DataFrame({"seq": ["AA", "AA"]})
    with pytest.raises(ValueError, match="Duplicate"):
        HammingGraph(df_dup, sequence_col="seq", alphabet=DNA_ALPHABET)

    # Boolean radius rejected
    df = pl.DataFrame({"seq": ["AA", "AC"]})
    with pytest.raises(TypeError, match="integer"):
        HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, hamming_radius=True)  # type: ignore
