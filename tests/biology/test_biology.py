"""Tests for biology package: Alphabets, FitnessLandscape, and FitnessLandscapeTopology."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.biology import (
    AMINO_ACIDS_20,
    AMINO_ACIDS_21,
    AMINO_ACIDS_23,
    DNA_ALPHABET,
    Alphabet,
    FitnessLandscape,
    FitnessLandscapeTopology,
)
from topographer.graph import HammingGraph


def test_alphabet_invariants() -> None:
    assert len(DNA_ALPHABET) == 4
    assert len(AMINO_ACIDS_20) == 20
    assert len(AMINO_ACIDS_21) == 21
    assert len(AMINO_ACIDS_23) == 23
    assert "%" in AMINO_ACIDS_23
    assert "^" in AMINO_ACIDS_23
    assert "&" in AMINO_ACIDS_23
    assert "*" not in AMINO_ACIDS_23

    # Duplicate symbols rejected
    with pytest.raises(ValueError, match="Duplicate"):
        Alphabet(("A", "A"))


def test_fitness_landscape_construction_and_topology() -> None:
    df = pl.DataFrame(
        {
            "seq": ["AA", "AC"],
            "fitness": [0.1, 0.9],
        }
    )
    hg = HammingGraph(
        df, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fitness",)
    )
    fl = FitnessLandscape(hg, scalar_attr="fitness")
    assert fl.alphabet is DNA_ALPHABET
    assert fl.sequence_attr == "sequence"

    fl_top = FitnessLandscapeTopology(fl)
    join = fl_top.join_tree()
    assert join.kind == "join"
    assert fl_top.landscape is fl
