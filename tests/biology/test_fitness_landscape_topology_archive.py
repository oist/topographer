"""Workspace round trip of FitnessLandscapeTopology."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from topographer.biology import Alphabet, FitnessLandscape, FitnessLandscapeTopology
from topographer.graph import HammingGraph, ScalarNetwork
from topographer.statistics import bootstrap_persistence_band
from topographer.topology import ScalarNetworkTopology


def make_landscape() -> FitnessLandscape:
    alphabet = Alphabet(("0", "1"))
    table = pl.DataFrame(
        {
            "seq": ["000", "001", "010", "011", "100", "101", "110", "111"],
            "fitness": [0.5, 0.0, 0.6, 1.0, 0.1, 0.7, 0.9, 0.8],
            "se": [0.05, 0.02, 0.08, 0.03, 0.06, 0.04, 0.07, 0.01],
        }
    )
    graph = HammingGraph(
        table, sequence_col="seq", alphabet=alphabet, vertex_attribute_cols=("fitness", "se")
    )
    return FitnessLandscape(graph, scalar_attr="fitness")


def make_topology() -> FitnessLandscapeTopology:
    landscape = make_landscape()
    topology = FitnessLandscapeTopology(landscape)
    topology.join_tree().simplify(0.15)
    topology.split_tree()
    topology.set_band(
        bootstrap_persistence_band(
            landscape, filtration="superlevel", n_draws=5, seed=2, se_attr="se"
        )
    )
    return topology


def test_landscape_topology_round_trip_restores_the_subclasses(tmp_path: Path) -> None:
    original = make_topology()
    original.write(tmp_path / "ws")

    restored = FitnessLandscapeTopology.read(tmp_path / "ws")

    assert type(restored) is FitnessLandscapeTopology
    assert type(restored.landscape) is FitnessLandscape
    assert restored.landscape.alphabet.symbols == original.landscape.alphabet.symbols
    assert restored.landscape.sequence_attr == original.landscape.sequence_attr
    assert restored.summary().equals(original.summary())
    assert restored.band == original.band
    assert restored.join_tree().persistence().equals(original.join_tree().persistence())
    assert sorted(restored.join_tree().simplifications) == [0.15]
    assert restored.roughness_to_slope_ratio() == original.roughness_to_slope_ratio()
    assert restored.sign_epistasis_fraction() == original.sign_epistasis_fraction()


def test_base_read_of_a_landscape_workspace_returns_the_plain_classes(tmp_path: Path) -> None:
    original = make_topology()
    original.write(tmp_path / "ws")

    restored = ScalarNetworkTopology.read(tmp_path / "ws")

    assert type(restored) is ScalarNetworkTopology
    assert type(restored.network) is ScalarNetwork
    assert restored.band == original.band
    assert restored.join_tree().persistence().equals(original.join_tree().persistence())


def test_landscape_read_of_a_plain_workspace_is_refused(tmp_path: Path) -> None:
    make_landscape().write(tmp_path / "landscape")
    plain = ScalarNetwork.read(tmp_path / "landscape")
    ScalarNetworkTopology(plain).write(tmp_path / "ws")

    with pytest.raises(ValueError, match="landscape"):
        FitnessLandscapeTopology.read(tmp_path / "ws")
    with pytest.raises(ValueError, match=r"ScalarNetworkTopology\.read"):
        FitnessLandscapeTopology.read(tmp_path / "ws")
    assert type(ScalarNetworkTopology.read(tmp_path / "ws").network) is ScalarNetwork


def test_a_landscape_in_a_base_workspace_reads_back_as_a_landscape_topology(
    tmp_path: Path,
) -> None:
    ScalarNetworkTopology(make_landscape()).write(tmp_path / "ws")

    restored = FitnessLandscapeTopology.read(tmp_path / "ws")

    assert type(restored.landscape) is FitnessLandscape
