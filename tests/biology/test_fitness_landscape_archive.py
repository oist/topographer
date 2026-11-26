"""Round trip of FitnessLandscape.write and FitnessLandscape.read."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from topographer.biology import Alphabet, FitnessLandscape
from topographer.graph import Graph, HammingGraph, ScalarNetwork


def make_landscape() -> FitnessLandscape:
    alphabet = Alphabet(("0", "1"))
    table = pl.DataFrame(
        {
            "seq": ["00", "01", "10", "11"],
            "fitness": [1.0, -2.0, 3.0, 0.5],
            "se": [0.1, 0.1, 0.2, 0.2],
        }
    )
    graph = HammingGraph(
        table, sequence_col="seq", alphabet=alphabet, vertex_attribute_cols=("fitness", "se")
    )
    return FitnessLandscape(graph, scalar_attr="fitness")


def test_landscape_round_trip(tmp_path: Path) -> None:
    original = make_landscape()
    original.write(tmp_path / "fl")

    restored = FitnessLandscape.read(tmp_path / "fl")

    assert type(restored) is FitnessLandscape
    assert restored.alphabet.symbols == original.alphabet.symbols
    assert restored.sequence_attr == original.sequence_attr
    assert restored.vertex_ids == original.vertex_ids
    assert restored.scalar_attr == original.scalar_attr
    assert restored.graph.edge_list() == original.graph.edge_list()
    assert restored.roughness_to_slope_ratio() == original.roughness_to_slope_ratio()
    assert restored.sign_epistasis_fraction() == original.sign_epistasis_fraction()


def test_manifest_records_landscape_extras(tmp_path: Path) -> None:
    make_landscape().write(tmp_path / "fl")

    manifest = json.loads((tmp_path / "fl" / "manifest.json").read_text())
    assert manifest["network_class"] == "FitnessLandscape"
    assert manifest["fitness_landscape"] == {"sequence_attr": "sequence", "alphabet": ["0", "1"]}


def test_plain_read_of_a_landscape_folder_returns_a_scalar_network(tmp_path: Path) -> None:
    original = make_landscape()
    original.write(tmp_path / "fl")

    restored = ScalarNetwork.read(tmp_path / "fl")

    assert type(restored) is ScalarNetwork
    assert restored.vertex_ids == original.vertex_ids


def test_landscape_read_of_a_plain_network_folder_raises(tmp_path: Path) -> None:
    graph = Graph.from_tables(
        pl.DataFrame({"vid": ["a", "b"], "value": [0.1, 0.2]}),
        pl.DataFrame({"src": ["a"], "dst": ["b"]}),
        vertex_id_col="vid",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=["value"],
    )
    ScalarNetwork(graph, scalar_attr="value").write(tmp_path / "net")

    with pytest.raises(ValueError, match="fitness landscape"):
        FitnessLandscape.read(tmp_path / "net")


def test_from_scalar_network_keeps_identity_names() -> None:
    landscape = make_landscape()
    rebuilt = FitnessLandscape.from_scalar_network(
        landscape, sequence_attr="sequence", alphabet=landscape.alphabet
    )

    assert rebuilt.vertex_id_attr == landscape.vertex_id_attr
    assert rebuilt.edge_id_attr == landscape.edge_id_attr


def test_from_scalar_network_keeps_declared_edge_identities() -> None:
    network = ScalarNetwork(
        Graph.from_tables(
            pl.DataFrame({"vid": ["00", "01"], "value": [0.1, 0.2]}),
            pl.DataFrame({"s": ["00", "00"], "t": ["01", "01"], "k": ["x", "y"]}),
            vertex_id_col="vid",
            source_id_col="s",
            target_id_col="t",
            edge_id_col="k",
            vertex_attribute_cols=["value"],
            multigraph=True,
        ),
        scalar_attr="value",
    )

    rebuilt = FitnessLandscape.from_scalar_network(
        network, sequence_attr="vid", alphabet=Alphabet(("0", "1"))
    )

    assert (rebuilt.vertex_id_attr, rebuilt.edge_id_attr) == ("vid", "k")
