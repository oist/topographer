from __future__ import annotations

import polars as pl
import pytest
import rustworkx as rx

from topographer.biology import annotate_stop_codons, build_tree_lineage_metrics, has_stop_codon
from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("ADL", False),
        ("A*L", True),
        ("A%L", True),
        ("A^L", True),
        ("A&L", True),
        ("*%&^", True),
        (None, False),
    ],
)
def test_has_stop_codon(sequence: str | None, expected: bool) -> None:
    assert has_stop_codon(sequence) is expected


def test_annotate_stop_codons_adds_boolean_column_without_mutating_input() -> None:
    source = pl.DataFrame({"sequence_id": ["a", "b", "c"], "aa": ["ADL", "A*L", None]})

    result = annotate_stop_codons(source, aa_column="aa")

    assert source.columns == ["sequence_id", "aa"]
    assert result["sequence_id"].to_list() == ["a", "b", "c"]
    assert result["has_stop_codon"].to_list() == [False, True, False]
    assert result.schema["has_stop_codon"] == pl.Boolean


def test_annotate_stop_codons_rejects_missing_sequence_column() -> None:
    with pytest.raises(KeyError, match="aa"):
        annotate_stop_codons(pl.DataFrame({"sequence": ["A*L"]}), aa_column="aa")


def test_lineage_metrics_use_public_ids_declared_scalar_and_mapping() -> None:
    vertices = pl.DataFrame(
        {
            "variant": [10, 20, 30, 35, 40],
            "fitness": [0.9, 0.5, 0.8, 0.3, 0.1],
            "dna_sequence": ["AAA", "AAT", "ATT", "TTT", "TTA"],
            "aa_sequence": ["**K", "NKK", "*II", "K*K", "KKK"],
        }
    )
    edges = pl.DataFrame(
        {
            "source": [10, 20, 30, 35],
            "target": [20, 35, 20, 40],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="variant",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "dna_sequence", "aa_sequence"),
    )
    network = ScalarNetwork(graph, scalar_attr="fitness")
    tree = ScalarNetworkTopology(network).split_tree()

    result = build_tree_lineage_metrics(network, tree)

    ordinary = result.persistence_table.filter(pl.col("pair_kind") == "ordinary")
    extended = result.persistence_table.filter(pl.col("pair_kind") == "extended")
    assert ordinary["lineage_stop_codon_count"].to_list() == [1]
    assert ordinary["birth_vertex_id"].to_list() == [30]
    assert extended["lineage_vertex_count"].to_list() == [5]
    assert extended["lineage_stop_codon_count"].to_list() == [3]
    assert extended["birth_vertex_id"].to_list() == [10]
    assert set(result.critical_node_table["vertex_id"].to_list()) == {10, 20, 30, 40}
    rows = {row["source_vertex_id"]: row for row in result.branch_table.to_dicts()}
    assert set(rows) == {10, 20, 30}
    assert rows[20]["branch_vertex_count"] == 2
    assert rows[20]["branch_stop_codon_count"] == 1
    assert rows[20]["lineage_vertex_count"] == 4
    assert rows[20]["lineage_stop_codon_count"] == 3
    for leaf in (10, 30):
        assert rows[leaf]["branch_vertex_count"] == 1
        assert rows[leaf]["branch_stop_codon_count"] == 1
    assert result.branch_table.columns[:9] == [
        "source_tree_node_index",
        "target_tree_node_index",
        "source_vertex_id",
        "target_vertex_id",
        "branch_vertex_count",
        "branch_stop_codon_count",
        "lineage_vertex_count",
        "lineage_stop_codon_count",
        "has_stop_codon",
    ]
    labeled = build_tree_lineage_metrics(network, tree, label_attr="fitness")
    assert set(labeled.persistence_table["birth_label"].to_list()) == {"0.8", "0.9"}


def test_lineage_metrics_reject_network_with_different_declared_scalar() -> None:
    vertices = pl.DataFrame(
        {
            "id": [1, 2],
            "height": [0.0, 1.0],
            "other": [0.0, 1.0],
            "dna_sequence": ["AAA", "AAT"],
            "aa_sequence": ["KKK", "K*K"],
        }
    )
    edges = pl.DataFrame({"source": [1], "target": [2]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height", "other", "dna_sequence", "aa_sequence"),
    )
    network = ScalarNetwork(graph, scalar_attr="height")
    tree = ScalarNetworkTopology(ScalarNetwork(graph, scalar_attr="other")).join_tree()

    with pytest.raises(ValueError, match="does not match network scalar"):
        build_tree_lineage_metrics(network, tree)


def test_lineage_metrics_reject_same_scalar_tree_with_different_vertex_ids() -> None:
    def make_network(vertex_ids: list[str]) -> ScalarNetwork:
        vertices = pl.DataFrame(
            {
                "id": vertex_ids,
                "fitness": [0.0, 1.0],
                "dna_sequence": ["AA", "AT"],
                "aa_sequence": ["K*", "KK"],
            }
        )
        edges = pl.DataFrame({"source": [vertex_ids[0]], "target": [vertex_ids[1]]})
        graph = Graph.from_tables(
            vertices,
            edges,
            vertex_id_col="id",
            source_id_col="source",
            target_id_col="target",
            vertex_attribute_cols=("fitness", "dna_sequence", "aa_sequence"),
        )
        return ScalarNetwork(graph, scalar_attr="fitness")

    original_network = make_network(["a", "b"])
    tree = ScalarNetworkTopology(original_network).join_tree()
    different_network = make_network(["x", "y"])

    with pytest.raises(ValueError, match="vertex identity.*does not match"):
        build_tree_lineage_metrics(different_network, tree)


def test_lineage_metrics_preserve_tuple_vertex_ids() -> None:
    graph = rx.PyGraph()
    first = graph.add_node(
        {"vertex": ("variant", 2), "fitness": 0.0, "dna_sequence": "AA", "aa_sequence": "K*"}
    )
    second = graph.add_node(
        {"vertex": ("variant", 9), "fitness": 1.0, "dna_sequence": "AT", "aa_sequence": "KK"}
    )
    graph.add_edge(first, second, {})
    source_graph = Graph.from_rustworkx(
        graph,
        vertex_id_attr="vertex",
        vertex_attribute_cols=("fitness", "dna_sequence", "aa_sequence"),
    )
    network = ScalarNetwork(source_graph, scalar_attr="fitness")
    tree = ScalarNetworkTopology(network).join_tree()

    result = build_tree_lineage_metrics(network, tree)

    assert set(result.critical_node_table["vertex_id"].to_list()) == {
        ("variant", 2),
        ("variant", 9),
    }


def _four_vertex_network(*, dna: list[str] | None = None) -> ScalarNetwork:
    """Network e(1.0) and d(0.9) joining at s(0.5), then r(0.1) below s."""

    dna = dna or ["AAA", "TTT", "AAT", "ATA"]
    vertices = pl.DataFrame(
        {
            "id": ["e", "d", "s", "r"],
            "fitness": [1.0, 0.9, 0.5, 0.1],
            "dna_sequence": dna,
            "aa_sequence": ["K", "*", "N", "K"],
        }
    )
    edges = pl.DataFrame({"source": ["e", "d", "s"], "target": ["s", "s", "r"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "dna_sequence", "aa_sequence"),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def test_lineage_metrics_capture_pre_merge_and_extended_aggregates() -> None:
    network = _four_vertex_network()
    tree = ScalarNetworkTopology(network).split_tree()

    result = build_tree_lineage_metrics(network, tree)
    ordinary = result.persistence_table.filter(pl.col("pair_kind") == "ordinary").row(
        0, named=True
    )
    extended = result.persistence_table.filter(pl.col("pair_kind") == "extended").row(
        0, named=True
    )
    saddle_row = next(
        row for row in result.critical_node_table.to_dicts() if row["vertex_id"] == "s"
    )
    trunk = next(row for row in result.branch_table.to_dicts() if row["source_vertex_id"] == "s")

    assert ordinary["lineage_vertex_count"] == 1
    assert ordinary["lineage_stop_codon_count"] == 1
    assert ordinary["dna_hamming"] == 2
    assert ordinary["aa_hamming"] == 1
    assert ordinary["synonymous_endpoints"] == 0
    assert ordinary["horton_strahler_order"] == 1
    assert saddle_row["lineage_vertex_count"] == 3
    assert saddle_row["horton_strahler_order"] == 2
    assert trunk["lineage_vertex_count"] == 3
    assert extended["lineage_vertex_count"] == 4
    assert extended["lineage_stop_codon_count"] == 1
    assert extended["dna_hamming"] == 1
    assert extended["synonymous_endpoints"] == 1
    assert extended["horton_strahler_order"] == 2
    assert extended["dna_entropy_bits"] > 0.0
    assert extended["aa_entropy_bits"] > 0.0


def test_lineage_metrics_require_aligned_sequence_payloads() -> None:
    network = _four_vertex_network(dna=["AAA", "TTT", "AAT", "AT"])
    tree = ScalarNetworkTopology(network).split_tree()

    with pytest.raises(ValueError, match="dna_sequence.*same length"):
        build_tree_lineage_metrics(network, tree)


def test_parse_vertex_chain_json_normalizes_inputs() -> None:
    from topographer.biology import parse_vertex_chain_json

    assert parse_vertex_chain_json(None) == []
    assert parse_vertex_chain_json('[{"label": "A"}]') == [{"label": "A"}]
    original = [{"label": "B"}]
    copied = parse_vertex_chain_json(original)
    assert copied == original and copied[0] is not original[0]
    with pytest.raises(TypeError):
        parse_vertex_chain_json('{"label": "A"}')
