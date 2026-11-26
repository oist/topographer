"""Release-surface checks that do not depend on algorithm output."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_archive_helpers_are_not_exported_from_graph() -> None:
    graph = importlib.import_module("topographer.graph")
    for name in (
        "read_manifest",
        "write_manifest",
        "create_folder",
        "read_table",
        "require_keys",
        "make_column",
        "MANIFEST_NAME",
        "MERGE_TREE_FORMAT",
        "WORKSPACE_FORMAT",
    ):
        assert name not in graph.__all__
        assert not hasattr(graph, name)


def test_hamming_example_bundle_is_gone() -> None:
    hamming = importlib.import_module("topographer.examples.hamming")
    examples = importlib.import_module("topographer.examples")
    assert not hasattr(hamming, "HammingExampleBundle")
    assert not hasattr(examples, "HammingExampleBundle")
    assert not hasattr(hamming, "build_hamming_example_bundle")


def test_readme_quickstart_runs() -> None:
    import polars as pl

    from topographer.graph import Graph, ScalarNetwork
    from topographer.topology import ScalarNetworkTopology

    nodes = pl.DataFrame({"id": ["a", "b", "c"], "height": [0.2, 0.9, 0.4]})
    edges = pl.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    graph = Graph.from_tables(
        nodes,
        edges,
        vertex_id_col="id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=["height"],
    )
    network = ScalarNetwork(graph, scalar_attr="height")
    pairs = ScalarNetworkTopology(network).split_tree().persistence()
    assert pairs.height >= 1


def test_topology_modules_do_not_import_biology() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "topographer" / "topology"
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "topographer.biology" not in text, path.name
