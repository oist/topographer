from __future__ import annotations

from importlib.metadata import version as distribution_version
import importlib.util
import subprocess
import sys

import pytest

import topographer
import topographer.biology
import topographer.graph
import topographer.plotting
import topographer.statistics
import topographer.statistics.bootstrap
import topographer.topology


def test_root_package_all_exports() -> None:
    expected_all = [
        "Graph",
        "HammingGraph",
        "MergeTree",
        "MergeTreeMapping",
        "PersistenceResult",
        "ScalarNetwork",
        "ScalarNetworkTopology",
        "SequenceGraph",
        "__version__",
    ]
    assert sorted(topographer.__all__) == sorted(expected_all)


def test_root_package_exports_curated_primary_domain_api() -> None:
    assert topographer.__version__ == distribution_version("topographer")
    assert topographer.Graph is topographer.graph.Graph
    assert topographer.HammingGraph is topographer.graph.HammingGraph
    assert topographer.SequenceGraph is topographer.graph.SequenceGraph
    assert topographer.ScalarNetwork is topographer.graph.ScalarNetwork
    assert topographer.MergeTree is topographer.topology.MergeTree
    assert topographer.MergeTreeMapping is topographer.topology.MergeTreeMapping
    assert topographer.PersistenceResult is topographer.topology.PersistenceResult
    assert topographer.ScalarNetworkTopology is topographer.topology.ScalarNetworkTopology


def test_topology_submodule_all_exports() -> None:
    expected_all = [
        "BootstrapPersistenceBand",
        "MergeTree",
        "MergeTreeMapping",
        "PersistenceResult",
        "ScalarNetworkTopology",
        "TreeMetrics",
        "build_join_tree",
        "build_split_tree",
        "build_tree_location_tables",
        "build_tree_metrics",
        "compute_topological_summary",
        "diagram_bottleneck_distance",
        "diagram_wasserstein_distance",
        "reduce_merge_tree",
    ]
    assert sorted(topographer.topology.__all__) == sorted(expected_all)


def test_biology_submodule_all_exports() -> None:
    expected_all = [
        "AMINO_ACIDS_20",
        "AMINO_ACIDS_21",
        "AMINO_ACIDS_23",
        "Alphabet",
        "DNA_ALPHABET",
        "FitnessLandscape",
        "FitnessLandscapeTopology",
        "RNA_ALPHABET",
        "TreeLineageMetrics",
        "annotate_stop_codons",
        "build_tree_lineage_metrics",
        "has_stop_codon",
        "parse_vertex_chain_json",
        "roughness_to_slope_ratio",
        "sign_epistasis_fraction",
        "wildtype_accessibility_fraction",
    ]
    assert sorted(topographer.biology.__all__) == sorted(expected_all)


def test_graph_submodule_all_exports() -> None:
    expected_all = ["Graph", "HammingGraph", "ScalarNetwork", "SequenceGraph"]
    assert sorted(topographer.graph.__all__) == sorted(expected_all)


def test_statistics_submodule_all_exports() -> None:
    expected_all = [
        "BottleneckToNominal",
        "bootstrap",
        "bootstrap_persistence_band",
        "compute_bh_qvalues",
        "validate_se_attr",
    ]
    assert sorted(topographer.statistics.__all__) == sorted(expected_all)


def test_bootstrap_submodule_all_exports() -> None:
    expected_all = [
        "BootstrapChunk",
        "DrawStatistic",
        "DrawnNetwork",
        "combine",
        "fingerprint",
        "run",
    ]
    assert sorted(topographer.statistics.bootstrap.__all__) == sorted(expected_all)


def test_plotting_submodule_all_exports() -> None:
    expected_all = [
        "ColorScaleConfig",
        "PersistencePlotConfig",
        "ReferenceLineConfig",
        "TreeAnnotationLayer",
        "TreeMarkerConfig",
        "TreePlotConfig",
        "TreePlotData",
        "TreePresentationConfig",
        "build_scalar_plot_domain",
        "build_tree_plot_data",
        "compute_branch_local_swarm",
        "compute_fitness_histogram_table",
        "count_color_scale",
        "make_tree_layout",
        "plot_branch_span",
        "plot_branch_spans",
        "plot_extended_persistence",
        "plot_fitness_histogram",
        "plot_ordinary_persistence",
        "plot_persistence",
        "plot_tree",
        "relayout_tree_plot_data",
        "select_tree_root_node_index",
    ]
    assert sorted(topographer.plotting.__all__) == sorted(expected_all)


def test_domain_entities_accessible_via_submodules() -> None:
    assert hasattr(topographer.biology, "Alphabet")
    assert hasattr(topographer.biology, "FitnessLandscape")
    assert hasattr(topographer.biology, "FitnessLandscapeTopology")
    assert hasattr(topographer.biology, "roughness_to_slope_ratio")
    assert hasattr(topographer.biology, "sign_epistasis_fraction")
    assert hasattr(topographer.biology, "wildtype_accessibility_fraction")
    assert hasattr(topographer.topology, "MergeTree")
    assert hasattr(topographer.topology, "ScalarNetworkTopology")
    assert hasattr(topographer.topology, "compute_topological_summary")
    assert hasattr(topographer.topology, "diagram_bottleneck_distance")
    assert hasattr(topographer.topology, "diagram_wasserstein_distance")
    assert hasattr(topographer.plotting, "plot_tree")
    assert not any(name.endswith("_altair") for name in topographer.plotting.__all__)
    assert hasattr(topographer.plotting, "make_tree_layout")
    assert hasattr(topographer.graph, "Graph")


def test_plotting_not_imported_eagerly() -> None:
    cmd = [
        sys.executable,
        "-c",
        "import topographer, sys; assert 'topographer.plotting' not in sys.modules",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"


def test_experimental_mapper_and_legacy_exports_are_removed() -> None:
    assert importlib.util.find_spec("topographer.mapper") is None
    for removed_name in (
        "MapperResult",
        "build_mapper_graph",
        "build_merge_tree",
        "build_split_tree",
        "build_scalar_network_from_tables",
        "build_edge_filtration_table",
        "Alphabet",
        "FitnessLandscape",
        "TreePlotConfig",
        "plot_tree_altair",
    ):
        assert not hasattr(topographer, removed_name)


def test_submodules_are_importable() -> None:
    assert importlib.util.find_spec("topographer.graph") is not None
    assert importlib.util.find_spec("topographer.topology") is not None
    assert importlib.util.find_spec("topographer.biology") is not None
    assert importlib.util.find_spec("topographer.plotting") is not None
    assert importlib.util.find_spec("topographer.examples") is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "topographer.network",
        "topographer.tree",
        "topographer.persistence",
        "topographer.reeb",
        "topographer.builders",
        "topographer.filtration",
        "topographer.analysis",
        "topographer.plot",
        "topographer.plot_config",
        "topographer.layout",
        "topographer.biology.papkou2023",
        "topographer.examples._analysis",
        "topographer.examples._cube_assets",
        "topographer.examples._hypercube_assets",
        "topographer.examples._weinreich_assets",
        "topographer.examples._cube_demo",
        "topographer.examples._hypercube_demo",
        "topographer.examples._square_demo",
        "topographer.examples._demo_common",
        "topographer.graph._tables",
        "topographer.topology._spans",
        "topographer.topology._lineages",
        "topographer.topology._persistence_result",
        "topographer.topology._component_persistence",
        "topographer.topology._cycle_persistence",
        "topographer.biology._topology",
        "topographer.complex",
        "topographer.workflow_manifest",
        "topographer.graph.grouping",
        "topographer.graph.threshold_hierarchy",
        "topographer.topology.grouped_checkpoint",
        "topographer.topology._reeb",
        "topographer.topology._bdt",
        "topographer.examples.papkou2023",
        "topographer.examples.weinreich2006",
    ],
)
def test_retired_legacy_modules_raise_import_error(module_name: str) -> None:
    """Assert that importing legacy modules raises ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        __import__(module_name)


def test_retired_legacy_symbols_are_not_present() -> None:
    """Assert that ExampleInput, ExamplePresentation, ExampleTreeStage, and load_hypercube are absent."""
    import topographer.examples as ex

    for symbol in ("ExampleInput", "ExamplePresentation", "ExampleTreeStage", "load_hypercube"):
        assert not hasattr(ex, symbol), (
            f"{symbol} should no longer be present in topographer.examples"
        )


@pytest.mark.parametrize(
    "namespace",
    (
        topographer,
        topographer.biology,
        topographer.graph,
        topographer.topology,
        topographer.plotting,
    ),
)
def test_every_public_export_has_a_documented_example(namespace: object) -> None:
    """Public objects provide source material for a useful generated API page."""
    exported_names = namespace.__all__
    missing_examples = [
        name
        for name in exported_names
        if name != "__version__"
        and (
            (doc := getattr(getattr(namespace, name), "__doc__", None)) is None
            or "Examples" not in doc
        )
    ]
    assert missing_examples == []
