"""Characterization tests capturing canonical topology scientific results for Cube, and Hypercube."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.biology import Alphabet
from topographer.examples.cube import (
    build_two_peak_terrain_scalar_network,
    cube_edge_table,
    cube_node_table,
    two_peak_terrain_edge_table,
)
from topographer.examples.hamming import HAMMING_EXAMPLE_SCALARS
from topographer.graph import Graph, HammingGraph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology
from topographer.topology._tree_construction import build_merge_tree_sweep_trace


def test_cube_canonical_characterization() -> None:
    """Assert Cube scientific properties: 8 vertices, 12 edges, join & split trees."""
    nodes = cube_node_table()
    edges = cube_edge_table()

    # 1. Verification of inputs
    assert nodes.height == 8
    assert edges.height == 12
    vids = sorted(nodes["node_id"].to_list())
    assert vids == ["000", "001", "010", "011", "100", "101", "110", "111"]

    heights = dict(zip(nodes["node_id"].to_list(), nodes["height"].to_list(), strict=True))
    assert heights["001"] == 0.0
    assert heights["011"] == 1.0

    # Build Graph and ScalarNetwork
    graph = Graph.from_tables(
        nodes,
        edges,
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height", "label", "x", "y"),
        edge_attribute_cols=("distance", "capacity"),
    )
    assert graph.vertex_count == 8
    assert graph.edge_count == 12

    sn = ScalarNetwork(graph, scalar_attr="height")
    top = ScalarNetworkTopology(sn)

    # 2. Component persistence: ordinary (0.4), extended_positive (1.0), relative (0.1)
    comp = top.component_persistence()
    comp_df = comp.pairs
    assert comp_df.height == 3
    kinds = set(comp_df["pair_kind"].to_list())
    assert kinds == {"ordinary", "extended_positive", "relative"}

    ord_pair = comp_df.filter(pl.col("pair_kind") == "ordinary").row(0, named=True)
    assert ord_pair["birth_simplex"] == ["100"]
    assert ord_pair["death_simplex"] == ["000"]
    assert pytest.approx(ord_pair["persistence"]) == 0.4

    ext_pair = comp_df.filter(pl.col("pair_kind") == "extended_positive").row(0, named=True)
    assert ext_pair["birth_simplex"] == ["001"]
    assert ext_pair["death_simplex"] == ["011"]
    assert pytest.approx(ext_pair["persistence"]) == 1.0

    rel_pair = comp_df.filter(pl.col("pair_kind") == "relative").row(0, named=True)
    assert rel_pair["birth_simplex"] == ["110"]
    assert rel_pair["death_simplex"] == ["111"]
    assert pytest.approx(rel_pair["persistence"]) == 0.1

    # 3. Join and split tree persistence
    join_t = top.join_tree()
    join_pers = join_t.persistence()
    assert join_pers.height == 2
    assert join_pers.filter(pl.col("pair_kind") == "ordinary").height == 1
    assert join_pers.filter(pl.col("pair_kind") == "extended").height == 1

    split_t = top.split_tree()
    split_pers = split_t.persistence()
    assert split_pers.height == 2
    assert split_pers.filter(pl.col("pair_kind") == "ordinary").height == 1
    assert split_pers.filter(pl.col("pair_kind") == "extended").height == 1

    # 4. Simplification at 10% of scalar range (1.0 - 0.0 = 1.0 -> threshold = 0.1)
    threshold = 0.10 * (1.0 - 0.0)
    # Join tree ordinary persistence is 0.4 > 0.1 -> remains unchanged
    join_simp = join_t.simplify(threshold)
    assert join_simp.graph.num_nodes() == join_t.graph.num_nodes()

    # Split tree ordinary persistence is 0.1 -> prune <= 0.1 prunes the ordinary branch
    split_simp = split_t.simplify(threshold)
    assert split_simp.graph.num_nodes() < split_t.graph.num_nodes()


def test_hypercube_canonical_characterization() -> None:
    """Assert Hypercube scientific properties: 16 vertices, 32 edges."""
    scalars = HAMMING_EXAMPLE_SCALARS[4]
    alphabet = Alphabet("01")
    node_table = pl.DataFrame([{"seq": k, "scalar": v} for k, v in sorted(scalars.items())])
    hg = HammingGraph(
        node_table,
        sequence_col="seq",
        alphabet=alphabet,
        hamming_radius=1,
        vertex_attribute_cols=("scalar",),
    )
    assert hg.vertex_count == 16
    assert hg.edge_count == 32

    sn = ScalarNetwork(hg, scalar_attr="scalar")
    top = ScalarNetworkTopology(sn)

    # 1. Component persistence
    comp = top.component_persistence()
    # 1 extended_positive, 1 ordinary (from join), 3 relative (from split) = 5 total
    assert comp.pairs.height == 5
    assert comp.pairs.filter(pl.col("pair_kind") == "extended_positive").height == 1
    assert comp.pairs.filter(pl.col("pair_kind") == "ordinary").height == 1
    assert comp.pairs.filter(pl.col("pair_kind") == "relative").height == 3

    # 2. Join and split tree persistence
    join_t = top.join_tree()
    join_pers = join_t.persistence()
    assert join_pers.height == 2

    split_t = top.split_tree()
    split_pers = split_t.persistence()
    assert split_pers.height == 4

    # 3. Simplification at 10% range:
    max_s = max(scalars.values())
    min_s = min(scalars.values())
    threshold = 0.10 * (max_s - min_s)
    # Join tree ordinary persistence is ~0.279 > threshold (~0.088) -> unchanged
    join_simp = join_t.simplify(threshold)
    assert join_simp.graph.num_nodes() == join_t.graph.num_nodes()

    # Split tree has 2 ordinary pairs with persistence ~0.0588 < threshold -> simplifies
    split_simp = split_t.simplify(threshold)
    assert split_simp.graph.num_nodes() < split_t.graph.num_nodes()


@pytest.mark.slow
def test_cube_notebook_execution(tmp_path: pytest.TempPathFactory) -> None:
    """Verify that notebooks/cube.py executes idempotently as a Python app and exports all artifacts."""
    pytest.importorskip("marimo")
    import importlib.util
    from pathlib import Path

    nb_path = Path(__file__).parents[2] / "notebooks" / "cube.py"
    spec = importlib.util.spec_from_file_location("cube_notebook", nb_path)
    assert spec is not None and spec.loader is not None
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    outputs, defs = cube.app.run()
    assert "export_cube_artifacts" in defs
    export_fn = defs["export_cube_artifacts"]

    paths = export_fn(tmp_path)
    assert len(paths) >= 20
    assert paths["table_vertices"].is_file()
    assert paths["table_edges"].is_file()
    assert paths["table_persistence"].is_file()
    assert paths["chart_scalar_network_svg"].is_file()
    assert paths["chart_scalar_network_html"].is_file()
    assert paths["chart_canonical_persistence_svg"].is_file()
    assert paths["chart_join_tree_augmented_svg"].is_file()
    assert paths["chart_split_tree_augmented_svg"].is_file()


@pytest.mark.slow
def test_hypercube_notebook_execution(tmp_path: pytest.TempPathFactory) -> None:
    """Verify that notebooks/hypercube.py executes idempotently as a Python app and exports all artifacts."""
    pytest.importorskip("marimo")
    import importlib.util
    from pathlib import Path

    nb_path = Path(__file__).parents[2] / "notebooks" / "hypercube.py"
    spec = importlib.util.spec_from_file_location("hypercube_notebook", nb_path)
    assert spec is not None and spec.loader is not None
    hypercube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hypercube)

    outputs, defs = hypercube.app.run()
    assert "export_hypercube_artifacts" in defs
    export_fn = defs["export_hypercube_artifacts"]

    paths = export_fn(tmp_path)
    assert len(paths) >= 20
    assert paths["table_vertices"].is_file()
    assert paths["table_edges"].is_file()
    assert paths["table_persistence"].is_file()
    assert paths["chart_scalar_network_svg"].is_file()
    assert paths["chart_scalar_network_html"].is_file()
    assert paths["chart_canonical_persistence_svg"].is_file()
    assert paths["chart_join_tree_augmented_svg"].is_file()
    assert paths["chart_split_tree_augmented_svg"].is_file()
    assert paths["chart_split_tree_simplified_svg"].is_file()


def test_canonical_simplification_exact_and_zero_thresholds() -> None:
    """Assert <= threshold inclusive rule, exact threshold pruning, and <=0 rejection."""
    nodes = cube_node_table()
    edges = cube_edge_table()
    graph = Graph.from_tables(
        nodes,
        edges,
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height",),
    )
    sn = ScalarNetwork(graph, scalar_attr="height")
    top = ScalarNetworkTopology(sn)
    split_t = top.split_tree()

    # Split tree ordinary pair: persistence == 0.1
    split_pers = split_t.persistence()
    ord_rows = split_pers.filter(pl.col("pair_kind") == "ordinary")
    assert ord_rows.height == 1
    assert pytest.approx(ord_rows["persistence"][0]) == 0.1

    # Exact threshold = 0.1 must prune this branch because persistence <= threshold
    simp_exact = split_t.simplify(0.1)
    assert simp_exact.graph.num_nodes() < split_t.graph.num_nodes()

    # Threshold slightly less (0.099) must NOT prune
    simp_below = split_t.simplify(0.099)
    assert simp_below.graph.num_nodes() == split_t.graph.num_nodes()

    # Non-positive thresholds must raise ValueError
    with pytest.raises(ValueError, match="persistence_threshold must be positive"):
        split_t.simplify(0.0)

    with pytest.raises(ValueError, match="persistence_threshold must be positive"):
        split_t.simplify(-0.1)

    # Simplifying an already simplified tree must raise ValueError
    with pytest.raises(ValueError, match="Cannot simplify an already simplified tree"):
        simp_exact.simplify(0.2)


def test_tree_plot_data_orthogonal_segments_and_public_ids() -> None:
    """Assert TreePlotData computes orthogonal L-shaped segments and preserves vertex IDs."""
    from topographer.examples._presentation import build_tree_stage_plot_data

    nodes = cube_node_table()
    edges = cube_edge_table()
    graph = Graph.from_tables(
        nodes,
        edges,
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height",),
    )
    sn = ScalarNetwork(graph, scalar_attr="height")
    top = ScalarNetworkTopology(sn)
    join_t = top.join_tree()

    stage_data = build_tree_stage_plot_data(join_t.graph, tree_kind="join", scalar_attr="height")
    seg_df = stage_data.edge_segment_table
    assert seg_df.height > 0
    assert "x_start" in seg_df.columns
    assert "y_start" in seg_df.columns
    assert "x_end" in seg_df.columns
    assert "y_end" in seg_df.columns
    assert "segment_kind" in seg_df.columns

    # Every segment must be strictly orthogonal (either horizontal or vertical)
    for row in seg_df.to_dicts():
        is_vert = pytest.approx(row["x_start"]) == row["x_end"]
        is_horiz = pytest.approx(row["y_start"]) == row["y_end"]
        assert is_vert or is_horiz, f"Segment {row} is not orthogonal"


def test_two_peak_terrain_builds_with_one_split_saddle() -> None:
    """The two-peak terrain builds and has two maxima joined at one split saddle."""
    edges = two_peak_terrain_edge_table()
    undirected_pairs = [
        frozenset(pair) for pair in zip(edges["source"].to_list(), edges["target"].to_list())
    ]
    assert len(undirected_pairs) == len(set(undirected_pairs))

    terrain = build_two_peak_terrain_scalar_network()
    assert terrain.graph.num_nodes() == 20
    assert terrain.graph.num_edges() == 36

    trace = build_merge_tree_sweep_trace(terrain, ascending=False)
    events = trace.result.event_table["event"].value_counts().to_dicts()
    assert {row["event"]: row["count"] for row in events} == {
        "maximum": 2,
        "regular": 17,
        "split_saddle": 1,
    }
    saddle = trace.step_table.filter(pl.col("event") == "split_saddle").row(0, named=True)
    assert saddle["vertex_index"] == 12
    assert saddle["step_index"] == 12
    assert saddle["component_vertex_index"] == 0
