"""Unit tests for topological summary invariants and compute_topological_summary."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from topographer.examples import build_cube_scalar_network
from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology, compute_topological_summary


def _build_simple_network(node_values: list[float], edges: list[tuple[int, int]]) -> ScalarNetwork:
    v_df = pl.DataFrame(
        {
            "id": [f"v{i}" for i in range(len(node_values))],
            "scalar": node_values,
        }
    )
    e_df = pl.DataFrame(
        {
            "src": [f"v{u}" for u, _ in edges],
            "dst": [f"v{v}" for _, v in edges],
        }
    )
    g = Graph.from_tables(
        v_df,
        e_df,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    return ScalarNetwork(g, scalar_attr="scalar")


def test_compute_topological_summary_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="Expected ScalarNetwork"):
        compute_topological_summary(42)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected ScalarNetwork"):
        compute_topological_summary("invalid")  # type: ignore[arg-type]


def test_compute_topological_summary_cube_invariants() -> None:
    cube_net = build_cube_scalar_network()
    cube_topo = ScalarNetworkTopology(cube_net)

    summary_from_net = compute_topological_summary(cube_net)
    summary_from_topo = cube_topo.summary()

    assert summary_from_net.equals(summary_from_topo)
    assert summary_from_net.height == 1

    expected_cols = [
        "n_vertices",
        "n_edges",
        "n_peaks",
        "n_sinks",
        "n_split_saddles",
        "n_join_saddles",
        "max_persistence",
        "total_persistence",
        "persistence_entropy",
        "tree_asymmetry_depth_ratio",
        "tree_asymmetry_persistence_ratio",
    ]
    assert summary_from_net.columns == expected_cols

    row = summary_from_net.to_dicts()[0]
    assert row["n_vertices"] == 8
    assert row["n_edges"] == 12
    assert row["n_peaks"] == 2
    assert row["n_sinks"] == 2
    assert row["n_split_saddles"] == 1
    assert row["n_join_saddles"] == 1
    assert math.isclose(row["max_persistence"], 1.0)
    assert math.isclose(row["total_persistence"], 1.1)

    # Manual Shannon entropy on split persistence: [1.0, 0.1]
    p = np.array([1.0, 0.1])
    p_hat = p / np.sum(p)
    expected_entropy = float(-np.sum(p_hat * np.log2(p_hat)))
    assert math.isclose(row["persistence_entropy"], expected_entropy, rel_tol=1e-5)

    # Join tree has height 2, Split tree has height 2 -> depth ratio 1.0
    assert math.isclose(row["tree_asymmetry_depth_ratio"], 1.0)

    # Join total persistence: [1.0, 0.4] -> 1.4; Split total persistence: 1.1 -> 1.4 / 1.1
    expected_pers_ratio = 1.4 / 1.1
    assert math.isclose(row["tree_asymmetry_persistence_ratio"], expected_pers_ratio, rel_tol=1e-5)


def test_compute_topological_summary_linear_monotonic_path() -> None:
    # Monotonic path: 0.0 -> 1.0 -> 2.0 -> 3.0
    net = _build_simple_network([0.0, 1.0, 2.0, 3.0], [(0, 1), (1, 2), (2, 3)])
    summary = compute_topological_summary(net)

    row = summary.to_dicts()[0]
    assert row["n_vertices"] == 4
    assert row["n_edges"] == 3
    assert row["n_peaks"] == 1
    assert row["n_sinks"] == 1
    assert row["n_split_saddles"] == 0
    assert row["n_join_saddles"] == 0
    assert math.isclose(row["max_persistence"], 3.0)
    assert math.isclose(row["total_persistence"], 3.0)
    assert math.isclose(row["persistence_entropy"], 0.0)
    assert math.isclose(row["tree_asymmetry_depth_ratio"], 1.0)
    assert math.isclose(row["tree_asymmetry_persistence_ratio"], 1.0)


def test_compute_topological_summary_asymmetric_peaks_vs_sinks() -> None:
    # 3 peaks connected to a single valley node
    # v0 (0.0): valley
    # v1 (1.0), v2 (2.0), v3 (3.0): peaks
    # Split tree has 3 peaks meeting at split saddles
    # Join tree has 1 sink meeting root
    net = _build_simple_network([0.0, 1.0, 2.0, 3.0], [(0, 1), (0, 2), (0, 3)])
    summary = compute_topological_summary(net)

    row = summary.to_dicts()[0]
    assert row["n_vertices"] == 4
    assert row["n_edges"] == 3
    assert row["n_peaks"] == 3
    assert row["n_sinks"] == 1
    assert row["n_split_saddles"] == 1
    assert row["n_join_saddles"] == 0
    # Split tree has 3 branches (2 ordinary + 1 extended)
    # Join tree has only 1 extended branch (persistence 3.0)
    assert row["total_persistence"] > 3.0
    assert row["tree_asymmetry_persistence_ratio"] < 1.0
    assert row["persistence_entropy"] > 0.0


def test_compute_topological_summary_flat_scalar_landscape() -> None:
    # Flat scalar landscape: all vertices have identical scalar 0.0
    net = _build_simple_network([0.0, 0.0, 0.0], [(0, 1), (1, 2)])
    summary = compute_topological_summary(net)

    row = summary.to_dicts()[0]
    assert row["n_vertices"] == 3
    assert row["n_edges"] == 2
    assert math.isclose(row["max_persistence"], 0.0)
    assert math.isclose(row["total_persistence"], 0.0)
    assert math.isclose(row["persistence_entropy"], 0.0)
    assert not math.isnan(row["tree_asymmetry_depth_ratio"])
    assert not math.isnan(row["tree_asymmetry_persistence_ratio"])


def test_compute_topological_summary_determinism() -> None:
    cube_net = build_cube_scalar_network()
    runs = [compute_topological_summary(cube_net) for _ in range(5)]
    first = runs[0]
    for r in runs[1:]:
        assert first.equals(r)
