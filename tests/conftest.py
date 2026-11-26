"""Test configuration and canonical fixtures for topographer tests."""

from __future__ import annotations

from pathlib import Path
import sys

import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def single_saddle_diamond() -> ScalarNetwork:
    """Canonical 4-vertex diamond network with 1 saddle and 1 1-cycle.

    Topology:
        Vertices: v0 (0.0), v1 (0.4), v2 (0.6), v3 (1.0)
        Edges: (v0, v1), (v0, v2), (v1, v3), (v2, v3)
        Critical points: 1 minimum (v0: 0.0), 1 maximum (v3: 1.0), with two
        disjoint paths through v1 (0.4) and v2 (0.6).
        Topology: Single 1-cycle (beta_1 = 1) and 1 topological saddle.
    """
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2", "v3"],
            "scalar": [0.0, 0.4, 0.6, 1.0],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v0", "v1", "v2"],
            "dst": ["v1", "v2", "v3", "v3"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    return ScalarNetwork(graph, scalar_attr="scalar")


@pytest.fixture
def bifurcating_two_peaks() -> ScalarNetwork:
    """Canonical 5-vertex bifurcating network with two peaks merging at a saddle.

    Topology:
        Vertices: v0 (0.1), v1 (0.9), v2 (0.4), v3 (0.8), v4 (0.2)
        Edges: (v0, v2), (v1, v2), (v2, v3), (v2, v4)
        Critical points:
            - Maxima (peaks): v1 (0.9, global max) and v3 (0.8, secondary peak)
            - Minima: v0 (0.1, global min) and v4 (0.2, secondary min)
            - Saddle: v2 (0.4, bifurcating saddle where peaks and minima meet)
        Topology: Tree centered at saddle v2, 0 cycles (beta_1 = 0).
        Persistence:
            - Split tree: secondary peak v3 merges into v1 at saddle v2 (persistence = 0.4).
            - Join tree: secondary minimum v4 merges into v0 at saddle v2 (persistence = 0.2).
            - Global feature: minimum v0 pairs with maximum v1 (persistence = 0.8).
    """
    vertices = pl.DataFrame(
        {
            "id": ["v0", "v1", "v2", "v3", "v4"],
            "scalar": [0.1, 0.9, 0.4, 0.8, 0.2],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v0", "v1", "v2", "v2"],
            "dst": ["v2", "v2", "v3", "v4"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    return ScalarNetwork(graph, scalar_attr="scalar")
