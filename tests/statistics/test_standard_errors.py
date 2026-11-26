"""Tests for `validate_se_attr`."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
from topographer.statistics import validate_se_attr


def test_validate_se_attr_valid_with_zeros() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v1", "v2", "v3"],
            "fitness": [0.1, 0.5, 0.9],
            "se": [0.0, 0.05, 0.1],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v1", "v2"],
            "dst": ["v2", "v3"],
        }
    )
    g = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("fitness", "se"),
    )
    network = ScalarNetwork(g, scalar_attr="fitness")
    se_arr = validate_se_attr(network, se_attr="se")
    assert isinstance(se_arr, np.ndarray)
    assert se_arr.dtype == np.float64
    assert se_arr.shape == (3,)
    np.testing.assert_allclose(se_arr, [0.0, 0.05, 0.1])


@pytest.mark.parametrize(
    "invalid_val",
    [
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.01,
        "invalid",
    ],
)
def test_validate_se_attr_raises_value_error(invalid_val) -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v1", "v2", "v3"],
            "fitness": [0.1, 0.5, 0.9],
            "se": [0.1, 0.2, 0.3],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v1", "v2"],
            "dst": ["v2", "v3"],
        }
    )
    g = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("fitness", "se"),
    )
    network = ScalarNetwork(g, scalar_attr="fitness")
    idx = network._graph_snapshot.node_indices()[0]
    payload = network._graph_snapshot.get_node_data(idx)
    payload["se"] = invalid_val
    with pytest.raises(ValueError):
        validate_se_attr(network, se_attr="se")


def test_validate_se_attr_missing_attr() -> None:
    vertices = pl.DataFrame(
        {
            "id": ["v1", "v2"],
            "fitness": [0.1, 0.5],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["v1"],
            "dst": ["v2"],
        }
    )
    g = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("fitness",),
    )
    network = ScalarNetwork(g, scalar_attr="fitness")
    with pytest.raises(ValueError, match="not found"):
        validate_se_attr(network, se_attr="se")
