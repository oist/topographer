"""Filtration-specific H0 diagrams."""

import math
import random

from hypothesis import given, settings
from hypothesis import strategies as st
import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology


def topology(values, edges):
    graph = Graph.from_tables(
        pl.DataFrame({"id": list(range(len(values))), "scalar": values}),
        pl.DataFrame({"src": [a for a, _ in edges], "dst": [b for _, b in edges]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar",),
    )
    return ScalarNetworkTopology(ScalarNetwork(graph, scalar_attr="scalar"))


@pytest.mark.parametrize(
    ("filtration", "expected"),
    [
        ("sublevel", [(0.2, 0.5), (0.0, 1.0)]),
        ("superlevel", [(0.5, 0.2), (1.0, 0.0)]),
    ],
)
def test_path_and_null_essential(filtration, expected):
    workspace = topology([0.0, 0.5, 0.2, 1.0], [(0, 1), (1, 2), (2, 3)])
    capped = workspace.persistence_diagram(filtration).pairs
    assert list(zip(capped["birth_scalar"], capped["death_scalar"])) == expected
    assert capped["pair_kind"].to_list() == ["ordinary", "essential"]
    null = workspace.persistence_diagram(filtration, include_essential="null").pairs
    assert null.row(0, named=True) == capped.row(0, named=True)
    essential = null.row(1, named=True)
    assert math.isnan(essential["death_scalar"])
    assert math.isinf(essential["persistence"])
    assert essential["death_simplex"] == []
    assert capped.equals(workspace.persistence_diagram(filtration).pairs)


@settings(max_examples=30, derandomize=True, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_random_graphs_match_component_pairs_and_duality(seed):
    rng = random.Random(seed)
    size = rng.randint(2, 8)
    values = [rng.randrange(5) / 4 for _ in range(size)]
    edges = [(i, rng.randrange(i)) for i in range(1, size)]
    workspace = topology(values, edges)
    component = workspace.component_persistence().pairs
    for filtration, kind in [("sublevel", "ordinary"), ("superlevel", "relative")]:
        result = workspace.persistence_diagram(filtration).pairs
        ordinary = result.filter(pl.col("pair_kind") == "ordinary")
        old = component.filter(pl.col("pair_kind") == kind)
        columns = ["birth_simplex", "death_simplex", "birth_scalar", "death_scalar"]
        assert ordinary.select(columns).equals(old.select(columns))
        essential = result.filter(pl.col("pair_kind") == "essential").row(0, named=True)
        assert essential["persistence"] == max(values) - min(values)
        assert result["pair_id"].to_list() == list(range(result.height))
        assert all(
            (b <= d if filtration == "sublevel" else b >= d)
            for b, d in zip(result["birth_scalar"], result["death_scalar"])
        )
    dual = topology([-v for v in values], edges).persistence_diagram("superlevel").pairs
    actual = workspace.persistence_diagram("sublevel").pairs.filter(
        pl.col("pair_kind") == "ordinary"
    )
    dual = dual.filter(pl.col("pair_kind") == "ordinary")
    assert sorted(zip(actual["birth_scalar"], actual["death_scalar"])) == sorted(
        (-b, -d) for b, d in zip(dual["birth_scalar"], dual["death_scalar"])
    )


def test_invalid_options():
    workspace = topology([0.0, 1.0], [(0, 1)])
    with pytest.raises(ValueError, match="filtration"):
        workspace.persistence_diagram("bad")
    with pytest.raises(ValueError, match="include_essential"):
        workspace.persistence_diagram("sublevel", include_essential="bad")


@pytest.mark.parametrize("filtration", ["sublevel", "superlevel"])
def test_tied_extrema_follow_tree_extended_pair(filtration):
    workspace = topology([0.0, 1.0, 0.0, 1.0], [(0, 1), (1, 2), (2, 3)])
    tree = workspace.compute_merge_tree(filtration)
    extended = next(row for row in tree.persistence().to_dicts() if row["pair_kind"] == "extended")
    essential = (
        workspace.persistence_diagram(filtration)
        .pairs.filter(pl.col("pair_kind") == "essential")
        .row(0, named=True)
    )
    assert essential["birth_simplex"] == [extended["birth_vertex_index"]]
    assert essential["death_simplex"] == [extended["death_vertex_index"]]


def test_disconnected_network_is_rejected_before_diagram():
    with pytest.raises(ValueError, match="connected graph"):
        topology([0.0, 0.5, 1.0], [(0, 1)])


def test_cube_component_persistence_matches_base_commit():
    from topographer.examples import build_cube_scalar_network

    # Captured by executing the 586ad66 persistence implementation on the cube.
    rows = (
        ScalarNetworkTopology(build_cube_scalar_network()).component_persistence().pairs.to_dicts()
    )
    assert rows == [
        {
            "homology_dimension": 0,
            "pair_kind": "ordinary",
            "birth_simplex": ["100"],
            "death_simplex": ["000"],
            "birth_scalar": 0.1,
            "death_scalar": 0.5,
            "persistence": 0.4,
            "pair_id": 0,
        },
        {
            "homology_dimension": 0,
            "pair_kind": "extended_positive",
            "birth_simplex": ["001"],
            "death_simplex": ["011"],
            "birth_scalar": 0.0,
            "death_scalar": 1.0,
            "persistence": 1.0,
            "pair_id": 1,
        },
        {
            "homology_dimension": 1,
            "pair_kind": "relative",
            "birth_simplex": ["110"],
            "death_simplex": ["111"],
            "birth_scalar": 0.9,
            "death_scalar": 0.8,
            "persistence": 0.09999999999999998,
            "pair_id": 2,
        },
    ]
