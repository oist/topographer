"""Tests for the `BootstrapPersistenceBand` record and its network fingerprint."""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
import topographer.topology
from topographer.topology import BootstrapPersistenceBand
from topographer.topology._band import band_network_fingerprint

DISTANCES = (0.0, 0.1, 0.25, 0.3, 0.5, 0.05, 0.4, 0.2, 0.15, 0.35)


def make_band(**overrides) -> BootstrapPersistenceBand:
    options = {
        "quantile": 0.95,
        "filtration": "superlevel",
        "n_draws": len(DISTANCES),
        "seed": 3,
        "se_attr": "se",
        "network_fingerprint": "abc",
    }
    options.update(overrides)
    return BootstrapPersistenceBand.from_distances(DISTANCES, **options)


def path_network(
    scalars=(1.0, 0.0, 0.8), se=(0.0, 0.0, 0.1), edges=(("a", "b"), ("b", "c"))
) -> ScalarNetwork:
    ids = ["a", "b", "c"]
    graph = Graph.from_tables(
        pl.DataFrame({"id": ids, "scalar": list(scalars), "se": list(se)}),
        pl.DataFrame({"src": [u for u, _ in edges], "dst": [v for _, v in edges]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar", "se"),
    )
    return ScalarNetwork(graph, scalar_attr="scalar")


def test_exported_from_topology_not_top_level() -> None:
    assert "BootstrapPersistenceBand" in topographer.topology.__all__
    assert not hasattr(topographer, "BootstrapPersistenceBand")


def test_threshold_is_twice_the_radius() -> None:
    band = make_band()
    assert band.threshold == 2 * band.radius


def test_from_distances_equals_linear_quantile() -> None:
    for quantile in (0.1, 0.5, 0.9, 0.95, 0.99):
        band = make_band(quantile=quantile)
        assert band.radius == float(np.quantile(DISTANCES, quantile, method="linear"))
        assert band.distances == DISTANCES
        assert band.quantile == quantile


def test_from_distances_accepts_arrays_and_stores_a_tuple_of_floats() -> None:
    band = BootstrapPersistenceBand.from_distances(
        np.array(DISTANCES),
        quantile=0.9,
        filtration="sublevel",
        n_draws=len(DISTANCES),
        seed=0,
        se_attr="se",
        network_fingerprint="x",
    )
    assert isinstance(band.distances, tuple)
    assert all(type(value) is float for value in band.distances)


def test_at_quantile_keeps_draws_and_changes_radius() -> None:
    band = make_band(quantile=0.95)
    other = band.at_quantile(0.5)
    assert other.distances == band.distances
    assert other.quantile == 0.5
    assert other.radius == float(np.quantile(DISTANCES, 0.5, method="linear"))
    assert other.radius != band.radius
    assert other == make_band(quantile=0.5)
    assert dataclasses.replace(other, quantile=0.95, radius=band.radius) == band


def test_record_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_band().radius = 1.0  # type: ignore[misc]


def test_radius_mismatch_rejected() -> None:
    band = make_band()
    with pytest.raises(ValueError, match="radius"):
        dataclasses.replace(band, radius=band.radius + 1e-9)


@pytest.mark.parametrize("filtration", ["up", "", None, "Superlevel"])
def test_invalid_filtration_rejected(filtration) -> None:
    with pytest.raises(ValueError, match="filtration"):
        make_band(filtration=filtration)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_quantile_rejected(quantile) -> None:
    with pytest.raises(ValueError, match="quantile"):
        make_band(quantile=quantile)


@pytest.mark.parametrize("n_draws", [0, -1, 9, 11])
def test_invalid_n_draws_rejected(n_draws) -> None:
    with pytest.raises(ValueError, match="n_draws"):
        make_band(n_draws=n_draws)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01])
def test_invalid_distances_rejected(bad) -> None:
    distances = list(DISTANCES)
    distances[4] = bad
    with pytest.raises(ValueError, match="distances"):
        BootstrapPersistenceBand.from_distances(
            distances,
            quantile=0.9,
            filtration="superlevel",
            n_draws=len(distances),
            seed=0,
            se_attr="se",
            network_fingerprint="x",
        )


def test_fingerprint_is_stable_and_sensitive_to_every_input() -> None:
    base = band_network_fingerprint(path_network(), se_attr="se")
    assert base == band_network_fingerprint(path_network(), se_attr="se")
    assert len(base) == 64
    changed = {
        "scalar": path_network(scalars=(1.0, 0.0, 0.7)),
        "se": path_network(se=(0.0, 0.0, 0.2)),
        "edge": path_network(edges=(("a", "b"), ("a", "c"))),
    }
    for name, network in changed.items():
        assert band_network_fingerprint(network, se_attr="se") != base, name


def test_fingerprint_is_sensitive_to_vertex_ids() -> None:
    renamed = Graph.from_tables(
        pl.DataFrame({"id": ["a", "b", "z"], "scalar": [1.0, 0.0, 0.8], "se": [0.0, 0.0, 0.1]}),
        pl.DataFrame({"src": ["a", "b"], "dst": ["b", "z"]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar", "se"),
    )
    network = ScalarNetwork(renamed, scalar_attr="scalar")
    assert band_network_fingerprint(network, se_attr="se") != band_network_fingerprint(
        path_network(), se_attr="se"
    )


def test_fingerprint_rejects_a_missing_se_attr() -> None:
    with pytest.raises(ValueError, match="se_missing"):
        band_network_fingerprint(path_network(), se_attr="se_missing")
