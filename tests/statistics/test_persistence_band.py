"""Tests for the bootstrap persistence band statistic and function.

The diagram is the ordinary H0 pairs plus the capped global pair. The band
radius is the chosen quantile of the per-draw bottleneck distances, and the
simplification threshold is twice that radius.
"""

from __future__ import annotations

import itertools
import pickle
import random

import numpy as np
import polars as pl
import pytest

from topographer.examples.cube.network import cube_edge_table, cube_node_table
from topographer.graph import Graph, ScalarNetwork
from topographer.statistics import BottleneckToNominal, bootstrap, bootstrap_persistence_band
from topographer.statistics._persistence_band import _h0_diagram_points
from topographer.statistics.bootstrap import _draw_rng
from topographer.topology import (
    BootstrapPersistenceBand,
    ScalarNetworkTopology,
    diagram_bottleneck_distance,
)

N_DRAWS = 40
SLICE = 7
SEED = 11


def path_network(
    scalars=(1.0, 0.0, 0.8), se=(0.0, 0.0, 0.5), edges=(("a", "b"), ("b", "c"))
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


def cube_network(se: float = 0.12) -> ScalarNetwork:
    nodes = cube_node_table().with_columns(se=pl.lit(se))
    graph = Graph.from_tables(
        nodes,
        cube_edge_table(),
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height", "se"),
    )
    return ScalarNetwork(graph, scalar_attr="height")


def ring_network(se: float = 0.1) -> ScalarNetwork:
    """A 12-vertex ring with chords and scrambled scalars (several peaks and saddles)."""
    count = 12
    scalars = [float(x) for x in np.random.default_rng(3).permutation(count) / (count - 1)]
    ids = [f"v{i}" for i in range(count)]
    edges = [(i, (i + 1) % count) for i in range(count)] + [(0, 4), (2, 7), (3, 9)]
    graph = Graph.from_tables(
        pl.DataFrame({"id": ids, "scalar": scalars, "se": [se] * count}),
        pl.DataFrame({"src": [ids[a] for a, _ in edges], "dst": [ids[b] for _, b in edges]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar", "se"),
    )
    return ScalarNetwork(graph, scalar_attr="scalar")


def diagram_rows(network: ScalarNetwork, filtration: str) -> list[tuple[float, float]]:
    pairs = (
        ScalarNetworkTopology(network)
        .persistence_diagram(filtration, include_essential="capped")
        .pairs
    )
    return sorted(zip(pairs["birth_scalar"].to_list(), pairs["death_scalar"].to_list()))


# --- one helper builds the nominal and the per-draw diagrams -----------------------------------


@pytest.mark.parametrize("filtration", ["sublevel", "superlevel"])
@pytest.mark.parametrize("make", [path_network, cube_network, ring_network])
def test_diagram_helper_equals_public_persistence_diagram(make, filtration) -> None:
    network = make()
    points = _h0_diagram_points(network.graph, network.scalar_attr, filtration)
    assert sorted(map(tuple, points.tolist())) == diagram_rows(network, filtration)


def test_diagram_helper_works_on_a_scalar_only_draw_graph() -> None:
    network = ring_network()
    graph = bootstrap._scalar_only_graph(
        network.graph,
        "scalar",
        np.array([network._vertex_scalars[v] for v in network.vertex_ids]),
    )
    points = _h0_diagram_points(graph, "scalar", "superlevel")
    assert sorted(map(tuple, points.tolist())) == diagram_rows(network, "superlevel")


# --- hand-computed band ------------------------------------------------------------------------
#
# Path a - b - c, scalars (A, B, X) = (1.0, 0.0, X), where X is the drawn value of c (the only
# vertex with a nonzero standard error; a and b are fixed). Superlevel filtration: sweep from
# high to low, a component is born at a local maximum and dies when it merges into an elder
# (higher) component at a saddle. The essential pair is (global max, global min), capped.
#
#   X < 0:      a=1 > b=0 > c=X is monotone. One maximum (a), no saddle.
#               Ordinary: none.  Essential: (1.0, X)  [global max a, global min c].
#   0 < X < 1:  maxima a (1.0) and c (X), saddle b (0.0), global min b.
#               c is the younger peak and dies at b: ordinary (X, 0.0).
#               Essential: (1.0, 0.0).
#   X > 1:      c is the global maximum, a the younger peak and dies at b: ordinary (1.0, 0.0).
#               Essential: (X, 0.0).
#
# The nominal diagram is the case 0 < X < 1 at X = 0.8: {(0.8, 0.0), (1.0, 0.0)}.
# The bottleneck distance of two tiny diagrams is found by enumerating every matching. Each
# point of the first diagram goes to a distinct point of the second or to the diagonal; each
# unmatched point of the second goes to the diagonal. A matched pair costs the L-infinity
# distance, a point on the diagonal costs |birth - death| / 2, and the matching costs its
# largest term; the distance is the cheapest matching.

NOMINAL_X = 0.8


def closed_form_diagram(x: float) -> list[tuple[float, float]]:
    if x < 0.0:
        return [(1.0, x)]
    if x < 1.0:
        return [(x, 0.0), (1.0, 0.0)]
    return [(1.0, 0.0), (x, 0.0)]


def closed_form_case(x: float) -> str:
    return "x<0" if x < 0.0 else "0<x<1" if x < 1.0 else "x>1"


def bottleneck_by_enumeration(first, second) -> float:
    best = float("inf")
    slots = list(range(len(second))) + [None] * len(first)
    for assignment in set(itertools.permutations(slots, len(first))):
        costs = []
        used = set()
        for point, target in zip(first, assignment):
            if target is None:
                costs.append(abs(point[0] - point[1]) / 2)
            else:
                used.add(target)
                other = second[target]
                costs.append(max(abs(point[0] - other[0]), abs(point[1] - other[1])))
        costs.extend(abs(b - d) / 2 for k, (b, d) in enumerate(second) if k not in used)
        best = min(best, max(costs, default=0.0))
    return best


def hand_values(seed: int, n_draws: int) -> tuple[list[float], list[float]]:
    xs, distances = [], []
    for draw in range(n_draws):
        values = _draw_rng(seed, draw).normal(
            loc=np.array([1.0, 0.0, NOMINAL_X]), scale=np.array([0.0, 0.0, 0.5])
        )
        assert values[0] == 1.0 and values[1] == 0.0
        xs.append(float(values[2]))
        distances.append(
            bottleneck_by_enumeration(
                closed_form_diagram(float(values[2])), closed_form_diagram(NOMINAL_X)
            )
        )
    return xs, distances


def test_enumeration_oracle_on_pinned_cases() -> None:
    nominal = closed_form_diagram(NOMINAL_X)
    assert bottleneck_by_enumeration(nominal, nominal) == 0.0
    # X = -0.4: {(1, -0.4)} against {(0.8, 0), (1, 0)}: match (1, -0.4) with (1, 0) at cost 0.4
    # and send (0.8, 0) to the diagonal at cost 0.4.
    assert bottleneck_by_enumeration(closed_form_diagram(-0.4), nominal) == pytest.approx(0.4)
    # X = 1.5: {(1, 0), (1.5, 0)} against {(0.8, 0), (1, 0)}: (1, 0)->(0.8, 0) costs 0.2 and
    # (1.5, 0)->(1, 0) costs 0.5; the other matching costs max(0.5, 0.2)... both give 0.5.
    assert bottleneck_by_enumeration(closed_form_diagram(1.5), nominal) == pytest.approx(0.5)
    # X = 0.6: identity matching costs 0.2 and nothing is cheaper than moving (0.6, 0) by 0.2
    # (to the diagonal it costs 0.3, to (1, 0) it costs 0.4).
    assert bottleneck_by_enumeration(closed_form_diagram(0.6), nominal) == pytest.approx(0.2)
    for x in (-0.4, 0.3, 0.6, 1.5):
        library = diagram_bottleneck_distance(
            pl.DataFrame(
                closed_form_diagram(x), schema=["birth_scalar", "death_scalar"], orient="row"
            ),
            pl.DataFrame(nominal, schema=["birth_scalar", "death_scalar"], orient="row"),
        )
        assert library == pytest.approx(bottleneck_by_enumeration(closed_form_diagram(x), nominal))


def test_closed_form_diagram_matches_public_diagram_for_each_case() -> None:
    for x in (-0.4, 0.3, 0.8, 1.5):
        network = path_network(scalars=(1.0, 0.0, x), se=(0.0, 0.0, 0.0))
        assert diagram_rows(network, "superlevel") == sorted(closed_form_diagram(x))


def test_band_equals_hand_derivation() -> None:
    n_draws, seed = 200, 5
    xs, hand = hand_values(seed, n_draws)
    assert {closed_form_case(x) for x in xs} == {"x<0", "0<x<1", "x>1"}

    band = bootstrap_persistence_band(
        path_network(), filtration="superlevel", n_draws=n_draws, seed=seed
    )

    assert band.distances == pytest.approx(hand, abs=1e-12)
    assert band.radius == pytest.approx(float(np.quantile(hand, 0.95, method="linear")), abs=1e-12)
    assert band.threshold == pytest.approx(2 * band.radius, abs=1e-12)
    assert band.at_quantile(0.5).radius == pytest.approx(
        float(np.quantile(hand, 0.5, method="linear")), abs=1e-12
    )
    assert (band.filtration, band.n_draws, band.seed, band.quantile) == (
        "superlevel",
        n_draws,
        seed,
        0.95,
    )


# --- one call equals chunks --------------------------------------------------------------------


def slices(total: int, size: int) -> list[range]:
    return [range(start, min(start + size, total)) for start in range(0, total, size)]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_one_call_equals_shuffled_chunks(tmp_path, n_jobs) -> None:
    network = ring_network()
    one_call = bootstrap_persistence_band(
        network, filtration="superlevel", n_draws=N_DRAWS, seed=SEED
    )
    statistic = BottleneckToNominal("superlevel").prepare(network)
    paths = []
    for position, draws in enumerate(slices(N_DRAWS, SLICE)):
        path = tmp_path / f"chunk{position}.npz"
        bootstrap.run(
            network, statistic, n_draws=N_DRAWS, seed=SEED, draws=draws, n_jobs=n_jobs
        ).save(path)
        paths.append(path)
    random.Random(1).shuffle(paths)
    chunks = [bootstrap.BootstrapChunk.load(path) for path in paths]

    combined = bootstrap.combine(chunks, statistic)

    assert combined == one_call
    assert combined.distances == one_call.distances
    assert bootstrap.combine(chunks, statistic, quantile=0.9) == one_call.at_quantile(0.9)


def test_n_jobs_does_not_change_the_band() -> None:
    kwargs = {"filtration": "sublevel", "n_draws": 12, "seed": 2}
    network = ring_network()
    assert bootstrap_persistence_band(network, **kwargs) == bootstrap_persistence_band(
        network, n_jobs=2, **kwargs
    )


def test_progress_is_forwarded() -> None:
    seen = []
    bootstrap_persistence_band(
        ring_network(),
        filtration="superlevel",
        n_draws=4,
        seed=0,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (4, 4)


def test_band_records_se_attr_and_fingerprint() -> None:
    from topographer.topology._band import band_network_fingerprint

    network = ring_network()
    band = bootstrap_persistence_band(network, filtration="superlevel", n_draws=5, seed=1)
    assert band.se_attr == "se"
    assert band.network_fingerprint == band_network_fingerprint(network, se_attr="se")
    assert isinstance(band, BootstrapPersistenceBand)


def test_custom_se_attr_is_used_for_draws_and_statistic() -> None:
    ids = ["a", "b", "c"]
    graph = Graph.from_tables(
        pl.DataFrame(
            {"id": ids, "scalar": [1.0, 0.0, 0.8], "sigma": [0.0, 0.0, 0.5], "se": [9.0] * 3}
        ),
        pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar", "sigma", "se"),
    )
    network = ScalarNetwork(graph, scalar_attr="scalar")
    band = bootstrap_persistence_band(
        network, filtration="superlevel", n_draws=200, seed=5, se_attr="sigma"
    )
    assert band.se_attr == "sigma"
    assert band.distances == pytest.approx(hand_values(5, 200)[1], abs=1e-12)


# --- fingerprints ------------------------------------------------------------------------------


def stat_fingerprint(network: ScalarNetwork, filtration: str = "superlevel") -> str:
    return BottleneckToNominal(filtration).prepare(network).fingerprint()


def test_statistic_fingerprint_is_sensitive_to_its_inputs() -> None:
    base = stat_fingerprint(path_network())
    assert base == stat_fingerprint(path_network())
    assert stat_fingerprint(path_network(scalars=(1.0, 0.0, 0.7))) != base
    assert stat_fingerprint(path_network(se=(0.0, 0.0, 0.4))) != base
    assert stat_fingerprint(path_network(edges=(("a", "b"), ("a", "c")))) != base
    assert stat_fingerprint(path_network(), "sublevel") != base


def test_combining_chunks_from_a_different_setup_is_refused() -> None:
    first, second = path_network(), path_network(se=(0.0, 0.0, 0.4))
    stat_first = BottleneckToNominal("superlevel").prepare(first)
    stat_second = BottleneckToNominal("superlevel").prepare(second)
    chunk_first = bootstrap.run(first, stat_first, n_draws=4, seed=0, draws=range(2))
    chunk_second = bootstrap.run(second, stat_second, n_draws=4, seed=0, draws=range(2, 4))
    with pytest.raises(ValueError, match="different fingerprint"):
        bootstrap.combine([chunk_first, chunk_second], stat_first)
    chunk_all = bootstrap.run(first, stat_first, n_draws=4, seed=0)
    with pytest.raises(ValueError, match="statistic's fingerprint"):
        bootstrap.combine([chunk_all], stat_second)


def test_setup_fingerprint_matches_the_chunks() -> None:
    network = path_network()
    statistic = BottleneckToNominal("superlevel").prepare(network)
    chunk = bootstrap.run(network, statistic, n_draws=3, seed=0)
    assert chunk.fingerprint == bootstrap.fingerprint(network, statistic)
    assert chunk.counts == {}
    assert chunk.stats.shape == (3, 1)


# --- binding and errors ------------------------------------------------------------------------


def test_unbound_statistic_errors() -> None:
    statistic = BottleneckToNominal("superlevel")
    drawn = bootstrap.DrawnNetwork(
        graph=path_network().graph, scalar_attr="scalar", values=np.zeros(3)
    )
    with pytest.raises(ValueError, match="not bound"):
        statistic.measure(drawn)
    with pytest.raises(ValueError, match="not bound"):
        statistic.fingerprint()
    with pytest.raises(ValueError, match="not bound"):
        statistic.finish({}, np.zeros((1, 1)), 1, seed=0)


def test_prepare_is_idempotent_on_the_same_network_and_refuses_another() -> None:
    network = path_network()
    bound = BottleneckToNominal("superlevel").prepare(network)
    assert bound.prepare(network) is bound
    assert bound.prepare(path_network()) is bound
    with pytest.raises(ValueError, match="already bound"):
        bound.prepare(path_network(scalars=(1.0, 0.0, 0.7)))


def test_prepare_does_not_bind_the_original() -> None:
    statistic = BottleneckToNominal("superlevel")
    statistic.prepare(path_network())
    with pytest.raises(ValueError, match="not bound"):
        statistic.fingerprint()


def test_bound_statistic_pickles() -> None:
    statistic = BottleneckToNominal("sublevel").prepare(ring_network())
    clone = pickle.loads(pickle.dumps(statistic))
    assert clone.fingerprint() == statistic.fingerprint()


def test_add_is_order_independent() -> None:
    statistic = BottleneckToNominal("superlevel")
    assert statistic.add({}, {}) == {}


@pytest.mark.parametrize("filtration", ["up", "", None, "Superlevel"])
def test_invalid_filtration(filtration) -> None:
    with pytest.raises(ValueError, match="filtration"):
        BottleneckToNominal(filtration)
    with pytest.raises(ValueError, match="filtration"):
        bootstrap_persistence_band(path_network(), filtration=filtration, n_draws=2, seed=0)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.5, 2.0])
def test_invalid_quantile(quantile) -> None:
    with pytest.raises(ValueError, match="quantile"):
        bootstrap_persistence_band(
            path_network(), filtration="superlevel", n_draws=2, seed=0, quantile=quantile
        )


@pytest.mark.parametrize("n_draws", [0, -3, True])
def test_invalid_n_draws(n_draws) -> None:
    with pytest.raises(ValueError, match="n_draws"):
        bootstrap_persistence_band(
            path_network(), filtration="superlevel", n_draws=n_draws, seed=0
        )


def test_invalid_quantile_in_combine_is_refused() -> None:
    network = path_network()
    statistic = BottleneckToNominal("superlevel").prepare(network)
    chunk = bootstrap.run(network, statistic, n_draws=3, seed=0)
    with pytest.raises(ValueError, match="quantile"):
        bootstrap.combine([chunk], statistic, quantile=1.0)


def test_missing_se_attr_is_refused() -> None:
    with pytest.raises(ValueError, match="se_missing"):
        bootstrap_persistence_band(
            path_network(), filtration="superlevel", n_draws=2, seed=0, se_attr="se_missing"
        )


def test_non_network_is_refused() -> None:
    with pytest.raises(TypeError):
        bootstrap_persistence_band(object(), filtration="superlevel", n_draws=2, seed=0)


# --- zero standard errors ----------------------------------------------------------------------


def test_zero_standard_errors_give_a_zero_width_band_that_simplify_refuses() -> None:
    """All SE zero means no uncertainty was supplied: radius 0, threshold 0, no simplification.

    ``MergeTree.simplify`` keeps its existing rule that the threshold must be positive, so a
    zero-width band cannot be applied; the error is the signal that the band is empty.
    """
    network = cube_network(se=0.0)
    band = bootstrap_persistence_band(network, filtration="superlevel", n_draws=6, seed=0)
    assert band.radius == 0.0
    assert band.threshold == 0.0
    assert band.distances == (0.0,) * 6
    tree = ScalarNetworkTopology(network).compute_merge_tree("superlevel")
    with pytest.raises(ValueError, match="persistence_threshold must be positive"):
        tree.simplify(band.threshold)


# --- band-simplified tree on the cube ----------------------------------------------------------


@pytest.mark.parametrize(
    ("filtration", "se", "expected_kept"),
    [
        ("superlevel", 0.01, 1),
        ("superlevel", 0.04, 0),
        ("sublevel", 0.04, 1),
        ("sublevel", 0.2, 0),
    ],
)
def test_band_simplified_cube_tree_keeps_exactly_the_pairs_above_the_threshold(
    filtration, se, expected_kept
) -> None:
    network = cube_network(se=se)
    band = bootstrap_persistence_band(network, filtration=filtration, n_draws=60, seed=4)
    tree = ScalarNetworkTopology(network).compute_merge_tree(filtration)

    def ordinary(merge_tree) -> list[float]:
        pairs = merge_tree.persistence().filter(pl.col("pair_kind") == "ordinary")
        return pairs["persistence"].to_list()

    kept = ordinary(tree.simplify(band.threshold))
    pruned = [value for value in ordinary(tree) if value <= band.threshold]

    assert all(value > band.threshold for value in kept)
    assert sorted(kept + pruned) == sorted(ordinary(tree))
    assert len(kept) == expected_kept


def test_run_refuses_an_se_attr_that_differs_from_the_statistics() -> None:
    graph = Graph.from_tables(
        pl.DataFrame(
            {
                "id": ["a", "b", "c"],
                "scalar": [1.0, 0.0, 0.8],
                "se": [0.0, 0.0, 0.5],
                "alt": [0.1] * 3,
            }
        ),
        pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar", "se", "alt"),
    )
    network = ScalarNetwork(graph, scalar_attr="scalar")
    statistic = BottleneckToNominal("superlevel", se_attr="alt").prepare(network)
    with pytest.raises(ValueError, match="se_attr"):
        bootstrap.run(network, statistic, n_draws=2, seed=0, se_attr="se")
    bootstrap.run(network, statistic, n_draws=2, seed=0, se_attr="alt")
