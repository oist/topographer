"""Tests for the chunked, resumable bootstrap.

Combined slices of draws match one run bit for bit, and a saved chunk whose
version is not the current one is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import signal
import stat as stat_module
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from topographer.graph import Graph, ScalarNetwork
from topographer.statistics import bootstrap
from topographer.statistics.bootstrap import BootstrapChunk, DrawnNetwork, _draw_rng

N_DRAWS = 40
SLICE = 7
SEED = 11
HIGH = 0.9


def make_network(
    scalars: list[float], se: list[float], extra_edges: tuple[tuple[int, int], ...] = ()
) -> ScalarNetwork:
    """A connected ring with chords on vertices ``v0..`` with scalars and standard errors."""
    count = len(scalars)
    ids = [f"v{i}" for i in range(count)]
    edges = [(i, (i + 1) % count) for i in range(count)] + [(0, 4), (2, 7), (3, 9)]
    edges += list(extra_edges)
    graph = Graph.from_tables(
        pl.DataFrame({"id": ids, "scalar": scalars, "se": se}),
        pl.DataFrame({"src": [ids[a] for a, _ in edges], "dst": [ids[b] for _, b in edges]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("scalar", "se"),
    )
    return ScalarNetwork(graph, scalar_attr="scalar")


@pytest.fixture
def network() -> ScalarNetwork:
    scalars = [float(x) for x in np.random.default_rng(3).permutation(12) / 11]
    return make_network(scalars, [0.12] * 12)


@dataclass(frozen=True)
class ArgmaxResult:
    """What `ArgmaxStatistic.finish` returns."""

    argmax_frequency: pl.DataFrame
    high_vertices: np.ndarray
    draw_stats: pl.DataFrame
    n_draws: int
    seed: int
    mode: str
    scale: float


@dataclass(frozen=True)
class ArgmaxStatistic:
    """Test-local `DrawStatistic`: which vertex holds the extreme drawn scalar of each draw.

    Counts have two keys. ``argmax`` is a per-vertex tally of "is the argmax"
    (``mode="max"``) or "is the argmin" (``mode="min"``). ``high`` is a variable-length
    int64 array of the vertex indices whose drawn value exceeds ``HIGH`` (empty for some
    draws), joined by sorted concatenation so that `add` is
    order-independent. The stats row is
    ``[extreme value, argmax vertex]``. Like a real
    statistic it binds to one network in `prepare`, refuses another, and fingerprints its
    inputs. ``scale`` is a finish-only option so chunks can be finished again.
    """

    mode: str = "max"
    bound_to: str | None = None

    def prepare(self, network: ScalarNetwork) -> ArgmaxStatistic:
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}.")
        digest = hashlib.sha256()
        digest.update(repr(network.vertex_ids).encode())
        digest.update(np.asarray(network.graph.edge_list(), dtype="<i8").tobytes())
        network_hash = digest.hexdigest()
        if self.bound_to is not None:
            if self.bound_to != network_hash:
                raise ValueError("this statistic is already bound to another network.")
            return self
        return replace(self, bound_to=network_hash)

    def measure(self, network: DrawnNetwork) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if self.bound_to is None:
            raise ValueError("statistic is not bound to a network; call prepare(network) first.")
        values = network.values
        index = int(np.argmax(values) if self.mode == "max" else np.argmin(values))
        tally = np.zeros(len(values), dtype=np.int64)
        tally[index] = 1
        high = np.flatnonzero(values > HIGH).astype(np.int64)
        return {"argmax": tally, "high": high}, np.array([values[index], index], dtype=np.float64)

    def add(self, first, second):
        return {
            "argmax": first["argmax"] + second["argmax"],
            "high": np.sort(np.concatenate([first["high"], second["high"]])),
        }

    def finish(self, counts, stats, n_draws, **options) -> ArgmaxResult:
        scale = options.get("scale", 1.0)
        if not scale > 0:
            raise ValueError("scale must be positive.")
        frequency = counts["argmax"] / n_draws * scale
        return ArgmaxResult(
            argmax_frequency=pl.DataFrame(
                {"vertex": np.arange(len(frequency)), "frequency": frequency}
            ),
            high_vertices=counts["high"],
            draw_stats=pl.DataFrame(
                {"draw": np.arange(n_draws), "extreme": stats[:, 0], "vertex": stats[:, 1]}
            ),
            n_draws=n_draws,
            seed=options["seed"],
            mode=self.mode,
            scale=scale,
        )

    def fingerprint(self) -> str:
        if self.bound_to is None:
            raise ValueError("statistic is not bound to a network; call prepare(network) first.")
        return f"argmax:{self.mode}:{self.bound_to}"


def assert_results_identical(first: ArgmaxResult, second: ArgmaxResult) -> None:
    assert first.argmax_frequency.equals(second.argmax_frequency)
    assert np.array_equal(first.high_vertices, second.high_vertices)
    assert first.draw_stats.equals(second.draw_stats)
    assert (first.n_draws, first.seed, first.mode) == (second.n_draws, second.seed, second.mode)


def statistic() -> ArgmaxStatistic:
    return ArgmaxStatistic("max")


def slices() -> list[range]:
    return [range(a, min(a + SLICE, N_DRAWS)) for a in range(0, N_DRAWS, SLICE)]


# --- seeding ----------------------------------------------------------------------------------


def test_draw_rng_equals_spawned_child_state() -> None:
    children = np.random.SeedSequence(SEED).spawn(N_DRAWS)
    for draw in (0, 1, 17, N_DRAWS - 1):
        ours = _draw_rng(SEED, draw)
        theirs = np.random.default_rng(children[draw])
        assert ours.bit_generator.state == theirs.bit_generator.state
        assert np.array_equal(ours.normal(size=5), theirs.normal(size=5))


# --- chunked runs equal one run ---------------------------------------------------------------


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_slices_combine_bit_identically_to_one_run(
    network: ScalarNetwork, tmp_path: Path, n_jobs: int
) -> None:
    stat = statistic().prepare(network)
    whole = bootstrap.run(network, stat, n_draws=N_DRAWS, seed=SEED)
    expected = bootstrap.combine([whole], stat)

    paths = []
    for index, draws in enumerate(slices()):
        chunk = bootstrap.run(
            network, stat, n_draws=N_DRAWS, seed=SEED, draws=draws, n_jobs=n_jobs
        )
        assert chunk.draws.tolist() == list(draws)
        path = tmp_path / f"chunk{index}.npz"
        chunk.save(path)
        paths.append(path)
    assert len(slices()[-1]) < SLICE

    loaded = [BootstrapChunk.load(path) for path in reversed(paths)]
    combined = bootstrap.combine(loaded, stat)

    assert_results_identical(expected, combined)
    assert combined.seed == SEED
    assert combined.draw_stats["draw"].to_list() == list(range(N_DRAWS))
    for name in whole.counts:
        assert np.array_equal(whole.counts[name], sum_counts(loaded, stat)[name])


def sum_counts(chunks: list[BootstrapChunk], stat: ArgmaxStatistic):
    total = chunks[0].counts
    for chunk in chunks[1:]:
        total = stat.add(total, chunk.counts)
    return total


def test_parallel_whole_run_equals_serial(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    serial = bootstrap.run(network, stat, n_draws=N_DRAWS, seed=SEED)
    parallel = bootstrap.run(network, stat, n_draws=N_DRAWS, seed=SEED, n_jobs=2)
    assert np.array_equal(serial.stats, parallel.stats)
    for name in serial.counts:
        assert np.array_equal(serial.counts[name], parallel.counts[name])


def test_finish_options_apply_only_at_combine(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=10, seed=SEED)
    plain = bootstrap.combine([chunk], stat)
    scaled = bootstrap.combine([chunk], stat, scale=0.5)
    assert plain.scale == 1.0 and scaled.scale == 0.5
    assert plain.draw_stats.equals(scaled.draw_stats)
    assert scaled.argmax_frequency["frequency"].to_list() == pytest.approx(
        [0.5 * f for f in plain.argmax_frequency["frequency"].to_list()]
    )
    with pytest.raises(ValueError, match="scale"):
        bootstrap.combine([chunk], stat, scale=0.0)


# --- combine refusals -------------------------------------------------------------------------


def test_combine_refuses_mismatched_fingerprint(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    first = bootstrap.run(network, stat, n_draws=10, seed=SEED, draws=range(5))
    other = make_network([float(x) for x in np.linspace(0, 1, 12)], [0.12] * 12)
    second = bootstrap.run(
        other, statistic().prepare(other), n_draws=10, seed=SEED, draws=range(5, 10)
    )
    assert first.fingerprint != second.fingerprint
    with pytest.raises(ValueError, match="fingerprint"):
        bootstrap.combine([first, second], stat)


def test_fingerprint_depends_on_se(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    wider = make_network([network.graph.get_node_data(i)["scalar"] for i in range(12)], [0.2] * 12)
    one = bootstrap.run(network, stat, n_draws=2, seed=1)
    two = bootstrap.run(wider, stat, n_draws=2, seed=1)
    assert one.fingerprint != two.fingerprint


def test_combine_refuses_other_seed_or_n_draws(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    base = bootstrap.run(network, stat, n_draws=10, seed=SEED, draws=range(5))
    rest = bootstrap.run(network, stat, n_draws=10, seed=SEED, draws=range(5, 10))
    with pytest.raises(ValueError, match="seed"):
        bootstrap.combine([base, replace(rest, seed=SEED + 1)], stat)
    with pytest.raises(ValueError, match="n_draws"):
        bootstrap.combine([base, replace(rest, n_draws=11)], stat)


def test_combine_refuses_overlapping_draws(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    first = bootstrap.run(network, stat, n_draws=10, seed=SEED, draws=range(6))
    second = bootstrap.run(network, stat, n_draws=10, seed=SEED, draws=range(5, 10))
    with pytest.raises(ValueError, match="overlap on draws 5"):
        bootstrap.combine([first, second], stat)


def test_combine_refuses_missing_draws(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    first = bootstrap.run(network, stat, n_draws=10, seed=SEED, draws=range(4))
    second = bootstrap.run(network, stat, n_draws=10, seed=SEED, draws=range(6, 10))
    with pytest.raises(ValueError, match="2 of 10 draws are missing: 4-5"):
        bootstrap.combine([first, second], stat)


def test_combine_refuses_no_chunks_and_other_statistic(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=3, seed=SEED)
    with pytest.raises(ValueError, match="at least one chunk"):
        bootstrap.combine([], stat)
    other = ArgmaxStatistic("min").prepare(network)
    with pytest.raises(ValueError, match="statistic"):
        bootstrap.combine([chunk], other)
    with pytest.raises(ValueError, match="prepare"):
        bootstrap.combine([chunk], statistic())


def test_combine_refuses_chunk_inconsistent_with_itself(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=6, seed=SEED)
    with pytest.raises(ValueError, match="chunk 0 has 5 stats rows for 6 draws"):
        bootstrap.combine([replace(chunk, stats=chunk.stats[:-1])], stat)
    repeated = replace(chunk, draws=np.array([0, 1, 2, 3, 4, 4]))
    with pytest.raises(ValueError, match="chunk 0 lists the same draw more than once"):
        bootstrap.combine([repeated], stat)


# --- prepare binds one network ----------------------------------------------------------------


def test_bound_statistic_refuses_another_network_and_keeps_inputs(
    network: ScalarNetwork,
) -> None:
    bound = statistic().prepare(network)
    assert bound.prepare(network) is bound
    assert bound.mode == "max"
    other = make_network([float(x) for x in np.linspace(0, 1, 12)], [0.12] * 12, ((1, 6),))
    with pytest.raises(ValueError, match="already bound to another network"):
        bound.prepare(other)
    assert statistic().prepare(other) is not None


# --- fingerprint ------------------------------------------------------------------------------


def scalars_of(network: ScalarNetwork) -> list[float]:
    return [network.graph.get_node_data(i)["scalar"] for i in range(12)]


def test_fingerprint_equals_the_chunk_fingerprint(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=3, seed=SEED)
    assert bootstrap.fingerprint(network, stat) == chunk.fingerprint
    with pytest.raises(ValueError, match="prepare"):
        bootstrap.fingerprint(network, statistic())


def test_fingerprint_changes_with_edges_and_statistic_settings(
    network: ScalarNetwork,
) -> None:
    base = bootstrap.fingerprint(network, statistic().prepare(network))

    more_edges = make_network(scalars_of(network), [0.12] * 12, ((1, 6),))
    assert bootstrap.fingerprint(more_edges, statistic().prepare(more_edges)) != base

    smallest = ArgmaxStatistic("min").prepare(network)
    assert bootstrap.fingerprint(network, smallest) != base


def test_single_draw_bootstrap_combines(network: ScalarNetwork) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=1, seed=SEED)
    result = bootstrap.combine([chunk], stat)
    assert result.n_draws == 1
    assert result.draw_stats["draw"].to_list() == [0]
    assert result.argmax_frequency["frequency"].sum() == 1.0


# --- worker signals ---------------------------------------------------------------------------


def test_worker_initializer_ignores_stop_signals() -> None:
    code = (
        "import signal\n"
        "from topographer.statistics.bootstrap import _ignore_signals\n"
        "_ignore_signals()\n"
        "names = [n for n in ('SIGTERM', 'SIGUSR1', 'SIGINT') if hasattr(signal, n)]\n"
        "assert names\n"
        "assert all(signal.getsignal(getattr(signal, n)) is signal.SIG_IGN for n in names)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_IGN


# --- run arguments ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_draws": 0},
        {"n_draws": True},
        {"n_jobs": 0},
        {"seed": -1},
        {"seed": None},
        {"draws": range(11)},
        {"draws": range(-1, 3)},
        {"draws": range(3, 3)},
        {"draws": [1, 2]},
    ],
)
def test_run_rejects_invalid_arguments(network: ScalarNetwork, kwargs: dict) -> None:
    arguments = {"n_draws": 10, "seed": 1, **kwargs}
    with pytest.raises(ValueError):
        bootstrap.run(network, statistic(), **arguments)


# --- save and load ----------------------------------------------------------------------------


def test_save_load_round_trip(network: ScalarNetwork, tmp_path: Path) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=12, seed=SEED, draws=range(3, 9))
    path = tmp_path / "chunk.npz"
    chunk.save(path)
    loaded = BootstrapChunk.load(path)

    assert loaded.draws.tolist() == [3, 4, 5, 6, 7, 8]
    assert (loaded.n_draws, loaded.seed) == (12, SEED)
    assert loaded.fingerprint == chunk.fingerprint
    assert loaded.statistic_fingerprint == chunk.statistic_fingerprint
    assert np.array_equal(loaded.stats, chunk.stats)
    assert set(loaded.counts) == set(chunk.counts)
    for name, array in chunk.counts.items():
        assert np.array_equal(loaded.counts[name], array)
        assert loaded.counts[name].dtype == array.dtype


def test_save_leaves_no_temporary_file_and_replaces_atomically(
    network: ScalarNetwork, tmp_path: Path
) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=4, seed=1)
    path = tmp_path / "chunk.npz"
    chunk.save(path)
    chunk.save(path)
    assert [entry.name for entry in tmp_path.iterdir()] == ["chunk.npz"]


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows does not apply POSIX umask bits to the saved chunk",
)
def test_saved_chunk_mode_honours_the_umask(network: ScalarNetwork, tmp_path: Path) -> None:
    chunk = bootstrap.run(network, statistic().prepare(network), n_draws=2, seed=1)
    previous = os.umask(0o022)
    try:
        chunk.save(tmp_path / "chunk.npz")
        os.umask(0o077)
        chunk.save(tmp_path / "private.npz")
    finally:
        os.umask(previous)
    assert stat_module.S_IMODE((tmp_path / "chunk.npz").stat().st_mode) == 0o644
    assert stat_module.S_IMODE((tmp_path / "private.npz").stat().st_mode) == 0o600


def test_failed_save_keeps_old_file_and_removes_temporary(
    network: ScalarNetwork, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=4, seed=1)
    path = tmp_path / "chunk.npz"
    chunk.save(path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(bootstrap.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        chunk.save(path)
    assert path.read_bytes() == before
    assert [entry.name for entry in tmp_path.iterdir()] == ["chunk.npz"]


def test_load_rejects_unknown_version_and_foreign_files(
    network: ScalarNetwork, tmp_path: Path
) -> None:
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=3, seed=1)
    path = tmp_path / "chunk.npz"
    chunk.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["format_version"] = np.array(99)
    future = tmp_path / "future.npz"
    np.savez(future, **arrays)
    with pytest.raises(ValueError, match="version 99"):
        BootstrapChunk.load(future)

    foreign = tmp_path / "foreign.npz"
    np.savez(foreign, x=np.arange(3))
    with pytest.raises(ValueError, match="format_version"):
        BootstrapChunk.load(foreign)


# --- progress ---------------------------------------------------------------------------------


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_progress_callback_reports_done_and_total(network: ScalarNetwork, n_jobs: int) -> None:
    calls: list[tuple[int, int]] = []
    bootstrap.run(
        network,
        statistic(),
        n_draws=20,
        seed=SEED,
        draws=range(2, 14),
        n_jobs=n_jobs,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls
    assert all(total == 12 for _, total in calls)
    assert [done for done, _ in calls] == sorted(done for done, _ in calls)
    assert calls[-1] == (12, 12)
    if n_jobs == 1:
        assert [done for done, _ in calls] == list(range(1, 13))


# --- multiple count keys, empty arrays and the draw stream ------------------------------------


def node_order_arrays(network: ScalarNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Scalars and standard errors in node-index order, as the engine draws them."""
    graph = network.graph
    values = np.array([graph.get_node_data(i)["scalar"] for i in graph.node_indices()])
    se = np.array([graph.get_node_data(i)["se"] for i in graph.node_indices()])
    return values, se


def test_draw_stream_pins_recorded_extremes_and_high_vertices(network: ScalarNetwork) -> None:
    """Per-draw results equal the raw ``SeedSequence(seed, spawn_key=(draw,))`` normal draws."""
    values, se = node_order_arrays(network)
    stat = statistic().prepare(network)
    chunk = bootstrap.run(network, stat, n_draws=N_DRAWS, seed=SEED)

    expected_high = []
    for draw in range(N_DRAWS):
        drawn = _draw_rng(SEED, draw).normal(values, se)
        assert chunk.stats[draw, 0] == drawn.max()
        assert chunk.stats[draw, 1] == int(np.argmax(drawn))
        expected_high.append(np.flatnonzero(drawn > HIGH))

    for draw in (0, 5, 17, N_DRAWS - 1):
        assert chunk.stats[draw, 0] == max(_draw_rng(SEED, draw).normal(values, se))
    assert np.array_equal(chunk.counts["high"], np.sort(np.concatenate(expected_high)))


def test_high_counts_are_sometimes_empty_and_vary_in_length(network: ScalarNetwork) -> None:
    values, se = node_order_arrays(network)
    lengths = [
        int((_draw_rng(SEED, draw).normal(values, se) > HIGH).sum()) for draw in range(N_DRAWS)
    ]
    assert 0 in lengths
    assert len(set(lengths)) > 2
    assert set(bootstrap.run(network, statistic(), n_draws=1, seed=SEED).counts) == {
        "argmax",
        "high",
    }


def test_empty_array_counts_survive_save_load_and_combine(
    network: ScalarNetwork, tmp_path: Path
) -> None:
    values, se = node_order_arrays(network)
    empty_draws = [
        draw
        for draw in range(N_DRAWS)
        if not (_draw_rng(SEED, draw).normal(values, se) > HIGH).any()
    ]
    assert empty_draws
    stat = statistic().prepare(network)

    empty = empty_draws[0]
    chunk = bootstrap.run(network, stat, n_draws=N_DRAWS, seed=SEED, draws=range(empty, empty + 1))
    assert chunk.counts["high"].shape == (0,)
    path = tmp_path / "empty.npz"
    chunk.save(path)
    loaded = BootstrapChunk.load(path)
    assert loaded.counts["high"].shape == (0,)
    assert loaded.counts["high"].dtype == np.int64
    assert np.array_equal(loaded.counts["argmax"], chunk.counts["argmax"])

    rest = [d for d in range(N_DRAWS) if d != empty]
    others = [
        bootstrap.run(network, stat, n_draws=N_DRAWS, seed=SEED, draws=range(a, a + 1))
        for a in range(N_DRAWS)
        if a != empty
    ]
    assert len(rest) == len(others)
    combined = bootstrap.combine([loaded, *others], stat)
    whole = bootstrap.combine([bootstrap.run(network, stat, n_draws=N_DRAWS, seed=SEED)], stat)
    assert_results_identical(whole, combined)
