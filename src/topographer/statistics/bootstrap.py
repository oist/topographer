"""Chunked, resumable parametric bootstrap on a scalar network.

This is the network layer of the chunked bootstrap. It knows how to draw a landscape (every vertex scalar
replaced by ``Normal(scalar, se)``), how to run a slice of draws, how to save and load a slice,
and how to combine slices. What is measured on a drawn landscape belongs to a `DrawStatistic`,
and nothing here knows about trees, leaves or peaks.

Draw ``d`` of a run with seed ``s`` uses
``numpy.random.default_rng(numpy.random.SeedSequence(s, spawn_key=(d,)))``, which equals child
``d`` of ``SeedSequence(s).spawn(n_draws)``. A draw therefore depends on nothing but ``(s, d)``,
so any partition of ``0..n_draws-1`` into slices, run in any order or process, combines to the
result of one run, bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import functools
import hashlib
import multiprocessing
import numbers
import os
from pathlib import Path
import signal
import tempfile
from typing import Any, Protocol, runtime_checkable

import numpy as np
import rustworkx as rx

from topographer.graph import ScalarNetwork
from topographer.statistics._standard_errors import validate_se_attr

__all__ = ["BootstrapChunk", "DrawStatistic", "DrawnNetwork", "combine", "fingerprint", "run"]

Counts = Mapping[str, np.ndarray]

CHUNK_FORMAT_VERSION = 1
_CHUNKS_PER_WORKER = 4
_FOLD_EVERY = 32
_COUNT_PREFIX = "counts__"


@dataclass(frozen=True)
class DrawnNetwork:
    """One drawn landscape handed to `DrawStatistic.measure`.

    The graph is reused and overwritten by the next draw, so a statistic must not keep it.

    Attributes
    ----------
    graph : rx.PyGraph
        Rustworkx graph with node indices ``0..n-1`` and the network's edges. Every node
        payload is ``{scalar_attr: value}`` and nothing else.
    scalar_attr : str
        Payload key that holds the drawn value.
    values : np.ndarray
        The drawn values in node-index order, read-only.
    """

    graph: rx.PyGraph
    scalar_attr: str
    values: np.ndarray


@runtime_checkable
class DrawStatistic(Protocol):
    """What is measured on every drawn landscape, and how the measurements are combined.

    ``counts`` is a mapping of names to integer ``numpy`` arrays so a chunk can be stored
    without pickle. ``add`` must be exact, commutative and associative (integer sums) and give
    a canonical form, so combining in any grouping or order is bit-identical.
    """

    def prepare(self, network: ScalarNetwork) -> DrawStatistic:
        """Return a statistic bound to ``network`` (validated, picklable, fingerprintable).

        Called once by `run` before any draw. A statistic that needs nothing from the network
        returns itself.
        """
        ...

    def measure(self, network: DrawnNetwork) -> tuple[Counts, np.ndarray]:
        """Return ``(counts, stats_row)`` for one drawn landscape.

        ``stats_row`` is a one-dimensional float array of the same length for every draw.
        """
        ...

    def add(self, first: Counts, second: Counts) -> Counts:
        """Return the exact sum of two counts mappings."""
        ...

    def finish(self, counts: Counts, stats: np.ndarray, n_draws: int, **options: Any) -> Any:
        """Turn the summed counts and the ``(n_draws, k)`` stats rows into the final result.

        ``stats`` rows are in draw order ``0..n_draws-1``. `combine` always passes the
        bootstrap's ``seed`` as a keyword option, beside the caller's ``options``, which are
        applied only here so saved chunks can be finished again with other options.
        """
        ...

    def fingerprint(self) -> str:
        """Return a string that changes whenever a draw's measurement would change."""
        ...


@dataclass(frozen=True)
class BootstrapChunk:
    """The summed counts and per-draw statistics of a set of draws of one bootstrap.

    Attributes
    ----------
    counts : Mapping[str, np.ndarray]
        The statistic's counts summed over `draws`.
    stats : np.ndarray
        Float64 array with one row per entry of `draws`, in the same order.
    draws : np.ndarray
        Int64 draw ids covered, each in ``0..n_draws-1``.
    n_draws : int
        Total number of draws of the whole bootstrap this chunk belongs to.
    seed : int
        Seed of the whole bootstrap.
    fingerprint : str
        Hash of the network values, edges, standard errors and the statistic's fingerprint.
    statistic_fingerprint : str
        The statistic's own fingerprint, which `combine` checks against the statistic it gets.
    """

    counts: Counts
    stats: np.ndarray
    draws: np.ndarray
    n_draws: int
    seed: int
    fingerprint: str
    statistic_fingerprint: str

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the chunk to ``path`` as a versioned ``.npz``, atomically.

        The data go to a temporary file in the same directory, are flushed and synced, and
        replace ``path`` in one step, so a crash leaves either the old file or the new one and
        no partial file. The path is used exactly as given (no suffix is added). The file mode
        is ``0o666 & ~umask`` (``0o644`` under the usual umask), not the private ``0o600`` of a
        temporary file, so other users and jobs of the group can read saved chunks. The parent
        directory is synced after the replace where the platform allows it (best effort).
        """
        target = Path(path)
        arrays: dict[str, Any] = {
            "format_version": np.array(CHUNK_FORMAT_VERSION, dtype=np.int64),
            "n_draws": np.array(self.n_draws, dtype=np.int64),
            "seed": np.array(str(self.seed)),
            "fingerprint": np.array(self.fingerprint),
            "statistic_fingerprint": np.array(self.statistic_fingerprint),
            "draws": np.asarray(self.draws, dtype=np.int64),
            "stats": np.asarray(self.stats, dtype=np.float64),
        }
        for name, array in self.counts.items():
            arrays[_COUNT_PREFIX + name] = np.asarray(array)
        handle = tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        )
        temporary = Path(handle.name)
        try:
            with handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o666 & ~_current_umask())
            os.replace(temporary, target)
            _sync_directory(target.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BootstrapChunk:
        """Read a chunk written by `save`.

        Raises
        ------
        ValueError
            If the file has no format version, an unknown one, or lacks a required entry.
        """
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "format_version" not in names:
                raise ValueError(f"{path} is not a bootstrap chunk: no format_version.")
            version = int(archive["format_version"])
            if version != CHUNK_FORMAT_VERSION:
                raise ValueError(
                    f"{path} has chunk format version {version}; this topographer reads "
                    f"version {CHUNK_FORMAT_VERSION}."
                )
            required = {
                "n_draws",
                "seed",
                "fingerprint",
                "statistic_fingerprint",
                "draws",
                "stats",
            }
            if required - names:
                raise ValueError(f"{path} is missing entries {sorted(required - names)}.")
            counts = {
                name[len(_COUNT_PREFIX) :]: archive[name]
                for name in sorted(names)
                if name.startswith(_COUNT_PREFIX)
            }
            return cls(
                counts=counts,
                stats=archive["stats"],
                draws=archive["draws"],
                n_draws=int(archive["n_draws"]),
                seed=int(str(archive["seed"])),
                fingerprint=str(archive["fingerprint"]),
                statistic_fingerprint=str(archive["statistic_fingerprint"]),
            )


def _current_umask() -> int:
    """Return the process umask (it can only be read by setting it, so set it back at once)."""
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _sync_directory(directory: Path) -> None:
    """Flush a directory entry to disk after a rename; best effort, skipped without support."""
    flag = getattr(os, "O_DIRECTORY", None)
    if flag is None:
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY | flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


# --- drawing ----------------------------------------------------------------------------------


def _draw_rng(seed: int, draw: int) -> np.random.Generator:
    """Return the generator of draw ``draw``: child ``draw`` of ``SeedSequence(seed).spawn``."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(draw,)))


@dataclass(frozen=True)
class _Setup:
    """Inputs shared by every draw."""

    graph: rx.PyGraph
    scalar_attr: str
    values: np.ndarray
    se: np.ndarray
    statistic: DrawStatistic
    seed: int


def _node_order_values(graph: rx.PyGraph, scalar_attr: str) -> np.ndarray:
    """Return vertex scalars in node-index order, requiring indices ``0..n-1``."""
    indices = [int(index) for index in graph.node_indices()]
    if indices != list(range(len(indices))):
        raise ValueError("Network node indices must be contiguous from 0.")
    return np.array([graph.get_node_data(i)[scalar_attr] for i in indices], dtype=np.float64)


def _scalar_only_graph(graph: rx.PyGraph, scalar_attr: str, values: np.ndarray) -> rx.PyGraph:
    """Return a graph with the same nodes and edges whose payloads hold only the scalar.

    Draws rewrite the scalar payloads in place and every tree build copies them, so the draw
    graph carries nothing else. Node indices ``0..n-1`` and the edge order are kept.
    """
    minimal = rx.PyGraph(multigraph=graph.multigraph)
    minimal.add_nodes_from([{scalar_attr: float(value)} for value in values])
    minimal.add_edges_from_no_data(graph.edge_list())
    return minimal


def _measure_draw(setup: _Setup, draw: int) -> tuple[Counts, np.ndarray]:
    drawn = _draw_rng(setup.seed, draw).normal(loc=setup.values, scale=setup.se)
    graph = setup.graph
    for node, value in zip(graph.node_indices(), drawn):
        graph[node][setup.scalar_attr] = float(value)
    drawn.setflags(write=False)
    counts, row = setup.statistic.measure(DrawnNetwork(graph, setup.scalar_attr, drawn))
    return counts, np.asarray(row, dtype=np.float64).ravel()


def _fold(statistic: DrawStatistic, accumulated: Counts | None, pending: list[Counts]) -> Counts:
    """Add the pending per-draw counts into ``accumulated`` (one big add per batch)."""
    batch = functools.reduce(statistic.add, pending)
    return batch if accumulated is None else statistic.add(accumulated, batch)


def _run_draws(
    setup: _Setup, ids: Sequence[int], on_draw: Callable[[], None] | None = None
) -> tuple[Counts, np.ndarray]:
    """Run draws ``ids`` in order; return their summed counts and stats rows."""
    accumulated: Counts | None = None
    pending: list[Counts] = []
    rows: list[np.ndarray] = []
    for draw in ids:
        counts, row = _measure_draw(setup, int(draw))
        pending.append(counts)
        rows.append(row)
        if len(pending) >= _FOLD_EVERY:
            accumulated, pending = _fold(setup.statistic, accumulated, pending), []
        if on_draw is not None:
            on_draw()
    if pending:
        accumulated = _fold(setup.statistic, accumulated, pending)
    return accumulated, np.vstack(rows)  # type: ignore[return-value]


_WORKER_SETUP: _Setup | None = None


_IGNORED_SIGNALS = ("SIGTERM", "SIGUSR1", "SIGINT")


def _ignore_signals() -> None:
    """Make this process ignore SIGTERM, SIGUSR1 and SIGINT (those the platform has).

    A scheduler such as Slurm signals the whole process group. Workers that die on it make the
    pool raise ``BrokenProcessPool`` in the parent, so workers ignore these signals and the
    parent decides when to stop.
    """
    for name in _IGNORED_SIGNALS:
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, signal.SIG_IGN)


def _init_worker(setup: _Setup) -> None:
    """Ignore stop signals and store the shared draw inputs once per worker process."""
    global _WORKER_SETUP
    _ignore_signals()
    _WORKER_SETUP = setup


def _run_chunk(start: int, ids: Sequence[int]) -> tuple[int, Counts, np.ndarray]:
    """Worker task: run a contiguous slice of the requested draws."""
    if _WORKER_SETUP is None:
        raise RuntimeError("Worker was not initialised.")
    counts, stats = _run_draws(_WORKER_SETUP, ids)
    return start, counts, stats


def _sum_serial(
    setup: _Setup, ids: Sequence[int], progress: Callable[[int, int], None] | None
) -> tuple[Counts, np.ndarray]:
    done = 0

    def on_draw() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, len(ids))

    return _run_draws(setup, ids, on_draw)


def _sum_parallel(
    setup: _Setup,
    ids: Sequence[int],
    n_jobs: int,
    progress: Callable[[int, int], None] | None,
) -> tuple[Counts, np.ndarray]:
    """Run the draws in spawned worker processes; return summed counts and stats rows.

    The ``spawn`` start method is used because the parent has already started Polars threads,
    and forking a multi-threaded process can deadlock. Workers ignore SIGTERM, SIGUSR1 and
    SIGINT, so the parent alone decides when to stop.
    """
    size = max(1, -(-len(ids) // (n_jobs * _CHUNKS_PER_WORKER)))
    chunks = [(start, ids[start : start + size]) for start in range(0, len(ids), size)]
    stats_by_start: dict[int, np.ndarray] = {}
    total: Counts | None = None
    done = 0
    with ProcessPoolExecutor(
        max_workers=min(n_jobs, len(chunks)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(setup,),
    ) as pool:
        futures = [pool.submit(_run_chunk, start, chunk) for start, chunk in chunks]
        try:
            for future in as_completed(futures):
                start, counts, chunk_stats = future.result()
                total = counts if total is None else setup.statistic.add(total, counts)
                stats_by_start[start] = chunk_stats
                done += len(chunk_stats)
                if progress is not None:
                    progress(done, len(ids))
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    stats = np.vstack([stats_by_start[start] for start in sorted(stats_by_start)])
    return total, stats  # type: ignore[return-value]


# --- validation -------------------------------------------------------------------------------


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    return int(seed)


def _draw_ids(draws: range | None, n_draws: int) -> np.ndarray:
    """Return the draw ids to run, checking that they lie in ``0..n_draws-1``."""
    if draws is None:
        return np.arange(n_draws, dtype=np.int64)
    if not isinstance(draws, range):
        raise ValueError("draws must be a range or None.")
    if len(draws) == 0:
        raise ValueError("draws is empty: there is nothing to run.")
    if min(draws) < 0 or max(draws) >= n_draws:
        raise ValueError(f"draws must lie within 0..{n_draws - 1}, got {draws!r}.")
    return np.array(draws, dtype=np.int64)


def _setup_fingerprint(
    network: ScalarNetwork,
    values: np.ndarray,
    se: np.ndarray,
    statistic_fingerprint: str,
) -> str:
    """Hash everything that decides a draw: values, standard errors, edges and statistic."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(se, dtype="<f8").tobytes())
    edges = np.asarray(network.graph.edge_list(), dtype="<i8").reshape(-1, 2)
    digest.update(edges.tobytes())
    digest.update(statistic_fingerprint.encode())
    return digest.hexdigest()


def fingerprint(network: ScalarNetwork, statistic: DrawStatistic, *, se_attr: str = "se") -> str:
    """Return the setup fingerprint that `run` stores in every chunk it makes.

    It hashes the vertex values, the standard errors, the edge list and the statistic's own
    fingerprint, so it tells whether chunks belong to one bootstrap of ``network`` without
    running a draw.

    Parameters
    ----------
    network : ScalarNetwork
        Network that is (or was) bootstrapped.
    statistic : DrawStatistic
        The statistic **already bound** to ``network``, that is ``statistic.prepare(network)``;
        an unbound statistic raises ``ValueError`` from its own ``fingerprint``.
    se_attr : str, default "se"
        Vertex attribute with the standard errors.

    Returns
    -------
    str
        Equal to ``run(network, statistic, ...).fingerprint``.
    """
    se = validate_se_attr(network, se_attr=se_attr)
    values = _node_order_values(network.graph, network.scalar_attr)
    return _setup_fingerprint(network, values, se, statistic.fingerprint())


# --- public API -------------------------------------------------------------------------------


def run(
    network: ScalarNetwork,
    statistic: DrawStatistic,
    *,
    se_attr: str = "se",
    n_draws: int,
    seed: int,
    draws: range | None = None,
    n_jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> BootstrapChunk:
    """Run draws of a parametric bootstrap and return their summed measurements.

    Every draw replaces each vertex scalar by ``Normal(scalar, se)`` independently and calls
    ``statistic.measure`` on the drawn network. Draw ``d`` uses
    ``default_rng(SeedSequence(seed, spawn_key=(d,)))``, so ``draws=range(a, b)`` of an
    ``n_draws`` bootstrap gives exactly the draws ``a..b-1`` of the full run, and chunks of
    any partition combine (see `combine`) to the result of one run, bit for bit. The input
    network is never modified.

    With ``n_jobs > 1`` the workers are started with the ``spawn`` method, so a script that
    calls this function must guard its entry point with ``if __name__ == "__main__":``.

    Parameters
    ----------
    network : ScalarNetwork
        Network whose scalar attribute is the landscape that is drawn around.
    statistic : DrawStatistic
        What to measure on each draw. ``statistic.prepare(network)`` is called first.
    se_attr : str, default "se"
        Vertex attribute with finite, non-negative standard errors (zero keeps a vertex fixed).
    n_draws : int
        Size of the whole bootstrap, at least 1.
    seed : int
        Non-negative integer seed of the whole bootstrap.
    draws : range | None, default None
        Draw ids to run, within ``0..n_draws-1``; all of them by default.
    n_jobs : int, default 1
        Worker processes; 1 runs in-process.
    progress : Callable[[int, int], None] | None, default None
        Called as ``progress(done, total)`` after each draw (``n_jobs == 1``) or each finished
        group of draws (``n_jobs > 1``); ``total`` is the number of draws of this call.

    Returns
    -------
    BootstrapChunk
        Summed counts, per-draw stats rows with their real draw ids, and the setup fingerprint.

    Raises
    ------
    ValueError
        For an invalid ``n_draws``, ``seed``, ``n_jobs`` or ``draws`` (booleans are rejected),
        invalid standard errors, a statistic prepared with a different ``se_attr``, or whatever
        ``statistic.prepare`` rejects.
    TypeError
        If ``network`` is not a ``ScalarNetwork``.
    """
    n_draws = _positive_int("n_draws", n_draws)
    n_jobs = _positive_int("n_jobs", n_jobs)
    seed = _validate_seed(seed)
    ids = _draw_ids(draws, n_draws)
    se = validate_se_attr(network, se_attr=se_attr)
    bound = statistic.prepare(network)
    _check_statistic_se_attr(bound, se_attr)

    graph = network.graph
    values = _node_order_values(graph, network.scalar_attr)
    setup = _Setup(
        graph=_scalar_only_graph(graph, network.scalar_attr, values),
        scalar_attr=network.scalar_attr,
        values=values,
        se=se,
        statistic=bound,
        seed=seed,
    )
    statistic_fingerprint = bound.fingerprint()
    if n_jobs == 1 or len(ids) == 1:
        counts, stats = _sum_serial(setup, ids, progress)
    else:
        counts, stats = _sum_parallel(setup, ids, n_jobs, progress)
    return BootstrapChunk(
        counts={name: np.asarray(array) for name, array in counts.items()},
        stats=stats,
        draws=ids,
        n_draws=n_draws,
        seed=seed,
        fingerprint=_setup_fingerprint(network, values, se, statistic_fingerprint),
        statistic_fingerprint=statistic_fingerprint,
    )


def _check_statistic_se_attr(statistic: DrawStatistic, se_attr: str) -> None:
    """Raise ``ValueError`` if the statistic declares an ``se_attr`` other than ``se_attr``."""
    declared = getattr(statistic, "se_attr", None)
    if declared is not None and declared != se_attr:
        raise ValueError(
            f"the statistic was prepared with se_attr={declared!r} but run was given "
            f"se_attr={se_attr!r}; they must agree."
        )


def _ranges(ids: np.ndarray, limit: int = 5) -> str:
    """Describe sorted draw ids as ``a-b`` ranges, truncated to ``limit`` ranges."""
    breaks = np.flatnonzero(np.diff(ids) != 1) + 1
    runs = [(int(run[0]), int(run[-1])) for run in np.split(ids, breaks)]
    shown = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs[:limit])
    return shown + (", ..." if len(runs) > limit else "")


def _check_chunk(position: int, chunk: BootstrapChunk) -> None:
    """Raise ``ValueError`` if a chunk is inconsistent with itself."""
    draws = np.asarray(chunk.draws)
    stats = np.asarray(chunk.stats)
    if stats.ndim != 2 or len(stats) != len(draws):
        raise ValueError(
            f"chunk {position} has {len(stats)} stats rows for {len(draws)} draws: "
            "it is corrupt or was edited."
        )
    if len(np.unique(draws)) != len(draws):
        raise ValueError(f"chunk {position} lists the same draw more than once.")


def combine(chunks: Iterable[BootstrapChunk], statistic: DrawStatistic, **finish_options: Any):
    """Combine chunks that together cover every draw once, and finish the statistic.

    ``statistic`` must be bound to the network the chunks were run on, that is the result of
    ``statistic.prepare(network)``; its fingerprint must match the chunks'. The counts are
    summed with ``statistic.add``, the stats rows are put in draw order, and the result is
    ``statistic.finish(counts, stats, n_draws, seed=seed, **finish_options)``. Finish options such as a
    core cutoff are applied only here, so saved chunks can be combined again with others.

    Parameters
    ----------
    chunks : Iterable[BootstrapChunk]
        Chunks of one bootstrap, in any order.
    statistic : DrawStatistic
        The bound statistic the chunks were run with.
    **finish_options
        Passed to ``statistic.finish``.

    Returns
    -------
    Any
        Whatever ``statistic.finish`` returns.

    Raises
    ------
    ValueError
        If there are no chunks; the chunks differ in fingerprint, ``n_draws``, seed or
        statistic fingerprint; a chunk has a different number of stats rows than draws or repeats
        a draw; two chunks share a draw; or a draw is missing.
    """
    chunks = list(chunks)
    if not chunks:
        raise ValueError("combine needs at least one chunk.")
    first = chunks[0]
    for position, chunk in enumerate(chunks):
        _check_chunk(position, chunk)
    for position, chunk in enumerate(chunks[1:], start=1):
        for field_name in ("fingerprint", "n_draws", "seed"):
            if getattr(chunk, field_name) != getattr(first, field_name):
                raise ValueError(
                    f"chunk {position} has a different {field_name} than chunk 0: chunks come "
                    "from different bootstraps."
                )
    expected = statistic.fingerprint()
    if first.statistic_fingerprint != expected:
        raise ValueError(
            "the statistic's fingerprint does not match the chunks': they were run with a "
            "different statistic or reference."
        )

    ids = np.concatenate([np.asarray(chunk.draws, dtype=np.int64) for chunk in chunks])
    if ids.min() < 0 or ids.max() >= first.n_draws:
        raise ValueError(f"chunk draws must lie within 0..{first.n_draws - 1}.")
    unique, repeats = np.unique(ids, return_counts=True)
    if (repeats > 1).any():
        raise ValueError(f"chunks overlap on draws {_ranges(unique[repeats > 1])}.")
    missing = np.setdiff1d(np.arange(first.n_draws), unique)
    if missing.size:
        raise ValueError(
            f"{missing.size} of {first.n_draws} draws are missing: {_ranges(missing)}."
        )

    order = np.argsort(ids)
    stats = np.vstack([np.asarray(chunk.stats) for chunk in chunks])[order]
    counts = functools.reduce(statistic.add, [chunk.counts for chunk in chunks])
    return statistic.finish(counts, stats, first.n_draws, seed=first.seed, **finish_options)
