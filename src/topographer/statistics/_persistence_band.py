"""Bootstrap persistence band (Fasy et al. 2014) on the chunked bootstrap engine.

`BottleneckToNominal` is the per-draw statistic: the bottleneck distance between a draw's
diagram and the nominal network's diagram. Both diagrams are the ordinary H0 pairs plus the
capped global pair. `bootstrap_persistence_band` runs the engine and returns the
`BootstrapPersistenceBand` record of the topology layer. The band covers scalar perturbations
on a fixed graph only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from typing import Any, Literal

import numpy as np
import polars as pl
import rustworkx as rx

from topographer.graph import ScalarNetwork
from topographer.statistics import bootstrap
from topographer.statistics.bootstrap import DrawnNetwork
from topographer.topology import (
    BootstrapPersistenceBand,
    ScalarNetworkTopology,
    band_network_fingerprint,
    build_join_tree,
    build_split_tree,
    build_tree_extended_persistence_pair_table,
    diagram_bottleneck_distance,
    reduce_merge_tree,
)

__all__ = ["BottleneckToNominal", "bootstrap_persistence_band"]

_FILTRATIONS = ("sublevel", "superlevel")

# What a diagram contains and how two diagrams are compared; part of the statistic fingerprint.
_DIAGRAM_CONVENTION = "h0:ordinary+essential-capped;bottleneck:all-kinds,linf"

Counts = Mapping[str, np.ndarray]


def _check_filtration(filtration: object) -> None:
    if filtration not in _FILTRATIONS:
        raise ValueError(f"filtration must be 'sublevel' or 'superlevel', got {filtration!r}.")


def _h0_diagram_points(
    graph: rx.PyGraph, scalar_attr: str, filtration: Literal["sublevel", "superlevel"]
) -> np.ndarray:
    """Return the H0 diagram of a graph as ``(birth, death)`` rows: ordinary and capped essential.

    This is the one construction behind both the nominal diagram and every draw's diagram. It
    builds the tree exactly as ``ScalarNetworkTopology.persistence_diagram(filtration,
    include_essential="capped")`` does, but needs only a rustworkx graph whose node payloads
    hold ``scalar_attr``, so it runs on the scalar-only graph of a `DrawnNetwork`.
    """
    build, kind = (
        (build_split_tree, "split") if filtration == "superlevel" else (build_join_tree, "join")
    )
    reduced = reduce_merge_tree(build(graph, scalar_attr=scalar_attr).tree)
    table = build_tree_extended_persistence_pair_table(reduced, tree_kind=kind)
    return np.column_stack(
        (
            table["birth_scalar"].to_numpy().astype(np.float64),
            table["death_scalar"].to_numpy().astype(np.float64),
        )
    )


def _as_diagram(points: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"birth_scalar": points[:, 0], "death_scalar": points[:, 1]})


class BottleneckToNominal:
    """Per-draw statistic of the bootstrap persistence band: bottleneck distance to the nominal.

    For each bootstrap draw it builds the draw's H0 persistence diagram (ordinary pairs plus
    the capped global pair) for one filtration and returns its bottleneck distance to the
    nominal network's diagram, as computed by `topographer.topology.diagram_bottleneck_distance`
    over all pair kinds with the L-infinity ground metric. There are no count arrays and one
    stats column. It implements the ``DrawStatistic`` protocol of
    `topographer.statistics.bootstrap`, so draws can be run in chunks, saved, and combined.

    ``prepare(network)`` returns a new statistic bound to ``network`` (it keeps the nominal
    diagram and a hash of the network); preparing the bound statistic again on the same
    network returns it, on another network raises ``ValueError``. ``measure``, ``finish`` and
    ``fingerprint`` need a bound statistic. ``finish(counts, stats, n_draws, *, seed,
    quantile=0.95)`` returns the `BootstrapPersistenceBand`; ``quantile`` is applied only
    there, so saved chunks can be combined at any band quantile.

    Parameters
    ----------
    filtration : {"sublevel", "superlevel"}
        ``"superlevel"`` uses the split tree, ``"sublevel"`` the join tree.
    se_attr : str, default "se"
        Vertex attribute with the standard errors. It must be the same attribute that is given
        to `topographer.statistics.bootstrap.run`.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.graph import Graph, ScalarNetwork
    >>> from topographer.statistics import BottleneckToNominal, bootstrap
    >>> graph = Graph.from_tables(
    ...     pl.DataFrame(
    ...         {"id": ["a", "b", "c"], "x": [1.0, 0.0, 0.8], "se": [0.0, 0.0, 0.1]}
    ...     ),
    ...     pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]}),
    ...     vertex_id_col="id",
    ...     source_id_col="src",
    ...     target_id_col="dst",
    ...     vertex_attribute_cols=("x", "se"),
    ... )
    >>> network = ScalarNetwork(graph, scalar_attr="x")
    >>> statistic = BottleneckToNominal("superlevel").prepare(network)
    >>> chunk = bootstrap.run(network, statistic, n_draws=10, seed=1)
    >>> band = bootstrap.combine([chunk], statistic, quantile=0.9)
    >>> band.n_draws, band.quantile
    (10, 0.9)
    """

    def __init__(
        self, filtration: Literal["sublevel", "superlevel"], *, se_attr: str = "se"
    ) -> None:
        _check_filtration(filtration)
        self.filtration = filtration
        self.se_attr = se_attr
        self._nominal: np.ndarray | None = None
        self._network_fingerprint: str | None = None

    def _require_bound(self) -> tuple[np.ndarray, str]:
        if self._nominal is None or self._network_fingerprint is None:
            raise ValueError("statistic is not bound to a network; call prepare(network) first.")
        return self._nominal, self._network_fingerprint

    def prepare(self, network: ScalarNetwork) -> BottleneckToNominal:
        """Return a statistic bound to ``network``; raise if this one is bound to another."""
        # The public diagram validates the network (simple graph); the helper builds the
        # nominal diagram so that it and every draw's diagram come from one construction.
        ScalarNetworkTopology(network).persistence_diagram(self.filtration)
        network_hash = band_network_fingerprint(network, se_attr=self.se_attr)
        if self._network_fingerprint is not None:
            if self._network_fingerprint != network_hash:
                raise ValueError("this statistic is already bound to another network.")
            return self
        bound = BottleneckToNominal(self.filtration, se_attr=self.se_attr)
        bound._nominal = _h0_diagram_points(network.graph, network.scalar_attr, self.filtration)
        bound._nominal.setflags(write=False)
        bound._network_fingerprint = network_hash
        return bound

    def measure(self, network: DrawnNetwork) -> tuple[Counts, np.ndarray]:
        """Return ``({}, [distance])`` for one drawn landscape."""
        nominal, _ = self._require_bound()
        points = _h0_diagram_points(network.graph, network.scalar_attr, self.filtration)
        distance = diagram_bottleneck_distance(_as_diagram(points), _as_diagram(nominal))
        return {}, np.array([distance], dtype=np.float64)

    def add(self, first: Counts, second: Counts) -> Counts:
        """Return the sum of two counts mappings; there are no counts, so it is empty."""
        return {}

    def finish(
        self,
        counts: Counts,
        stats: np.ndarray,
        n_draws: int,
        *,
        seed: int,
        quantile: float = 0.95,
        **options: Any,
    ) -> BootstrapPersistenceBand:
        """Build the `BootstrapPersistenceBand` from the per-draw distances (stats column 0)."""
        _, network_hash = self._require_bound()
        if options:
            raise TypeError(f"unexpected finish options: {sorted(options)}.")
        return BootstrapPersistenceBand.from_distances(
            np.asarray(stats, dtype=np.float64)[:, 0],
            quantile=quantile,
            filtration=self.filtration,
            n_draws=n_draws,
            seed=seed,
            se_attr=self.se_attr,
            network_fingerprint=network_hash,
        )

    def fingerprint(self) -> str:
        """Hash the filtration, diagram convention, nominal diagram, network and ``se_attr``."""
        nominal, network_hash = self._require_bound()
        digest = hashlib.sha256()
        digest.update(f"band:{self.filtration}:{_DIAGRAM_CONVENTION}:{self.se_attr}:".encode())
        digest.update(np.ascontiguousarray(nominal, dtype="<f8").tobytes())
        digest.update(network_hash.encode())
        return digest.hexdigest()


def bootstrap_persistence_band(
    network: ScalarNetwork,
    *,
    filtration: Literal["sublevel", "superlevel"],
    n_draws: int,
    seed: int,
    quantile: float = 0.95,
    se_attr: str = "se",
    n_jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> BootstrapPersistenceBand:
    """Compute the bootstrap persistence band of a scalar network (Fasy et al. 2014).

    Each bootstrap draw replaces every vertex scalar by ``Normal(scalar, se)`` on the fixed
    graph. Its H0 persistence diagram (ordinary pairs plus the capped global pair) is compared
    with the nominal network's diagram by the bottleneck distance. The radius ``c`` is the
    ``quantile`` quantile (linear interpolation) of the distances, and the bootstrapped
    persistence threshold is ``2c``: a peak survives if and only if its ordinary H0
    persistence is greater than ``2c``, which is what ``MergeTree.simplify(band.threshold)``
    keeps. If every standard error is zero the radius is 0 and the band is empty; then
    ``MergeTree.simplify`` rejects the threshold, as it does for any non-positive one.

    The draws run on the chunked engine of `topographer.statistics.bootstrap`. For a long
    run, use ``BottleneckToNominal(...).prepare(network)`` with ``bootstrap.run`` slices and
    ``bootstrap.combine``; the result equals this function's, bit for bit.

    Limits. The band reflects scalar perturbations on a fixed graph only. It omits
    uncertainty from graph construction, from thresholds and from the estimation of the
    standard errors. It provides neither calibrated peak-level coverage nor a false-discovery
    rate.

    Reference: Fasy, B. T., Lecci, F., Rinaldo, A., Wasserman, L., Balakrishnan, S. and
    Singh, A. (2014), "Confidence sets for persistence diagrams", Annals of Statistics
    42(6).

    Parameters
    ----------
    network : ScalarNetwork
        Connected simple network; its scalar attribute is the nominal landscape.
    filtration : {"sublevel", "superlevel"}
        ``"superlevel"`` (split tree, peaks) or ``"sublevel"`` (join tree, valleys).
    n_draws : int
        Number of bootstrap draws, at least 1.
    seed : int
        Non-negative integer seed (draw ``d`` uses ``SeedSequence(seed, spawn_key=(d,))``).
    quantile : float, default 0.95
        Band quantile, in ``(0, 1)``.
    se_attr : str, default "se"
        Vertex attribute with finite, non-negative standard errors.
    n_jobs : int, default 1
        Worker processes; with more than one, guard the script entry point with
        ``if __name__ == "__main__":``.
    progress : Callable[[int, int], None] | None, default None
        Called as ``progress(done, total)``.

    Returns
    -------
    BootstrapPersistenceBand
        Radius, threshold and the per-draw distances.

    Raises
    ------
    ValueError
        For an invalid ``filtration``, ``quantile``, ``n_draws``, ``seed`` or ``n_jobs``, or
        invalid standard errors.
    TypeError
        If ``network`` is not a ``ScalarNetwork``.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.examples.cube.network import cube_edge_table, cube_node_table
    >>> from topographer.graph import Graph, ScalarNetwork
    >>> from topographer.statistics import bootstrap_persistence_band
    >>> from topographer.topology import ScalarNetworkTopology
    >>> graph = Graph.from_tables(
    ...     cube_node_table().with_columns(se=pl.lit(0.04)),
    ...     cube_edge_table(),
    ...     vertex_id_col="node_id",
    ...     source_id_col="source",
    ...     target_id_col="target",
    ...     vertex_attribute_cols=("height", "se"),
    ... )
    >>> network = ScalarNetwork(graph, scalar_attr="height")
    >>> band = bootstrap_persistence_band(
    ...     network, filtration="superlevel", n_draws=20, seed=1
    ... )
    >>> band.n_draws, band.quantile, band.threshold == 2 * band.radius
    (20, 0.95, True)
    >>> tree = ScalarNetworkTopology(network).compute_merge_tree("superlevel")
    >>> band_simplified = tree.simplify(band.threshold)
    """
    _check_filtration(filtration)
    statistic = BottleneckToNominal(filtration, se_attr=se_attr).prepare(network)
    chunk = bootstrap.run(
        network,
        statistic,
        se_attr=se_attr,
        n_draws=n_draws,
        seed=seed,
        n_jobs=n_jobs,
        progress=progress,
    )
    return bootstrap.combine([chunk], statistic, quantile=quantile)
