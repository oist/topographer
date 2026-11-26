"""The bootstrap persistence band record (Fasy et al. 2014) and its network fingerprint."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
import hashlib
import math
import numbers
from typing import Literal, Self

import numpy as np

from topographer.graph import ScalarNetwork
from topographer.topology._scalar_view import network_graph, vertex_scalar_values

__all__ = ["BootstrapPersistenceBand"]

_FILTRATIONS = ("sublevel", "superlevel")


def _validated(
    filtration: object, quantile: object, n_draws: object, distances: Iterable[float]
) -> tuple[float, ...]:
    """Check the fields shared by the constructor and `from_distances`; return the distances."""
    if filtration not in _FILTRATIONS:
        raise ValueError(f"filtration must be 'sublevel' or 'superlevel', got {filtration!r}.")
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, numbers.Real)
        or not 0.0 < quantile < 1.0
    ):
        raise ValueError(f"quantile must be a number in (0, 1), got {quantile!r}.")
    if isinstance(n_draws, bool) or not isinstance(n_draws, numbers.Integral) or n_draws < 1:
        raise ValueError(f"n_draws must be a positive integer, got {n_draws!r}.")
    values = tuple(float(value) for value in distances)
    if len(values) != n_draws:
        raise ValueError(f"n_draws is {n_draws} but there are {len(values)} distances.")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("distances must be finite and non-negative.")
    return values


def _radius(distances: tuple[float, ...], quantile: float) -> float:
    return float(np.quantile(np.asarray(distances, dtype=np.float64), quantile, method="linear"))


@dataclass(frozen=True)
class BootstrapPersistenceBand:
    """The bootstrap persistence band of one bootstrap on a scalar network (Fasy et al. 2014).

    The record holds the per-draw bottleneck distances between each bootstrap draw's H0
    persistence diagram and the nominal network's diagram, the radius ``c`` (their ``quantile``
    quantile, linear interpolation) and the bootstrapped persistence threshold ``2c``. A peak
    survives the band if and only if its ordinary H0 persistence is greater than the
    threshold, which is what ``MergeTree.simplify(band.threshold)`` keeps.

    The band reflects scalar perturbations on a fixed graph only. It omits uncertainty from
    graph construction, from thresholds and from the estimation of the standard errors. It
    provides neither calibrated peak-level coverage nor a false-discovery rate.

    Reference: Fasy, B. T., Lecci, F., Rinaldo, A., Wasserman, L., Balakrishnan, S. and
    Singh, A. (2014), "Confidence sets for persistence diagrams", Annals of Statistics
    42(6).

    Attributes
    ----------
    filtration : {"sublevel", "superlevel"}
        Filtration of the diagrams; ``"superlevel"`` is the split tree, ``"sublevel"`` the
        join tree.
    quantile : float
        Band quantile, in ``(0, 1)``.
    radius : float
        The radius ``c``, ``numpy.quantile(distances, quantile, method="linear")``.
    n_draws : int
        Number of bootstrap draws.
    seed : int
        Seed of the bootstrap.
    se_attr : str
        Vertex attribute that held the standard errors.
    network_fingerprint : str
        Hash of the vertex ids, scalars, standard errors and edges of the bootstrapped network.
    distances : tuple[float, ...]
        Per-draw bottleneck distances, in draw order.

    Examples
    --------
    >>> from topographer.topology import BootstrapPersistenceBand
    >>> band = BootstrapPersistenceBand.from_distances(
    ...     [0.0, 0.1, 0.2, 0.3, 0.4],
    ...     quantile=0.5,
    ...     filtration="superlevel",
    ...     n_draws=5,
    ...     seed=7,
    ...     se_attr="se",
    ...     network_fingerprint="example",
    ... )
    >>> round(band.radius, 12), round(band.threshold, 12)
    (0.2, 0.4)
    >>> band.at_quantile(0.75).radius == 0.3
    True
    """

    filtration: Literal["sublevel", "superlevel"]
    quantile: float
    radius: float
    n_draws: int
    seed: int
    se_attr: str
    network_fingerprint: str
    distances: tuple[float, ...]

    def __post_init__(self) -> None:
        distances = _validated(self.filtration, self.quantile, self.n_draws, self.distances)
        object.__setattr__(self, "distances", distances)
        if self.radius != _radius(distances, self.quantile):
            raise ValueError(
                f"radius {self.radius!r} is not the {self.quantile} quantile of the distances "
                f"({_radius(distances, self.quantile)!r})."
            )

    @property
    def threshold(self) -> float:
        """The bootstrapped persistence threshold, ``2 * radius``."""
        return 2.0 * self.radius

    @classmethod
    def from_distances(
        cls,
        distances: Iterable[float],
        *,
        quantile: float,
        filtration: Literal["sublevel", "superlevel"],
        n_draws: int,
        seed: int,
        se_attr: str,
        network_fingerprint: str,
    ) -> Self:
        """Build the band from per-draw distances; the radius is their linear quantile."""
        values = _validated(filtration, quantile, n_draws, distances)
        return cls(
            filtration=filtration,
            quantile=quantile,
            radius=_radius(values, quantile),
            n_draws=n_draws,
            seed=seed,
            se_attr=se_attr,
            network_fingerprint=network_fingerprint,
            distances=values,
        )

    def at_quantile(self, quantile: float) -> Self:
        """Return the band of the same draws at another band quantile."""
        values = _validated(self.filtration, quantile, self.n_draws, self.distances)
        return replace(self, quantile=quantile, radius=_radius(values, quantile))


def band_network_fingerprint(network: ScalarNetwork, *, se_attr: str) -> str:
    """Hash the vertex ids, scalars, standard errors and edges of ``network``.

    Vertices are taken in canonical (node-index) order and edges in edge-index order. The
    standard errors are read from the vertex payloads without further validation; the
    bootstrap engine validates them.
    """
    graph = network_graph(network)
    scalars_by_id = vertex_scalar_values(network)
    se = []
    for index in graph.node_indices():
        payload = graph.get_node_data(index)
        if payload is None or se_attr not in payload:
            raise ValueError(f"Standard error attribute {se_attr!r} not found on vertex payload.")
        se.append(float(payload[se_attr]))
    scalars = [scalars_by_id[vertex_id] for vertex_id in network.vertex_ids]
    digest = hashlib.sha256()
    digest.update(repr(network.vertex_ids).encode())
    digest.update(np.asarray(scalars, dtype="<f8").tobytes())
    digest.update(np.asarray(se, dtype="<f8").tobytes())
    digest.update(np.asarray(graph.edge_list(), dtype="<i8").reshape(-1, 2).tobytes())
    return digest.hexdigest()
