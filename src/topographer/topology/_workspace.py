"""ScalarNetworkTopology selective derivation workspace."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, ClassVar, Literal, Self

import polars as pl

from topographer.graph import ScalarNetwork
from topographer.graph._archive import (
    WORKSPACE_FORMAT,
    create_folder,
    read_manifest,
    read_table,
    require_keys,
    write_manifest,
)
from topographer.topology._band import BootstrapPersistenceBand, band_network_fingerprint
from topographer.topology._merge_tree import MergeTree, construct_oriented_merge_tree
from topographer.topology._network_persistence import (
    PersistenceResult,
    compute_component_persistence,
    compute_persistence_diagram,
)
from topographer.topology._scalar_view import network_graph, vertex_scalar_values


def _validate_simple_graph_for_persistence(network: ScalarNetwork) -> None:
    rx_graph = network_graph(network)
    for u, v in rx_graph.edge_list():
        if u == v:
            raise ValueError(
                "Network extended persistence requires a simple graph without self-loops."
            )
    if getattr(rx_graph, "multigraph", False) or rx_graph.num_edges() > 0:
        seen_edges: set[tuple[int, int]] = set()
        for u, v in rx_graph.edge_list():
            canonical = (u, v) if u <= v else (v, u)
            if canonical in seen_edges:
                raise ValueError(
                    "Network extended persistence requires a simple graph without parallel edges."
                )
            seen_edges.add(canonical)


NETWORK_FOLDER = "network"
TREES_FOLDER = "trees"
BAND_TABLE = "band-distances.parquet"
_BAND_CONVENTIONS = {
    "distance": "bottleneck_linf_h0_capped_essential",
    "quantile_method": "linear",
}
_BAND_FIELDS = (
    "filtration",
    "quantile",
    "radius",
    "n_draws",
    "seed",
    "se_attr",
    "network_fingerprint",
)
_TREE_KINDS = {"join": "sublevel", "split": "superlevel"}


class ScalarNetworkTopology:
    """Workspace for deriving join and split merge trees from a ScalarNetwork.


    Trees are selected by filtration: ``"sublevel"`` gives the join tree and
    ``"superlevel"`` the split tree.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> topology = ScalarNetworkTopology(build_cube_scalar_network())
    >>> topology.compute_merge_tree("sublevel").kind
    'join'
    >>> topology.compute_merge_tree("superlevel").kind
    'split'

    """

    _network_type: ClassVar[type[ScalarNetwork]] = ScalarNetwork

    def __init__(self, network: ScalarNetwork) -> None:
        if not isinstance(network, ScalarNetwork):
            raise TypeError(f"Expected ScalarNetwork, got {type(network).__name__}")
        self._network = network
        self._join_tree: MergeTree | None = None
        self._split_tree: MergeTree | None = None
        self._band: BootstrapPersistenceBand | None = None

    @property
    def network(self) -> ScalarNetwork:
        return self._network

    @property
    def band(self) -> BootstrapPersistenceBand | None:
        """The attached bootstrap persistence band, or ``None``."""
        return self._band

    def set_band(self, band: BootstrapPersistenceBand) -> None:
        """Attach the bootstrap persistence band computed for this network.

        Raises
        ------
        TypeError
            If ``band`` is not a ``BootstrapPersistenceBand``.
        ValueError
            If a band is already attached (a band is never replaced) or the band was computed
            for a different network (its fingerprint, recomputed here, does not match).

        Examples
        --------
        >>> import polars as pl
        >>> from topographer.graph import Graph, ScalarNetwork
        >>> from topographer.topology import BootstrapPersistenceBand
        >>> from topographer.topology._band import band_network_fingerprint
        >>> network = ScalarNetwork(
        ...     Graph.from_tables(
        ...         pl.DataFrame({"vid": ["a", "b"], "value": [0.1, 0.9], "se": [0.01, 0.02]}),
        ...         pl.DataFrame({"src": ["a"], "dst": ["b"]}),
        ...         vertex_id_col="vid", source_id_col="src", target_id_col="dst",
        ...         vertex_attribute_cols=["value", "se"],
        ...     ),
        ...     scalar_attr="value",
        ... )
        >>> band = BootstrapPersistenceBand.from_distances(
        ...     [0.1, 0.2], quantile=0.5, filtration="superlevel", n_draws=2, seed=1,
        ...     se_attr="se", network_fingerprint=band_network_fingerprint(network, se_attr="se"),
        ... )
        >>> topology = ScalarNetworkTopology(network)
        >>> topology.band is None
        True
        >>> topology.set_band(band)
        >>> topology.band is band
        True
        """
        if not isinstance(band, BootstrapPersistenceBand):
            raise TypeError(f"Expected BootstrapPersistenceBand, got {type(band).__name__}")
        if self._band is not None:
            raise ValueError("A band is already attached; a workspace holds one band.")
        fingerprint = band_network_fingerprint(self._network, se_attr=band.se_attr)
        if fingerprint != band.network_fingerprint:
            raise ValueError(
                "The band was computed for a different network (fingerprint differs)."
            )
        self._band = band

    def write(self, path: str | os.PathLike[str]) -> None:
        """Write the network, the cached merge trees and the band (if set) to a new folder.

        Only trees that were already computed are stored; nothing is computed here. The folder
        holds ``network/`` (a network folder), ``trees/join`` and ``trees/split`` (tree
        folders, for the cached kinds), ``band-distances.parquet`` (with a band) and
        ``manifest.json``, written last.

        Raises
        ------
        TypeError
            If the network cannot be stored (see ``ScalarNetwork.write``).
        FileExistsError
            If ``path`` already exists.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from topographer.examples import build_cube_scalar_network
        >>> topology = ScalarNetworkTopology(build_cube_scalar_network())
        >>> _ = topology.split_tree()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     topology.write(Path(tmp) / "workspace")
        ...     sorted(p.name for p in (Path(tmp) / "workspace").iterdir())
        ['manifest.json', 'network', 'trees']
        """
        folder = create_folder(path)
        try:
            self._write_contents(folder)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def _write_contents(self, folder: Path) -> None:
        self._network.write(folder / NETWORK_FOLDER)
        trees = {"join": self._join_tree, "split": self._split_tree}
        stored = [kind for kind, tree in trees.items() if tree is not None]
        if stored:
            (folder / TREES_FOLDER).mkdir()
        for kind in stored:
            trees[kind].write(folder / TREES_FOLDER / kind)  # type: ignore[union-attr]
        band = self._band
        if band is not None:
            pl.DataFrame(
                [
                    pl.Series("draw", range(band.n_draws), dtype=pl.Int64),
                    pl.Series("distance", band.distances, dtype=pl.Float64),
                ]
            ).write_parquet(folder / BAND_TABLE)
        write_manifest(
            folder,
            WORKSPACE_FORMAT,
            {
                "workspace_class": type(self).__name__,
                "trees": stored,
                "band": None if band is None else _band_record(band),
            },
        )

    @classmethod
    def _read_network(cls, folder: Path) -> ScalarNetwork:
        """Read the nested network folder as this class's network type."""
        return cls._network_type.read(folder)

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> Self:
        """Read a workspace folder written by `write`; nothing is recomputed.

        The network is restored as the class's network type (a plain ``ScalarNetwork`` here,
        a ``FitnessLandscape`` for ``FitnessLandscapeTopology``), the stored trees become the
        cached trees and the band is attached through the checks of `set_band`.

        ``read`` returns the class it is called on: reading a ``FitnessLandscapeTopology``
        folder with ``ScalarNetworkTopology.read`` is allowed and returns the plain
        ``ScalarNetworkTopology`` and ``ScalarNetwork``, dropping the landscape extras.

        Raises
        ------
        ValueError
            If the folder is not a workspace folder (the message names the ``read`` to call),
            a part is missing or invalid, a tree does not belong to the network (kind,
            ``scalar_attr``, vertex count, scalars or vertex identities differ), or the band
            conventions, quantile or fingerprint do not match.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from topographer.examples import build_cube_scalar_network
        >>> topology = ScalarNetworkTopology(build_cube_scalar_network())
        >>> _ = topology.split_tree()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     topology.write(Path(tmp) / "workspace")
        ...     restored = ScalarNetworkTopology.read(Path(tmp) / "workspace")
        >>> restored.split_tree().persistence().equals(topology.split_tree().persistence())
        True
        """
        folder = Path(path)
        manifest = read_manifest(folder, WORKSPACE_FORMAT)
        require_keys(manifest, ("workspace_class", "trees", "band"))
        kinds = manifest["trees"]
        if (
            not isinstance(kinds, list)
            or len(set(kinds)) != len(kinds)
            or not set(kinds) <= set(_TREE_KINDS)
        ):
            raise ValueError(f"The manifest trees must list 'join' and/or 'split', got {kinds!r}.")
        network = cls._read_network(folder / NETWORK_FOLDER)
        workspace = cls(network)  # type: ignore[arg-type]
        for kind in kinds:
            tree = MergeTree.read(folder / TREES_FOLDER / kind)
            _check_tree_matches_network(tree, kind, network)
            if kind == "join":
                workspace._join_tree = tree
            else:
                workspace._split_tree = tree
        if manifest["band"] is not None:
            workspace.set_band(_read_band(folder, manifest["band"]))
        return workspace

    def join_tree(self) -> MergeTree:
        """Equivalent to ``compute_merge_tree("sublevel")``."""
        return self.compute_merge_tree("sublevel")

    def split_tree(self) -> MergeTree:
        """Equivalent to ``compute_merge_tree("superlevel")``."""
        return self.compute_merge_tree("superlevel")

    def compute_merge_tree(
        self,
        filtration: Literal["sublevel", "superlevel"],
        *,
        tie_break: Literal["vertex_index"] = "vertex_index",
    ) -> MergeTree:
        """Return the cached merge tree for a sublevel or superlevel filtration.

        Equal scalar values are ordered by vertex index. The sublevel filtration
        yields a join tree; the superlevel filtration yields a split tree.
        """
        if filtration not in ("sublevel", "superlevel"):
            raise ValueError(f"filtration must be 'sublevel' or 'superlevel', got {filtration!r}")
        if tie_break != "vertex_index":
            raise ValueError(f"tie_break must be 'vertex_index', got {tie_break!r}")
        if filtration == "sublevel":
            if self._join_tree is None:
                self._join_tree = construct_oriented_merge_tree(self._network, kind="join")
            return self._join_tree
        if self._split_tree is None:
            self._split_tree = construct_oriented_merge_tree(self._network, kind="split")
        return self._split_tree

    def persistence_diagram(
        self,
        filtration: Literal["sublevel", "superlevel"],
        *,
        include_essential: Literal["capped", "null"] = "capped",
    ) -> PersistenceResult:
        """Return H0 pairs in native units for the selected filtration.

        Birth is at most death for sublevel and at least death for superlevel.
        The essential pair dies at the opposite extremum when capped, or has
        NaN death and infinite persistence when null. Ties use vertex index,
        applied to the negated scalar for superlevel filtrations. The network
        must be connected and have no self-loops or parallel edges.

        Examples
        --------
        >>> from topographer.examples import build_cube_scalar_network
        >>> topology = ScalarNetworkTopology(build_cube_scalar_network())
        >>> pairs = topology.persistence_diagram("superlevel").pairs
        >>> pairs.select("pair_kind", "birth_scalar", "death_scalar").rows()
        [('ordinary', 0.9, 0.8), ('essential', 1.0, 0.0)]
        """
        if filtration not in ("sublevel", "superlevel"):
            raise ValueError(f"filtration must be 'sublevel' or 'superlevel', got {filtration!r}")
        if include_essential not in ("capped", "null"):
            raise ValueError(
                f"include_essential must be 'capped' or 'null', got {include_essential!r}"
            )
        _validate_simple_graph_for_persistence(self._network)
        return compute_persistence_diagram(self, filtration, include_essential=include_essential)

    def component_persistence(self) -> PersistenceResult:
        """Compute extended component persistence adapted from join and split trees."""
        _validate_simple_graph_for_persistence(self._network)
        return compute_component_persistence(self)

    def summary(self) -> pl.DataFrame:
        """Compute a single-row Polars DataFrame of topological invariants."""
        from topographer.topology._summary import compute_topological_summary

        return compute_topological_summary(self)


def _band_record(band: BootstrapPersistenceBand) -> dict[str, Any]:
    """The manifest entry of a band; the seed is a decimal string (it may exceed 64 bits)."""
    return {
        "filtration": band.filtration,
        "quantile": band.quantile,
        "radius": band.radius,
        "n_draws": band.n_draws,
        "seed": str(band.seed),
        "se_attr": band.se_attr,
        "network_fingerprint": band.network_fingerprint,
        **_BAND_CONVENTIONS,
    }


def _read_band(folder: Path, record: Any) -> BootstrapPersistenceBand:
    """Rebuild a band from its manifest record and distance table through its constructor."""
    if not isinstance(record, dict):
        raise ValueError("The manifest band must be null or an object.")
    require_keys(record, (*_BAND_FIELDS, *_BAND_CONVENTIONS))
    for key, expected in _BAND_CONVENTIONS.items():
        if record[key] != expected:
            raise ValueError(
                f"Unsupported band {key} {record[key]!r}; this version reads {expected!r}."
            )
    seed = record["seed"]
    if type(seed) is not str or not seed.isdecimal():
        raise ValueError(f"The band seed must be a non-negative decimal string, got {seed!r}.")
    table = read_table(folder, BAND_TABLE, ["draw", "distance"])
    draws = table["draw"].to_list()
    if draws != list(range(len(draws))) or table["distance"].null_count() > 0:
        raise ValueError("The band distance table must hold draws 0..n-1 in order, without nulls.")
    return BootstrapPersistenceBand(
        filtration=record["filtration"],
        quantile=record["quantile"],
        radius=record["radius"],
        n_draws=record["n_draws"],
        seed=int(seed),
        se_attr=record["se_attr"],
        network_fingerprint=record["network_fingerprint"],
        distances=tuple(table["distance"].to_list()),
    )


def _check_tree_matches_network(tree: MergeTree, kind: str, network: ScalarNetwork) -> None:
    """Cheap O(n) check that a restored tree belongs to the restored network."""
    where = f"Stored {kind} tree"
    if tree.kind != kind:
        raise ValueError(f"{where} folder holds a {tree.kind} tree.")
    if tree.scalar_attr != network.scalar_attr:
        raise ValueError(
            f"{where} has scalar_attr {tree.scalar_attr!r}, the network has "
            f"{network.scalar_attr!r}."
        )
    vertex_ids = network.vertex_ids
    scalars = vertex_scalar_values(network)
    graph = tree._augmented_graph
    if graph.num_nodes() != len(vertex_ids):
        raise ValueError(
            f"{where} has {graph.num_nodes()} vertices, the network has {len(vertex_ids)}."
        )
    for graph in (tree._augmented_graph, tree._graph):
        for index in graph.node_indices():
            payload = graph.get_node_data(index)
            vertex = payload["vertex_index"]
            vertex_id = vertex_ids[vertex]
            if payload.get("scalar") != scalars[vertex_id]:
                raise ValueError(
                    f"{where} scalar at vertex_index {vertex} differs from the network."
                )
            if payload.get("vertex_id") != str(vertex_id):
                raise ValueError(
                    f"{where} vertex_id at vertex_index {vertex} differs from the network."
                )
