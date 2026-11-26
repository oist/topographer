"""MergeTree immutable domain object for join and split trees, and tree construction seam."""

from __future__ import annotations

from collections import deque
import os
from typing import Any, Literal, Self
import warnings

import polars as pl
import rustworkx as rx

from topographer.graph import ScalarNetwork, copy_graph_with_payloads
from topographer.topology._mapping import MergeTreeMapping
from topographer.topology._persistence import (
    build_tree_branch_span_table,
    build_tree_extended_persistence_pair_table,
)
from topographer.topology._simplification import (
    build_induced_subtree,
    prune_leaf_branches_by_ordinary_persistence,
)
from topographer.topology._tree_construction import build_join_tree, build_split_tree
from topographer.topology._tree_location import build_tree_location_tables
from topographer.topology._tree_reduction import reduce_merge_tree


def _get_edge_data_safe(tree: rx.PyDiGraph, u: int, v: int) -> dict | None:
    try:
        return tree.get_edge_data(u, v)
    except Exception:
        pass
    try:
        return tree.get_edge_data(v, u)
    except Exception:
        return None


def _orient_root_to_leaves(tree: rx.PyDiGraph, root_idx: int) -> rx.PyDiGraph:
    """Orient tree edges structurally from root toward leaves.

    Children are visited in ascending node-index order so that edge insertion
    order is deterministic across builds; ``PyGraph.neighbors`` itself returns
    an unspecified, per-call order.
    """
    undirected = tree.to_undirected()
    oriented = rx.PyDiGraph()
    for idx in tree.node_indices():
        oriented.add_node(tree.get_node_data(idx))

    # BFS from root, children in ascending node-index order
    visited = {root_idx}
    queue = deque([root_idx])
    while queue:
        curr = queue.popleft()
        for nbr in sorted(undirected.neighbors(curr)):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
                edge_data = _get_edge_data_safe(tree, curr, nbr)
                oriented.add_edge(curr, nbr, edge_data or {})
    return oriented


def _directed_sink(tree: rx.PyDiGraph) -> int:
    """Return the unique node without outgoing edges of a leaf-to-root merge tree.

    The sink is the structural root and does not depend on scalar ties or node
    order. A tree without exactly one sink is malformed and is rejected rather
    than guessed.
    """
    sinks = [int(idx) for idx in tree.node_indices() if tree.out_degree(idx) == 0]
    if len(sinks) != 1:
        raise ValueError("A merge-tree result must have exactly one directed root.")
    return sinks[0]


def construct_oriented_merge_tree(
    network: ScalarNetwork,
    *,
    kind: Literal["join", "split"],
) -> MergeTree:
    """Construct an oriented join or split merge tree from a ScalarNetwork."""
    if kind not in {"join", "split"}:
        raise ValueError(f"kind must be 'join' or 'split', got {kind!r}")

    if kind == "join":
        raw_res = build_join_tree(network.graph, scalar_attr=network.scalar_attr)
    else:
        raw_res = build_split_tree(network.graph, scalar_attr=network.scalar_attr)

    reduced_tree = reduce_merge_tree(raw_res.tree)
    root_node = _directed_sink(reduced_tree)
    oriented = _orient_root_to_leaves(reduced_tree, root_node)
    raw_root = raw_res.root_node_index
    oriented_aug = _orient_root_to_leaves(raw_res.tree, raw_root)
    v_ids = network.vertex_ids
    for g in (oriented, oriented_aug):
        for idx in g.node_indices():
            data = dict(g.get_node_data(idx))
            v_idx = data.get("vertex_index")
            if v_idx is not None and 0 <= int(v_idx) < len(v_ids):
                vid_str = str(v_ids[int(v_idx)])
                data["vertex_id"] = vid_str
                data["label"] = vid_str
                g[idx] = data

    loc_tables = build_tree_location_tables(oriented)
    mapping = MergeTreeMapping(
        branch_vertices=loc_tables.branch_member_table,
        critical_vertices=loc_tables.critical_node_table,
    )
    return _build_merge_tree(
        kind=kind,
        graph=oriented,
        augmented_graph=oriented_aug,
        event_table=raw_res.event_table,
        mapping=mapping,
        scalar_attr=network.scalar_attr,
    )


def _build_merge_tree(
    *,
    kind: Literal["join", "split"],
    graph: rx.PyDiGraph,
    augmented_graph: rx.PyDiGraph,
    event_table: pl.DataFrame,
    mapping: MergeTreeMapping,
    scalar_attr: str,
    original: MergeTree | None = None,
    persistence_threshold: float | None = None,
) -> MergeTree:
    """Build a tree from internally owned construction and provenance data."""
    return MergeTree(
        kind=kind,
        graph=graph,
        augmented_graph=augmented_graph,
        event_table=event_table,
        mapping=mapping,
        scalar_attr=scalar_attr,
        original=original,
        persistence_threshold=persistence_threshold,
        _owned=True,
    )


class MergeTree:
    """Immutable domain object representing a join or split merge tree.

    Examples
    --------
    >>> from topographer.topology import MergeTree
    >>> issubclass(MergeTree, object)
    True
    """

    def __init__(
        self,
        *,
        kind: Literal["join", "split"],
        graph: rx.PyDiGraph,
        augmented_graph: rx.PyDiGraph | None = None,
        event_table: pl.DataFrame,
        mapping: MergeTreeMapping,
        scalar_attr: str,
        original: MergeTree | None = None,
        persistence_threshold: float | None = None,
        _owned: bool = False,
    ) -> None:
        if not _owned and (
            augmented_graph is not None
            or original is not None
            or persistence_threshold is not None
        ):
            warnings.warn(
                "The augmented_graph, original, and persistence_threshold constructor "
                "arguments are deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
        # ``_owned`` is private: internal builders pass graphs nobody else mutates, so the
        # defensive payload copy is skipped. Public callers always get the copy.
        if kind not in {"join", "split"}:
            raise ValueError(f"kind must be 'join' or 'split', got {kind!r}")
        self._kind: Literal["join", "split"] = kind
        take = (lambda g: g) if _owned else copy_graph_with_payloads
        self._graph: rx.PyDiGraph = take(graph)
        self._augmented_graph: rx.PyDiGraph = take(
            augmented_graph if augmented_graph is not None else graph
        )
        self._events: pl.DataFrame = event_table
        self._mapping: MergeTreeMapping = mapping
        self._scalar_attr: str = scalar_attr
        self._original: MergeTree | None = original
        self._persistence_threshold: float | None = persistence_threshold

        self._simplifications: dict[float, MergeTree] = {}
        self._cached_spans: pl.DataFrame | None = None
        self._cached_persistence: pl.DataFrame | None = None
        self._cached_lineages: Any | None = None

    @property
    def kind(self) -> Literal["join", "split"]:
        """Return 'join' or 'split'."""
        return self._kind

    @property
    def graph(self) -> rx.PyDiGraph:
        """Return defensive directed Rustworkx copy of the reduced tree.

        The copy has its own node and edge payloads, so mutating it never alters this tree.
        """
        return copy_graph_with_payloads(self._graph)

    @property
    def augmented_graph(self) -> rx.PyDiGraph:
        """Return defensive directed Rustworkx copy retaining every input vertex.

        The copy has its own node and edge payloads, so mutating it never alters this tree.
        """
        return copy_graph_with_payloads(self._augmented_graph)

    @property
    def mapping(self) -> MergeTreeMapping:
        """Return MergeTreeMapping partition."""
        return self._mapping

    @property
    def events(self) -> pl.DataFrame:
        """Return event table."""
        return self._events

    @property
    def simplifications(self) -> dict[float, MergeTree]:
        """Return cached simplified tree variants."""
        return dict(self._simplifications)

    @property
    def original(self) -> MergeTree | None:
        """Return the original unsimplified tree, or None if this is the original."""
        return self._original

    @property
    def scalar_attr(self) -> str:
        return self._scalar_attr

    def _leaves_to_root_graph(self) -> rx.PyDiGraph:
        """Orient edges from leaves toward root for legacy persistence routines.

        Payloads are shared with the internal tree; treat the result as read-only.
        """
        reversed_graph = rx.PyDiGraph()
        for idx in self._graph.node_indices():
            reversed_graph.add_node(self._graph.get_node_data(idx))
        for u, v, data in self._graph.weighted_edge_list():
            reversed_graph.add_edge(v, u, data)
        return reversed_graph

    def spans(self) -> pl.DataFrame:
        """Return cached branch spans table."""
        if self._cached_spans is None:
            extremum_event = "minimum" if self._kind == "join" else "maximum"
            saddle_event = "join_saddle" if self._kind == "join" else "split_saddle"
            self._cached_spans = build_tree_branch_span_table(
                self._leaves_to_root_graph(),
                extremum_event=extremum_event,
                saddle_event=saddle_event,
            )
        return self._cached_spans

    def persistence(self) -> pl.DataFrame:
        """Return cached extended and ordinary persistence pairs table."""
        if self._cached_persistence is None:
            self._cached_persistence = build_tree_extended_persistence_pair_table(
                self._leaves_to_root_graph(),
                tree_kind=self._kind,
            )
        return self._cached_persistence

    def lineages(self) -> pl.DataFrame:
        """Return generic topological lineages; an alias of :meth:`persistence`."""
        return self.persistence()

    def write(self, path: str | os.PathLike[str]) -> None:
        """Write this tree and its cached simplifications to a new folder.

        The folder holds the reduced tree, the augmented tree, the event table, the surviving
        nodes of every cached simplification and a ``manifest.json`` written last (format
        ``"topographer.merge_tree"``). A simplified tree cannot be written itself.

        Raises
        ------
        ValueError
            If this tree is a simplified variant; write ``tree.original`` instead.
        FileExistsError
            If ``path`` already exists.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from topographer.examples import build_cube_scalar_network
        >>> from topographer.topology import MergeTree, ScalarNetworkTopology
        >>> tree = ScalarNetworkTopology(build_cube_scalar_network()).split_tree()
        >>> _ = tree.simplify(0.5)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tree.write(Path(tmp) / "tree")
        ...     restored = MergeTree.read(Path(tmp) / "tree")
        >>> restored.kind, sorted(restored.simplifications)
        ('split', [0.5])
        """
        from topographer.topology._tree_archive import write_tree

        write_tree(self, path)

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> Self:
        """Read a tree folder written by `write`.

        The reduced and augmented trees, the events and every stored simplification are
        restored without a sweep or a reduction; the vertex mapping is recomputed from the
        restored tree. ``simplify(t)`` for a stored threshold returns the restored variant.

        Raises
        ------
        ValueError
            If the folder, manifest or a table is missing, the format or version is not
            supported, or a table is inconsistent (the message names the problem).

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from topographer.examples import build_cube_scalar_network
        >>> from topographer.topology import MergeTree, ScalarNetworkTopology
        >>> tree = ScalarNetworkTopology(build_cube_scalar_network()).join_tree()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tree.write(Path(tmp) / "tree")
        ...     restored = MergeTree.read(Path(tmp) / "tree")
        >>> restored.persistence().equals(tree.persistence())
        True
        """
        from topographer.topology._tree_archive import read_tree

        return read_tree(path)  # type: ignore[return-value]

    def simplify(self, persistence_threshold: float) -> MergeTree:
        """Return simplified MergeTree variant owned and cached by original."""
        if self._original is not None:
            raise ValueError("Cannot simplify an already simplified tree.")
        if persistence_threshold <= 0:
            raise ValueError("persistence_threshold must be positive.")

        if persistence_threshold in self._simplifications:
            return self._simplifications[persistence_threshold]

        pruned_res = prune_leaf_branches_by_ordinary_persistence(
            self._leaves_to_root_graph(),
            max_persistence=persistence_threshold,
            tree_kind=self._kind,
        )
        return self._build_simplification(pruned_res.kept_nodes, persistence_threshold)

    def _build_simplification(self, kept_nodes: Any, persistence_threshold: float) -> MergeTree:
        """Build, cache and return the variant that keeps the reduced-tree nodes ``kept_nodes``.

        The one code path shared by :meth:`simplify` and by reading a saved tree: induced
        subgraph with original payloads, re-orientation from the root, vertex mapping.
        """
        simplified_rx = build_induced_subtree(self._leaves_to_root_graph(), kept_nodes)

        root_node = _directed_sink(simplified_rx)
        oriented_simplified = _orient_root_to_leaves(simplified_rx, root_node)

        loc_tables = build_tree_location_tables(oriented_simplified)
        new_mapping = MergeTreeMapping(
            branch_vertices=loc_tables.branch_member_table,
            critical_vertices=loc_tables.critical_node_table,
        )

        variant = _build_merge_tree(
            kind=self._kind,
            graph=oriented_simplified,
            augmented_graph=self._augmented_graph,
            event_table=self._events,
            mapping=new_mapping,
            scalar_attr=self._scalar_attr,
            original=self,
            persistence_threshold=persistence_threshold,
        )
        self._simplifications[persistence_threshold] = variant
        return variant
