"""Cumulative sequence annotations for reduced join and split trees."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import heapq
import json
from typing import Any

import polars as pl
import rustworkx as rx

from topographer.biology._sequences import (
    hamming_distance,
    has_stop_codon,
    parse_vertex_chain_json,
    shannon_entropy_bits_from_counts,
)
from topographer.graph import ScalarNetwork
from topographer.topology import (
    MergeTree,
    _fold_tree,
    _losing_children,
    _pair_value,
    _rooted_merge_tree,
    _RootedTree,
    build_tree_extended_persistence_pair_table,
    build_tree_metrics,
)

__all__ = ["TreeLineageMetrics", "build_tree_lineage_metrics"]


@dataclass(frozen=True, slots=True)
class TreeLineageMetrics:
    """Sequence-aware branch, critical-node, and persistence annotations.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.biology import TreeLineageMetrics
    >>> empty = pl.DataFrame()
    >>> metrics = TreeLineageMetrics(empty, empty, empty)
    >>> metrics.branch_table.is_empty()
    True
    """

    branch_table: pl.DataFrame
    critical_node_table: pl.DataFrame
    persistence_table: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _SequenceAggregate:
    """Immutable per-component sequence statistics; empty counts mean identity."""

    stop_codon_count: int
    dna_counts: tuple[Counter[str], ...]
    aa_counts: tuple[Counter[str], ...]

    @classmethod
    def empty(cls) -> _SequenceAggregate:
        return cls(0, (), ())

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, dna_key: str, aa_key: str
    ) -> _SequenceAggregate:
        dna = str(payload[dna_key])
        aa = str(payload[aa_key])
        return cls(
            int(has_stop_codon(aa)),
            tuple(Counter({symbol: 1}) for symbol in dna),
            tuple(Counter({symbol: 1}) for symbol in aa),
        )

    def combine(self, other: _SequenceAggregate) -> _SequenceAggregate:
        if not other.dna_counts and not other.aa_counts:
            return _SequenceAggregate(
                self.stop_codon_count + other.stop_codon_count, self.dna_counts, self.aa_counts
            )
        if not self.dna_counts and not self.aa_counts:
            return _SequenceAggregate(
                self.stop_codon_count + other.stop_codon_count, other.dna_counts, other.aa_counts
            )
        return _SequenceAggregate(
            self.stop_codon_count + other.stop_codon_count,
            tuple(a + b for a, b in zip(self.dna_counts, other.dna_counts, strict=True)),
            tuple(a + b for a, b in zip(self.aa_counts, other.aa_counts, strict=True)),
        )

    def metrics(self) -> dict[str, int | float | bool]:
        return {
            "lineage_stop_codon_count": self.stop_codon_count,
            "has_stop_codon": self.stop_codon_count > 0,
            "dna_entropy_bits": _total_entropy(self.dna_counts),
            "aa_entropy_bits": _total_entropy(self.aa_counts),
        }


def build_tree_lineage_metrics(
    network: ScalarNetwork,
    tree: MergeTree,
    *,
    dna_key: str = "dna_sequence",
    aa_key: str = "aa_sequence",
    label_attr: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TreeLineageMetrics:
    """Build cumulative sequence metrics for a network's merge tree.

    Sequence attributes are read from the declared network graph and assigned
    through the merge tree mapping. Output rows carry public vertex IDs wherever
    a row represents an input vertex; tree-local node indices remain only as
    references to merge-tree events. A sequence containing several stop symbols
    contributes once. Ordinary pairs use the losing pre-saddle component;
    extended pairs use the complete root component.

    Parameters
    ----------
    network
        Source scalar network with aligned DNA and amino-acid vertex attributes.
    tree
        Merge tree derived from this network, with matching mapping and scalars.
    dna_key
        Name of the DNA sequence attribute.
    aa_key
        Name of the amino-acid sequence attribute used for stop detection.
    label_attr
        Attribute used for persistence endpoint labels; defaults to ``dna_key``.
    progress_callback
        Optional callback receiving completed and total event counts.

    Returns
    -------
    TreeLineageMetrics
        Inspectable branch, critical-node, and persistence tables.
        ``branch_vertex_count`` and ``branch_stop_codon_count``
        include the branch's starting critical vertex and its regular
        vertices and exclude the ending saddle; ``lineage_vertex_count`` and
        ``lineage_stop_codon_count`` are cumulative over the component before
        the next event. The root vertex is never on a branch. Public endpoint
        identities preserve their original types.

    Raises
    ------
    ValueError
        If the tree and network disagree or sequence attributes are missing
        or unaligned.
    TypeError
        If inputs or sequence attributes have unsupported types.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.biology import build_tree_lineage_metrics
    >>> from topographer.graph import Graph, ScalarNetwork
    >>> from topographer.topology import ScalarNetworkTopology
    >>> vertices = pl.DataFrame(
    ...     {
    ...         "id": [0, 1, 2],
    ...         "fitness": [0.0, 1.0, 0.5],
    ...         "dna_sequence": ["GCT", "GCA", "TAA"],
    ...         "aa_sequence": ["A", "A", "*"],
    ...     }
    ... )
    >>> edges = pl.DataFrame({"source": [0, 0], "target": [1, 2]})
    >>> graph = Graph.from_tables(
    ...     vertices,
    ...     edges,
    ...     vertex_id_col="id",
    ...     source_id_col="source",
    ...     target_id_col="target",
    ...     vertex_attribute_cols=("fitness", "dna_sequence", "aa_sequence"),
    ... )
    >>> network = ScalarNetwork(graph, scalar_attr="fitness")
    >>> tree = ScalarNetworkTopology(network).split_tree()
    >>> metrics = build_tree_lineage_metrics(network, tree)
    >>> metrics.branch_table.columns[:6]
    ['source_tree_node_index', 'target_tree_node_index', 'source_vertex_id', 'target_vertex_id', 'branch_vertex_count', 'branch_stop_codon_count']
    """
    if not isinstance(network, ScalarNetwork):
        raise TypeError("network must be a ScalarNetwork.")
    if not isinstance(tree, MergeTree):
        raise TypeError("tree must be a MergeTree.")
    if tree.scalar_attr != network.scalar_attr:
        raise ValueError(
            f"Tree scalar attribute {tree.scalar_attr!r} does not match network scalar "
            f"attribute {network.scalar_attr!r}."
        )

    # Tree construction records normalized graph indices internally. The
    # defensive network graph is rebuilt in the same canonical vertex order;
    # resolve indices to original IDs through the public vertex_ids sequence.
    graph = network.graph
    vertex_payloads = {
        int(index): dict(graph.get_node_data(index)) for index in graph.node_indices()
    }
    public_id_by_index = dict(zip(graph.node_indices(), network.vertex_ids, strict=True))
    tree_graph = tree.graph
    mapping = tree.mapping
    branch_members: dict[tuple[int, int], set[int]] = {}
    for row in mapping.branch_vertices.iter_rows(named=True):
        branch_members.setdefault(
            (int(row["source_tree_node_index"]), int(row["target_tree_node_index"])), set()
        ).add(int(row["vertex_index"]))
    critical_members = {
        int(row["tree_node_index"]): int(row["vertex_index"])
        for row in mapping.critical_vertices.iter_rows(named=True)
    }
    tree_critical_members = {
        int(index): int(data["vertex_index"])
        for index in tree_graph.node_indices()
        if (data := dict(tree_graph.get_node_data(index))).get("vertex_index") is not None
    }
    if critical_members != tree_critical_members:
        raise ValueError("Merge-tree critical mapping does not match its tree graph.")

    def enrich(payload: dict[str, Any], *, include_identity: bool = True) -> dict[str, Any]:
        result = dict(payload)
        vertex_index = result.get("vertex_index")
        if vertex_index is None:
            return result
        source = vertex_payloads.get(int(vertex_index))
        if source is None:
            raise ValueError(
                f"Tree mapping references unknown network vertex index {vertex_index!r}."
            )
        if network.scalar_attr not in source:
            raise ValueError(
                f"Network vertex is missing scalar attribute {network.scalar_attr!r}."
            )
        if "scalar" in result and float(result["scalar"]) != float(source[network.scalar_attr]):
            raise ValueError(
                f"Merge tree scalar at vertex {vertex_index!r} does not match the network."
            )
        result.update(
            {key: source[key] for key in (dna_key, aa_key, label_attr or dna_key) if key in source}
        )
        vertex_id = public_id_by_index[int(vertex_index)]
        original_vertex_id = result.get("vertex_id")
        if original_vertex_id is not None and str(original_vertex_id) != str(vertex_id):
            raise ValueError(
                f"Merge tree vertex identity {original_vertex_id!r} does not match "
                f"network vertex identity {vertex_id!r}."
            )
        if include_identity:
            result["vertex_id"] = vertex_id
            result["label"] = str(vertex_id)
        else:
            result.pop("vertex_id", None)
        return result

    reversed_tree = rx.PyDiGraph()
    for index in tree_graph.node_indices():
        reversed_tree.add_node(enrich(dict(tree_graph.get_node_data(index))))
    for source, target, data in tree_graph.weighted_edge_list():
        edge_payload = dict(data or {})
        chain = parse_vertex_chain_json(edge_payload.get("vertex_chain_json"))
        expected = branch_members.get((int(source), int(target)), set())
        if chain:
            observed = {int(item["vertex_index"]) for item in chain[1:-1]}
            if expected != observed:
                raise ValueError("Merge-tree branch mapping does not match its edge chain.")
            edge_payload["vertex_chain_json"] = json.dumps(
                [enrich(item, include_identity=False) for item in chain]
            )
        elif expected:
            raise ValueError("Merge-tree branch mapping has members absent from its edge chain.")
        reversed_tree.add_edge(int(target), int(source), edge_payload)

    result = _build_sequence_tables(
        tree,
        reversed_tree,
        dna_key=dna_key,
        aa_key=aa_key,
        label_attr=label_attr or dna_key,
        progress_callback=progress_callback,
    )
    public_id_by_tree_node = {
        int(index): enrich(dict(tree_graph.get_node_data(index))).get("vertex_id")
        for index in tree_graph.node_indices()
    }
    branch = result.branch_table
    if not branch.is_empty():
        sources = branch["source_tree_node_index"].to_list()
        targets = branch["target_tree_node_index"].to_list()
        branch = branch.with_columns(
            _identity_series(
                "source_vertex_id", [public_id_by_tree_node[int(index)] for index in sources]
            ),
            _identity_series(
                "target_vertex_id", [public_id_by_tree_node[int(index)] for index in targets]
            ),
        )
        branch = branch.select(
            [
                "source_tree_node_index",
                "target_tree_node_index",
                "source_vertex_id",
                "target_vertex_id",
                *[
                    column
                    for column in branch.columns
                    if column
                    not in (
                        "source_tree_node_index",
                        "target_tree_node_index",
                        "source_vertex_id",
                        "target_vertex_id",
                    )
                ],
            ]
        )
    if not result.critical_node_table.is_empty():
        node_indices = result.critical_node_table["tree_node_index"].to_list()
        critical = result.critical_node_table.with_columns(
            _identity_series(
                "vertex_id", [public_id_by_tree_node[int(index)] for index in node_indices]
            )
        )
    else:
        critical = result.critical_node_table
    if not result.persistence_table.is_empty():
        births = result.persistence_table["birth_tree_node_index"].to_list()
        deaths = result.persistence_table["death_tree_node_index"].to_list()
        persistence = result.persistence_table.with_columns(
            _identity_series(
                "birth_vertex_id", [public_id_by_tree_node[int(index)] for index in births]
            ),
            _identity_series(
                "death_vertex_id", [public_id_by_tree_node[int(index)] for index in deaths]
            ),
        )
    else:
        persistence = result.persistence_table
    return TreeLineageMetrics(branch, critical, persistence)


def _identity_series(name: str, values: list[Any]) -> pl.Series:
    """Preserve arbitrary hashable graph identities at the Polars boundary."""
    return pl.Series(name, values, dtype=pl.Object)


def _build_sequence_tables(
    tree: MergeTree,
    leaf_to_root: Any,
    *,
    dna_key: str,
    aa_key: str,
    label_attr: str,
    progress_callback: Callable[[int, int], None] | None,
) -> TreeLineageMetrics:
    """Fold sequence statistics over the shared rooted tree and join topology counts.

    ``leaf_to_root`` is the enriched leaf-to-root copy of the tree graph. Branch
    rows describe the active component immediately before its next critical
    event; critical-node rows describe the component after that event has been
    added. Ordinary persistence rows use the losing pre-saddle aggregate, while
    extended rows use the complete root aggregate.
    """

    topology = build_tree_metrics(tree)
    rooted = _rooted_merge_tree(tree)
    node_data = {
        int(node): dict(leaf_to_root.get_node_data(node)) for node in leaf_to_root.node_indices()
    }
    chains: dict[int, list[dict[str, Any]]] = {}
    for child, parent, data in leaf_to_root.weighted_edge_list():
        chains[int(child)] = parse_vertex_chain_json(dict(data).get("vertex_chain_json")) or [
            node_data[int(child)],
            node_data[int(parent)],
        ]
    _validate_payloads(
        [*node_data.values(), *(payload for chain in chains.values() for payload in chain)],
        dna_key=dna_key,
        aa_key=aa_key,
    )
    extremum_event = "minimum" if tree.kind == "join" else "maximum"
    for node, kids in rooted.children.items():
        if not kids and str(node_data[node].get("event")) != extremum_event:
            raise ValueError(f"Merge-tree source node {node} must be a {extremum_event}.")

    def vertex_value(node: int) -> _SequenceAggregate:
        return _SequenceAggregate.from_payload(node_data[node], dna_key=dna_key, aa_key=aa_key)

    def regular_value(child: int) -> _SequenceAggregate:
        aggregate = _SequenceAggregate.empty()
        for payload in chains[child][1:-1]:
            aggregate = aggregate.combine(
                _SequenceAggregate.from_payload(payload, dna_key=dna_key, aa_key=aa_key)
            )
        return aggregate

    fold = _fold_tree(
        rooted,
        vertex_value=vertex_value,
        regular_value=regular_value,
        combine=_SequenceAggregate.combine,
        progress_callback=progress_callback,
    )

    branch_rows: list[dict[str, Any]] = []
    for node in _parent_order(rooted):
        for child in rooted.children[node]:
            child_data, node_payload = node_data[child], node_data[node]
            branch_rows.append(
                {
                    "source_tree_node_index": child,
                    "target_tree_node_index": node,
                    "synonymous_endpoints": int(child_data[aa_key] == node_payload[aa_key]),
                    "dna_hamming": hamming_distance(child_data[dna_key], node_payload[dna_key]),
                    "aa_hamming": hamming_distance(child_data[aa_key], node_payload[aa_key]),
                    "branch_stop_codon_count": fold.branch_local[child].stop_codon_count,
                    **fold.branch_lineage[child].metrics(),
                }
            )
    critical_rows = [
        {"tree_node_index": node, **aggregate.metrics()}
        for node, aggregate in sorted(fold.post_node.items())
    ]
    scalar = {node: float(data["scalar"]) for node, data in node_data.items()}
    vertex = {node: int(data.get("vertex_index", node)) for node, data in node_data.items()}
    losing = _losing_children(rooted, tree_kind=tree.kind, scalar=scalar, vertex=vertex)
    persistence = build_tree_extended_persistence_pair_table(
        leaf_to_root, tree_kind=tree.kind, label_attr=label_attr
    )
    pair_rows = []
    for row in persistence.to_dicts():
        birth, death = int(row["birth_tree_node_index"]), int(row["death_tree_node_index"])
        aggregate = _pair_value(
            row, losing=losing, branch_lineage=fold.branch_lineage, post_node=fold.post_node
        )
        row.update(aggregate.metrics())
        row["synonymous_endpoints"] = int(node_data[birth][aa_key] == node_data[death][aa_key])
        row["dna_hamming"] = hamming_distance(node_data[birth][dna_key], node_data[death][dna_key])
        row["aa_hamming"] = hamming_distance(node_data[birth][aa_key], node_data[death][aa_key])
        pair_rows.append(row)

    metric_columns = [
        "lineage_stop_codon_count",
        "has_stop_codon",
        "dna_entropy_bits",
        "aa_entropy_bits",
    ]
    branch = _join_topology(
        pl.DataFrame(branch_rows),
        topology.branch_table,
        keys=["source_tree_node_index", "target_tree_node_index"],
        leading=["source_tree_node_index", "target_tree_node_index"],
        trailing=["synonymous_endpoints", "dna_hamming", "aa_hamming"],
        metric_columns=metric_columns,
    )
    critical = _join_topology(
        pl.DataFrame(critical_rows),
        topology.critical_node_table,
        keys=["tree_node_index"],
        leading=["tree_node_index"],
        trailing=[],
        metric_columns=metric_columns,
    )
    pairs = pl.DataFrame(pair_rows)
    pairs = _join_topology(
        pairs,
        topology.persistence_table,
        keys=["birth_tree_node_index", "death_tree_node_index"],
        leading=persistence.columns,
        trailing=["synonymous_endpoints", "dna_hamming", "aa_hamming"],
        metric_columns=metric_columns,
    )
    return TreeLineageMetrics(branch, critical, pairs)


def _join_topology(
    table: pl.DataFrame,
    topology: pl.DataFrame,
    *,
    keys: list[str],
    leading: list[str],
    trailing: list[str],
    metric_columns: list[str],
) -> pl.DataFrame:
    """Left-join topology counts onto a sequence table and fix the column order."""

    if table.is_empty():
        return table
    topology_columns = [
        column
        for column in ("branch_vertex_count", "lineage_vertex_count", "horton_strahler_order")
        if column in topology.columns
    ]
    joined = table.join(
        topology.select([*keys, *topology_columns]),
        on=keys,
        how="left",
        maintain_order="left",
    )
    if joined.height != table.height or any(
        joined[column].null_count() for column in topology_columns
    ):
        raise ValueError("Tree topology metrics do not match the annotated tree rows.")
    counts = [
        column
        for column in (
            "branch_vertex_count",
            "branch_stop_codon_count",
            "lineage_vertex_count",
            "lineage_stop_codon_count",
        )
        if column in joined.columns
    ]
    metrics = [column for column in metric_columns if column not in counts]
    return joined.select([*leading, *counts, *metrics, "horton_strahler_order", *trailing])


def _parent_order(rooted: _RootedTree) -> list[int]:
    """Order nodes smallest-index-first once all their children are done."""

    remaining = {node: len(kids) for node, kids in rooted.children.items()}
    ready = [node for node, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        parent = rooted.parent[node]
        if parent is not None:
            remaining[parent] -= 1
            if remaining[parent] == 0:
                heapq.heappush(ready, parent)
    return order


def _validate_payloads(payloads: Any, *, dna_key: str, aa_key: str) -> None:
    lengths: dict[str, set[int]] = {dna_key: set(), aa_key: set()}
    for payload in payloads:
        for key in (dna_key, aa_key):
            if payload.get(key) is None:
                raise ValueError(f"Missing required {key!r} sequence in represented tree vertex.")
            if not isinstance(payload[key], str):
                raise TypeError(f"Required {key!r} sequence must be a string.")
            lengths[key].add(len(payload[key]))
    for key, observed in lengths.items():
        if len(observed) > 1:
            raise ValueError(f"All represented {key!r} sequences must have the same length.")


def _total_entropy(site_counts: tuple[Counter[str], ...]) -> float:
    return sum(shannon_entropy_bits_from_counts(list(counts.values())) for counts in site_counts)
