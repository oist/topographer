"""FitnessLandscape specialization of ScalarNetwork."""

from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
from scipy import linalg

from topographer.biology._alphabet import Alphabet
from topographer.graph import Graph, HammingGraph, ScalarNetwork, SequenceGraph
from topographer.topology import (
    MergeTree,
    ScalarNetworkTopology,
    _register_landscape_summary,
    compute_topological_summary,
)


class FitnessLandscape(ScalarNetwork):
    """A ScalarNetwork specialized for biological sequence fitness landscapes.

    Examples
    --------
    >>> from topographer.biology import FitnessLandscape, DNA_ALPHABET
    >>> from topographer.graph import HammingGraph
    >>> import polars as pl
    >>> df = pl.DataFrame({"seq": ["AA", "AC"], "fit": [0.1, 0.9]})
    >>> hg = HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fit",))
    >>> fl = FitnessLandscape(hg, scalar_attr="fit")
    >>> fl.scalar_attr
    'fit'
    """

    def __init__(
        self,
        graph: Graph,
        *,
        scalar_attr: str,
        edge_scalar_attr: str | None = None,
        sequence_attr: str | None = None,
        alphabet: Alphabet | None = None,
    ) -> None:
        if isinstance(graph, (HammingGraph, SequenceGraph)):
            resolved_alphabet = graph.alphabet
            resolved_seq_attr = "sequence"
        elif isinstance(graph, Graph):
            if sequence_attr is None:
                raise ValueError(
                    "sequence_attr is required when constructing FitnessLandscape from generic Graph."
                )
            if alphabet is None:
                raise ValueError(
                    "alphabet is required when constructing FitnessLandscape from generic Graph."
                )
            resolved_alphabet = alphabet
            resolved_seq_attr = sequence_attr
        else:
            raise TypeError(f"Expected Topographer Graph instance, got {type(graph).__name__}.")

        super().__init__(graph, scalar_attr=scalar_attr, edge_scalar_attr=edge_scalar_attr)

        self._alphabet = resolved_alphabet
        self._sequence_attr = resolved_seq_attr

        # Validate sequence attribute on every node
        for idx in self._graph_snapshot.node_indices():
            payload = self._graph_snapshot.get_node_data(idx)
            if resolved_seq_attr not in payload:
                raise ValueError(
                    f"Sequence attribute {resolved_seq_attr!r} missing on vertex payload."
                )
            seq = payload[resolved_seq_attr]
            if not isinstance(seq, str):
                raise TypeError("Sequence must be a string.")
            self._alphabet.validate_sequence(seq)

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet

    @property
    def sequence_attr(self) -> str:
        return self._sequence_attr

    def _manifest_extras(self) -> dict[str, Any]:
        """Store the sequence attribute and alphabet symbols beside the network."""
        return {
            "fitness_landscape": {
                "sequence_attr": self._sequence_attr,
                "alphabet": list(self._alphabet.symbols),
            }
        }

    @classmethod
    def _from_graph(cls, graph: Graph, manifest: dict[str, Any]) -> Self:
        """Restore a landscape from the rebuilt graph; the manifest must hold landscape extras."""
        extras = manifest.get("fitness_landscape")
        if (
            not isinstance(extras, dict)
            or not isinstance(extras.get("sequence_attr"), str)
            or not isinstance(extras.get("alphabet"), list)
        ):
            raise ValueError(
                "The folder holds no fitness landscape extras (sequence_attr and alphabet); "
                "read it with ScalarNetwork.read instead."
            )
        return cls(
            graph,
            scalar_attr=manifest["scalar_attr"],
            edge_scalar_attr=manifest["edge_scalar_attr"],
            sequence_attr=extras["sequence_attr"],
            alphabet=Alphabet(tuple(extras["alphabet"])),
        )

    @classmethod
    def from_scalar_network(
        cls,
        network: ScalarNetwork,
        *,
        sequence_attr: str,
        alphabet: Alphabet,
    ) -> FitnessLandscape:
        """Create FitnessLandscape from existing ScalarNetwork if graph contains sequence data."""
        rx_graph = network.graph
        edge_id_attr = network.edge_id_attr
        edge_ids = (
            [rx_graph.get_edge_data_by_index(i)[edge_id_attr] for i in rx_graph.edge_indices()]
            if edge_id_attr is not None
            else None
        )
        return cls(
            Graph(
                rx_graph,
                vertex_id_attr=network.vertex_id_attr,
                edge_id_attr=edge_id_attr,
                edge_ids=edge_ids,
                vertex_ids=network.vertex_ids,
                vertex_id_to_index={vid: i for i, vid in enumerate(network.vertex_ids)},
                index_to_vertex_id={i: vid for i, vid in enumerate(network.vertex_ids)},
                vertex_attr_names={network.scalar_attr, sequence_attr},
                edge_attr_names={network.edge_scalar_attr} if network.edge_scalar_attr else set(),
            ),
            scalar_attr=network.scalar_attr,
            edge_scalar_attr=network.edge_scalar_attr,
            sequence_attr=sequence_attr,
            alphabet=alphabet,
        )

    def roughness_to_slope_ratio(self) -> float:
        """Compute the roughness-to-slope ratio of this fitness landscape."""
        return roughness_to_slope_ratio(self)

    def sign_epistasis_fraction(self) -> float:
        """Compute the fraction of Hamming squares exhibiting sign epistasis."""
        return sign_epistasis_fraction(self)

    def wildtype_accessibility_fraction(self) -> float:
        """Compute the fraction of monotonic paths from min to max increasing in fitness."""
        return wildtype_accessibility_fraction(self)

    def summary(self) -> pl.DataFrame:
        """Compute a single-row Polars DataFrame of topological invariants and ruggedness metrics."""

        return compute_topological_summary(self)


def roughness_to_slope_ratio(landscape: FitnessLandscape) -> float:
    """Compute the roughness-to-slope ratio (r/s) of a fitness landscape.

    Fits a multidimensional linear regression of fitness on sequence positions
    and returns the residual standard deviation over the average absolute
    slope (additive effect) per mutation.

    Parameters
    ----------
    landscape : FitnessLandscape
        The input biological fitness landscape.

    Returns
    -------
    float
        The roughness-to-slope ratio, 0.0 for purely additive landscapes.

    Examples
    --------
    >>> from topographer.biology import FitnessLandscape, DNA_ALPHABET, roughness_to_slope_ratio
    >>> from topographer.graph import HammingGraph
    >>> import polars as pl
    >>> df = pl.DataFrame({"seq": ["AA", "AC", "CA", "CC"], "fit": [1.0, 2.0, 3.0, 4.0]})
    >>> hg = HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fit",))
    >>> fl = FitnessLandscape(hg, scalar_attr="fit")
    >>> round(roughness_to_slope_ratio(fl), 6)
    0.0
    """
    if not isinstance(landscape, FitnessLandscape):
        raise TypeError(f"Expected FitnessLandscape, got {type(landscape).__name__}")

    graph = landscape._graph_snapshot
    if graph.num_nodes() <= 1:
        return 0.0

    seq_attr = landscape.sequence_attr
    scalar_attr = landscape.scalar_attr

    seqs: list[str] = []
    fits: list[float] = []
    for idx in graph.node_indices():
        payload = graph.get_node_data(idx)
        seqs.append(str(payload[seq_attr]))
        fits.append(float(payload[scalar_attr]))

    L = len(seqs[0])
    cols: list[list[float]] = []
    for j in range(L):
        chars = sorted({s[j] for s in seqs})
        for c in chars[1:]:
            cols.append([1.0 if s[j] == c else 0.0 for s in seqs])

    if not cols:
        return 0.0

    X = np.column_stack([np.ones(len(seqs))] + cols)
    y = np.array(fits, dtype=float)

    beta, _, _, _ = linalg.lstsq(X, y)
    res = y - X @ beta
    roughness = float(np.std(res))
    if np.isclose(roughness, 0.0, atol=1e-12):
        roughness = 0.0

    slope = float(np.mean(np.abs(beta[1:])))
    if slope == 0.0:
        return 0.0 if roughness == 0.0 else float("inf")

    return roughness / slope


def sign_epistasis_fraction(landscape: FitnessLandscape) -> float:
    """Compute the fraction of 2-step mutational paths (quadrangles/squares) with sign epistasis.

    Parameters
    ----------
    landscape : FitnessLandscape
        The input biological fitness landscape.

    Returns
    -------
    float
        Fraction of squares exhibiting simple or reciprocal sign epistasis,
        or 0.0 if no squares exist.

    Examples
    --------
    >>> from topographer.biology import FitnessLandscape, DNA_ALPHABET, sign_epistasis_fraction
    >>> from topographer.graph import HammingGraph
    >>> import polars as pl
    >>> df = pl.DataFrame({"seq": ["AA", "AC", "CA", "CC"], "fit": [1.0, 2.0, 3.0, 4.0]})
    >>> hg = HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fit",))
    >>> fl = FitnessLandscape(hg, scalar_attr="fit")
    >>> sign_epistasis_fraction(fl)
    0.0
    """
    if not isinstance(landscape, FitnessLandscape):
        raise TypeError(f"Expected FitnessLandscape, got {type(landscape).__name__}")

    from topographer.biology._sequences import build_categorical_hamming_faces

    faces = build_categorical_hamming_faces(landscape, sequence_key=landscape.sequence_attr)
    square_faces = [f for f in faces if f.payload.get("face_kind") == "square"]
    if not square_faces:
        return 0.0

    graph = landscape._graph_snapshot
    scalar_attr = landscape.scalar_attr

    sign_epistasis_count = 0
    for sq in square_faces:
        u, v, x, w = sq.vertices
        fu = float(graph.get_node_data(u)[scalar_attr])
        fv = float(graph.get_node_data(v)[scalar_attr])
        fx = float(graph.get_node_data(x)[scalar_attr])
        fw = float(graph.get_node_data(w)[scalar_attr])

        delta_i_u = fv - fu
        delta_i_w = fx - fw
        delta_j_u = fw - fu
        delta_j_v = fx - fv

        if (delta_i_u * delta_i_w < 0.0) or (delta_j_u * delta_j_v < 0.0):
            sign_epistasis_count += 1

    return float(sign_epistasis_count) / float(len(square_faces))


def wildtype_accessibility_fraction(landscape: FitnessLandscape) -> float:
    """Compute the fraction of monotonic paths from global minimum to global maximum increasing in fitness.

    Parameters
    ----------
    landscape : FitnessLandscape
        The input biological fitness landscape.

    Returns
    -------
    float
        Fraction of monotonic paths strictly monotonically increasing in fitness.
        Returns 1.0 if global minimum and maximum are the same genotype.

    Notes
    -----
    Exact ties follow the ascending sweep's symbolic order (fitness, vertex index).
    The start is the first tied global minimum, the lowest vertex index, and the end
    is the last tied global maximum, the highest vertex index. They are the birth
    and death vertices of the extended H0 pair reported by
    ``ScalarNetworkTopology.component_persistence()``; the end is also the
    join-tree root. Vertex index follows the sorted native identity order.

    Examples
    --------
    >>> from topographer.biology import FitnessLandscape, DNA_ALPHABET, wildtype_accessibility_fraction
    >>> from topographer.graph import HammingGraph
    >>> import polars as pl
    >>> df = pl.DataFrame({"seq": ["AA", "AC", "CA", "CC"], "fit": [1.0, 2.0, 3.0, 4.0]})
    >>> hg = HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fit",))
    >>> fl = FitnessLandscape(hg, scalar_attr="fit")
    >>> wildtype_accessibility_fraction(fl)
    1.0
    """
    if not isinstance(landscape, FitnessLandscape):
        raise TypeError(f"Expected FitnessLandscape, got {type(landscape).__name__}")

    graph = landscape._graph_snapshot
    if graph.num_nodes() == 0:
        return 0.0
    if graph.num_nodes() == 1:
        return 1.0

    seq_attr = landscape.sequence_attr
    scalar_attr = landscape.scalar_attr

    nodes = [
        (idx, str(payload[seq_attr]), float(payload[scalar_attr]))
        for idx in graph.node_indices()
        for payload in (graph.get_node_data(idx),)
    ]

    min_node = min(nodes, key=lambda item: (item[2], item[0]))
    max_node = max(nodes, key=lambda item: (item[2], item[0]))
    min_seq, min_fit = min_node[1], min_node[2]
    max_seq, max_fit = max_node[1], max_node[2]

    if min_seq == max_seq:
        return 1.0
    if min_fit >= max_fit:
        return 0.0

    diff_positions = [i for i in range(len(min_seq)) if min_seq[i] != max_seq[i]]
    D = len(diff_positions)
    if D == 0:
        return 1.0
    if D > 20:
        raise ValueError(
            f"Distance {D} between global min and max is too large for exact path enumeration."
        )

    seq_to_fitness = {item[1]: item[2] for item in nodes}
    total_paths = math.factorial(D)

    def make_seq_for_mask(mask: int) -> str:
        chars = list(min_seq)
        for bit in range(D):
            if (mask >> bit) & 1:
                chars[diff_positions[bit]] = max_seq[diff_positions[bit]]
        return "".join(chars)

    dp: dict[int, int] = {0: 1}

    for size in range(1, D + 1):
        for comb in combinations(range(D), size):
            mask = sum(1 << bit for bit in comb)
            s = make_seq_for_mask(mask)
            if s not in seq_to_fitness:
                continue
            fit_s = seq_to_fitness[s]
            acc_count = 0
            for bit in comb:
                prev_mask = mask ^ (1 << bit)
                if dp.get(prev_mask, 0) > 0:
                    prev_s = make_seq_for_mask(prev_mask)
                    if fit_s > seq_to_fitness[prev_s]:
                        acc_count += dp[prev_mask]
            if acc_count > 0:
                dp[mask] = acc_count

    full_mask = (1 << D) - 1
    accessible_paths = dp.get(full_mask, 0)
    return float(accessible_paths) / float(total_paths)


class FitnessLandscapeTopology(ScalarNetworkTopology):
    """Topology workspace specialized for FitnessLandscapes.

    Examples
    --------
    >>> from topographer.biology import FitnessLandscape, FitnessLandscapeTopology, DNA_ALPHABET
    >>> from topographer.graph import HammingGraph
    >>> import polars as pl
    >>> df = pl.DataFrame({"seq": ["AA", "AC"], "fit": [0.1, 0.9]})
    >>> hg = HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fit",))
    >>> fl = FitnessLandscape(hg, scalar_attr="fit")
    >>> fl_top = FitnessLandscapeTopology(fl)
    >>> fl_top.landscape is fl
    True
    """

    _network_type = FitnessLandscape

    @classmethod
    def _read_network(cls, folder: Path) -> FitnessLandscape:
        """Read the nested network; point to ``ScalarNetworkTopology.read`` for a plain one."""
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
        except (OSError, ValueError):
            manifest = None  # let the network reader report the missing or invalid manifest
        if isinstance(manifest, dict) and "fitness_landscape" not in manifest:
            raise ValueError(
                "The workspace holds no fitness landscape extras (sequence_attr and alphabet); "
                "read it with ScalarNetworkTopology.read instead."
            )
        return super()._read_network(folder)

    def __init__(self, landscape: FitnessLandscape) -> None:
        if not isinstance(landscape, FitnessLandscape):
            raise TypeError(f"Expected FitnessLandscape, got {type(landscape).__name__}")
        super().__init__(landscape)
        self._landscape = landscape

    @property
    def landscape(self) -> FitnessLandscape:
        return self._landscape

    def roughness_to_slope_ratio(self) -> float:
        """Compute the roughness-to-slope ratio of the underlying fitness landscape."""
        return roughness_to_slope_ratio(self._landscape)

    def sign_epistasis_fraction(self) -> float:
        """Compute the fraction of Hamming squares exhibiting sign epistasis."""
        return sign_epistasis_fraction(self._landscape)

    def wildtype_accessibility_fraction(self) -> float:
        """Compute the fraction of monotonic paths from min to max increasing in fitness."""
        return wildtype_accessibility_fraction(self._landscape)

    def summary(self) -> pl.DataFrame:
        """Compute a single-row Polars DataFrame of topological invariants and ruggedness metrics."""

        return compute_topological_summary(self)

    def annotate_tree_sequences(self, tree: MergeTree) -> pl.DataFrame:
        """Convenience method returning sequence annotations for tree mapping branch vertices.

        Examples
        --------
        >>> from topographer.biology import FitnessLandscape, FitnessLandscapeTopology, DNA_ALPHABET
        >>> from topographer.graph import HammingGraph
        >>> import polars as pl
        >>> df = pl.DataFrame({"seq": ["AA", "AC"], "fit": [0.1, 0.9]})
        >>> hg = HammingGraph(df, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fit",))
        >>> fl = FitnessLandscape(hg, scalar_attr="fit")
        >>> fl_top = FitnessLandscapeTopology(fl)
        >>> join = fl_top.compute_merge_tree("sublevel")
        >>> res = fl_top.annotate_tree_sequences(join)
        >>> isinstance(res, pl.DataFrame)
        True
        """
        return tree.mapping.branch_vertices


def _landscape_summary_columns(target: Any) -> dict[str, float] | None:
    """Columns that ``compute_topological_summary`` appends for a fitness landscape."""
    if isinstance(target, FitnessLandscapeTopology):
        landscape: FitnessLandscape | None = target.landscape
    elif isinstance(target, FitnessLandscape):
        landscape = target
    elif isinstance(target, ScalarNetworkTopology) and isinstance(
        target.network, FitnessLandscape
    ):
        landscape = target.network
    else:
        return None
    return {
        "roughness_to_slope_ratio": roughness_to_slope_ratio(landscape),
        "sign_epistasis_fraction": sign_epistasis_fraction(landscape),
        "wildtype_accessibility_fraction": wildtype_accessibility_fraction(landscape),
    }


_register_landscape_summary(_landscape_summary_columns)
