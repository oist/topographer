"""Sequence helpers: stop codons, Hamming distance, entropy, and Hamming faces."""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl

from topographer.biology._faces import Face

__all__ = [
    "annotate_stop_codons",
    "build_categorical_hamming_faces",
    "hamming_distance",
    "has_stop_codon",
    "parse_vertex_chain_json",
    "shannon_entropy_bits_from_counts",
]

_STOP_CODON_SYMBOLS: frozenset[str] = frozenset({"*", "%", "^", "&"})


def has_stop_codon(sequence: str | None) -> bool:
    """Return whether an amino-acid sequence contains an encoded stop codon.

    Parameters
    ----------
    sequence
        Amino-acid sequence string to test.

    Returns
    -------
    bool
        ``True`` if any character in ``sequence`` is one of the supported stop
        symbols (``*``, ``%``, ``^``, or ``&``); ``False`` otherwise.

    Examples
    --------
    >>> from topographer.biology import has_stop_codon
    >>> has_stop_codon("ADL")
    False
    >>> has_stop_codon("A*L")
    True
    """

    if sequence is None:
        return False
    return bool(_STOP_CODON_SYMBOLS.intersection(str(sequence)))


def annotate_stop_codons(
    table: pl.DataFrame,
    *,
    aa_column: str = "aa_sequence",
    target_column: str = "has_stop_codon",
) -> pl.DataFrame:
    """Add a boolean stop-codon indicator column to a sequence table.

    Parameters
    ----------
    table
        Polars DataFrame containing an amino-acid sequence column.
    aa_column
        Name of the amino-acid sequence column.
    target_column
        Name of the boolean result column to add.

    Returns
    -------
    pl.DataFrame
        Table with ``target_column`` added.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.biology import annotate_stop_codons
    >>> df = pl.DataFrame({"aa_sequence": ["ADL", "A*L"]})
    >>> annotate_stop_codons(df)["has_stop_codon"].to_list()
    [False, True]
    """

    if table.is_empty():
        return table.with_columns(pl.Series(target_column, [], dtype=pl.Boolean))
    if aa_column not in table.columns:
        raise KeyError(f"Column {aa_column!r} is not present in table.")
    pattern = r"[*%^&]"
    return table.with_columns(
        pl.col(aa_column).str.contains(pattern).fill_null(False).alias(target_column)
    )


def hamming_distance(seq_a: str | None, seq_b: str | None) -> int | None:
    """Count differing positions in two equal-length sequences.

    Parameters
    ----------
    seq_a, seq_b
        Strings to compare. ``None`` and unequal-length inputs are unsupported.

    Returns
    -------
    int | None
        The number of mismatched positions, or ``None`` when comparison is invalid.

    Examples
    --------
    >>> hamming_distance("AAC", "AGC")
    1
    >>> hamming_distance("AA", "AAA") is None
    True
    """

    if seq_a is None or seq_b is None:
        return None
    if len(seq_a) != len(seq_b):
        return None
    return sum(char_a != char_b for char_a, char_b in zip(seq_a, seq_b, strict=True))


def shannon_entropy_bits_from_counts(counts: list[int]) -> float:
    """Compute Shannon entropy in bits from observed category counts.

    Parameters
    ----------
    counts
        Non-negative category counts; zero counts contribute nothing.

    Returns
    -------
    float
        ``-sum(p * log2(p))``; an empty or non-positive total returns ``0.0``.

    Examples
    --------
    >>> shannon_entropy_bits_from_counts([1, 1])
    1.0
    >>> shannon_entropy_bits_from_counts([0, 0])
    0.0
    """

    total = sum(int(count) for count in counts)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = int(count) / total
        entropy -= probability * math.log2(probability)
    return float(entropy)


def parse_vertex_chain_json(vertex_chain_json: Any) -> list[dict[str, Any]]:
    """Normalize a GraphML vertex-chain value into payload dictionaries.

    Parameters
    ----------
    vertex_chain_json
        ``None``, a JSON list string, or an already-decoded list.

    Returns
    -------
    list[dict[str, Any]]
        Independent dictionary copies of chain payloads.

    Raises
    ------
    TypeError
        If the decoded value is not a list.

    Examples
    --------
    >>> from topographer.biology import parse_vertex_chain_json
    >>> parse_vertex_chain_json('[{"label": "A"}]')
    [{'label': 'A'}]
    >>> parse_vertex_chain_json(None)
    []
    """

    if vertex_chain_json is None:
        return []
    if isinstance(vertex_chain_json, str):
        parsed = json.loads(vertex_chain_json)
    else:
        parsed = vertex_chain_json
    if not isinstance(parsed, list):
        raise TypeError("vertex_chain_json must decode to a list of vertex payload dictionaries.")
    return [dict(row) for row in parsed]


def build_categorical_hamming_faces(
    network: Any,
    *,
    sequence_key: str = "sequence",
) -> tuple[Any, ...]:
    """Derive elementary polygonal faces from a categorical Hamming sequence network.

    Generates elementary square 4-cycles arising from independent mutations at two positions,
    and elementary triangular 3-cycles from multi-state mutations at a single position.

    Parameters
    ----------
    network:
        Validated ScalarNetwork whose vertex payloads contain sequence strings.
    sequence_key:
        Payload key for the sequence string (or DNA sequence).

    Returns
    -------
    tuple[Face, ...]
        Ordered sequence of Face objects.

    Examples
    --------
    >>> import rustworkx as rx
    >>> from topographer.graph import Graph, ScalarNetwork
    >>> from topographer.biology._sequences import build_categorical_hamming_faces
    >>> g = rx.PyGraph()
    >>> a = g.add_node({"sequence": "00", "f": 0.0}); b = g.add_node({"sequence": "01", "f": 1.0})
    >>> c = g.add_node({"sequence": "10", "f": 2.0}); d = g.add_node({"sequence": "11", "f": 3.0})
    >>> _ = g.add_edge(a, b, {}); _ = g.add_edge(a, c, {}); _ = g.add_edge(b, d, {}); _ = g.add_edge(c, d, {})
    >>> graph = Graph.from_rustworkx(g, vertex_id_attr="sequence", vertex_attribute_cols=("f",))
    >>> net = ScalarNetwork(graph, scalar_attr="f")
    >>> len(build_categorical_hamming_faces(net))
    1
    """
    graph = network.graph
    node_sequences: dict[int, str] = {}
    seq_to_node: dict[str, int] = {}

    for idx in graph.node_indices():
        payload = graph.get_node_data(idx)
        seq = payload.get(sequence_key) or payload.get("dna_sequence")
        if seq is None:
            raise ValueError(f"Vertex {idx} lacks sequence attribute {sequence_key!r}.")
        s_str = str(seq)
        node_sequences[idx] = s_str
        seq_to_node[s_str] = idx

    faces: list[Face] = []
    seen_face_sets: set[frozenset[int]] = set()

    # 1. Square faces from independent pairs of mutations at two positions (i < j)
    # For each sequence s, and each pair of positions (i, j), look for s_i, s_j, and s_ij
    nodes = list(graph.node_indices())
    for u in nodes:
        s = node_sequences[u]
        L = len(s)
        for i in range(L):
            for j in range(i + 1, L):
                # Find all neighbors mutating position i
                neighbors_i = [
                    v
                    for v in graph.neighbors(u)
                    if node_sequences[v][i] != s[i]
                    and all(node_sequences[v][k] == s[k] for k in range(L) if k != i)
                ]
                # Find all neighbors mutating position j
                neighbors_j = [
                    w
                    for w in graph.neighbors(u)
                    if node_sequences[w][j] != s[j]
                    and all(node_sequences[w][k] == s[k] for k in range(L) if k != j)
                ]

                for v in neighbors_i:
                    s_v = node_sequences[v]
                    for w in neighbors_j:
                        s_w = node_sequences[w]
                        # Target sequence must have s_v[i] at position i and s_w[j] at position j
                        s_target_chars = list(s)
                        s_target_chars[i] = s_v[i]
                        s_target_chars[j] = s_w[j]
                        s_target = "".join(s_target_chars)

                        if s_target in seq_to_node:
                            x = seq_to_node[s_target]
                            # Check that x is connected to v and w
                            if (graph.has_edge(v, x) or graph.has_edge(x, v)) and (
                                graph.has_edge(w, x) or graph.has_edge(x, w)
                            ):
                                face_nodes = frozenset([u, v, x, w])
                                if face_nodes not in seen_face_sets:
                                    seen_face_sets.add(face_nodes)
                                    # Canonical simple cycle (u, v, x, w)
                                    faces.append(
                                        Face(
                                            vertices=(u, v, x, w),
                                            face_id=f"sq_{i}_{j}_{u}",
                                            payload={
                                                "positions": (i, j),
                                                "face_kind": "square",
                                            },
                                        )
                                    )

    # 2. Triangular faces from multi-state single positions
    for u in nodes:
        s = node_sequences[u]
        L = len(s)
        for i in range(L):
            # All neighbors differing only at position i
            neighbors_pos_i = [
                v
                for v in graph.neighbors(u)
                if hamming_distance(s, node_sequences[v]) == 1
                and all(node_sequences[v][k] == s[k] for k in range(L) if k != i)
            ]
            for idx_a in range(len(neighbors_pos_i)):
                for idx_b in range(idx_a + 1, len(neighbors_pos_i)):
                    v = neighbors_pos_i[idx_a]
                    w = neighbors_pos_i[idx_b]
                    if graph.has_edge(v, w) or graph.has_edge(w, v):
                        face_nodes = frozenset([u, v, w])
                        if face_nodes not in seen_face_sets:
                            seen_face_sets.add(face_nodes)
                            faces.append(
                                Face(
                                    vertices=(u, v, w),
                                    face_id=f"tri_{i}_{u}",
                                    payload={
                                        "position": i,
                                        "face_kind": "triangle",
                                    },
                                )
                            )

    return tuple(faces)
